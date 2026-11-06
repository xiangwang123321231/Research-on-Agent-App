"""Bounded JS AST literals via a pinned, isolated parser; never execute APK code.

Parsing has no imposed time deadline. Missing Node and unsupported syntax
are explicit coverage gaps, never a noisy blob fallback.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from business_facts import llm_literal_prefilter_pattern

MAX_LITERAL_CHARS = 16 * 1024
WORKER = Path(__file__).with_name("js_literal_worker.cjs")


@dataclass(frozen=True)
class SourceLiteral:
    value: str
    start: int
    end: int
    line: int = 0
    column: int = 0
    module_id: str = ''
    module_byte_offset: int = 0


def javascript_payload(text: str, diagnostics: dict | None = None, *, modules=None, facts=False):
    info = diagnostics if diagnostics is not None else {}
    started = time.monotonic()
    info.update(status="complete", parser="babel-7.26.7", syntax_status="pending",
                scanned_chars=0, literals=0, offset_unit="utf16_code_units")
    try:
        data = text.encode("utf-8")
        arguments = ['--literal-hints', llm_literal_prefilter_pattern()]
        if modules is not None:
            data = json.dumps({'modules': [{'id': m.module_id, 'byte_offset': m.byte_offset, 'source': m.text,
                                           'root': getattr(m, 'asset_root', '')}
                                           for m in modules]}, ensure_ascii=False).encode('utf-8')
            arguments.append('--modules')
        if facts:
            arguments.append('--facts')
        executable = os.environ.get("LLM_JS_NODE") or shutil.which("node")
        if not executable:
            info.update(status="partial", reason="node_unavailable")
            return
        environment = dict(os.environ)
        environment.pop("NODE_OPTIONS", None)
        environment.pop("NODE_PATH", None)
        try:
            process = subprocess.run(
                [executable, str(WORKER), *arguments],
                input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=environment,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
            )
        except OSError:
            info.update(status="partial", reason="node_unavailable")
            return
        if process.returncode:
            info.update(status="partial", reason="parser_process_failed", return_code=process.returncode)
            return
        try:
            payload = json.loads(process.stdout)
            candidates = payload.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError("invalid_worker_output")
        except (ValueError, KeyError, TypeError):
            info.update(status="partial", reason="invalid_parser_output")
            return
        info.update({k: v for k, v in payload.items() if k not in {'candidates', 'facts'}})
        info["candidate_literals"] = len(candidates)
        return payload
    finally:
        info["elapsed_seconds"] = round(time.monotonic() - started, 6)


def javascript_literals(text: str, diagnostics: dict | None = None, *, modules=None):
    payload = javascript_payload(text, diagnostics, modules=modules)
    for item in (payload or {}).get('candidates', []):
        if not isinstance(item.get('value'), str) or len(item['value']) > MAX_LITERAL_CHARS:
            continue
        yield SourceLiteral(item['value'], item['start'], item['end'], item['line'], item['column'],
                            item.get('module_id', ''), item.get('module_byte_offset', 0))
