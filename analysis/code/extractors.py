"""Independent signal extractors — one per analysis layer.

Each extractor is a self-contained class with an ``extract()`` method
that returns ``list[RawSignal]``.  Extractors do **not** depend on each
other and can be tested / benchmarked / replaced individually.
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Iterator

from app_scope import AppCodeScope, expand_shared_app_scope, resolve_app_scope
from business_facts import (scan_llm_business_literal, scan_llm_business_text,
                            scan_llm_literal, scan_llm_json_config)
from hermes_strings import is_hermes_bytecode, hermes_string_literals
from ram_bundle import is_ram_bundle, ram_modules
from signal_types import (
    RawSignal,
    SignalRelation,
    SignalRole,
    SignalScope,
    SignalType,
    SignalVerification,
)
from source_facts import without_comments
from artifact_facts import font_format, is_license_document
from device_model_facts import generation_config_facts, generic_runtime_fact
from bounded_literals import javascript_literals
import time
from webview_context import llm_html_context, page_observation
from registry import (
    # Keyword / pattern constants
    SPECIAL_TOKEN_HINTS,
    PAD_HINTS, DYNAMIC_DOWNLOAD_HINTS, LICENSE_HINTS,
    NATIVE_HINTS,
    LITERT_LM_PACKAGE,
    is_litert_lm_native_library,
    litert_lm_native_symbols,
    MODEL_FILE_EXTENSIONS,
    STRONG_LOCAL_MODEL_HINTS,
    LOCAL_MODEL_TEXT_HINTS, STRONG_MODEL_TEXT_HINTS,
    BACKEND_MARKER_PATS,
    # Shared utilities
    decode_bytes,
    extract_urls,
    classify_url_signal,
    scan_for_api_keys,
    prompt_template_markers,
    scan_aicore_permissions,
    scan_for_baseurl_override,
    scan_for_serverless_url,
    scan_url_for_business_ai_endpoint,
    scan_for_retrofit_ai_paths,
    scan_for_local_split_ai_endpoint,
    framework_keyword_matches,
    scan_blob_for_business_ai_endpoint,
    scan_for_api_key_input_prompt,
    decode_manifest_xml,
    iter_known_sdk_packages,
    iter_contextual_sdk_identities,
    extract_contextual_llm_model_ids,
    extract_contextual_llm_model_evidence,
    extract_binary_llm_model_evidence,
    extract_bounded_binary_model_evidence,
    extract_llm_schema_components,
    extract_resource_model_selector_ids,
    tokenizer_llm_context_components,
    is_reference_only_source,
    is_known_non_llm_model_artifact,
    is_opaque_model_binary,
    has_strong_llm_context,
    match_api_service,
    match_web_service,
    scan_binary_for_known_api_domains,
    _iter_binary_http_urls,
)


def _is_lottie_animation_json(name: str, text: str) -> bool:
    """Identify structured Bodymovin/Lottie JSON before model-text scans."""
    if not name.lower().endswith(".json") or not text.lstrip().startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    required = {"v", "fr", "ip", "op", "w", "h"}
    if not required.issubset(payload):
        return False
    return isinstance(payload.get("v"), str) and any(
        isinstance(payload.get(key), list) for key in ("layers", "assets")
    )


_RELATION_WINDOW_RADIUS = 4 * 1024
_ENDPOINT_CONTEXT_RADIUS = 768
# These formats cannot contain source/configuration text that is useful to
# the classifier.  Checking the suffix before ``decode_bytes`` prevents a
# printable-looking image or media payload from entering the expensive text
# scanners.
_NON_TEXT_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
    ".mp3", ".mp4", ".m4a", ".wav", ".ogg", ".aac", ".flac", ".3gp",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".zip", ".gz", ".7z", ".rar", ".tar", ".bz2", ".xz",
    ".dex", ".so", ".a", ".o", ".class",
})
# Flutter AOT binaries keep platform-channel method names rather than the
# Java-shaped ``webview.loadUrl(...)`` call.  These markers identify the
# WebView load bridge without treating arbitrary ``url`` strings as a call.
_BINARY_WEBVIEW_MARKERS = (
    "webview.loadurl",
    "webviewhostapi.loadurl",
    "loadurlfrominstance",
    "get:loadurl",
)
def _context_around_value(
    text: str,
    value: str,
    radius: int = _ENDPOINT_CONTEXT_RADIUS,
    lower_text: str | None = None,
) -> str:
    """Return the first bounded window containing an exact observed value."""
    lower = lower_text if lower_text is not None else text.lower()
    position = lower.find(value.lower())
    if position < 0:
        return ""
    return text[
        max(0, position - radius):
        min(len(text), position + len(value) + radius)
    ]


def _source_webview_calls(text: str):
    """Yield calls, arguments and their enclosing block, excluding strings.

    This fallback does not establish runtime receiver types. Typed Smali is
    the stronger path; source receiver names/declarations retain candidates.
    """
    text = without_comments(text)
    masked = re.sub(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
                    lambda m: " " * len(m[0]), text)
    blocks: list[tuple[int, int]] = []
    stack: list[int] = []
    for position, char in enumerate(masked):
        if char == "{":
            stack.append(position)
        elif char == "}" and stack:
            blocks.append((stack.pop(), position + 1))
    pattern = re.compile(r"\b([\w$]+)\s*\.\s*(loadUrl|loadData|loadDataWithBaseURL)\s*\(", re.I)
    for call in pattern.finditer(masked):
        receiver, method = call[1], call[2]
        method = {"loadurl": "loadUrl", "loaddata": "loadData", "loaddatawithbaseurl": "loadDataWithBaseURL"}[method.lower()]
        # A method-name match on an arbitrary downloader is not a WebView.
        if ("webview" not in receiver.lower()
                and not re.search(r"\b(?:WebView|WebViewController)\s+" + re.escape(receiver) + r"\b", masked)
                and not re.search(r"\b(?:val|var)\s+" + re.escape(receiver) + r"\s*:\s*WebView\b", masked)):
            continue
        start = call.end()
        depth, argument_start, arguments = 1, start, []
        end = None
        for position in range(start, min(len(masked), start + 64 * 1024)):
            char = masked[position]
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
                if depth == 0:
                    arguments.append(text[argument_start:position].strip())
                    end = position + 1
                    break
            elif char == "," and depth == 1:
                arguments.append(text[argument_start:position].strip())
                argument_start = position + 1
        if end is None:
            continue
        enclosing = [(left, right) for left, right in blocks if left < call.start() < right]
        if enclosing:
            left, right = min(enclosing, key=lambda span: span[1] - span[0])
            context = text[max(left, call.start() - _RELATION_WINDOW_RADIUS):min(right, end + _RELATION_WINDOW_RADIUS)]
        else:
            context = text[max(0, call.start() - _RELATION_WINDOW_RADIUS):min(len(text), end + _RELATION_WINDOW_RADIUS)]
        yield method, arguments, context


def _scan_webview_relations(
    text: str,
    source: str,
    layer: str,
    scope: SignalScope,
    binary_file: bool = False,
) -> Iterator[RawSignal]:
    """Retain relevant targets and candidates; never upgrade co-occurrence."""
    if binary_file:
        lower = text.lower()
        markers = [m for m in _BINARY_WEBVIEW_MARKERS if m in lower]
        if not markers:
            return
        for url in dict.fromkeys(_iter_binary_http_urls(text)):
            observation = page_observation(
                url, source, layer, scope, call="unresolved",
                binding="binary_file_cooccurrence", source_kind="binary_webview_cooccurrence",
                extra=tuple(f"webview_marker={m}" for m in markers),
            )
            if observation:
                yield observation
        return
    if not re.search(r"\bload(?:Url|Data|DataWithBaseURL)\s*\(", text, re.I):
        return
    for method, arguments, context in _source_webview_calls(text):
        index = 1 if method == "loadDataWithBaseURL" else 0
        if len(arguments) <= index:
            continue
        argument = arguments[index]
        literal = re.fullmatch(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", argument, re.S)
        if literal:
            try:
                value = json.loads(argument) if argument.startswith('"') else argument[1:-1]
            except ValueError:
                continue
            if method != "loadUrl":
                if llm_html_context(value):
                    yield RawSignal(
                        SignalType.WEBVIEW_CHAT_PAGE, "inline_html", layer, source,
                        role=SignalRole.RELATION, scope=scope,
                        verification=SignalVerification.CANDIDATE, source_kind="source_webview_inline_html",
                        components=(f"webview_call={method}", "binding=literal_argument",
                                    "relevance_hint=llm_html_context"),
                    )
                continue
            observation = page_observation(
                value, source, layer, scope, call=method, binding="literal_argument",
                source_kind="source_webview_call", context=context,
            )
            if observation:
                yield observation
        else:
            if method != "loadUrl":
                # Unknown HTML cannot be promoted by its base origin.
                continue
            for url in dict.fromkeys(extract_urls(context)):
                observation = page_observation(
                    url, source, layer, scope, call=method, binding="unresolved_source_context",
                    source_kind="source_webview_cooccurrence",
                )
                if observation:
                    yield observation


def _scan_llm_json_schema(
    text: str,
    source: str,
    layer: str,
    scope: SignalScope,
) -> Iterator[RawSignal]:
    """Find a distinctive LLM request/response schema in a local window."""
    lower = text.lower()
    anchors = tuple(
        match.start()
        for match in re.finditer(
            r"(?<![a-z0-9_])(?:messages|max_tokens|max_output_tokens|"
            r"temperature|top_p|choices|finish_reason|prompt)(?![a-z0-9_])",
            lower,
        )
    )
    if not anchors:
        return
    # Objects and allocation identities already establish local scope. A text
    # window may cut an allocation off from its branch or common controls.
    complete = extract_llm_schema_components(text)
    if complete:
        yield RawSignal(SignalType.LLM_JSON_SCHEMA, f'{source}:{len(complete)}',
                        layer, source, role=SignalRole.RELATION, scope=scope,
                        relation=SignalRelation.JSON_FIELDS_LLM_API,
                        components=tuple(f'json_field={key}' for key in complete))
        return
    best: tuple[str, ...] = ()
    for position in anchors:
        window = text[
            max(0, position - _RELATION_WINDOW_RADIUS):
            min(len(text), position + _RELATION_WINDOW_RADIUS)
        ]
        components = extract_llm_schema_components(window)
        if len(components) > len(best):
            best = components

    if not best:
        return
    yield RawSignal(
        SignalType.LLM_JSON_SCHEMA,
        f"{source}:{len(best)}",
        layer,
        source,
        role=SignalRole.RELATION,
        scope=scope,
        relation=SignalRelation.JSON_FIELDS_LLM_API,
        components=tuple(f"json_field={key}" for key in best),
    )


# ═══════════════════════════════════════════════════════════════════
# Layer 1: ZIP direct scan
# ═══════════════════════════════════════════════════════════════════

class ZIPExtractor:
    """Extract signals directly from the APK ZIP file.

    This layer works without APKTool / JADX. Large JS literals additionally
    use an optional, bounded Node parser; missing Node is a coverage gap.
    """

    _BINARY_API_CONFIG_EXTENSIONS = frozenset({
        ".bundle", ".cfg", ".conf", ".dat", ".html", ".htm", ".ini",
        ".js", ".jsbundle", ".json", ".properties", ".toml", ".txt",
        ".wasm", ".xml", ".yaml", ".yml",
    })
    _BINARY_API_CONFIG_MAX_BYTES = 64 * 1024 * 1024

    @classmethod
    def _should_scan_binary_api_hosts(cls, filename: str, size: int) -> bool:
        lower = filename.replace("\\", "/").lower()
        if re.fullmatch(r"classes\d*\.dex", lower):
            return True
        if lower.startswith("lib/") and lower.endswith(".so"):
            return True
        if lower == "resources.arsc" or lower.endswith("/global-metadata.dat"):
            return True
        if not lower.startswith(("assets/", "res/raw/")):
            return False
        return (
            Path(lower).suffix in cls._BINARY_API_CONFIG_EXTENSIONS
            and size <= cls._BINARY_API_CONFIG_MAX_BYTES
        )

    def __init__(self) -> None:
        self._reset_diagnostics()

    def _reset_diagnostics(self) -> None:
        self.diagnostics = {
            "status": "success", "error_count": 0, "errors": [],
            "errors_truncated": False, "read_entries": 0,
            "excluded_artifact_counts": {}, "large_text_files": [],
            "large_script_budget": {"limit_seconds": None,
                "file_limit_seconds": None, "byte_limit": None,
                "file_limit": None, "spent_seconds": 0.0,
                "attempted_files": 0, "attempted_bytes": 0, "skipped_files": 0},
        }

    def _attempt(self, source: str, stage: str, operation):
        """Isolate one scanner failure without discarding earlier yields.

        Exception messages may contain credential values, so diagnostics
        expose only source, stage and exception class, capped at 20 entries.
        """
        try:
            return operation()
        except Exception as exc:
            self.diagnostics["status"] = "partial"
            self.diagnostics["error_count"] += 1
            errors = self.diagnostics["errors"]
            if len(errors) < 20:
                errors.append({"source_file": source, "stage": stage,
                               "error_type": type(exc).__name__})
            else:
                self.diagnostics["errors_truncated"] = True
            return None

    def extract(self, apk_path: Path) -> tuple[list[RawSignal], int, int]:
        """Return signals, archive entry count and decoded text count.

        Failures are reported in ``diagnostics`` even when no signal is
        found. Each scanner is isolated; successfully read entries, earlier
        signals and subsequent scanners/entries survive a partial failure.
        """
        self._reset_diagnostics()
        signals: list[RawSignal] = []
        text_decoded = 0
        zf = self._attempt(str(apk_path), "open", lambda: zipfile.ZipFile(apk_path))
        if zf is None:
            self.diagnostics["status"] = "failed"
            return signals, 0, 0
        with zf:
            entries = zf.infolist()
            for info in entries:
                lower = info.filename.lower()
                if (lower.startswith('assets/') and lower.endswith(('.js', '.mjs', '.cjs', '.jsbundle', '.bundle'))
                        and info.file_size > 2 * 1024 * 1024):
                    # Probe only the header before applying JS budgets. Hermes
                    # keeps its existing structured bytecode path and limits.
                    started = time.monotonic()
                    header = self._attempt(info.filename, "header", lambda: self._read_prefix(zf, info, 8))
                    if header is None:
                        continue
                    if not is_hermes_bytecode(header) and not is_ram_bundle(header):
                        self.diagnostics['large_script_budget']['spent_seconds'] += time.monotonic() - started
                        recovered, decoded = self._extract_large_script_entry(zf, info)
                        signals.extend(recovered)
                        text_decoded += decoded
                        continue
                data = self._attempt(info.filename, "read", lambda: zf.read(info))
                if data is None:
                    continue
                self.diagnostics["read_entries"] += 1

                detected_font = font_format(data)
                if detected_font:
                    counts = self.diagnostics["excluded_artifact_counts"]
                    counts[detected_font] = counts.get(detected_font, 0) + 1
                    continue

                def emit(stage, producer):
                    def run():
                        # Most scanners return an iterator; ``_scan_blob``
                        # writes directly into ``signals`` and returns None.
                        # Accept both contracts so a successful mutating scan
                        # does not become a false ZIP failure.
                        produced = producer()
                        if produced is not None:
                            signals.extend(produced)

                    self._attempt(info.filename, stage, run)

                lower = info.filename.lower()
                if is_hermes_bytecode(data):
                    # Compressed HBC storage is not a source/config file.
                    # Unsupported/damaged formats remain a diagnostic gap.
                    emit('hermes_string_table', lambda: self._scan_hermes(info.filename, data))
                    continue
                if is_ram_bundle(data):
                    info_record = {'source_file': info.filename, 'size_bytes': len(data),
                                   'format': 'indexed_ram_bundle', 'coverage': 'module_literals_only'}
                    self.diagnostics['large_text_files'].append(info_record)
                    def read_ram():
                        try:
                            modules = ram_modules(data)
                        except (ValueError, UnicodeError):
                            info_record.update(status='partial', reason='invalid_ram_bundle')
                            raise
                        for literal in javascript_literals('', info_record, modules=modules):
                            for observed in self._scan_hermes_literal(info.filename, literal.value):
                                signals.append(replace(observed, source_kind='ram_string_literal',
                                    components=(*observed.components, f'module_id={literal.module_id}',
                                        f'module_byte_offset={literal.module_byte_offset}',
                                        f'char_offset={literal.start}', f'char_end={literal.end}',
                                        'offset_unit=utf16_code_units', 'parser=babel-7.26.7')))
                    emit('ram_bundle', read_ram)
                    if info_record.get('status') != 'complete':
                        self.diagnostics['status'] = 'partial'
                    continue
                non_llm = is_known_non_llm_model_artifact(info.filename)
                # DEX/SO/Unity metadata are binary string pools.  They can
                # pass the printable-text heuristic even though their
                # unrelated strings are not a source file or JSON document.
                # Keep their URL/key/blob scanners, but never run structured
                # text relations (notably LLM JSON schema) on decoded pools.
                code_artifact = (
                    re.fullmatch(r"classes\d*\.dex", lower) is not None
                    or (lower.startswith("lib/") and lower.endswith(".so"))
                    or lower.endswith("/global-metadata.dat")
                )
                resource_binary = lower == "resources.arsc"
                # resources.arsc is a compiled global string table, not a
                # source/config file. It must not feed schema relations that
                # combine unrelated resource keys, while its credential and
                # registered-host observations remain useful below.
                code_artifact = code_artifact or resource_binary
                opaque = is_opaque_model_binary(info.filename)
                non_text_binary = any(lower.endswith(ext)
                                      for ext in _NON_TEXT_BINARY_EXTENSIONS)
                text = ("" if non_llm or code_artifact or non_text_binary
                        else self._attempt(
                    info.filename, "decode", lambda: decode_bytes(data))) or ""
                visual = bool(text and _is_lottie_animation_json(info.filename, text))
                if text and is_license_document(text, info.filename):
                    counts = self.diagnostics['excluded_artifact_counts']
                    counts['license_document'] = counts.get('license_document', 0) + 1
                    # Keep the existing narrow LLM license-reference fact;
                    # document URLs and model mentions are not functionality.
                    if any(hint in text.lower() for hint in LICENSE_HINTS):
                        signals.append(RawSignal(
                            SignalType.LICENSE_KEYWORD, info.filename, 'zip', info.filename,
                            role=SignalRole.REFERENCE, scope=SignalScope.FILE,
                            source_kind='license_document_reference'))
                    continue
                if text:
                    text_decoded += 1
                    if not visual:
                        emit("text", lambda: self._scan_text(
                            info.filename, text, scan_model_content=not opaque))
                emit("path", lambda: self._scan_path(info.filename, info.file_size, data[:1024 * 1024 + 8]))
                if not visual and lower.startswith((
                    "assets/", "res/raw/", "res/xml/", "meta-inf/"
                )):
                    emit("special_file", lambda: self._scan_special_file(info.filename, text))
                if info.filename == "AndroidManifest.xml":
                    emit("manifest", lambda: self._scan_binary_manifest(data))

                code_binary = not non_llm and len(data) <= 16 * 1024 * 1024 and (
                    (lower.startswith("lib/") and lower.endswith(".so"))
                    or re.fullmatch(r"classes\d*\.dex", lower)
                    or lower.endswith("/global-metadata.dat")
                )
                if code_binary:
                    emit("binary_custom_urls", lambda: scan_blob_for_business_ai_endpoint(
                        data, info.filename))
                # Known-host scanning is independent of text decoding and
                # the 16 MiB limit on general blob scans. Run it once.
                if not visual and not non_llm and self._should_scan_binary_api_hosts(
                    info.filename, info.file_size
                ):
                    emit("binary_api_hosts", lambda: scan_binary_for_known_api_domains(
                        data, info.filename, "zip"))
                if resource_binary and not non_llm and len(data) <= 16 * 1024 * 1024:
                    emit("resource_blob", lambda: self._scan_blob(
                        data,
                        info.filename,
                        signals,
                        scan_known_api_domains=False,
                        scan_model_content=False,
                        scan_custom_urls=True,
                    ))
                blob_scan = not text and not non_llm and len(data) <= 16 * 1024 * 1024 and (
                    lower.startswith(("assets/", "res/raw/"))
                    or (lower.startswith("lib/") and lower.endswith(".so"))
                    or re.fullmatch(r"classes\d*\.dex", lower)
                )
                if blob_scan:
                    self._attempt(info.filename, "blob", lambda: self._scan_blob(
                        data, info.filename, signals, scan_known_api_domains=False,
                        scan_model_content=not opaque, scan_custom_urls=not code_binary))
            if (entries and not self.diagnostics["read_entries"]
                    and not self.diagnostics["large_script_budget"]["skipped_files"]):
                self.diagnostics["status"] = "failed"
            return signals, len(entries), text_decoded

    # ── per-file scanners ──────────────────────────────────────────

    @staticmethod
    def _read_prefix(zf, entry, size):
        with zf.open(entry) as stream:
            return stream.read(size)

    def _extract_large_script_entry(self, zf, entry):
        budget = self.diagnostics['large_script_budget']
        info = {'source_file': entry.filename, 'size_bytes': entry.file_size,
                'coverage': 'atomic_literals_only', 'status': 'pending'}
        records = self.diagnostics['large_text_files']
        if len(records) < 20:
            records.append(info)
        else:
            self.diagnostics['large_text_files_truncated'] = True
        started = time.monotonic()
        budget['attempted_files'] += 1
        budget['attempted_bytes'] += entry.file_size
        signals, decoded = [], 0
        try:
            data = self._attempt(entry.filename, 'large_script_read',
                                 lambda: zf.read(entry))
            if data is None:
                info.update(status='partial', reason='source_read_error')
            else:
                self.diagnostics['read_entries'] += 1
                text = data.decode('utf-8-sig')
                if '\x00' in text:
                    raise ValueError('not_source_text')
                decoded = 1
                def scan():
                    signals.extend(self._scan_large_script(entry.filename, text, info))
                    return True
                if not self._attempt(entry.filename, 'large_script_literals', scan):
                    info.update(status='partial', reason='literal_scan_error')
        except (UnicodeError, ValueError):
            info.update(status='partial', reason='unsupported_script_encoding')
        finally:
            elapsed = time.monotonic() - started
            budget['spent_seconds'] += elapsed
            info['elapsed_seconds'] = round(elapsed, 6)
            if info['status'] != 'complete':
                self.diagnostics['status'] = 'partial'
        return signals, decoded

    def _scan_large_script(self, source: str, text: str, diagnostics: dict) -> Iterator[RawSignal]:
        started = time.monotonic()
        literals = javascript_literals(text, diagnostics)
        try:
            for literal in literals:
                value = literal.value
                for signal in scan_llm_literal(value, source, 'zip'):
                    yield replace(signal, source_kind='source_string_literal',
                                  components=(*signal.components, 'coverage=atomic_literal',
                                              'parser=babel-7.26.7', 'offset_unit=utf16_code_units',
                                              f'char_offset={literal.start}', f'char_end={literal.end}',
                                              f'source_line={literal.line}', f'source_column={literal.column}'))
        finally:
            literals.close()
            diagnostics['elapsed_seconds'] = round(time.monotonic() - started, 6)

    def _scan_hermes(self, source: str, data: bytes) -> Iterator[RawSignal]:
        for literal in hermes_string_literals(data):
            for signal in self._scan_hermes_literal(source, literal.value):
                # The table index is provenance, never a call/scope binding.
                yield replace(signal, source_kind='hermes_string_literal',
                              components=(*signal.components, f'string_index={literal.index}'))

    def _scan_hermes_literal(self, source: str, value: str) -> Iterator[RawSignal]:
        yield from scan_llm_literal(value, source, 'zip', 'hermes_string_literal')

    def _scan_path(self, name: str, size_bytes: int, header: bytes | None = None) -> Iterator[RawSignal]:
        layer = "zip"
        lower = name.lower()

        if any(lower.endswith(h) for h in MODEL_FILE_EXTENSIONS):
            from artifact_facts import model_header_facts
            from signal_types import SignalVerification
            facts = model_header_facts(header or b"")
            # Keep filename observations auditable, but an invalid or unread
            # artifact must not be treated as a validated model file.
            valid = facts["status"] == "header_valid"
            if lower.endswith("mlc-chat-config.json") and header:
                try:
                    config = json.loads(decode_bytes(header))
                    valid = isinstance(config, dict) and "model_type" in config and "model_config" in config
                    facts = {"format": "mlc_chat_config", "status": "config_structure_observed" if valid else "invalid"}
                except (ValueError, RecursionError):
                    valid = False
            yield RawSignal(
                SignalType.MODEL_FILE_EXTENSION,
                f"{name}::{size_bytes}",
                layer,
                name,
                role=SignalRole.ARTIFACT,
                scope=SignalScope.VALUE,
                verification=SignalVerification.VERIFIED if valid else SignalVerification.CANDIDATE,
                source_kind="model_artifact",
                components=(f"artifact_format={facts['format']}", f"artifact_check={facts['status']}"),
            )

        if lower.startswith("lib/") and lower.endswith(".so"):
            yield from generic_runtime_fact(name)
            if (any(h in lower for h in STRONG_LOCAL_MODEL_HINTS)
                    or is_litert_lm_native_library(name)):
                yield RawSignal(
                    SignalType.NATIVE_LLM_LIBRARY,
                    name,
                    layer,
                    name,
                    role=SignalRole.ARTIFACT,
                    scope=SignalScope.VALUE,
                )

    def _scan_text(self, source: str, text: str,
                   scan_model_content: bool = True) -> Iterator[RawSignal]:
        layer = "zip"
        if scan_model_content:
            yield from scan_llm_business_text(text, source, layer)
            yield from scan_llm_json_config(text, source, layer)
            yield from generation_config_facts(text, source, layer)
        lower = text.lower()
        tokenizer_components = (
            tokenizer_llm_context_components(text, source)
            if scan_model_content and not is_reference_only_source(source)
            else ()
        )

        sdk_matches: set[str] = set()
        if not is_reference_only_source(source):
            sdk_matches = set(iter_known_sdk_packages(text))
            sdk_matches.update(iter_contextual_sdk_identities(text))
            for pkg in sorted(sdk_matches):
                yield RawSignal(SignalType.KNOWN_LLM_SDK_PACKAGE, pkg, layer, source)
        if scan_model_content and LITERT_LM_PACKAGE in sdk_matches:
            yield RawSignal(SignalType.LLM_FRAMEWORK_KEYWORD,
                            LITERT_LM_PACKAGE, layer, source)

        yield from scan_aicore_permissions(text, source, layer)

        native_hits = tuple(h for h in NATIVE_HINTS if h in lower)
        if not is_reference_only_source(source):
            native_hits += litert_lm_native_symbols(text)
        if scan_model_content and native_hits:
            yield RawSignal(
                SignalType.NATIVE_LLM_FUNCTION,
                source,
                layer,
                source,
                role=SignalRole.REFERENCE,
                scope=SignalScope.FILE,
                components=tuple(f"native_symbol={hit}" for hit in native_hits),
            )
            if any(framework_keyword_matches(h, lower) for h in ("llama", "mediapipe", "executorch",
                                         "ggml", "mnn", "mlc", "qwen")):
                yield RawSignal(
                    SignalType.LLM_FRAMEWORK_KEYWORD,
                    source,
                    layer,
                    source,
                    role=SignalRole.REFERENCE,
                    scope=SignalScope.FILE,
                )
            if any(h in lower for h in ("ort", "onnxruntime", "qnn", "snpe", "tflite", "nnapi")):
                # Require co-occurrence with LLM engine keywords.
                # TFLite alone is used by camera / OCR / face-detection
                # apps — only relevant when an LLM engine is also present.
                _LLM_ENGINE_KW = (
                    "llama.cpp", "executorch", "gguf", "mediapipe genai",
                    "mlc-chat-config", "litertlm", "mnn_llm")
                engine_hits = tuple(kw for kw in _LLM_ENGINE_KW if framework_keyword_matches(kw, lower))
                if tokenizer_components:
                    engine_hits += ("contextual_tokenizer",)
                if engine_hits:
                    runtime_hits = tuple(
                        kw for kw in ("ort", "onnxruntime", "qnn", "snpe", "tflite", "nnapi")
                        if kw in lower
                    )
                    yield RawSignal(
                        SignalType.ONNX_TFLITE_RUNTIME,
                        source,
                        layer,
                        source,
                        role=SignalRole.RELATION,
                        scope=SignalScope.FILE,
                        relation=SignalRelation.RUNTIME_LLM_CONTEXT,
                        components=tuple(
                            [*(f"runtime={kw}" for kw in runtime_hits),
                             *(f"llm_context={kw}" for kw in engine_hits)]
                        ),
                    )

        yield from _scan_llm_json_schema(
            text, source, layer, SignalScope.FILE)

        # Model IDs require nearby configuration, API, or loading context.
        if scan_model_content:
            for model_id, context_components in extract_contextual_llm_model_evidence(text):
                yield RawSignal(
                    SignalType.KNOWN_LLM_MODEL_NAME,
                    model_id,
                    layer,
                    source,
                    role=SignalRole.RELATION,
                    scope=SignalScope.FILE,
                    relation=SignalRelation.MODEL_ID_USAGE_CONTEXT,
                    components=(f"model_id={model_id}", *context_components),
                )
            if tokenizer_components:
                yield RawSignal(
                    SignalType.LLM_FRAMEWORK_KEYWORD,
                    "tokenizer",
                    layer,
                    source,
                    role=SignalRole.RELATION,
                    scope=SignalScope.FILE,
                    relation=SignalRelation.TOKENIZER_LLM_CONTEXT,
                    components=tokenizer_components,
                )

        # Special tokens
        for tok in SPECIAL_TOKEN_HINTS:
            if tok in text:
                yield RawSignal(SignalType.LLM_SPECIAL_TOKEN, tok, layer, source)

        # Prompt templates
        for marker in prompt_template_markers(text):
            yield RawSignal(
                SignalType.LLM_PROMPT_TEMPLATE,
                marker,
                layer,
                source,
                role=SignalRole.REFERENCE,
                scope=SignalScope.VALUE,
                components=('prompt_evidence=instruction_literal',),
            )

        # URLs (skip known noise files that contain domain lists)
        _NOISE_FILES = {"publicsuffixdatabase.list"}
        _has_llm_signal = False
        if not any(nf in source.lower() for nf in _NOISE_FILES):
            # ── LLM context gate ────────────────────────────────────
            # SERVERLESS_URL (Vercel/Supabase) and BACKEND_PROXY_KEYWORD
            # (worker/proxy/gateway) are noisy when the file has zero
            # LLM payload context — every REST app uses Vercel; every
            # Android app uses WorkManager.  Require at least one LLM
            # keyword in the same file before emitting them.
            _has_llm_signal = has_strong_llm_context(text)

            for url in extract_urls(text):
                url_context = _context_around_value(
                    text, url, lower_text=lower)
                for sig in classify_url_signal(source, url, layer):
                    # Gate BACKEND_PROXY_KEYWORD — only emit when the
                    # file also contains LLM payload keywords.
                    if (sig.signal_type == SignalType.BACKEND_PROXY_KEYWORD
                            and not _has_llm_signal):
                        continue
                    yield sig
                yield from scan_url_for_business_ai_endpoint(
                    url, source, layer, url_context, SignalScope.FILE)
            if _has_llm_signal:
                # This scan examines the complete file once.  Keeping it
                # outside the URL loop avoids rescanning the same text once
                # for every URL (up to the extractor's URL limit).
                yield from scan_for_serverless_url(text, source, layer)
            # Recover a separated base URL and LLM route only within this
            # decoded source file and a bounded local window. This covers
            # JavaScript/config bundles without restoring APK-wide string
            # co-occurrence across unrelated libraries.
            yield from scan_for_local_split_ai_endpoint(text, source, layer)

        # New signal types — API keys, baseurl override, key input prompt
        yield from scan_for_api_keys(text, source, layer)
        if _has_llm_signal:
            yield from scan_for_baseurl_override(text, source, layer)
            yield from scan_for_api_key_input_prompt(text, source, layer)

        yield from _scan_webview_relations(
            text, source, layer, SignalScope.FILE)

    def _scan_special_file(self, name: str, text: str) -> Iterator[RawSignal]:
        layer = "zip"
        lower = name.lower()
        lowered_text = text.lower()

        if lower.startswith(("assets/", "res/raw/")):
            # Lottie files frequently embed large base64 images. Random byte
            # sequences inside those images can spell tokens such as ``ggml``;
            # their validated animation schema makes them non-model assets.
            if _is_lottie_animation_json(name, text):
                return
            if is_known_non_llm_model_artifact(name):
                return
            if "debugprobeskt.bin" in lower:
                return
            if "publicsuffixdatabase.list" in lower:
                return  # public suffix list — full of domains, none LLM-related
            # TFLite files in assets/ are MLKit / OCR / CV models,
            # never LLM models.  Skip to avoid false positives on
            # MODEL_FILE_REFERENCE and DYNAMIC_DOWNLOAD_HINT.
            if lower.endswith(".tflite"):
                return
            # Binary assets that can never be LLM models — images,
            # fonts, videos, animations, documents.  Their latin-1
            # decoded content produces random substrings that falsely
            # match "gguf", "ggml", "tokenizer" etc.
            _NON_MODEL_EXT = {
                ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
                ".mp3", ".mp4", ".wav", ".ogg", ".aac",
                ".svg", ".svga",
                ".pdf", ".doc", ".docx", ".ppt", ".pptx",
                ".ttf", ".otf", ".woff", ".woff2", ".eot",
                ".dex", ".so",
            }
            if any(lower.endswith(ext) for ext in _NON_MODEL_EXT):
                return
            scan_model_content = not is_opaque_model_binary(name)
            tokenizer_components = tokenizer_llm_context_components(text, name)
            model_context_hits = tuple(
                tok for tok in (
                    "gguf", "ggml", "safetensors", "llama.cpp",
                    "executorch", "mediapipe genai", "mlc-chat-config",
                )
                if framework_keyword_matches(tok, lowered_text)
            )
            model_context_hits += tuple(
                f"tokenizer:{component}" for component in tokenizer_components
            )
            model_context_hits += tuple(
                f"model_id:{model_id}"
                for model_id in extract_contextual_llm_model_ids(text)
            )
            # HTML assets are web documents (including error pages), not
            # model/config files.  Their prose, embedded scripts, and
            # base64 data can contain words such as ``ggml``, ``tokenizer``
            # or ``sentencepiece`` by coincidence.  URLs and API evidence
            # from the same HTML have already been handled by _scan_text;
            # only suppress the model-artifact inference here.
            model_text_allowed = not lower.endswith((".html", ".htm"))
            if model_text_allowed and scan_model_content:
                # A reference must be a path value, not an arbitrary four-byte
                # substring in prose, base64 or unknown binary assets.
                references = re.findall(r'''["']([^"'\r\n]+\.(?:gguf|ggml))["']''', text, re.I)
                if any(not re.search(r"\s|://|[<>]", p) for p in references):
                    yield RawSignal(
                        SignalType.MODEL_FILE_REFERENCE,
                        name,
                        layer,
                        name,
                        role=SignalRole.REFERENCE,
                        scope=SignalScope.FILE,
                    )
            if any(tok in lowered_text for tok in LICENSE_HINTS):
                yield RawSignal(
                    SignalType.LICENSE_KEYWORD,
                    name,
                    layer,
                    name,
                    role=SignalRole.REFERENCE,
                    scope=SignalScope.FILE,
                )
            pad_hits = tuple(h for h in PAD_HINTS if h in lowered_text)
            if scan_model_content and pad_hits and model_context_hits:
                yield RawSignal(
                    SignalType.ASSET_DELIVERY_HINT,
                    name,
                    layer,
                    name,
                    role=SignalRole.RELATION,
                    scope=SignalScope.FILE,
                    relation=SignalRelation.ASSET_DELIVERY_MODEL_CONTEXT,
                    components=tuple(
                        [*(f"delivery_hint={hit}" for hit in pad_hits),
                         *(f"model_context={hit}" for hit in model_context_hits)]
                    ),
                )
            if (scan_model_content
                    and any(h in lowered_text for h in DYNAMIC_DOWNLOAD_HINTS)):
                # Guard: require co-occurrence with an LLM keyword
                # in the same file — "weights" / "download_url" match
                # in binary noise (fonts, OCR models) otherwise.
                download_hits = tuple(
                    h for h in DYNAMIC_DOWNLOAD_HINTS if h in lowered_text
                )
                if model_context_hits:
                    yield RawSignal(
                        SignalType.DYNAMIC_DOWNLOAD_HINT,
                        name,
                        layer,
                        name,
                        role=SignalRole.RELATION,
                        scope=SignalScope.FILE,
                        relation=SignalRelation.MODEL_DOWNLOAD_CONTEXT,
                        components=tuple(
                            [*(f"download_hint={hit}" for hit in download_hits),
                             *(f"model_context={hit}" for hit in model_context_hits)]
                        ),
                    )

        if lower.startswith("meta-inf/"):
            # Skip JAR signature files — they contain base64 hashes
            # whose latin-1 decoding produces random substrings that
            # can falsely match SDK package names.
            _SIG_EXTS = {".sf", ".rsa", ".mf", ".dsa", ".ec"}
            if any(lower.endswith(ext) for ext in _SIG_EXTS):
                return

            lowered = lowered_text
            if any(tok in lowered for tok in LICENSE_HINTS):
                yield RawSignal(
                    SignalType.LICENSE_KEYWORD,
                    name,
                    layer,
                    name,
                    role=SignalRole.REFERENCE,
                    scope=SignalScope.FILE,
                )
            # LICENSE/NOTICE mentions are intentionally kept only as
            # LICENSE_KEYWORD. They are not delivery or download evidence.

    def _scan_binary_manifest(self, data: bytes) -> Iterator[RawSignal]:
        layer = "zip"
        xml_text = decode_manifest_xml(data)
        if not xml_text:
            return
        lower = xml_text.lower()

        yield from scan_aicore_permissions(xml_text, 'AndroidManifest.xml', layer, manifest=True)
        for pkg in iter_known_sdk_packages(xml_text):
            yield RawSignal(SignalType.KNOWN_LLM_SDK_PACKAGE,
                            pkg, layer, "AndroidManifest.xml")

    def _scan_blob(self, data: bytes, filename: str,
                   signals: list[RawSignal],
                   scan_known_api_domains: bool = True,
                   scan_model_content: bool = True,
                   scan_custom_urls: bool = True) -> None:
        # Font files and other non-code binaries produce random latin-1
        # substrings that falsely match LLM keywords (e.g. "mistral" in
        # a .ttf glyph table).  Skip them.
        _FONT_EXTS = {".ttf", ".otf", ".woff", ".woff2", ".eot"}
        if font_format(data) or any(filename.lower().endswith(ext) for ext in _FONT_EXTS):
            return

        try:
            raw = data.decode("latin-1", errors="ignore")
        except Exception:
            return
        layer = "zip"

        if scan_known_api_domains:
            signals.extend(scan_binary_for_known_api_domains(
                data, filename, layer))
        sdk_matches = set(iter_known_sdk_packages(raw))
        sdk_matches.update(iter_contextual_sdk_identities(raw))
        for pkg in sorted(sdk_matches):
            if len(pkg) >= 8:
                signals.append(RawSignal(SignalType.KNOWN_LLM_SDK_PACKAGE,
                                         pkg, layer, filename, detail="blob_scan"))
        for tok in SPECIAL_TOKEN_HINTS:
            if tok in raw:
                signals.append(RawSignal(SignalType.LLM_SPECIAL_TOKEN, tok,
                                         layer, filename, detail="blob_scan"))
        signals.extend(_scan_webview_relations(
            raw, filename, layer, SignalScope.FILE, binary_file=True))
        # DEX/SO/Hermes string tables place unrelated field names next to each
        # other.  Without a function-aware decoder their proximity is not a
        # reproducible schema relation; decoded text/config and JADX cover the
        # structured cases.
        # Binary string pools often contain isolated marketing or dependency
        # names. Prefer local configuration/API/loading context, with the
        # narrower explicit-identity fallback handled below.
        if scan_model_content:
            # DEX/SO/Unity metadata do not preserve source-method or field
            # boundaries.  A file-wide contextual scan therefore joins
            # adjacent native string fragments (for example
            # ``gemma17load_arch...``) into fabricated model IDs.  Their
            # model evidence is handled only by the bounded identity
            # fallback below.  Text-like asset blobs retain the contextual
            # scan because they can be structured model configuration files.
            code_binary = (
                re.fullmatch(r"classes\d*\.dex", filename.lower()) is not None
                or (filename.lower().startswith("lib/")
                    and filename.lower().endswith(".so"))
                or filename.lower().endswith("/global-metadata.dat")
            )
            contextual_models = (
                extract_bounded_binary_model_evidence(raw)
                if code_binary else extract_contextual_llm_model_evidence(raw)
            )
            emitted_model_ids: set[str] = set()
            for model_id, context_components in contextual_models:
                emitted_model_ids.add(model_id)
                signals.append(RawSignal(
                    SignalType.KNOWN_LLM_MODEL_NAME,
                    model_id,
                    layer,
                    filename,
                    detail="blob_scan",
                    role=SignalRole.RELATION,
                    scope=SignalScope.FILE,
                    relation=SignalRelation.MODEL_ID_USAGE_CONTEXT,
                    components=(f"model_id={model_id}", *context_components),
                ))
            # Compiled Flutter/native code frequently separates the model
            # literal from its provider/package identity.  Recover only
            # family-matched model IDs when an explicit LLM identity occurs
            # in the same binary; generic model-shaped substrings remain
            # filtered out.
            for model_id, context_components in extract_binary_llm_model_evidence(raw):
                if model_id in emitted_model_ids:
                    continue
                signals.append(RawSignal(
                    SignalType.KNOWN_LLM_MODEL_NAME,
                    model_id,
                    layer,
                    filename,
                    detail="binary_file_identity",
                    role=SignalRole.RELATION,
                    scope=SignalScope.FILE,
                    relation=SignalRelation.MODEL_ID_USAGE_CONTEXT,
                    components=(f"model_id={model_id}", *context_components),
                ))
        # Hardcoded API keys in binary blobs
        signals.extend(list(scan_for_api_keys(raw, filename, layer)))
        # Local custom-domain + LLM-path relation for compiled string tables,
        # including Flutter/Dart constants in libapp.so.
        if scan_custom_urls:
            signals.extend(scan_blob_for_business_ai_endpoint(raw, filename))


# ═══════════════════════════════════════════════════════════════════
# Layer 2: APKTool decoded resources
# ═══════════════════════════════════════════════════════════════════

class APKToolExtractor:
    """Extract signals from APKTool-decoded resources.

    This is the *only* layer that can read compiled string resources
    (``res/values/strings.xml``).
    """

    def extract(self, output_dir: Path) -> list[RawSignal]:
        layer = "apktool"
        signals: list[RawSignal] = []
        if not output_dir or not output_dir.exists():
            return signals

        manifest = output_dir / "AndroidManifest.xml"
        if manifest.exists():
            try:
                text = manifest.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            signals.extend(scan_aicore_permissions(text, str(manifest), layer, manifest=True))
        assets_prefix = str(output_dir / "assets").lower()
        for root, _, files in os.walk(output_dir):
            for filename in files:
                fp = Path(root) / filename
                rel_lower = str(fp.relative_to(output_dir)).lower()

                if rel_lower.startswith("res") and filename.lower().endswith(".xml"):
                    try:
                        content = fp.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        content = ""
                    for key, value in re.findall(
                        r'<string\s+name\s*=\s*"([^"]+)"\s*>(.*?)</string\s*>',
                        content, re.IGNORECASE | re.DOTALL,
                    ):
                        self._classify_string_value(key, value, str(fp), signals)

                lower_fp = str(fp).lower()
                if lower_fp.startswith(assets_prefix) and any(
                    h in lower_fp for h in MODEL_FILE_EXTENSIONS):
                    signals.append(RawSignal(
                        SignalType.MODEL_FILE_EXTENSION,
                        str(fp),
                        layer,
                        str(fp),
                        role=SignalRole.ARTIFACT,
                        scope=SignalScope.VALUE,
                    ))
        return signals

    def _classify_string_value(self, key: str, value: str, source_file: str,
                                signals: list[RawSignal]) -> None:
        layer = "apktool"
        vlower = value.lower()
        # UI/resource values are already separated by the resource reader.
        # Their explicit LLM meaning does not depend on a request call graph.
        signals.extend(replace(s, detail=f'string_key:{key}')
                       for s in scan_llm_business_literal(value, source_file, layer,
                                                          'resource_string_literal'))
        for marker in prompt_template_markers(value):
            signals.append(RawSignal(SignalType.LLM_PROMPT_TEMPLATE, marker, layer, source_file,
                                     detail=f'string_key:{key}', role=SignalRole.REFERENCE,
                                     scope=SignalScope.VALUE, source_kind='resource_string_literal',
                                     components=('prompt_evidence=instruction_literal',)))

        for url in extract_urls(value):
            if match_api_service(url, classification_only=True):
                signals.append(RawSignal(
                    SignalType.KNOWN_LLM_API_DOMAIN, url, layer, source_file,
                    detail=f"string_key:{key}"))

        for url in extract_urls(value):
            if match_web_service(url):
                signals.append(RawSignal(
                    SignalType.KNOWN_LLM_WEB_DOMAIN, url, layer, source_file,
                    detail=f"string_key:{key}"))

        structured_value = f'{key}="{value}"'
        has_llm_context = has_strong_llm_context(structured_value)

        if (any(p.search(vlower) for p in BACKEND_MARKER_PATS)
                or any(m in vlower for m in ("api_base_url", "prompts_base_url"))):
            if has_llm_context:
                signals.append(RawSignal(SignalType.BACKEND_PROXY_KEYWORD, value.strip(),
                                         layer, source_file, detail=f"string_key:{key}"))

        model_evidence = {
            model_id: set(context_components)
            for model_id, context_components
            in extract_contextual_llm_model_evidence(structured_value)
        }
        for model_id in extract_resource_model_selector_ids(key, value):
            model_evidence.setdefault(model_id, set()).add(
                "model_context=resource_selector")
        for model_id, context_components in sorted(model_evidence.items()):
            signals.append(RawSignal(
                SignalType.KNOWN_LLM_MODEL_NAME, model_id, layer,
                source_file, detail=f"string_key:{key}",
                role=SignalRole.RELATION,
                scope=SignalScope.CONFIG,
                relation=SignalRelation.MODEL_ID_USAGE_CONTEXT,
                components=tuple(
                    [f"config_key={key}", f"model_id={model_id}",
                     *sorted(context_components)]
                )))

        for tok in SPECIAL_TOKEN_HINTS:
            if tok in value:
                signals.append(RawSignal(SignalType.LLM_SPECIAL_TOKEN, tok,
                                         layer, source_file, detail=f"string_key:{key}"))

        download_hits = tuple(
            h for h in ("huggingface.co", "hf.co", "model.safetensors", ".gguf")
            if h in vlower
        )
        model_context = tuple(
            h for h in (
                "llm", "tokenizer", "sentencepiece", "gguf", "ggml",
                "llama", "qwen", "gemma", "mistral", "deepseek",
            )
            if h in key.lower() or h in vlower
        )
        if extract_contextual_llm_model_ids(structured_value):
            model_context += ("configured_model_id",)
        if download_hits and model_context:
            signals.append(RawSignal(
                SignalType.DYNAMIC_DOWNLOAD_HINT,
                value.strip(),
                layer,
                source_file,
                detail=f"string_key:{key}",
                role=SignalRole.RELATION,
                scope=SignalScope.CONFIG,
                relation=SignalRelation.MODEL_DOWNLOAD_CONTEXT,
                components=tuple(
                    [f"config_key={key}",
                     *(f"download_hint={hit}" for hit in download_hits),
                     *(f"model_context={hit}" for hit in model_context)]
                ),
            ))

        if any(h in vlower for h in LICENSE_HINTS):
            signals.append(RawSignal(
                SignalType.LICENSE_KEYWORD,
                f"apktool_string:{key}",
                layer,
                source_file,
                detail=value.strip()[:200],
                role=SignalRole.REFERENCE,
                scope=SignalScope.CONFIG,
            ))

        # New signal types in string resources — gate BASEURL_OVERRIDE,
        # SERVERLESS_URL and API_KEY_INPUT_PROMPT on LLM keyword presence
        # in the key name or value.  Non-AI apps also have "base_url"
        # configs (ad SDKs, payment gateways) and generic "APIKey:" prompts.
        # Include the resource key in the bounded credential context.  The
        # Google AIza prefix is shared by Gemini, Maps, Firebase and many other
        # products, so a key such as ``gemini_api_key`` is useful while a
        # generic ``google_api_key`` is not an LLM signal.
        signals.extend(list(scan_for_api_keys(
            structured_value, source_file, layer)))
        if has_llm_context:
            signals.extend(list(scan_for_baseurl_override(value, source_file, layer)))
            signals.extend(list(scan_for_serverless_url(
                value, source_file, layer, context_text=structured_value)))
            signals.extend(list(scan_for_api_key_input_prompt(value, source_file, layer)))
        for u in extract_urls(value):
            signals.extend(list(scan_url_for_business_ai_endpoint(
                u, source_file, layer, structured_value, SignalScope.CONFIG)))


# ═══════════════════════════════════════════════════════════════════
# Layer 3: JADX decompiled source
# ═══════════════════════════════════════════════════════════════════

class JADXExtractor:
    """Extract signals from JADX-decompiled Java/Kotlin source.

    Scans are scoped to the app's own package tree to avoid false
    positives from AndroidX / Kotlin / bundled SDK code.
    """

    def extract(self, output_dir: Path, app_scope: AppCodeScope | str | None,
                has_assets: bool) -> list[RawSignal]:
        layer = "jadx"
        signals: list[RawSignal] = []
        self.diagnostics = {'files_read': 0, 'read_errors': 0,
                            'files_with_unreconstructed_methods': 0,
                            'decoder_declarations': 0, 'decoded_facts': 0}
        if not output_dir or not output_dir.exists():
            return signals

        scope = (
            app_scope
            if isinstance(app_scope, AppCodeScope)
            else resolve_app_scope(app_scope)
        )
        scope = expand_shared_app_scope(scope, output_dir)
        if not scope.usable:
            return signals

        from constant_decoders import decoder_declarations, decoded_literals, has_constant_calls
        declarations, ambiguous, encoded_sources = set(), set(), []

        for fp in scope.iter_jadx_files(output_dir):
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                self.diagnostics['read_errors'] += 1
                continue
            self.diagnostics['files_read'] += 1
            if 'Method not decompiled:' in text or 'method decompiled incorrectly' in text.lower():
                self.diagnostics['files_with_unreconstructed_methods'] += 1
            lower = text.lower()
            if 'byte[]' in text and '^' in text:
                for decoder in decoder_declarations(text):
                    if decoder in declarations:
                        ambiguous.add(decoder)
                    declarations.add(decoder)
            if 'byte[]' in text and re.search(r'new\s+byte\[\]\s*\{', text):
                if has_constant_calls(text):
                    encoded_sources.append((str(fp), text))
            signals.extend(scan_llm_business_text(text, str(fp), layer))

            sdk_matches = set(iter_known_sdk_packages(text))
            sdk_matches.update(iter_contextual_sdk_identities(text))
            for package in sorted(sdk_matches):
                signals.append(RawSignal(SignalType.KNOWN_LLM_SDK_PACKAGE,
                                         package, layer, str(fp)))
            if LITERT_LM_PACKAGE in sdk_matches:
                signals.append(RawSignal(SignalType.LLM_FRAMEWORK_KEYWORD,
                                         LITERT_LM_PACKAGE, layer, str(fp)))
            native_hits = litert_lm_native_symbols(text)
            if native_hits:
                signals.append(RawSignal(
                    SignalType.NATIVE_LLM_FUNCTION, str(fp), layer, str(fp),
                    role=SignalRole.REFERENCE, scope=SignalScope.FILE,
                    components=tuple(f"native_symbol={hit}" for hit in native_hits),
                ))
            for model_id, context_components in extract_contextual_llm_model_evidence(text):
                signals.append(RawSignal(
                    SignalType.KNOWN_LLM_MODEL_NAME,
                    model_id,
                    layer,
                    str(fp),
                    role=SignalRole.RELATION,
                    scope=SignalScope.FILE,
                    relation=SignalRelation.MODEL_ID_USAGE_CONTEXT,
                    components=(f"model_id={model_id}", *context_components),
                ))
            signals.extend(scan_aicore_permissions(text, str(fp), layer))
            for tok in ("gguf", "mlc-chat-config", "llama.cpp",
                         "mediapipe genai", "executorch"):
                if framework_keyword_matches(tok, lower):
                    signals.append(RawSignal(SignalType.LLM_FRAMEWORK_KEYWORD,
                                             tok, layer, str(fp)))
                    break
            tokenizer_components = tokenizer_llm_context_components(text, str(fp))
            if tokenizer_components:
                signals.append(RawSignal(
                    SignalType.LLM_FRAMEWORK_KEYWORD,
                    "tokenizer",
                    layer,
                    str(fp),
                    role=SignalRole.RELATION,
                    scope=SignalScope.FILE,
                    relation=SignalRelation.TOKENIZER_LLM_CONTEXT,
                    components=tokenizer_components,
                ))
            signals.extend(_scan_webview_relations(
                text, str(fp), layer, SignalScope.FILE))
            signals.extend(_scan_llm_json_schema(
                text, str(fp), layer, SignalScope.FILE))
            # New signal types — API keys always scanned; everything else
            # gated on LLM keyword presence to avoid noise from non-AI apps.
            signals.extend(list(scan_for_api_keys(text, str(fp), layer)))
            if has_strong_llm_context(text):
                signals.extend(list(scan_for_baseurl_override(text, str(fp), layer)))
                signals.extend(list(scan_for_serverless_url(text, str(fp), layer)))
                signals.extend(list(scan_for_api_key_input_prompt(text, str(fp), layer)))
            for u in extract_urls(text):
                # The custom-endpoint scanner deliberately excludes known
                # providers; classify their API/web identities separately.
                for signal in classify_url_signal(str(fp), u, layer):
                    if signal.signal_type in (
                        SignalType.KNOWN_LLM_API_DOMAIN,
                        SignalType.KNOWN_LLM_WEB_DOMAIN,
                    ):
                        signals.append(signal)
                signals.extend(list(scan_url_for_business_ai_endpoint(
                    u,
                    str(fp),
                    layer,
                    _context_around_value(text, u, lower_text=lower),
                    SignalScope.CLASS,
                )))
            signals.extend(list(scan_for_local_split_ai_endpoint(
                text, str(fp), layer)))
            signals.extend(list(scan_for_retrofit_ai_paths(text, str(fp), layer)))

        self.diagnostics['decoder_declarations'] = len(declarations-ambiguous)
        for source, text in encoded_sources:
            for value, decoder, offset in decoded_literals(text, declarations-ambiguous):
                for signal in scan_llm_literal(value, source, layer):
                    signals.append(replace(signal, source_kind='static_decoded_literal',
                                           components=(*signal.components, 'decode=xor_repeating_constant_key',
                                                       f'decoder_method={decoder}', f'char_offset={offset}')))
                    self.diagnostics['decoded_facts'] += 1
        self.diagnostics['signals'] = len(signals)
        return signals
