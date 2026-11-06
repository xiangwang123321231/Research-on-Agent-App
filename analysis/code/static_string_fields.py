"""Read-only final static String recovery with a bounded decoded-Smali write audit."""
from __future__ import annotations
from pathlib import Path
import re
from smali_literals import decode_smali_string
from webview_fields import METHOD, STRING, MOVE, without_comments, register_normalizer, written_registers

FIELD_READ = re.compile(r"(?m)^\s*sget-object\s+[vp]\d+,\s*(L[^;\s]+;->[^:\s]+:Ljava/lang/String;)\s*$")
WRITE = re.compile(r"(?m)^\s*sput-object\s+[vp]\d+,\s*(L[^;\s]+;->[^:\s]+:Ljava/lang/String;)\s*$")
MAX_AUDIT_BYTES = 256 * 1024 * 1024
MAX_AUDIT_FILES = 65536


def recover_static_strings(root: Path, methods: dict, diagnostics: dict | None = None, *, wanted_fields=None) -> dict[str, str]:
    info = diagnostics if diagnostics is not None else {}
    wanted = (set(wanted_fields) if wanted_fields is not None else
              {key for method in methods.values() if method.app_owned
               for key in FIELD_READ.findall(without_comments(method.body))})
    info.update(status="not_needed", fields_requested=len(wanted), fields_recovered=0,
                files_checked=0, bytes_checked=0)
    if not wanted:
        return {}
    candidates, expected_writes, writes = {}, {}, {}
    declared = set()
    info["status"] = "complete"
    try:
        for dex in sorted(root.glob("smali*")):
            if not dex.is_dir():
                continue
            for path in sorted(dex.rglob("*.smali")):
                size = path.stat().st_size
                if (info["files_checked"] >= MAX_AUDIT_FILES or size > 8 * 1024 * 1024
                        or info["bytes_checked"] + size > MAX_AUDIT_BYTES):
                    info.update(status="partial", reason="static_write_audit_budget")
                    return {}
                text = path.read_text(encoding="utf-8")
                info["files_checked"] += 1
                info["bytes_checked"] += size
                if "Ljava/lang/String;" not in text:
                    continue
                text = without_comments(text)
                for key in WRITE.findall(text):
                    if key in wanted:
                        writes[key] = writes.get(key, 0) + 1
                owner = re.search(r"(?m)^\s*\.class\b[^\n]*\s(L[^;\s]+;)\s*$", text)
                if not owner or not any(k.startswith(owner[1] + "->") for k in wanted):
                    continue
                local = {}
                for declaration in re.findall(r"(?m)^\s*\.field\s+([^\n]+)", text):
                    match = re.fullmatch(r'(.*?)\s+([^:\s]+:Ljava/lang/String;)(?:\s*=\s*("(?:\\.|[^"\\])*"))?', declaration)
                    if not match or not {"static", "final"}.issubset(match[1].split()):
                        continue
                    key = owner[1] + "->" + match[2]
                    if key not in wanted:
                        continue
                    if key in declared:
                        info.update(status="partial", reason="duplicate_field_declaration")
                        return {}
                    declared.add(key)
                    local[key] = None
                    if match[3]:
                        try:
                            local[key] = decode_smali_string(match[3])
                            expected_writes[key] = 0
                        except ValueError:
                            pass
                # A straight-line initializer may set a final field once.
                for method in METHOD.finditer(text):
                    if "<clinit>()V" not in method[1]:
                        continue
                    body = method[2]
                    if re.search(r"(?m)^\s*(?:if-|goto|:|throw|invoke-|\.catch|packed-switch|sparse-switch)", body):
                        continue
                    registers = {}
                    normalize = register_normalizer(method[1], body)
                    assigned = {}
                    for raw in body.splitlines():
                        line = raw.strip()
                        if not line or line.startswith("."):
                            continue
                        if line.startswith("return"):
                            break
                        if m := STRING.fullmatch(line):
                            try:
                                registers[normalize(m[1])] = decode_smali_string(m[2])
                            except ValueError:
                                registers.pop(normalize(m[1]), None)
                        elif m := MOVE.fullmatch(line):
                            registers[normalize(m[1])] = registers.get(normalize(m[2]))
                        elif m := re.fullmatch(r"sput-object\s+([vp]\d+),\s*(L\S+)", line):
                            if m[2] in local:
                                assigned.setdefault(m[2], []).append(registers.get(normalize(m[1])))
                        else:
                            for dest in written_registers(line):
                                registers.pop(normalize(dest), None)
                    for key, values in assigned.items():
                        if len(values) == 1 and isinstance(values[0], str) and len(values[0]) <= 4096:
                            local[key] = values[0]
                            expected_writes[key] = 1
                candidates.update({k: v for k, v in local.items() if isinstance(v, str) and len(v) <= 4096})
    except (OSError, UnicodeError):
        info.update(status="partial", reason="static_write_audit_unreadable")
        return {}
    result = {key: value for key, value in candidates.items()
              if writes.get(key, 0) == expected_writes.get(key, -1)}
    info["fields_recovered"] = len(result)
    return result

