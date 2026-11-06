"""Bounded final-instance String summaries for typed WebView arguments.

Only the same literal assigned on every supported constructor is eligible.
Other writes or an incomplete decoded-Smali audit disable new summaries.
Reflection, native writes and missing splits are outside this static analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from smali_literals import decode_smali_string

METHOD = re.compile(r"(?ms)^\s*\.method\s+([^\n]+)\n(.*?)^\s*\.end method")
STRING = re.compile(r'const-string(?:/jumbo)?\s+([vp]\d+),\s*("(?:\\.|[^"\\])*")$')
MOVE = re.compile(r"move-object(?:/from16|/16)?\s+([vp]\d+),\s*([vp]\d+)$")
FIELD_READ = re.compile(r"iget-object\s+([vp]\d+),\s*([vp]\d+),\s*"
                        r"(L[^;\s]+;->[^\s:]+:Ljava/lang/String;)$")
_FIELD_WRITE = re.compile(r"iput-object\s+([vp]\d+),\s*([vp]\d+),\s*"
                          r"(L[^;\s]+;->[^\s:]+:Ljava/lang/String;)$")
_WRITES = re.compile(r"(?m)^\s*[is]put(?:-[\w]+)?\s+[^\n]*,\s*"
                     r"(L[^;\s]+;->[^\s:]+:Ljava/lang/String;)\s*$")
_COMMENT = re.compile(r'"(?:\\.|[^"\\])*"|#[^\n]*')
MAX_AUDIT_FILES = 32768
MAX_AUDIT_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class ConstructorField:
    value: str
    owner: str
    field: str
    source: str
    constructors: tuple[str, ...]

    @property
    def components(self) -> tuple[str, ...]:
        return (f"field={self.field}", f"field_source={self.source}",
                "field_receiver=this", "field_policy=final_constructor_constant",
                "field_write_audit=decoded_smali", *(
                    f"field_constructor={header}" for header in self.constructors))


def without_comments(text: str) -> str:
    return _COMMENT.sub(lambda m: "" if m[0].startswith("#") else m[0], text)


def register_normalizer(header: str, body: str):
    local = re.search(r"(?m)^\s*\.locals\s+(\d+)\s*$", body)
    offset = int(local[1]) if local else None
    if offset is None:
        total = re.search(r"(?m)^\s*\.registers\s+(\d+)\s*$", body)
        descriptor = re.search(r"\(([^)]*)\)", header)
        if total and descriptor:
            params = re.findall(r"\[*L[^;]+;|\[*[ZBCSIFJD]", descriptor[1])
            slots = sum(2 if p in {"J", "D"} else 1 for p in params)
            offset = int(total[1]) - slots - (0 if "static" in header.split() else 1)

    def normalize(name: str) -> str:
        if offset is not None and name.startswith("v") and int(name[1:]) >= offset:
            return "p" + str(int(name[1:]) - offset)
        return name
    return normalize


def written_register(line: str) -> str | None:
    if line.startswith(("invoke-", "iput", "sput", "aput", "check-cast", "monitor-",
                        "if-", "return", "throw", "filled-new-array", "fill-array-data")):
        return None
    dest = re.match(r"[\w/-]+\s+([vp]\d+)(?:,|$)", line)
    return dest[1] if dest else None


def written_registers(line: str) -> tuple[str, ...]:
    dest = written_register(line)
    if dest is None:
        return ()
    op = line.split()[0]
    wide = (op.startswith(("move-wide", "move-result-wide", "const-wide", "iget-wide",
                           "sget-wide", "aget-wide"))
            or re.fullmatch(r"\w+-to-(?:long|double)", op)
            or re.fullmatch(r"(?:add|sub|mul|div|rem|and|or|xor|shl|shr|ushr|neg|not)-"
                            r"(?:long|double)(?:/2addr)?", op))
    return (dest, dest[0] + str(int(dest[1:]) + 1)) if wide else (dest,)


def stable_this(header: str, body: str) -> bool:
    if "static" in header.split():
        return False
    normalize = register_normalizer(header, body)
    return not any(normalize(dest) == "p0" for line in body.splitlines()
                   for dest in written_registers(line.strip()))


def _class_constants(owner: str, path: Path, text: str) -> dict[str, ConstructorField]:
    text = without_comments(text)
    fields = {}
    for match in re.finditer(r"(?m)^\s*\.field\s+([^\n]+)$", text):
        declaration = match[1].strip()
        parts = declaration.split()
        if ("final" in parts and "static" not in parts and "=" not in declaration
                and parts[-1].endswith(":Ljava/lang/String;")):
            fields[owner + "->" + parts[-1]] = []
    if not fields:
        return {}
    methods = list(METHOD.finditer(text))
    if len(methods) != len(re.findall(r"(?m)^\s*\.method\s", text)):
        return {}
    reads = {match[3] for method in methods
             if any(marker in method[2] for marker in ("->loadUrl(", "->loadData(", "->loadDataWithBaseURL(",
                                                       "->launchUrl(", "->launchTrustedWebActivity(", "TwaLauncher;->launch("))
             for line in method[2].splitlines() if (match := FIELD_READ.fullmatch(line.strip()))}
    fields = {key: values for key, values in fields.items() if key in reads}
    if not fields:
        return {}
    constructors = [m for m in methods if re.search(r"(?:^|\s)<init>\(", m[1])]
    if not constructors:
        return {}
    for method in constructors:
        header, body = method[1], method[2]
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if ("static" in header.split() or not lines or lines[-1] != "return-void"
                or any(line.startswith((":", "if-", "goto", "packed-switch", "sparse-switch",
                                        ".catch", "throw")) for line in lines)
                or any(line.startswith("return") for line in lines[:-1])
                or any(re.match(r"invoke-direct(?:/range)?\s", line)
                       and owner + "-><init>(" in line for line in lines)):
            return {}
        normalize = register_normalizer(header, body)
        values: dict[str, str] = {}
        aliases = {"p0"}
        assignments: dict[str, list[str | None]] = {key: [] for key in fields}
        for line in lines:
            literal, move = STRING.fullmatch(line), MOVE.fullmatch(line)
            write = _FIELD_WRITE.fullmatch(line)
            if write and write[3] in fields:
                assignments[write[3]].append(values.get(normalize(write[1]))
                                              if normalize(write[2]) in aliases else None)
            if literal:
                target = normalize(literal[1])
                values.pop(target, None)
                try:
                    values[target] = decode_smali_string(literal[2])
                except ValueError:
                    pass
                aliases.discard(target)
            elif move:
                target, source = normalize(move[1]), normalize(move[2])
                value, is_this = values.get(source), source in aliases
                values.pop(target, None)
                aliases.discard(target)
                if value is not None:
                    values[target] = value
                if is_this:
                    aliases.add(target)
            else:
                for dest in written_registers(line):
                    values.pop(normalize(dest), None)
                    aliases.discard(normalize(dest))
        for key, assigned in assignments.items():
            fields[key].append(assigned[0] if len(assigned) == 1 else None)
    result = {}
    all_writes = list(_WRITES.findall(text))
    for key, assigned in fields.items():
        if (all(value is not None for value in assigned) and len(set(assigned)) == 1
                and all_writes.count(key) == len(constructors)):
            result[key] = ConstructorField(assigned[0], owner, key, str(path),
                                           tuple(m[1].strip() for m in constructors))
    return result


def recover_constructor_fields(root: Path, sources: dict[str, tuple[Path, str]],
                               diagnostics: dict | None = None) -> dict[str, ConstructorField]:
    """Read outside classes only to veto writes, never to widen app evidence."""
    info = diagnostics if diagnostics is not None else {}
    info.update(status="not_needed", candidate_fields=0, eligible_fields=0,
                files_checked=0, bytes_checked=0)
    candidates = {}
    for owner, (path, text) in sources.items():
        try:
            candidates.update(_class_constants(owner, path, text))
        except (ValueError, IndexError):
            continue
    info["candidate_fields"] = len(candidates)
    if not candidates:
        return {}
    info["status"] = "complete"
    counts = {key: 0 for key in candidates}
    invalid = set()
    try:
        for dex_root in sorted(root.glob("smali*")):
            if not dex_root.is_dir():
                continue
            for path in dex_root.rglob("*.smali"):
                size = path.stat().st_size
                if (info["files_checked"] >= MAX_AUDIT_FILES or size > MAX_FILE_BYTES
                        or info["bytes_checked"] + size > MAX_AUDIT_BYTES):
                    info.update(status="partial", reason="write_audit_budget")
                    return {}
                text = path.read_text(encoding="utf-8")
                info["files_checked"] += 1
                info["bytes_checked"] += size
                # Most dependency files cannot reference these exact fields.
                # Avoid full lexical/regex work without skipping the read audit.
                if not any(key in text for key in candidates):
                    continue
                for key in _WRITES.findall(without_comments(text)):
                    if key in candidates:
                        counts[key] += 1
                        if path != Path(candidates[key].source):
                            invalid.add(key)
    except (OSError, UnicodeError):
        info.update(status="partial", reason="write_audit_unreadable")
        return {}
    result = {key: value for key, value in candidates.items()
              if key not in invalid and counts[key] == len(value.constructors)}
    info["eligible_fields"] = len(result)
    return result
