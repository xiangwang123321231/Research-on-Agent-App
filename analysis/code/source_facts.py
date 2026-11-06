"""Bounded syntax facts, without APK identities or category decisions.

This is a conservative adapter, not a complete Java/Kotlin/JS parser. Missing
syntax is unknown; words in comments, packages or unrelated objects are never
used to invent a serialized request.
"""
from __future__ import annotations

from collections import defaultdict
from bisect import bisect_right
import json
import re


_TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\r\n]*|/\*[\s\S]*?\*/')


def without_comments(text: str) -> str:
    return _TOKENS.sub(lambda m: " " * len(m[0]) if m[0].startswith(("//", "/*")) else m[0], text)


def smali_serialized_field_names(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for field in re.finditer(r'(?m)^\s*\.field\s+([^\n]*?)\b([\w$]+):[^\n]+', text):
        if re.search(r'\b(?:static|transient)\b', field[1]):
            continue
        tail = text[field.end():]
        stop = re.search(r"(?m)^\s*\.(?:field|method|end field)\b", tail)
        annotation = tail[:stop.start()] if stop else ""
        serialized = re.search(r'\b(?:name|value)\s*=\s*"([^"\\]+)"', annotation)
        fields[field[2]] = (serialized[1] if serialized else field[2]).lower()
    owner = re.search(r'(?m)^\s*\.class\b[^\n]*\s(L[^;\s]+;)\s*$', text)
    aliases: dict[str, set[str | None]] = defaultdict(set)
    # Kotlin/Moshi and other serializers can annotate constructor parameters
    # instead of fields. Bind their names through actual writes to this DTO.
    for method in re.finditer(r'(?ms)^\s*\.method\s+[^\n]*\bconstructor\s+<init>[^\n]*\n(.*?)^\s*\.end method', text):
        body = method[1]
        if not owner or re.search(r'(?m)^\s*(?:if-|goto|packed-switch|sparse-switch)', body):
            continue
        parameters = {}
        for param in re.finditer(r'(?ms)^\s*\.param\s+(p\d+)[^\n]*\n(.*?)^\s*\.end param', body):
            annotation = re.search(r'(?s)\.annotation\s+runtime\s+L[^;]+;(.*?)\.end annotation', param[2])
            name = re.search(r'\b(?:name|value)\s*=\s*"([^"\\]+)"', annotation[1]) if annotation else None
            if name:
                parameters[param[1]] = name[1].lower()
        bindings = {}
        for line in body.splitlines():
            move = re.match(r'\s*move(?:-object|-wide)?(?:/from16|/16)?\s+([vp]\d+),\s*([vp]\d+)\s*$', line)
            write = re.match(r'\s*iput(?:-[\w]+)?\s+([vp]\d+),\s*p0,\s*(L[^;]+;)->([\w$]+):', line)
            if move:
                parameters[move[1]] = parameters.get(move[2])
            elif write and write[2] == owner[1] and write[3] in fields:
                bindings[write[3]] = parameters.get(write[1])
            else:
                dest = re.match(r'\s*[\w/-]+\s+([vp]\d+)(?:,|$)', line)
                if dest and not line.strip().startswith(('invoke-', 'iput', 'sput', 'aput')):
                    parameters.pop(dest[1], None)
        for name, alias in bindings.items():
            aliases[name].add(alias)
    for name, values in aliases.items():
        if len(values) == 1 and None not in values:
            fields[name] = next(iter(values))
        elif len(values) > 1:
            fields.pop(name, None)
    return fields


def smali_serialized_fields(text: str) -> frozenset[str]:
    return frozenset(smali_serialized_field_names(text).values())


def smali_serialized_field_types(text: str, names: dict[str, str] | None = None) -> dict[str, frozenset[str]]:
    """Bind DTO composition to the actual serialized field, not a word bag."""
    if names is None:
        names = smali_serialized_field_names(text)
    result: dict[str, set[str]] = defaultdict(set)
    for field in re.finditer(r'(?m)^\s*\.field\s+[^\n]*?\b([\w$]+):([^\n]+)', text):
        if field[1] not in names:
            continue
        tail = text[field.end():]
        stop = re.search(r'(?m)^\s*\.(?:field|method|end field)\b', tail)
        annotation = tail[:stop.start()] if stop else ''
        signature = re.search(r'(?s)\.annotation\s+system\s+Ldalvik/annotation/Signature;(.*?)\.end annotation', annotation)
        type_text = field[2].split('=', 1)[0] + (signature[1] if signature else '')
        result[names[field[1]]].update(re.findall(r'L[^;<>\s"{}]+;', type_text))
    return {name: frozenset(types) for name, types in result.items()}


def serialized_field_groups(text: str) -> list[frozenset[str]]:
    """Return independent JSON objects, DTO classes and JSON-write receivers."""
    text = without_comments(text)
    groups: list[frozenset[str]] = []
    decoder = json.JSONDecoder()
    # Decode embedded JSON, but do not union sibling request/config objects.
    end = 0
    attempts = 0
    for match in re.finditer(r"\{", text):
        if match.start() < end:
            continue
        attempts += 1
        if attempts > 512:
            break
        try:
            obj, length = decoder.raw_decode(text[match.start():])
        except (ValueError, RecursionError):
            continue
        end = match.start() + length

        def visit(value, depth=0):
            if depth > 16:
                return
            if isinstance(value, dict):
                fields = {str(k).lower() for k in value}
                messages = value.get("messages")
                if isinstance(messages, list):
                    for message in messages[:128]:
                        if isinstance(message, dict):
                            fields.update(str(k).lower() for k in message)
                            if message.get("role") in ("system", "user", "assistant"):
                                fields.add(message["role"])
                if value.get("role") in ("system", "user", "assistant"):
                    fields.add(value["role"])
                groups.append(frozenset(fields))
                for child in value.values():
                    visit(child, depth + 1)
            elif isinstance(value, list):
                for child in value[:128]:
                    visit(child, depth + 1)
        visit(obj)

    # Class-level Java fields only: exclude package/import and method locals.
    stack: list[str | None] = []
    class_fields: dict[str, set[str]] = defaultdict(set)
    token_re = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{};]')
    start = 0
    serial = 0
    for token in token_re.finditer(text):
        if token[0][0] in "\"'":
            continue
        statement = text[start:token.start()]
        if token[0] == "{":
            if re.search(r"\b(?:class|interface|record)\s+[\w$]+", statement):
                serial += 1
                stack.append(str(serial))
            else:
                stack.append(None)
        elif token[0] == "}":
            if stack:
                stack.pop()
        elif stack and stack[-1] is not None:
            field = re.search(r'(?:^|\s)([\w.$]+(?:\s*<[^;{}]+>)?(?:\[\])?)\s+([\w$]+)\s*(?:=[\s\S]*)?$', statement.strip())
            declaration = re.sub(r'"(?:\\.|[^"\\])*"', '""', statement)
            if field and not re.search(r"\b(?:package|import|return|throw|static|transient)\b", declaration):
                names = re.findall(r'@[^\s(]+\(\s*(?:name|value)\s*=\s*"([^"\\]+)"\s*\)|@(?:SerializedName|JsonProperty)\(\s*"([^"\\]+)"\s*\)', statement)
                name = next((a or b for a, b in names), field[2])
                class_fields[stack[-1]].add(name.lower())
        start = token.end()
    groups.extend(frozenset(v) for v in class_fields.values())

    # Smali .field declarations and their serialization annotations.
    smali_fields = smali_serialized_fields(text)
    if smali_fields:
        groups.append(frozenset(smali_fields))

    # Explicit JSON writes, separated by receiver. No lexical word bag.
    writes: dict[tuple, set[str]] = defaultdict(set)
    positions, scopes, nesting = [0], [()], []
    for token in token_re.finditer(text):
        if token[0] == "{":
            nesting.append(token.start())
        elif token[0] == "}" and nesting:
            nesting.pop()
        else:
            continue
        positions.append(token.end())
        scopes.append(tuple(nesting))
    allocation_positions: dict[str, list[int]] = defaultdict(list)
    # Any assignment invalidates the previous allocation, including aliases
    # and unknown values. This adapter only joins an explicit new object.
    for allocation in re.finditer(r'\b([\w$]+)\s*=(?!=)', text):
        allocation_positions[allocation[1]].append(allocation.start())
    for declaration in re.finditer(r'\b(?:JSONObject|JsonObject)\s+([\w$]+)\s*;', text):
        allocation_positions[declaration[1]].append(declaration.start(1))
    for assignments in allocation_positions.values():
        assignments.sort()
    for m in re.finditer(r'\b([\w$]+)\.(?:put|addProperty)\(\s*["\']([\w_]+)["\']\s*,', text):
        scope = scopes[bisect_right(positions, m.start()) - 1]
        # Reassigning a local receiver starts a new object, even in one method.
        assignments = allocation_positions[m[1]]
        position = bisect_right(assignments, m.start()) - 1
        allocation = assignments[position] if position >= 0 else -1
        writes[(scope, m[1], allocation)].add(m[2].lower())
    # Same allocation + ancestor block: common writes before/after an if apply
    # to that branch. Sibling branches are NEVER unioned. A new allocation,
    # unknown assignment or a parameter has no such inferred relationship.
    for (scope, receiver, allocation), fields in list(writes.items()):
        if allocation < 0 or not re.match(r'[\w$]+\s*=\s*new\s+', text[allocation:]):
            continue
        combined = set(fields)
        for depth in range(len(scope)):
            combined.update(writes.get((scope[:depth], receiver, allocation), ()))
        groups.append(frozenset(combined))
    constants: dict[str, str] = {}
    allocations: dict[str, int] = {}
    method_scope = 0
    for line in text.splitlines():
        if line.strip().startswith((".method", ".end method", "if-", "goto", ":")):
            constants.clear()
            method_scope += 1
            allocations.clear()
        new = re.search(r'new-instance\s+([vp]\d+),', line)
        if new:
            allocations[new[1]] = allocations.get(new[1], 0) + 1
        m = re.search(r'const-string(?:/jumbo)?\s+([vp]\d+),\s*"([\w_]+)"', line)
        if m:
            constants[m[1]] = m[2].lower()
        m = re.search(r'invoke-\S+\s+\{([^}]+)\},\s*Lorg/json/JSONObject;->put\(', line)
        if m:
            regs = [v.strip() for v in m[1].split(",")]
            if len(regs) >= 2 and regs[1] in constants:
                writes[(method_scope, regs[0], allocations.get(regs[0], 0))].add(constants[regs[1]])
        if not m:
            dest = re.match(r"\s*[\w/-]+\s+([vp]\d+)(?:,|$)", line)
            if dest and not line.strip().startswith(("const-string", "iput", "sput", "aput", "invoke-")):
                constants.pop(dest[1], None)
    groups.extend(frozenset(v) for v in writes.values())
    return groups
