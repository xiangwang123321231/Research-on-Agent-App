"""Recognize a complete pure constant-byte-array XOR decoder statically.

Only calls to an unambiguous declaration matching the entire supported
template and two fresh literal arrays qualify. APK code is never executed.
"""
import re
from source_facts import without_comments

_DECL = re.compile(r'public\s+static\s+String\s+([\w$]+)\s*\(byte\[\]\s+(\w+),\s*byte\[\]\s+(\w+)\)\s*\{')
_ARRAY = r'new\s+byte\[\]\s*\{([\s\d,+-]*)\}'
_CALL = re.compile(r'(?<![\w$.])([\w$]+(?:\.[\w$]+)*)\s*\(\s*' + _ARRAY + r'\s*,\s*' + _ARRAY + r'\s*\)')


def _code(text):
    return re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                  lambda m:' '*len(m[0]), without_comments(text))


def decoder_declarations(text):
    code = _code(text)
    package = re.search(r'(?m)^\s*package\s+([\w.]+)\s*;', code)
    owner = re.search(r'\b(?:public\s+)?(?:final\s+)?class\s+([\w$]+)', code)
    if not owner:
        return []
    qualified = ((package[1]+'.') if package else '') + owner[1]
    found = []
    for m in _DECL.finditer(code):
        depth, end = 1, m.end()
        while end < len(code) and depth:
            if code[end] == '{': depth += 1
            if code[end] == '}': depth -= 1
            end += 1
        if depth:
            continue
        body = code[m.end():end-1]
        pat = (r'\s*int\s+(\w+)\s*=\s*'+re.escape(m[2])+r'\.length;\s*'
               r'int\s+(\w+)\s*=\s*'+re.escape(m[3])+r'\.length;\s*'
               r'int\s+(\w+)\s*=\s*0;\s*int\s+(\w+)\s*=\s*0;\s*'
               r'while\s*\(\s*\3\s*<\s*\1\s*\)\s*\{\s*'
               r'if\s*\(\s*\4\s*>=\s*\2\s*\)\s*\{\s*\4\s*=\s*0;\s*\}\s*'
               +re.escape(m[2])+r'\[\3\]\s*=\s*\(byte\)\s*\(\s*'
               +re.escape(m[2])+r'\[\3\]\s*\^\s*'+re.escape(m[3])+r'\[\4\]\s*\);\s*'
               r'\3\+\+;\s*\4\+\+;\s*\}\s*return\s+new\s+String\(\s*'
               +re.escape(m[2])+r',\s*(?:java\.nio\.charset\.)?StandardCharsets\.UTF_8\s*\);\s*')
        if re.fullmatch(pat, body):
            found.append(qualified+'.'+m[1])
    return found


def decoded_literals(text, declarations):
    code = _code(text)
    package = re.search(r'(?m)^\s*package\s+([\w.]+)\s*;', code)
    owner = re.search(r'\b(?:public\s+)?(?:final\s+)?class\s+([\w$]+)', code)
    imports = re.findall(r'(?m)^\s*import\s+([\w.]+)\s*;', code)
    for m in _CALL.finditer(code):
        name = m[1]
        candidates = {name}
        if package:
            candidates.add(package[1]+'.'+name)
        if '.' not in name and owner:
            candidates.add(((package[1]+'.') if package else '')+owner[1]+'.'+name)
        if '.' in name:
            cls, method = name.split('.', 1)
            if re.search(r'\b[\w.$]+(?:\[\])?\s+'+re.escape(cls)+r'\s*(?:[=,;)]|\[)', code):
                continue  # A local/parameter may shadow the class qualifier.
            candidates.update(i+'.'+method for i in imports if i.rsplit('.',1)[-1] == cls)
        matched = candidates & declarations
        if len(matched) != 1:
            continue
        try:
            data, key = [[int(v.strip()) for v in group.split(',') if v.strip()] for group in (m[2],m[3])]
            if not key or not all(-128 <= v <= 127 for arr in (data,key) for v in arr):
                continue
            value = bytes((v & 255) ^ (key[i % len(key)] & 255) for i,v in enumerate(data)).decode('utf-8')
            if any(ord(c) < 32 and c not in '\t\r\n' for c in value):
                continue
        except (ValueError, UnicodeError):
            continue
        yield value, next(iter(matched)), m.start()


def has_constant_calls(text):
    return bool(_CALL.search(_code(text)))
