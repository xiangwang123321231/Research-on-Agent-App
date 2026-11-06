"""Local constant string compositions; no execution or proximity-only joins.

Only literals, aliases, parentheses and final comma-expression values are
resolved. Reassignments, parameter shadowing and other expressions stay
unknown. This complements the existing request/call analyzers.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import re

_ID = r'[A-Za-z_$][\w$]*'
_TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`|//[^\r\n]*|/\*[\s\S]*?\*/')
_ASSIGN = re.compile(r'(?<![\w.$])(' + _ID + r')\s*=(?!=|>)')


def _decode(token):
    if token.startswith('"'):
        try:
            return json.loads(token)
        except ValueError:
            return None
    value = token[1:-1]
    if re.search(r'\\[^\\/\'"`]', value):
        return None
    return re.sub(r'\\([\\/\'"`])', r'\1', value)


def split_endpoint_literal_pairs(text: str, radius: int):
    """Yield base/route values only from explicit local string compositions."""
    tokens = [m for m in _TOKENS.finditer(text) if not m[0].startswith(('//', '/*'))]
    # Preserve offsets while hiding comments and all quoted contents from
    # assignment/use matching. In particular, URL query text is never code.
    mask = _TOKENS.sub(lambda m: ' ' * len(m[0]), text)
    starts = [m.start() for m in tokens]
    by_start = {m.start():m for m in tokens}
    positions, scopes, stack = [0], [()], []
    for m in re.finditer(r'[{}]', mask):
        if m[0] == '{':
            stack.append(m.start())
        elif stack:
            stack.pop()
        positions.append(m.end())
        scopes.append(tuple(stack))

    def scope_at(position):
        return scopes[bisect_right(positions,position)-1]

    def ancestor(parent, child):
        return child[:len(parent)] == parent

    parameters = defaultdict(set)
    for m in re.finditer(r'\bfunction\b[^(){};]*\(([^)]*)\)\s*\{|(?:\(([^)]*)\)|(' + _ID + r'))\s*=>\s*\{', mask):
        names = m[1] if m[1] is not None else m[2] if m[2] is not None else m[3]
        parameters[scope_at(m.end())].update(re.findall(_ID,names))
    for m in re.finditer(r'\b(?:void|String|boolean|int|long|Object)\s+' + _ID + r'\s*\(([^)]*)\)\s*\{', mask):
        parameters[scope_at(m.end())].update(re.findall(_ID,m[1]))
    assignments = defaultdict(list)
    for m in _ASSIGN.finditer(mask):
        assignments[m[1]].append(m)

    def scan_expression(start, stop):
        stack, commas = [], []
        i = start
        while i < stop:
            token = by_start.get(i)
            if token:
                i = token.end()
                continue
            c = mask[i]
            if c in '([{':
                stack.append(c)
            elif c in ')]}':
                if not stack:
                    break
                if stack.pop() != {')':'(', ']':'[', '}':'{'}[c]:
                    return i, []
            elif not stack and c in ',;':
                if c == ',':
                    commas.append(i)
                break
            i += 1
        return i, commas

    def constant(expression, active, position):
        expression = expression.strip()
        if len(expression) < 2:
            resolved = resolve(expression, position, active)
            return resolved[0] if resolved else None
        if _TOKENS.fullmatch(expression) and expression[0] in '\'"`':
            value = _decode(expression)
            return value if value is not None and '${' not in value else None
        if re.fullmatch(_ID, expression):
            resolved = resolve(expression, position, active)
            return resolved[0] if resolved else None
        # A parenthesized sequence has the value of its final expression.
        # Do not evaluate a call, ternary, property access or binary operator.
        if expression.startswith('(') and expression.endswith(')'):
            local_mask = _TOKENS.sub(lambda m:' ' * len(m[0]), expression)
            depth, comma = 0, None
            for i,c in enumerate(local_mask):
                if c in '([{':
                    depth += 1
                elif c in ')]}':
                    depth -= 1
                    if depth == 0 and i != len(expression)-1:
                        return None
                elif c == ',' and depth == 1:
                    comma = i
            if depth == 0:
                return constant(expression[(comma+1) if comma is not None else 1:-1], active, position)
        return None

    def resolve(name, position, active=frozenset()):
        use_scope = scope_at(position)
        definitions = [m for m in assignments.get(name, ())
                       if ancestor(scope_at(m.start()),use_scope)]
        if not definitions:
            return None
        owner = max((scope_at(m.start()) for m in definitions), key=len)
        definitions = [m for m in definitions if scope_at(m.start()) == owner]
        identity = (name,owner)
        if len(definitions) != 1 or identity in active or definitions[0].start() >= position:
            return None
        if any(name in names and ancestor(owner,scope) and ancestor(scope,use_scope)
               for scope,names in parameters.items()):
            return None
        # A write in a nested branch/closure can mutate this binding. A
        # separately declared child binding belongs to that child's scope.
        for m in assignments.get(name, ()):
            target = scope_at(m.start())
            if target == owner or not ancestor(owner,target):
                continue
            if re.search(r'\b(?:const|let|var|String)\s*$', mask[max(0,m.start()-32):m.start()]):
                continue
            if any(name in names and ancestor(scope,target) and ancestor(owner,scope)
                   for scope,names in parameters.items()):
                continue
            return None
        escaped = re.escape(name)
        mutation = (r'(?<![\w$])' + escaped
                    + r'\s*(?:\+\+|--|(?:[+*/%&|^\-]|&&|\|\||\?\?)=)'
                    + r'|(?:\+\+|--)\s*' + escaped + r'(?![\w$])')
        if any(ancestor(owner,scope_at(m.start())) for m in re.finditer(mutation, mask)):
            return None
        # An expression-bodied arrow has no brace scope in this adapter.
        if re.search(r'(?:\([^)]*(?<![\w$])' + escaped + r'(?![\w$])[^)]*\)|(?<![\w$])' + escaped + r')\s*=>\s*(?!\s*\{)', mask):
            return None
        m = definitions[0]
        end,_ = scan_expression(m.end(), len(text))
        try:
            value = constant(text[m.end():end], active | {identity},m.start())
            return (value,m.start()) if value is not None else None
        except RecursionError:
            return None

    def pair(base, route, position, binding, base_position):
        if (not isinstance(base,str) or not base.lower().startswith(('http://','https://'))
                or not isinstance(route,str) or not route.startswith('/')
                or route.startswith('//') or abs(position-base_position)>radius):
            return
        yield base,route,position,binding

    for token in tokens:
        value = _decode(token[0])
        if value is None:
            continue
        template = re.match(r'^\$\{(' + _ID + r')\}(/[^`]*?)$', value)
        if token[0].startswith('`') and template:
            name = template[1]
            route = template[2].split('${',1)[0]
            resolved = resolve(name,token.start())
            if resolved:
                yield from pair(resolved[0],route,token.start(),'template_constant',resolved[1])
        if not value.startswith('/') or value.startswith('//'):
            continue
        prefix = mask[max(0,token.start()-256):token.start()]
        name = re.search(r'(?<![\w.$])(' + _ID + r')\s*\+\s*$', prefix)
        if name:
            resolved = resolve(name[1],token.start())
            if resolved:
                yield from pair(resolved[0],value,token.start(),'constant_concatenation',resolved[1])
        index = bisect_left(starts,token.start())-1
        if index>=0:
            previous=tokens[index]
            if text[previous.end():token.start()].strip()=='+':
                yield from pair(_decode(previous[0]),value,token.start(),
                                'literal_concatenation',previous.start())
    # Both operands can themselves be named literal constants.
    for m in re.finditer(r'(?<![\w.$])(' + _ID + r')\s*\+\s*(' + _ID + r')(?![\w$])', mask):
        base,route=resolve(m[1],m.start()),resolve(m[2],m.start())
        if base and route:
            yield from pair(base[0],route[0],m.start(),'constant_concatenation',base[1])
