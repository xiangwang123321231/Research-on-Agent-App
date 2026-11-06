"""Local typed generation interfaces, including TVM packed-function adapters.

Only real Smali instructions participate. Constants flow along local CFG
edges; joins keep equal facts only. Cross-method handles must be initialized
consistently by every constructor and have no writes outside constructors.
This is not whole-app reachability or a claim of runtime execution.
"""
from __future__ import annotations

from collections import deque
import re

from smali_literals import decode_smali_string
from webview_fields import METHOD, register_normalizer, stable_this, without_comments, written_registers
from signal_types import RawSignal, SignalRole, SignalScope, SignalVerification, SignalType


INVOKE = re.compile(r'invoke-([\w/-]+)\s+\{([^}]*)\},\s*(L[^;\s]+;->[\w$<>]+\([^)]*\)\S+)$')
OWNER = re.compile(r'(?m)^\s*\.class\b[^\n]*\s(L[^;\s]+;)\s*$')
_ANNOTATION = re.compile(r'(?ms)^\s*\.(?:sub)?annotation\b.*?^\s*\.end (?:sub)?annotation\s*$')
_STRING = re.compile(r'const-string(?:/jumbo)?\s+([vp]\d+),\s*("(?:\\.|[^"\\])*")$')
_FIELD = re.compile(r'(iget-object|iput-object)\s+([vp]\d+),\s*([vp]\d+),\s*(L[^;]+;->[^\s:]+:L[^;]+;)$')
_FACTORY = 'Lorg/apache/tvm/Function;->getFunction(Ljava/lang/String;)Lorg/apache/tvm/Function;'
_FACTORY_NAMES = frozenset({'mlc.llm_chat_create', 'mlc_chat_create'})
_OPERATIONS = frozenset({'prefill', 'decode', 'get_message', 'reload', 'unload', 'reset_chat', 'stopped'})


def clean_smali(text):
    return _ANNOTATION.sub('', without_comments(text))


def invoke_registers(registers):
    if '..' in registers:
        m = re.fullmatch(r'\s*([vp])(\d+)\s*\.\.\s*\1(\d+)\s*', registers)
        if not m or int(m[3]) < int(m[2]):
            return []
        return [m[1] + str(i) for i in range(int(m[2]), int(m[3]) + 1)]
    return [v.strip() for v in registers.split(',') if re.fullmatch(r'[vp]\d+', v.strip())]


def typed_generation_runtime(target):
    """Exact owner + generation operation, never method-name-only matching."""
    if re.fullmatch(
            r'Lcom/google/mlkit/genai/prompt/GenerativeModel;->generateContent\('
            r'Lcom/google/mlkit/genai/prompt/GenerateContentRequest;'
            r'(?:Lkotlin/coroutines/Continuation;)?\)'
            r'(?:Ljava/lang/Object;|Lcom/google/mlkit/genai/prompt/GenerateContentResponse;|Lcom/google/android/gms/tasks/Task;)', target):
        return 'mlkit_genai_prompt'
    if re.fullmatch(r'Lcom/google/mediapipe/tasks/genai/llminference/LlmInference;->'
                    r'generateResponse(?:Async)?\(Ljava/lang/String;(?:L[^;]+;)*\)'
                    r'(?:Ljava/lang/String;|V|Lcom/google/common/util/concurrent/ListenableFuture;)', target):
        return 'mediapipe_llm'
    # Existing exact LiteRT-LM JNI identity, with generation operations only.
    if re.fullmatch(r'Lcom/google/ai/edge/litertlm/LiteRtLmJni;->'
                    r'native(?:GenerateContent(?:Stream)?|SendMessage(?:Async)?)\([^)]*\)\S+', target):
        return 'litert_lm'
    return None


def referenced_owners(text):
    """Concrete invoked constructors/methods eligible for a one-hop audit."""
    return {m[3].split(';->', 1)[0] + ';' for line in clean_smali(text).splitlines()
            if (m := INVOKE.fullmatch(line.strip()))}


def _step(line, state, normalize, owner):
    """Small finite value domain: strings, this, TVM handles, module results."""
    out = dict(state)
    pending = out.pop('$result', None)
    match = _STRING.fullmatch(line)
    if match:
        try:
            value = ('string', decode_smali_string(match[2]))
        except ValueError:
            value = None
        out.pop(normalize(match[1]), None)
        if value is not None:
            out[normalize(match[1])] = value
        return out
    move = re.fullmatch(r'move-object(?:/from16|/16)?\s+([vp]\d+),\s*([vp]\d+)', line)
    result = re.fullmatch(r'move-result-object\s+([vp]\d+)', line)
    field = _FIELD.fullmatch(line)
    invoke = INVOKE.fullmatch(line)
    value = None
    dest = None
    if move:
        dest, value = normalize(move[1]), state.get(normalize(move[2]))
    elif result:
        dest, value = normalize(result[1]), pending
    elif field:
        receiver = state.get(normalize(field[3]))
        key = '@' + field[4]
        if field[1] == 'iget-object':
            dest = normalize(field[2])
            value = state.get(key) if receiver == ('this', owner) else None
        elif field[4].startswith(owner + '->'):
            out.pop(key, None)
            if receiver == ('this', owner) and (value := state.get(normalize(field[2]))):
                out[key] = value
    elif invoke:
        regs = [normalize(r) for r in invoke_registers(invoke[2])]
        args = [state.get(r) for r in regs]
        target = invoke[3]
        if (invoke[1].startswith('static') and target == _FACTORY and len(args) == 1
                and args[0] and args[0][0] == 'string' and args[0][1] in _FACTORY_NAMES):
            value = ('function', 'create_engine')
        elif args and args[0]:
            receiver = args[0]
            if (target.startswith('Lorg/apache/tvm/Function;->pushArg(')
                    and target.endswith('Lorg/apache/tvm/Function;') and receiver[0] == 'function'):
                value = receiver
            elif target == 'Lorg/apache/tvm/Function;->invoke()Lorg/apache/tvm/TVMValue;' and receiver[0] == 'function':
                value = ('result', receiver[1])
            elif (target == 'Lorg/apache/tvm/TVMValue;->asModule()Lorg/apache/tvm/Module;'
                    and receiver == ('result', 'create_engine')):
                value = ('module', 'mlc_llm')
            elif (re.fullmatch(r'Lorg/apache/tvm/Module;->getFunction\(Ljava/lang/String;Z?\)Lorg/apache/tvm/Function;', target)
                    and receiver == ('module', 'mlc_llm') and len(args) >= 2
                    and args[1] and args[1][0] == 'string' and args[1][1] in _OPERATIONS):
                value = ('function', args[1][1])
        if value:
            out['$result'] = value
    else:
        for register in written_registers(line):
            out.pop(normalize(register), None)
    if dest:
        out.pop(dest, None)
        if value:
            out[dest] = value
    return out


def _flow(header, body, owner, fields):
    lines = [v.strip() for v in body.splitlines() if v.strip() and not v.strip().startswith('.')]
    normalize = register_normalizer(header, body)
    labels = {line: i for i, line in enumerate(lines) if line.startswith(':')}
    initial = dict(fields)
    if 'static' not in header.split():
        initial['p0'] = ('this', owner)
    states = [None for _ in lines]
    if not lines:
        return lines, states, normalize
    # Exception/switch payload edges are not reconstructed by this reader.
    # Independent typed calls still survive; dynamic-handle bindings do not.
    if '.catch' in body or any(v.startswith(('packed-switch', 'sparse-switch')) for v in lines):
        return lines, states, normalize
    states[0] = initial
    queue = deque([0])
    while queue:
        i = queue.popleft()
        line = lines[i]
        state = states[i]
        if line.startswith(':'):
            out = dict(state)
        else:
            out = _step(line, state, normalize, owner)
        if line.startswith(('return', 'throw')):
            successors = []
        elif line.startswith('goto'):
            successors = [labels[line.split()[-1]]] if line.split()[-1] in labels else []
        elif line.startswith('if-'):
            target = line.split()[-1]
            successors = ([i + 1] if i + 1 < len(lines) else []) + ([labels[target]] if target in labels else [])
        else:
            successors = [i + 1] if i + 1 < len(lines) else []
        for nxt in successors:
            previous = states[nxt]
            merged = dict(out) if previous is None else {k: v for k, v in previous.items() if out.get(k) == v}
            if previous != merged:
                states[nxt] = merged
                queue.append(nxt)
    return lines, states, normalize


def _constructor_fields(methods, owner, text):
    constructors = [m for m in methods if m[1].split()[-1].startswith('<init>(')]
    if not constructors:
        return {}
    candidates = None
    private_fields = {'@' + owner + '->' + m[1].split()[-1]
                      for m in re.finditer(r'(?m)^\s*\.field\s+([^\n]+)$', text)
                      if 'private' in m[1].split() and 'static' not in m[1].split() and '=' not in m[1]}
    for method in constructors:
        if not stable_this(method[1], method[2]):
            return {}
        lines, states, _ = _flow(method[1], method[2], owner, {})
        exits = [s for line, s in zip(lines, states) if line == 'return-void' and s is not None]
        if not exits:
            return {}
        for state in exits:
            handles = {k: v for k, v in state.items() if k in private_fields
                       and v[0] in {'function', 'module'}}
            candidates = handles if candidates is None else {k: v for k, v in candidates.items() if handles.get(k) == v}
    # No borrowing initialization from one constructor when another omits it,
    # or when a setter/writer can replace it later.
    for method in methods:
        if method in constructors:
            continue
        for line in method[2].splitlines():
            match = _FIELD.fullmatch(line.strip())
            if match and match[1] == 'iput-object':
                candidates.pop('@' + match[4], None)
    return candidates or {}


def extract_device_calls(text, source, diagnostics=None):
    text = clean_smali(text)
    match = OWNER.search(text)
    if not match:
        return []
    owner = match[1]
    methods = list(METHOD.finditer(text))
    fields = _constructor_fields(methods, owner, text) if 'Lorg/apache/tvm/' in text else {}
    signals = []
    for method in methods:
        if ('Lorg/apache/tvm/' not in method[2]
                and not any(typed_generation_runtime(m[3]) for line in method[2].splitlines()
                            if (m := INVOKE.fullmatch(line.strip())))):
            continue
        signature = method[1].split()[-1]
        initial = {} if signature.startswith('<init>(') else fields
        lines, states, normalize = _flow(method[1], method[2], owner, initial)
        if diagnostics is not None and 'Lorg/apache/tvm/' in method[2] and (
                '.catch' in method[2] or any(v.startswith(('packed-switch', 'sparse-switch')) for v in lines)):
            diagnostics['unsupported_runtime_binding_methods'] = diagnostics.get('unsupported_runtime_binding_methods', 0) + 1
        # When a CFG form is unsupported, preserve a reproducible factory
        # lookup from one basic block as a reference. Never promote that
        # fallback into a generated-text invocation or field summary.
        local = {}
        seen = set()
        for number, (line, state) in enumerate(zip(lines, states)):
            before_local = local
            local = {} if line.startswith((':', 'if-', 'goto', 'return', 'throw',
                                           'packed-switch', 'sparse-switch')) else _step(line, local, normalize, owner)
            invoke = INVOKE.fullmatch(line)
            if not invoke:
                continue
            target = invoke[3]
            runtime = typed_generation_runtime(target)
            if runtime and runtime != 'litert_lm' and not invoke[1].startswith(('virtual', 'interface')):
                runtime = None
            operation = 'text_generation' if runtime else None
            role = SignalRole.CALL
            evidence = 'typed_sdk_invocation'
            if not runtime and state is not None:
                after = _step(line, state, normalize, owner)
                result = after.get('$result')
                if result and result[0] == 'function' and result[1] in _OPERATIONS | {'create_engine'}:
                    runtime, operation = 'mlc_llm', result[1]
                    role, evidence = SignalRole.REFERENCE, 'typed_runtime_function_lookup'
                elif (target == 'Lorg/apache/tvm/Function;->invoke()Lorg/apache/tvm/TVMValue;'
                      and result and result[0] == 'result'):
                    runtime, operation, evidence = 'mlc_llm', result[1], 'typed_runtime_function_invoke'
            elif not runtime and state is None and target == _FACTORY:
                result = _step(line, before_local, normalize, owner).get('$result')
                if result == ('function', 'create_engine'):
                    runtime, operation = 'mlc_llm', 'create_engine'
                    role, evidence = SignalRole.REFERENCE, 'typed_runtime_function_lookup'
            if not runtime or (runtime, operation, role) in seen:
                continue
            seen.add((runtime, operation, role))
            signals.append(RawSignal(
                SignalType.ON_DEVICE_GENERATION_USAGE, runtime, 'apktool', source,
                role=role, scope=SignalScope.METHOD, verification=SignalVerification.VERIFIED,
                source_kind='smali_device_generation_interface',
                components=(f'runtime={runtime}', 'inference_location=same_device',
                            f'operation={operation}', f'binding={evidence}',
                            'evidence=static_invoke' if role == SignalRole.CALL else 'evidence=interface_reference',
                            f'caller={owner}->{signature}', f'target={target}',
                            f'instruction_index={number}', *(
                                ('runtime_function_family=mlc.llm_chat_create',) if runtime == 'mlc_llm' else ())),
            ))
    return signals
