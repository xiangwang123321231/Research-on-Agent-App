"""Protocol and serialized type summaries for bounded Smali value analysis.

The summaries are framework mechanisms, never business domain/class lists.
An interface declaration alone is a candidate, not an endpoint relation.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
import re

from registry import _GENERATION_PARAMETER_KEYS, _is_llm_schema
from source_facts import smali_serialized_field_names, smali_serialized_field_types
from signal_types import RawSignal, SignalRelation, SignalRole, SignalScope, SignalType, SignalVerification

_CLASS = re.compile(r"(?m)^\s*\.class\b[^\n]*\s(L[^;\s]+;)\s*$")
_METHOD = re.compile(r"(?ms)^\s*\.method\s+([^\n]+)\n(.*?)^\s*\.end method")
_TYPES = re.compile(r"L[^;<>\s\"{}]+;")
_CALL = re.compile(r'(?m)^\s*invoke-[\w/-]+\s+\{[^}]*\},\s*(L[^;]+;->\S+)')
_PARAM = re.compile(r'(?ms)^\s*\.param\s+p(\d+)[^\n]*\n(.*?)^\s*\.end param')
_ANNOTATION = re.compile(r'(?ms)^\s*\.annotation\s+runtime\s+(L[^;]+;)(.*?)^\s*\.end annotation')
# A provider namespace alone also contains uploads, billing and history APIs.
# Lexical candidates need an inference action and request/stream evidence.
_INFERENCE_ROUTE = re.compile(
    r'(?:^|/)(?:v\d+/)?chat/completions?(?:[/?#]|$)|'
    r'(?:^|/)v\d+/(?:responses|completions?)(?:[/?#]|$)|'
    r'(?:^|[/.:])generatecontent(?:[/?#]|$)|'
    r'(?:^|/)(?:chatgpt|llm|gpt|claude|gemini)/(?:'
    r'send(?:gpt|prompt|message)|chat|completions?|responses?|stream|generate|infer|ask|message)'
    r'(?:[/?#.]|$)', re.I)
_TEXT_INPUT_FIELDS = frozenset({'prompt', 'messages', 'message', 'question', 'query', 'input', 'text', 'content'})
_CONTROL_FIELD_NAMES = frozenset(name.replace('_', '') for name in _GENERATION_PARAMETER_KEYS)


def _dto_type(owner: str) -> bool:
    return not owner.startswith(('Ljava/', 'Ljavax/', 'Landroid/', 'Landroidx/', 'Lkotlin/', 'Lkotlinx/', 'Lretrofit'))


def _signature_parameters(signature: str) -> tuple[str, ...] | None:
    """Split JVM generic parameters without mixing nested input and output."""
    if '(' not in signature or len(signature) > 16384:
        return None
    pos = signature.index('(') + 1
    parameters = []
    while pos < len(signature) and signature[pos] != ')':
        start = pos
        while pos < len(signature) and signature[pos] == '[':
            pos += 1
        if pos >= len(signature):
            return None
        if signature[pos] in 'ZBCSIJFD':
            pos += 1
        elif signature[pos] in 'LT':
            depth = 0
            while pos < len(signature):
                char = signature[pos]
                if char == '<':
                    depth += 1
                    if depth > 64:
                        return None
                elif char == '>':
                    depth -= 1
                    if depth < 0:
                        return None
                elif char == ';' and depth == 0:
                    pos += 1
                    break
                elif char in '()':
                    return None
                pos += 1
            else:
                return None
        else:
            return None
        parameters.append(signature[start:pos])
    return tuple(parameters) if pos < len(signature) and signature[pos] == ')' else None


def contract_type_references(text: str) -> set[str]:
    """Dependency priority hints, never protocol or LLM observations.

    Runtime-annotated abstract methods cover renamed HTTP annotations too.
    The actual protocol index still verifies HTTP semantics independently.
    """
    references = set()
    for method in _METHOD.finditer(text):
        if ' abstract ' not in f' {method[1]} ' or not _ANNOTATION.search(method[2]):
            continue
        declaration = method[1].rsplit(None, 1)[-1]
        references.update(_TYPES.findall(declaration))
        for annotation in re.finditer(r'(?s)\.annotation\s+system\s+Ldalvik/annotation/Signature;(.*?)\.end annotation', method[2]):
            signature = ''.join(re.findall(r'"([^"\\]*)"', annotation[1]))
            references.update(_TYPES.findall(signature))
    return references


@dataclass(frozen=True)
class ApiMethod:
    key: str
    route: str
    body_slot: int
    body_type: str
    response_types: tuple[str, ...]
    fields: tuple[str, ...] = ()
    request_kind: str = "body"
    parameter_fields: tuple[tuple[int, str], ...] = ()
    evidence: tuple[str, ...] = ()
    streaming: bool = False
    request_fields: tuple[str, ...] = ()
    response_fields: tuple[str, ...] = ()


@dataclass
class ProtocolIndex:
    api_methods: dict[str, ApiMethod] = field(default_factory=dict)
    all_api_methods: dict[str, ApiMethod] = field(default_factory=dict)
    candidate_api_methods: dict[str, ApiMethod] = field(default_factory=dict)
    fields: dict[str, frozenset[str]] = field(default_factory=dict)
    type_edges: dict[str, frozenset[str]] = field(default_factory=dict)
    field_types: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)
    invocations: dict[str, set[str]] = field(default_factory=dict)
    missing_types: set[str] = field(default_factory=set)
    recovered_type_count: int = 0
    proxy_factories: set[str] = field(default_factory=lambda: {"Lretrofit2/Retrofit;", "Lretrofit/RestAdapter;"})
    parse_errors: int = 0
    limited: bool = False
    source_paths: dict[str, str] = field(default_factory=dict)
    field_readers: dict[tuple[str, str], set[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, paths: list[Path], limit: int | None = None, *, available_paths: list[Path] | None = None,
              recovery_limit: int | None = None) -> ProtocolIndex:
        index = cls(limited=limit is not None and len(paths) > limit)
        candidates = []
        coroutine_parameters = {}
        continuation_shapes = set()
        type_parents = {}
        method_records = []
        http_annotations = {f"Lretrofit{version}/http/{verb};" for version in ("", "2") for verb in ("POST", "PUT", "GET", "PATCH")}
        body_annotations = {"Lretrofit/http/Body;", "Lretrofit2/http/Body;"}

        def read_type(path, expected=None):
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                index.parse_errors += 1
                return None
            match = _CLASS.search(text)
            if not match:
                return None
            owner = match[1]
            if expected is not None and owner != expected:
                return None
            names = smali_serialized_field_names(text)
            index.source_paths[owner] = str(path)
            for method in _METHOD.finditer(text):
                # An invoked getter must read the actual declared field. Its
                # obfuscated method name supplies no serialization evidence.
                reads = re.findall(r'(?m)^\s*iget-object\s+(\w+),\s*p0,\s*'+re.escape(owner)+r'->([\w$]+):', method[2])
                if len(reads) == 1 and reads[0][1] in names and not _CALL.search(method[2]):
                    reg, field_name = reads[0]
                    accessor_body = re.sub(r'(?ms)^\s*\.annotation\b.*?^\s*\.end annotation\s*$', '', method[2])
                    instructions = [line.strip() for line in accessor_body.splitlines()
                                    if line.strip() and not line.strip().startswith(('.', '#'))]
                    if (len(instructions) == 2 and instructions[0].startswith('iget-object ')
                            and instructions[1] == 'return-object '+reg):
                        key = owner+'->'+method[1].rsplit(None,1)[-1]
                        index.field_readers.setdefault((owner, names[field_name]), set()).add(key)
            index.fields[owner] = frozenset(names.values())
            index.field_types[owner] = smali_serialized_field_types(text, names)
            index.type_edges[owner] = frozenset(t for types in index.field_types[owner].values() for t in types) - {owner}
            type_parents[owner] = set(re.findall(r'(?m)^\s*\.(?:implements|super)\s+(L[^;\s]+;)\s*$', text))
            # R8 may rename the Continuation class but preserve its public
            # methods. Require the stdlib metadata and both declarations;
            # a random generic input/callback is not a coroutine result.
            metadata = re.search(r'(?ms)^\s*\.annotation\s+runtime\s+Lkotlin/Metadata;(.*?)^\s*\.end annotation', text)
            if (' interface ' in f' {match[0]} '
                    and metadata and re.search(r'"kotlin-stdlib"', metadata[1])
                    and re.search(r'(?m)^\s*\.method\s+[^\n]*\babstract\s+resumeWith\(Ljava/lang/Object;\)V\s*$', text)
                    and re.search(r'(?m)^\s*\.method\s+[^\n]*\babstract\s+getContext\(\)L[^;\s]+;\s*$', text)):
                continuation_shapes.add(owner)
            return owner, text

        for path in sorted(paths)[:limit]:
            record = read_type(path)
            if record is None:
                continue
            owner, text = record
            if "Ljava/lang/reflect/Proxy;->newProxyInstance(" in text:
                index.proxy_factories.add(owner)
            for method in _METHOD.finditer(text):
                declaration, body = method[1], method[2]
                if " abstract " in f" {declaration} " and ".annotation runtime" in body:
                    method_records.append((owner, declaration, body))
                else:
                    source = owner + '->' + declaration.rsplit(None, 1)[-1]
                    for target in _CALL.findall(body):
                        if _dto_type(target.split('->', 1)[0]):
                            index.invocations.setdefault(target, set()).add(source)
                # Recover annotation semantics from the framework's own
                # dispatch, so renaming POST/Body classes does not break it.
                if "Ljava/lang/annotation/Annotation;" in declaration or "Ljava/lang/annotation/Annotation;" in body:
                    dispatch = list(re.finditer(r"(?m)^\s*instance-of\s+\w+,\s*\w+,\s*(L[^;]+;)", body))
                    for number, branch in enumerate(dispatch):
                        end = dispatch[number+1].start() if number+1 < len(dispatch) else len(body)
                        block = body[branch.end():min(end, branch.end()+4096)]
                        if re.search(r'(?m)^\s*const-string(?:/jumbo)?\s+\w+,\s*"(?:POST|PUT|GET|PATCH)"', block):
                            http_annotations.add(branch[1])
                        if re.search(r'(?m)^\s*const-string(?:/jumbo)?\s+\w+,\s*"[^"\n]*@Body\b', block):
                            body_annotations.add(branch[1])
        for owner, declaration, body in method_records:
                signature = declaration.rsplit(None, 1)[-1]
                if "(" not in signature:
                    continue
                # HTTP annotations are protocol facts, not string cooccurrence.
                # Parameter annotations cannot masquerade as method routes.
                method_body = _PARAM.sub('', body)
                routes = [value[1] for annotation in _ANNOTATION.finditer(method_body)
                          if annotation[1] in http_annotations
                          and (value := re.search(r'(?m)^\s*value\s*=\s*"([^"\\]+)"', annotation[2]))]
                if len(routes) != 1:
                    continue
                params = re.findall(r"\[*(?:L[^;]+;|[ZBCSIJFD])", signature.split("(", 1)[1].split(")", 1)[0])
                # .param uses Dalvik slots; wide primitive parameters take two.
                current = 0 if " static " in f" {declaration} " else 1
                slots = {}
                for typ in params:
                    slots[current] = typ
                    current += 2 if typ in {"J", "D"} else 1
                body_slots, form_fields = [], []
                for param in _PARAM.finditer(body):
                    slot = int(param[1])
                    if slot not in slots:
                        continue
                    for annotation in _ANNOTATION.finditer(param[2]):
                        if annotation[1] in body_annotations:
                            body_slots.append(slot)
                        if annotation[1] in {'Lretrofit/http/Field;', 'Lretrofit2/http/Field;'}:
                            name = re.search(r'(?m)^\s*value\s*=\s*"([^"\\]+)"', annotation[2])
                            if name:
                                form_fields.append((slot, name[1].lower()))
                method_annotations = {a[1] for a in _ANNOTATION.finditer(method_body)}
                form_encoded = bool(method_annotations & {'Lretrofit/http/FormUrlEncoded;', 'Lretrofit2/http/FormUrlEncoded;'})
                streaming = bool(method_annotations & {'Lretrofit/http/Streaming;', 'Lretrofit2/http/Streaming;'})
                if len(body_slots) == 1 and not form_fields and not form_encoded and slots[body_slots[0]].startswith('L'):
                    slot, body_type, kind = body_slots[0], slots[body_slots[0]], 'body'
                elif not body_slots and form_fields and form_encoded:
                    slot, body_type, kind = -1, '', 'form'
                elif not body_slots and not form_fields and not form_encoded:
                    slot, body_type, kind = -1, '', 'none'
                else:
                    continue
                generic_signature = re.search(r'(?s)\.annotation\s+system\s+Ldalvik/annotation/Signature;(.*?)\.end annotation', body)
                generic = ''.join(re.findall(r'"([^"\\]*)"', generic_signature[1])) if generic_signature else ''
                # A method Signature includes parameter generics too. They
                # cannot supply response choices/usage for a prompt request.
                annotation_types = set(_TYPES.findall(generic.split(')', 1)[1])) if ')' in generic else set()
                returns = signature.split(")", 1)[1]
                response = annotation_types | set(_TYPES.findall(returns))
                response -= set(params)
                response = {t for t in response if _dto_type(t)}
                key = owner + '->' + signature
                generic_params = _signature_parameters(generic)
                # Suspend methods return Object at the JVM level. The final
                # non-HTTP parameter alone may carry their logical response.
                annotated_slots = {int(p[1]) for p in _PARAM.finditer(body)
                                   if any(a[1] in body_annotations or re.fullmatch(r'Lretrofit2?/http/[^;]+;', a[1])
                                          for a in _ANNOTATION.finditer(p[2]))}
                if (returns == 'Ljava/lang/Object;' and params and generic_params
                        and len(generic_params) == len(params)
                        and max(slots) not in annotated_slots):
                    last = generic_params[-1]
                    erased = last.split('<', 1)[0] + ';' if '<' in last else last
                    argument = last.split('<-', 1)[1][:-2] if '<-' in last and last.endswith('>;') else ''
                    if (erased == params[-1] and argument
                            and _signature_parameters('(' + argument + ')Ljava/lang/Object;') == (argument,)):
                        carried_types = tuple(sorted(t for t in _TYPES.findall(argument) if _dto_type(t)))
                        if carried_types:
                            coroutine_parameters[key] = (params[-1], carried_types)
                candidates.append(ApiMethod(key, routes[0], slot, body_type, tuple(sorted(response)),
                                            request_kind=kind, parameter_fields=tuple(form_fields), streaming=streaming))

        # Index file paths without reading the entire APK again. A path is only
        # a lookup hint; read_type verifies its .class before using any fields.
        type_paths: dict[str, list[Path]] = {}
        for path in available_paths or ():
            parts = path.parts
            root = next((i for i, part in enumerate(parts) if re.fullmatch(r'smali(?:_classes\d+)?', part)), None)
            if root is not None:
                # APKTool preserves case-colliding class names with .1/.2
                # aliases on Windows. This is only a path hint; probe .class
                # on demand instead of rereading every aliased file in the APK.
                hint = '/'.join(re.sub(r'\.\d+$', '', part) for part in '/'.join(parts[root+1:])[:-6].split('/'))
                type_paths.setdefault(('L' + hint + ';').casefold(), []).append(path)
        attempted = set()
        expanded_depths = {}
        recovery_reads = 0

        def recovered_path(owner):
            nonlocal recovery_reads
            options = sorted(set(type_paths.get(owner.casefold(), ())))
            if not options:
                return None
            if len(options) == 1:
                if recovery_limit is not None and recovery_reads >= recovery_limit:
                    index.limited = True
                    return None
                recovery_reads += 1
                return options[0]
            matches = []
            unknown_header = False
            for path in options:
                if recovery_limit is not None and recovery_reads >= recovery_limit:
                    index.limited = True
                    return None
                recovery_reads += 1
                try:
                    with path.open(encoding='utf-8') as stream:
                        match = _CLASS.search(''.join(stream.readline() for _ in range(8)))
                except (OSError, UnicodeError):
                    index.parse_errors += 1
                    unknown_header = True
                    continue
                if not match:
                    unknown_header = True
                if match and match[1] == owner:
                    matches.append(path)
            # Differently cased classes are distinct JVM types; duplicate
            # declarations of the same exact type remain ambiguous.
            return matches[0] if len(matches) == 1 and not unknown_header else None

        def recover(owner, depth=0, follow=False):
            if not _dto_type(owner):
                return
            if owner not in index.fields and owner not in attempted:
                attempted.add(owner)
                path = recovered_path(owner)
                if path and read_type(path, expected=owner):
                    index.recovered_type_count += 1
            if owner not in index.fields:
                index.missing_types.add(owner)
                return
            if not follow:
                return
            if expanded_depths.get(owner, 6) <= depth:
                return
            expanded_depths[owner] = depth
            if depth >= 5:
                if any(_dto_type(t) and t not in index.fields for t in index.type_edges.get(owner, ())):
                    index.limited = True
                return
            for child in sorted(index.type_edges.get(owner, ())):
                if child != owner:
                    recover(child, depth+1, follow=True)

        def is_continuation(owner, depth=0):
            if owner == 'Lkotlin/coroutines/Continuation;':
                return True
            if depth >= 3 or not _dto_type(owner):
                return False
            recover(owner)
            return owner in continuation_shapes or any(
                is_continuation(parent, depth+1) for parent in sorted(type_parents.get(owner, ()))
                if parent != owner and parent != 'Ljava/lang/Object;')

        def field_path(owners, names):
            """Follow one actual response field chain, not a union of DTOs."""
            for owner in owners:
                recover(owner)
                if names[0] not in index.fields.get(owner, ()):
                    continue
                if len(names) == 1:
                    return True
                children = index.field_types.get(owner, {}).get(names[0], ())
                if field_path(children, names[1:]):
                    return True
            return False

        for api in sorted(candidates, key=lambda a: (not bool(_INFERENCE_ROUTE.search(a.route)), a.key)):
            coroutine = coroutine_parameters.get(api.key)
            if coroutine and is_continuation(coroutine[0]):
                api = replace(api, response_types=tuple(sorted(set(api.response_types) | set(coroutine[1]))))
            index.all_api_methods[api.key] = api
            if api.body_type:
                recover(api.body_type)
            # Request controls must belong to the body object itself. Do not
            # union model/messages/temperature from unrelated nested DTOs.
            request = index.fields.get(api.body_type, frozenset())
            if api.request_kind == 'form':
                request = frozenset(name for _, name in api.parameter_fields)
            if {'model', 'messages'}.issubset(request):
                for typ in index.field_types.get(api.body_type, {}).get('messages', ()):
                    recover(typ)
            response = set()
            if 'prompt' in request:
                for typ in api.response_types:
                    recover(typ, 1, follow=True)
                    response.update(index.expanded_fields(typ))
            # Custom wrappers: require token-accounted text choices, bound to
            # the method's own response type, not another bundled DTO.
            wrapped = ("prompt" in request and {"choices", "usage"}.issubset(response)
                       and {"prompt_tokens", "completion_tokens", "total_tokens"}.issubset(response))
            # model/messages is related when messages actually contains a
            # role/content DTO. Unrelated sibling fields cannot complete it.
            message_shape = any({'role', 'content'}.issubset(index.fields.get(t, ()))
                                for t in index.field_types.get(api.body_type, {}).get('messages', ()))
            typed_messages = {'model', 'messages'}.issubset(request) and message_shape
            # Camel-case controls are recognized only inside this associated
            # body structure. Preserve their observed wire names in output.
            controlled_messages = typed_messages and any(name.replace('_', '') in _CONTROL_FIELD_NAMES for name in request)
            nested_controls = set()
            for name in request & {'generationconfig', 'generation_config'}:
                for typ in index.field_types.get(api.body_type, {}).get(name, ()):
                    recover(typ)
                    nested_controls.update(f'{name}.{field}' for field in index.fields.get(typ, ())
                                           if field.replace('_', '') in _CONTROL_FIELD_NAMES)
            text_response = field_path(api.response_types, ('candidates', 'content', 'parts', 'text'))
            generation_contract = 'prompt' in request and bool(nested_controls) and text_response
            if _is_llm_schema(set(request)) or wrapped or controlled_messages or generation_contract:
                evidence = ('typed_message_structure_with_control',) if controlled_messages else ('serialized_request_response_contract' if wrapped else 'serialized_request_contract',)
                if generation_contract:
                    evidence = ('bound_text_generation_contract',)
                message_fields = {'role', 'content'} if typed_messages else set()
                verified = replace(api, fields=tuple(sorted(request | response | message_fields | nested_controls)), evidence=evidence,
                                   request_fields=tuple(sorted(request | message_fields | nested_controls)), response_fields=tuple(sorted(response)))
                index.api_methods[api.key] = verified
                index.all_api_methods[api.key] = verified
            elif typed_messages:
                candidate = replace(api, fields=tuple(sorted(request | {'role', 'content'})),
                                    request_fields=tuple(sorted(request | {'role', 'content'})),
                                    evidence=('typed_message_structure',))
                index.candidate_api_methods[api.key] = candidate
                index.all_api_methods[api.key] = candidate
            elif _INFERENCE_ROUTE.search(api.route):
                inputs = set(request) | {name for _, name in api.parameter_fields}
                if 'dialogs' in request:
                    for typ in index.field_types.get(api.body_type, {}).get('dialogs', ()):
                        recover(typ, 1, follow=True)
                        dialog_fields = index.expanded_fields(typ)
                        if 'role' in dialog_fields and dialog_fields & {'content', 'text'}:
                            inputs.update(dialog_fields)
                if inputs & _TEXT_INPUT_FIELDS or (api.request_kind == 'none' and api.streaming):
                    candidate = replace(api, fields=tuple(sorted(inputs)), request_fields=tuple(sorted(inputs)),
                                        evidence=('inference_route_with_stream' if not inputs else 'inference_route_with_text_input',))
                    index.candidate_api_methods[api.key] = candidate
                    index.all_api_methods[api.key] = candidate
        return index

    def usage_signals(self, allowed_sources: set[str] | None = None) -> list[RawSignal]:
        """Emit referenced protocol facts even when a client URL is unknown.

        Declarations in an unused SDK do not produce observations. These
        signals describe an invoke and its declared contract, not execution,
        client ownership, request values or an embedding category.
        """
        signals = []
        for key, api in sorted({**self.candidate_api_methods, **self.api_methods}.items()):
            for source in sorted(self.invocations.get(key, ())):
                if allowed_sources is not None and source not in allowed_sources:
                    continue
                structural = key in self.api_methods
                signals.append(RawSignal(
                    SignalType.LLM_JSON_SCHEMA if structural and api.request_kind == 'body' and 'bound_text_generation_contract' not in api.evidence else SignalType.LLM_HTTP_API_USAGE,
                    key, 'apktool', source,
                    detail='invoked_declared_http_contract; endpoint_binding_unknown',
                    role=SignalRole.RELATION, scope=SignalScope.METHOD,
                    relation=SignalRelation.HTTP_API_LLM_CONTEXT,
                    components=(f'api_method={key}', f'api_route={api.route}', f'request_kind={api.request_kind}',
                                'binding=interface_invocation', 'endpoint_binding=unknown',
                                *(f'declared_request_field={name}' for name in api.request_fields),
                                *(f'declared_response_field={name}' for name in api.response_fields),
                                *(f'protocol_evidence={item}' for item in api.evidence)),
                    verification=SignalVerification.OBSERVED if structural else SignalVerification.CANDIDATE,
                    source_kind='smali_http_contract',
                ))
        return signals

    def referenced_sources(self, app_owner, depth: int = 5) -> set[str]:
        """Follow invoked methods; a referenced class does not admit all SDK methods."""
        outgoing: dict[str, set[str]] = {}
        for target, sources in self.invocations.items():
            for source in sources:
                outgoing.setdefault(source, set()).add(target)
        seen = {source for source in outgoing if app_owner(source.split('->', 1)[0])}
        frontier = seen.copy()
        for _ in range(depth):
            frontier = {target for source in frontier for target in outgoing.get(source, ())} - seen
            seen.update(frontier)
            if not frontier:
                break
        return seen

    def response_signals(self, allowed_sources):
        """An accessed typed response graph; no endpoint/request assertion.

        Follow choices -> delta/message -> content through real field types.
        An unused DTO, sibling DTO names, or a bare content field is insufficient.
        """
        result = []
        for owner, fields in sorted(self.fields.items()):
            if not {'choices', 'model'}.issubset(fields):
                continue
            paths = set()
            for choice in self.field_types.get(owner, {}).get('choices', ()):
                if 'finish_reason' not in self.fields.get(choice, ()):
                    continue
                for branch in ('delta', 'message'):
                    for leaf in self.field_types.get(choice, {}).get(branch, ()):
                        if 'content' in self.fields.get(leaf, ()):
                            paths.add('choices.'+branch+'.content')
            if not paths:
                continue
            callers = {source for getter in self.field_readers.get((owner, 'choices'), ())
                       for source in self.invocations.get(getter, ()) if source in allowed_sources}
            for source in sorted(callers):
                result.append(RawSignal(
                    SignalType.LLM_JSON_SCHEMA, owner, 'apktool', self.source_paths[owner],
                    role=SignalRole.REFERENCE, scope=SignalScope.METHOD,
                    source_kind='smali_response_structure',
                    components=(f'source_method={source}', f'response_type={owner}',
                                'binding=accessed_response_type',
                                *(f'response_path={p}' for p in sorted(paths)))))
        return result

    def expanded_fields(self, owner: str, depth: int = 5) -> frozenset[str]:
        seen, frontier, result = set(), {owner}, set()
        for _ in range(depth + 1):
            next_frontier = set()
            for typ in sorted(frontier):
                if typ in seen:
                    continue
                if len(seen) >= 256:
                    self.limited = True
                    return frozenset(result)
                seen.add(typ)
                if typ not in self.fields and _dto_type(typ):
                    self.missing_types.add(typ)
                result.update(self.fields.get(typ, ()))
                next_frontier.update(self.type_edges.get(typ, ()))
            frontier = next_frontier - seen
            if not frontier:
                break
        if frontier:
            self.limited = True
        return frozenset(result)
