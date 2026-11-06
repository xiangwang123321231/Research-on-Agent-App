"""Bounded LLM-related business facts; no category or execution inference."""
from __future__ import annotations

import json
import re
from typing import Iterator
from urllib.parse import parse_qsl, urlsplit

from registry import (extract_llm_model_ids, normalize_llm_model_id, model_property_reference, is_reference_only_source,
                      is_reference_endpoint_url, is_custom_service_url, _AUXILIARY_HOST_LABELS,
                      iter_known_sdk_packages, scan_for_api_keys, SPECIAL_TOKEN_HINTS,
                      prompt_template_markers, classify_url_signal,
                      scan_url_for_business_ai_endpoint, llm_host_candidate_evidence,
                      SDK_REGISTRY, PROMPT_TEMPLATE_PATTERNS, endpoint_noise_reason,
                      llm_config_label_marker, classify_llm_route,
                      is_plausible_api_route_literal)
from signal_types import RawSignal, SignalType, SignalRole, SignalScope, SignalVerification


# Comments are consumed separately so documentation cannot become a request.
_LITERALS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`|//[^\r\n]*|/\*[\s\S]*?\*/')
_HINT = re.compile(r'gpt|claude|gemini|deepseek|qwen|llama|mixtral|mistral|gemma|phi|grok|glm|command-r|assistant|chat|AI|artificial.intelligence|\bLLM\b|language.model|人工智能|大语言模型|大語言模型|智能.{0,4}(?:聊天|对话|對話)|/(?:v\d+/(?:responses|completions?)|generatecontent)(?:[/?:#]|$)|[?&](?:gen|prompt|input|query|text)=', re.I)
# This is a transport/performance prefilter, never relevance evidence. It must
# also admit camelCase serialized fields and operation names such as getLLMReply.
_HINT = re.compile(_HINT.pattern + r'|llm', re.I)
_CAPABILITY = re.compile(r'\b(?:supports?|includes?|powered\s+by|subscribe|subscription)\b|구독|支持|订阅|支援', re.I)
_OPERATION = re.compile(
    r'\bAI[- ]generated\s+(?:chat|text|response|content|reply)\b|'
    r'^(?:your\s+)?AI(?:\s+guide)?[-_ ]?chat$|'
    r'(?:^|/)chat-agent(?:/|$)|'
    r'^/(?:[\w.-]+/)*assistants/(?:messages|send)(?:/|$)|'
    r'^assistant:(?:stream_starts|stream_ends|response_received)$|'
    r'^[A-Za-z]*AssistantMessage(?:Item|Component)?$|'
    r'^LLM$', re.I)
_GENERATION_INSTRUCTION = re.compile(
    r'^(?:please\s+)?(?:write|generate|compose|draft|rewrite|summari[sz]e|translate)\s+'
    r'(?:[\w\[\]{}<>-]+\s+){0,8}'
    r'(?:captions?|emails?|stories|story|poems?|essays?|articles?|paragraphs?|'
    r'text|replies|reply|responses?|sentences?|summar(?:y|ies)|descriptions?|posts?|letters?)\b', re.I)
_NON_LANGUAGE = re.compile(r'\b(?:images?|pictures?|photos?|visuals?|videos?|sql|database|files?|logs?)\b', re.I)

# Explicit text/conversation capability in one short UI/configuration value.
# Neither an arbitrary "AI" nor "chat" alone supplies this observation.
_TEXT_AI_OPERATION = re.compile(
    r'\b(?:AI|artificial intelligence)[-_ ]+(?:chat(?:bot)?|conversational?\s+assistant|writing\s+assistant)\b|'
    r'\b(?:chat|talk|converse)\s+with\s+(?:(?:an?|your)\s+)?(?:AI|artificial intelligence)\b|'
    r'\b(?:large language model|LLM)[-_ ]+(?:chat|assistant|text generation)\b|'
    r'(?:人工智能|大语言模型|大語言模型|AI)[\s的]*(?:聊天|对话|對話|写作|寫作)|'
    r'(?:与|與|和)(?:人工智能|AI)(?:聊天|对话|對話)', re.I)
_NEGATED_OPERATION = re.compile(r'\b(?:no|not|without|disable[ds]?)\b|不支持|不使用|禁用', re.I)


def _route_words(value: str) -> set[str]:
    tokens = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', value)
    tokens = re.sub(r'([a-z])([A-Z])', r'\1_\2', tokens)
    return set(re.findall(r'[a-z0-9]+', tokens.lower()))


def llm_summary_route_reference(value: str) -> bool:
    """A summary resource/operation in explicit LLM route context.

    Audio/transcription may be a parent workflow. The summary must occur in
    the terminal resource/operation, not just elsewhere in the APK/path.
    This permits a reference, never infers a generation call or endpoint.
    """
    if not value.startswith('/') or value.startswith('//'):
        return False
    # Query values can contain credentials, model names or task words. None
    # supplies route purpose; retain only the path in any resulting signal.
    path = value.split('?', 1)[0].split('#', 1)[0]
    words = _route_words(path)
    if not words & {'llm', 'chatgpt', 'gpt', 'gemini', 'claude'}:
        return False
    if words & {'help', 'docs', 'documentation', 'example', 'examples', 'billing',
                'quota', 'history', 'delete', 'logs', 'image', 'images', 'embedding', 'embeddings'}:
        return False
    segments = [s for s in path.split('/') if s and not re.fullmatch(r'\{[^{}]+\}|:[\w]+|\d+', s)]
    task_words = _route_words('/'.join(segments[-2:]))
    return bool(task_words & {'summary', 'summaries', 'summarize', 'summarise',
                              'summarization', 'summarisation'})


def dedicated_llm_reference(value: str) -> bool:
    """A named language-generation operation, never a request binding.

    Segment boundaries exclude chatgptish and llmachine. Documentation and
    management routes must not acquire inference relevance from a brand.
    """
    if not value.startswith('/') or '?' in value or '#' in value:
        return False
    if llm_summary_route_reference(value):
        return True
    words = _route_words(value)
    return bool(words & {'llm', 'chatgpt', 'gpt', 'gemini', 'claude'}
                and words & {'ask', 'reply', 'response', 'responses', 'completion',
                             'completions', 'infer', 'inference', 'generate'}
                and not words & {'help', 'docs', 'documentation', 'example', 'examples',
                                 'billing', 'quota', 'history', 'delete', 'logs',
                                 'image', 'images', 'audio', 'embedding', 'embeddings'})


def llm_serialized_name(name: str) -> bool:
    return bool(re.fullmatch(r'(?:llm|gpt)[_-][A-Za-z0-9_]+|(?:llm|gpt)[A-Z][A-Za-z0-9]*', name))


def llm_literal_prefilter_pattern() -> str:
    """A portable JS/Python superset derived from the actual fact rules.

    The parser uses this only to reduce transport volume. SDK/token additions
    must not require maintaining a second hand-written registry in JavaScript.
    """
    parts = [_HINT.pattern, _TEXT_AI_OPERATION.pattern,
             r'sk-|hf_|gsk_|xai-|AIza|pplx-|r8_',
             *(p.pattern for p in PROMPT_TEMPLATE_PATTERNS),
             *(re.escape(t) for t in sorted(SPECIAL_TOKEN_HINTS)),
             *(re.escape(s.package).replace(r'\.', r'[./]') for s in SDK_REGISTRY
               if s.classification_signal)]
    return '|'.join(f'(?:{part})' for part in parts)


def scan_llm_business_literal(value: str, source: str, layer: str,
                              source_kind: str = 'source_string_literal') -> Iterator[RawSignal]:
    """Scan exactly one string. Neighbouring strings never supply context."""
    if (not value or len(value) > 16 * 1024 or not _HINT.search(value)
            or is_reference_only_source(source)):
        return
    text = value.strip()
    # Atomic names or a single capability statement are observable model
    # references; neither asserts a configured/deployed/invoked model.
    model_text = re.sub(r'\b(GPT|Claude|Gemini)\s+(\d)', r'\1-\2', text, flags=re.I)
    models = extract_llm_model_ids(model_text)
    property_model = model_property_reference(model_text)
    if property_model:
        # A business key can mention a model and a subscription tier. Keep
        # that literal reference without guessing a deployed model variant
        # or letting each UI/counter key inflate the model-identity count.
        yield RawSignal(SignalType.LLM_OPERATION_CONTEXT, text, layer, source,
                        role=SignalRole.REFERENCE, scope=SignalScope.VALUE,
                        components=('business_fact=llm_model_property_reference',
                                    f'model_reference_prefix={property_model}',
                                    'identity_evidence=not_a_complete_model_id'),
                        source_kind=source_kind)
        selected_models = []
        model_context = 'application_property_key'
    elif (len(text) <= 1024 and not text.startswith(('{', '[')) and '://' not in text
            and not re.search(r'\.(?:png|jpe?g|svg|webp|gif)\b|\b(?:icons?|assets?)/', text, re.I)
            and _CAPABILITY.search(text)):
        selected_models = models
        model_context = 'capability_declaration'
    else:
        selected_models = [model for model in models if model == normalize_llm_model_id(model_text)]
        model_context = 'literal_reference'
    for model in selected_models:
        yield RawSignal(SignalType.KNOWN_LLM_MODEL_NAME, model, layer, source,
                        role=SignalRole.REFERENCE, scope=SignalScope.VALUE,
                        components=(f'model_id={model}', f'model_context={model_context}'),
                        source_kind=source_kind)
    operation = _OPERATION.search(text)
    # Suppress only a structurally identified non-UI occurrence. The same
    # acronym is a real category/UI value in source, Hermes and resources;
    # its case or unknown provenance alone cannot justify dropping it.
    if text.lower() == 'llm' and source_kind in {
            'kotlin_metadata_string', 'statistical_word_entry'}:
        operation = None
    if (not operation and len(text) <= 256 and not _NON_LANGUAGE.search(text)
            and not _NEGATED_OPERATION.search(text)):
        operation = _TEXT_AI_OPERATION.search(text)
    if not text.lower().startswith(('http://', 'https://')) and operation:
        fact = (text.split('?', 1)[0].split('#', 1)[0]
                if text.startswith('/') else operation[0])
        yield RawSignal(SignalType.LLM_OPERATION_CONTEXT, fact[:512], layer, source,
                        role=SignalRole.REFERENCE, scope=SignalScope.VALUE,
                        components=('business_fact=llm_related_operation',), source_kind=source_kind)
    elif (text.startswith('/') and is_plausible_api_route_literal(text)
          and (not re.search(r'(?:^|[/_-])(?:images?|audio|speech|embeddings?|transcriptions?)(?:[/_?-]|$)', text, re.I)
               or llm_summary_route_reference(text))
          and (classify_llm_route(text) == 'strong' or dedicated_llm_reference(text)
               or llm_summary_route_reference(text))):
        # A standalone protocol path is a related fact even when its base
        # cannot be paired. Do not fabricate an endpoint or a request call.
        yield RawSignal(
            SignalType.LLM_OPERATION_CONTEXT, text.split('?',1)[0].split('#',1)[0],
            layer, source, role=SignalRole.REFERENCE, scope=SignalScope.VALUE,
            components=('business_fact=llm_protocol_route',
                        'route_evidence=standalone_literal', *(
                            ('route_task=language_summary', 'operation_evidence=explicit_llm_summary_reference')
                            if llm_summary_route_reference(text) else ())), source_kind=source_kind)
    yield from _scan_generation_query(text, source, layer, source_kind)


def scan_llm_json_config(text: str, source: str, layer: str) -> Iterator[RawSignal]:
    """Preserve related links inside one JSON object, without inferring calls.

    Never borrow a label from a parent, sibling or another array item. A
    multi-origin object is ambiguous and is left to the ordinary URL rules.
    """
    if (is_reference_only_source(source) or not text.lstrip().startswith(('{', '['))
            or not _HINT.search(text)):
        return
    try:
        payload = json.loads(text)
    except (ValueError, RecursionError):
        return
    pending = [(payload, '')]
    while pending:
        node, path = pending.pop()
        if isinstance(node, list):
            pending.extend((value, f'{path}/{i}') for i, value in enumerate(node)
                           if isinstance(value, (dict, list)))
            continue
        if not isinstance(node, dict):
            continue
        pending.extend((value, path + '/' + str(key).replace('~', '~0').replace('/', '~1'))
                       for key, value in node.items() if isinstance(value, (dict, list)))
        strings = [(str(key), value.strip()) for key, value in node.items()
                   if isinstance(value, str)]
        urls = [(key, value) for key, value in strings
                if value.lower().startswith(('http://', 'https://'))
                and not any(c.isspace() for c in value)]
        if len({value for _, value in urls}) != 1:
            continue
        labels = [(key, llm_config_label_marker(value)) for key, value in strings
                  if len(value) <= 256 and '://' not in value]
        labels = [(key, marker) for key, marker in labels if marker]
        if not labels:
            continue
        key, url = urls[0]
        if (not is_custom_service_url(url) or is_reference_endpoint_url(url, source)
                or endpoint_noise_reason(url)):
            continue
        parsed = urlsplit(url)
        if parsed.username or parsed.password:
            continue
        host = parsed.hostname.lower()
        if host.split('.')[0] in _AUXILIARY_HOST_LABELS:
            continue
        label_key, marker = labels[0]
        # Keep only the observed URL (without potentially sensitive queries)
        # and the matching semantic marker, never arbitrary label/user data.
        endpoint = url.split('?', 1)[0].split('#', 1)[0]
        yield RawSignal(
            SignalType.BUSINESS_AI_ENDPOINT, endpoint, layer, source,
            role=SignalRole.CONFIG, scope=SignalScope.CONFIG,
            verification=SignalVerification.CANDIDATE,
            source_kind='json_llm_link_config',
            components=('route_strength=config_identity', 'binding=same_json_object',
                        'candidate_evidence=llm_link_config', f'custom_domain={host}',
                        f'llm_path={parsed.path or "/"}', f'config_object_path={path[:512]}',
                        f'config_url_field={key[:128]}', f'config_label_field={label_key[:128]}',
                        f'config_llm_marker={marker}'),
        )


def scan_llm_literal(value: str, source: str, layer: str,
                     source_kind: str = 'source_string_literal') -> Iterator[RawSignal]:
    """Common facts for a decoded literal, independent of its container.

    Do not join adjacent values or infer calls. Format readers add offsets;
    credentials remain redacted by the registry before leaving this function.
    """
    if not value or len(value) > 16 * 1024 or is_reference_only_source(source):
        return
    yield from scan_llm_business_literal(value, source, layer, source_kind)
    for package in iter_known_sdk_packages(value):
        yield RawSignal(SignalType.KNOWN_LLM_SDK_PACKAGE, package, layer, source,
                        source_kind=source_kind)
    if value.lower().startswith(('http://', 'https://')) and not any(c.isspace() for c in value):
        yield from (s for s in classify_url_signal(source, value, layer)
                    if s.signal_type in (SignalType.KNOWN_LLM_API_DOMAIN, SignalType.KNOWN_LLM_WEB_DOMAIN))
        yield from (s for s in scan_url_for_business_ai_endpoint(value, source, layer)
                    if 'route_strength=host_identity' not in s.components or llm_host_candidate_evidence(value))
    yield from scan_for_api_keys(value, source, layer)
    for token in sorted(SPECIAL_TOKEN_HINTS):
        if token in value:
            yield RawSignal(SignalType.LLM_SPECIAL_TOKEN, token, layer, source,
                            source_kind=source_kind)
    for marker in prompt_template_markers(value):
        yield RawSignal(SignalType.LLM_PROMPT_TEMPLATE, marker, layer, source,
                        role=SignalRole.REFERENCE, scope=SignalScope.VALUE,
                        components=('prompt_evidence=instruction_literal',), source_kind=source_kind)


def _scan_generation_query(value, source, layer, source_kind):
    if not value.lower().startswith(('http://', 'https://')) or len(value) > 8192:
        return
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        return
    if not parsed.hostname or parsed.username or parsed.password:
        return
    host = parsed.hostname.lower()
    if host.split('.')[0] in _AUXILIARY_HOST_LABELS or host.endswith('.firebaseio.com'):
        return
    if ':' in host:
        host = f'[{host}]'
    endpoint = f'{parsed.scheme.lower()}://{host}' + (f':{port}' if port else '') + (parsed.path or '/')
    if is_reference_endpoint_url(endpoint) or not is_custom_service_url(endpoint):
        return
    try:
        parameters = parse_qsl(parsed.query, max_num_fields=64)
    except ValueError:
        return
    for key, instruction in parameters:
        # Parameter names alone (gen/prompt/message) are never sufficient.
        # No key, user identifier or input is copied to signal components.
        if (key.lower() not in {'gen', 'prompt', 'input', 'query', 'text'}
                or len(instruction) > 1024 or not _GENERATION_INSTRUCTION.search(instruction)
                or _NON_LANGUAGE.search(instruction)):
            continue
        proof = (f'request_field={key.lower()}', 'request_evidence=text_generation_query')
        yield RawSignal(SignalType.BUSINESS_AI_ENDPOINT, endpoint, layer, source,
                        role=SignalRole.LITERAL, scope=SignalScope.VALUE,
                        components=(f'custom_domain={parsed.hostname.lower()}',
                                    f'llm_path={parsed.path or "/"}', *proof), source_kind=source_kind)
        # Preserve the instruction shape, not dynamic user text or query data.
        match = _GENERATION_INSTRUCTION.search(instruction)
        yield RawSignal(SignalType.LLM_PROMPT_TEMPLATE, instruction[:match.end()], layer, source,
                        role=SignalRole.LITERAL, scope=SignalScope.VALUE,
                        components=proof, source_kind=source_kind)


def scan_llm_business_text(text: str, source: str, layer: str) -> Iterator[RawSignal]:
    """Read bounded quoted values in source/config, including dynamic Url props.

    Literal URL fragments retain their observed query instruction; their
    eventual interpolation/call is unknown. Do not concatenate literals.
    """
    if is_reference_only_source(source) or not _HINT.search(text):
        return
    # A wire-field annotation is different from a variable or Kotlin Metadata.
    # This retains identity only, without asserting a full request schema.
    for match in re.finditer(r'@(?:[\w$.]+\.)?(?:SerialName|SerializedName|JsonProperty)\(\s*"([^"\\]+)"\s*\)', text):
        name = match[1]
        if llm_serialized_name(name):
            yield RawSignal(SignalType.LLM_OPERATION_CONTEXT, name, layer, source,
                            role=SignalRole.REFERENCE, scope=SignalScope.VALUE,
                            source_kind='serialized_field_reference',
                            components=('business_fact=llm_serialized_field',))
    metadata_ranges = []
    statistical_json = False
    if re.search(r'["\']llm["\']', text, re.I):
        # Mask comments without moving offsets, then recognize annotation
        # parentheses. Obfuscated Kotlin annotation names still expose their
        # d1/d2 metadata fields; no application/class identity is consulted.
        code = _LITERALS.sub(lambda m:' ' * len(m[0]) if m[0].startswith(('//','/*')) else m[0], text)
        for annotation in re.finditer(r'@([\w.$]+)\s*\(', code):
            depth = 1
            for token in re.finditer(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[()]', code[annotation.end():]):
                if token[0] == '(':
                    depth += 1
                elif token[0] == ')':
                    depth -= 1
                if depth == 0:
                    end = annotation.end() + token.end()
                    body = code[annotation.end():end]
                    if (annotation[1].split('.')[-1] == 'Metadata'
                            or re.search(r'\bd1\s*=', body) and re.search(r'\bd2\s*=', body)):
                        metadata_ranges.append((annotation.start(),end))
                    break
        if text.lstrip().startswith(('{','[')):
            try:
                json.loads(text)
                statistical_json = True
            except (ValueError,RecursionError):
                pass
    seen = set()
    for match in _LITERALS.finditer(text):
        token = match[0]
        if token.startswith(('//', '/*')) or len(token) > 16 * 1024 + 2:
            continue
        value = token[1:-1]
        if token.startswith('"'):
            try:
                value = json.loads(token)
            except ValueError:
                value = value.replace(r'\/', '/')
        else:
            value = value.replace(r'\/', '/')
        source_kind = 'source_string_literal'
        if value.strip().lower() == 'llm':
            if any(start <= match.start() < end for start,end in metadata_ranges):
                source_kind = 'kotlin_metadata_string'
            elif statistical_json and re.search(
                    r'"\d+(?:\.\d+)?/\d+(?:\.\d+)?"\s*:\s*$', text[max(0,match.start()-128):match.start()]):
                # A frequency/total key is evidence of a statistical table,
                # independently of its filename or parent property names.
                source_kind = 'statistical_word_entry'
        identity = (value,source_kind)
        if identity in seen:
            continue
        seen.add(identity)
        yield from scan_llm_business_literal(value, source, layer, source_kind)
