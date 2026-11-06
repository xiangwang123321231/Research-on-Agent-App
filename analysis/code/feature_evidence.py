"""Feature-level interpretations of retained evidence; never infer app labels.

Observations, typed calls and target bindings are deliberately separate. All
proof checks use one occurrence at a time, before feature-level deduplication.
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

from registry import (has_strong_llm_context, match_api_service,
                      match_web_service, is_reference_only_source)
from webview_context import is_auxiliary_page, web_signal_feature_eligible


def occurrences(signal: dict):
    return signal.get('evidence_instances') or [signal]


def sources(instance: dict):
    source = instance.get('source_file')
    return [str(source)] if source else [str(s) for s in instance.get('source_files', [])]


_SMALI_USAGE_EVIDENCE = frozenset({
    'typed_message_structure_with_control', 'serialized_request_contract',
    'serialized_request_response_contract', 'bound_text_generation_contract',
    'typed_message_structure', 'inference_route_with_text_input',
    'inference_route_with_stream',
})
_JS_USAGE_EVIDENCE = frozenset({
    'typed_messages_with_model_identity', 'bound_generation_request_schema',
    'bound_message_request_text_response', 'bound_generation_instruction_response',
})
_OPERATION_REFERENCE_FACTS = frozenset({
    'llm_related_operation', 'llm_protocol_route', 'llm_serialized_field',
    'llm_model_property_reference',
})


def _component_values(proof: set[str], key: str) -> set[str]:
    prefix = key + '='
    return {item[len(prefix):] for item in proof
            if item.startswith(prefix) and item[len(prefix):].strip()}


def programmatic_llm_usage(signal: dict) -> bool:
    """A retained interface invoke/typed request, never execution or ownership.

    Smali emits both schema and HTTP-usage records for invoked contracts.
    Qualify one occurrence at a time: an unrelated response reference or
    another request cannot lend its binding/protocol to this observation.
    General feature admission remains the vectorizer's responsibility.
    """
    if signal.get('signal_type') not in {'llm_http_api_usage', 'llm_json_schema'}:
        return False
    for instance in occurrences(signal):
        if (instance.get('role') != 'relation' or instance.get('scope') != 'method'
                or instance.get('relation') != 'http_api_with_llm_request_context'
                or not sources(instance)):
            continue
        proof = set(instance.get('components') or [])
        protocols = _component_values(proof, 'protocol_evidence')
        if instance.get('source_kind') == 'smali_http_contract':
            if (instance.get('verification') in {'observed', 'candidate', 'verified'}
                    and 'binding=interface_invocation' in proof
                    and _component_values(proof, 'api_method')
                    and _component_values(proof, 'request_kind')
                    and protocols & _SMALI_USAGE_EVIDENCE):
                return True
        elif instance.get('source_kind') == 'js_bound_request':
            if (instance.get('verification') == 'verified'
                    and {'binding=request_object', 'stage=static_invocation',
                         'operation=text_generation'} <= proof
                    and _component_values(proof, 'request_id')
                    and _component_values(proof, 'http_invocation')
                    and protocols & _JS_USAGE_EVIDENCE):
                return True
    return False


def llm_operation_reference(signal: dict) -> bool:
    """An explicit related reference, not a model identity or generation call."""
    if signal.get('signal_type') != 'llm_operation_context':
        return False
    for instance in occurrences(signal):
        refs = sources(instance)
        if (instance.get('role') != 'reference'
                or instance.get('scope') not in {
                    'value', 'config', 'method', 'local_method_graph', 'class', 'file'}
                or instance.get('verification') not in {'observed', 'verified'}
                or not refs or all(is_reference_only_source(ref) for ref in refs)):
            continue
        proof = set(instance.get('components') or [])
        if _component_values(proof, 'business_fact') & _OPERATION_REFERENCE_FACTS:
            return True
    return False


def host_identity(value: str) -> str:
    try:
        parsed = urlsplit(value if '://' in value else '//' + value)
        return (parsed.hostname or '').lower().rstrip('.') + (
            ':' + str(parsed.port) if parsed.port else '')
    except ValueError:
        return ''


def related_web_observation(signal: dict) -> bool:
    """Retain explicit LLM page observations without asserting target binding.

    A generic chat path, a WebView alone, or unrelated APK-wide cooccurrence
    still does not qualify. Binary references remain references, not calls.
    """
    if web_signal_feature_eligible(signal):
        return True
    for instance in occurrences(signal):
        if instance.get('scope') not in {'method', 'class', 'file'}:
            continue
        kind = instance.get('source_kind')
        if kind not in {'typed_smali_webview', 'smali_method_cooccurrence',
                        'source_webview_cooccurrence', 'binary_webview_cooccurrence'}:
            continue
        proof = set(instance.get('components') or [])
        urls = [p.split('=', 1)[1] for p in proof
                if p.startswith(('candidate_url=', 'target_url='))]
        if ('relevance_hint=known_llm_website' in proof
                and any(match_web_service(u) and not is_auxiliary_page(u) for u in urls)):
            return True
        if (kind in {'typed_smali_webview', 'smali_method_cooccurrence'}
                and instance.get('scope') in {'method', 'class'}
                and proof.intersection({'relevance_hint=llm_page_context',
                                        'relevance_hint=llm_html_context'})):
            return True
    return False


def typed_web_observation(signal: dict) -> bool:
    if web_signal_feature_eligible(signal):
        return True
    return any(i.get('source_kind') == 'typed_smali_webview'
               and i.get('scope') in {'method', 'class'}
               and any(str(c).startswith('webview_call=') for c in i.get('components', []))
               and related_web_observation({**signal, 'evidence_instances': [i]})
               for i in occurrences(signal))


def observed_web_hosts(signal: dict) -> set[str]:
    """An observed website identity is not a navigation relationship."""
    hosts = set()
    for instance in occurrences(signal):
        if not related_web_observation({**signal, 'evidence_instances': [instance]}):
            continue
        for c in instance.get('components') or []:
            if str(c).startswith(('candidate_url=', 'target_url=')):
                url = c.split('=', 1)[1]
                if match_web_service(url) and not is_auxiliary_page(url):
                    hosts.add(host_identity(url))
    return hosts - {''}


_ENDPOINT_ASSIGNMENT = re.compile(
    r'''\s*(?:[A-Za-z_$][\w$]*\.)?(?:base_?url|api_?base|api_?url|endpoint(?:_?url)?)\s*=\s*(["'])(https?://[^\s"']+)\1\s*;?\s*''', re.I)
_CREDENTIAL_LABEL = re.compile(
    r'\s*(?:(?:enter|set|your|configure|provide|input)\s+)?'
    r'(?:[A-Za-z0-9_-]+\s+){0,3}API[ _-]?key\s*[:：]?\s*', re.I)
_NON_LANGUAGE_ROUTE = re.compile(
    r'(?:^|/)(?:images?|audio|speech|transcriptions?|translations?|embeddings?)(?:/|$)', re.I)


def related_configuration(signal: dict) -> bool:
    """Accept concrete configuration facts, not setter/logging fragments."""
    value = str(signal.get('value') or '')
    st = signal.get('signal_type')
    for instance in occurrences(signal):
        refs = sources(instance)
        if not refs or all(is_reference_only_source(s) for s in refs):
            continue
        if st == 'baseurl_override':
            if instance.get('role') != 'config' or instance.get('source_kind') != 'config_marker':
                continue
            m = _ENDPOINT_ASSIGNMENT.fullmatch(value)
            if not m:
                continue
            url = m[2]
            try:
                p = urlsplit(url)
                if p.username or p.password or p.query or p.fragment:
                    continue
            except ValueError:
                continue
            # The registry is host-based for some providers, so explicitly
            # non-language routes need a separate check. Root configuration
            # identifies a service, not the purpose of all runtime requests.
            if (not _NON_LANGUAGE_ROUTE.search(p.path)
                    and match_api_service(url, classification_only=True)):
                return True
        elif st == 'api_key_input_prompt':
            if instance.get('source_kind') not in {'ui_prompt', 'resource_string_literal', 'text_marker'}:
                continue
            ui_source = any(re.search(r'(?:^|/)res/values[^/]*/strings\.xml$', s.replace('\\', '/'), re.I)
                            or (instance.get('source_kind') == 'ui_prompt'
                                and s.lower().endswith(('.html', '.htm'))) for s in refs)
            if ui_source and _CREDENTIAL_LABEL.fullmatch(value) and has_strong_llm_context(value):
                return True
    return False


def source_categories(signal: dict) -> tuple[bool, bool]:
    code = resource = False
    for instance in occurrences(signal):
        kind = instance.get('source_kind') or ''
        refs = sources(instance)
        if kind in {'smali_http_contract', 'hermes_string_literal', 'ram_string_literal'}:
            code = True
        for ref in refs:
            path = ref.lower().replace('\\', '/')
            code |= path.endswith(('.dex', '.smali', '.java', '.kt', '.so', '.js', '.mjs', '.cjs', '.jsx', '.tsx'))
            if kind == 'source_string_literal':
                code |= path.endswith(('.jsbundle', '.bundle')) or any(
                    str(c).startswith('script_asset=') for c in instance.get('components', []))
            if kind not in {'hermes_string_literal', 'ram_string_literal'}:
                resource |= path.endswith(('strings.xml', '.json', '.html', '.md', '.properties'))
    return bool(code), bool(resource)


def collapse_identity_facts(signals: list[dict]) -> list[dict]:
    """Dedup model inputs only, preserving each independently qualified proof.

    Calls and web relations deliberately remain occurrence-specific. No
    relationship can acquire validation from this union of sources.
    """
    domain_types = {'known_llm_api_domain', 'known_llm_web_domain'}
    identity_types = domain_types | {'business_ai_endpoint', 'llm_special_token',
                                    'llm_framework_keyword', 'native_llm_library',
                                    'llm_operation_context'}
    result, positions = [], {}
    for signal in signals:
        st = signal.get('signal_type')
        if st not in identity_types:
            result.append(signal)
            continue
        value = str(signal.get('value') or '').strip()
        identity = host_identity(value) if st in domain_types else value
        if st in {'llm_operation_context', 'llm_framework_keyword'}:
            identity = ' '.join(identity.lower().split())
        key = st, identity
        if not identity or key not in positions:
            clone = dict(signal)
            clone['evidence_instances'] = list(occurrences(signal))
            clone['source_files'] = list(signal.get('source_files') or [])
            result.append(clone)
            positions[key] = len(result) - 1
        else:
            target = result[positions[key]]
            target['evidence_instances'] += list(occurrences(signal))
            target['source_files'] = list(dict.fromkeys(target['source_files'] + list(signal.get('source_files') or [])))
    return result
