"""Semantic views of retained APK evidence, never app labels.

Inputs have passed admission per occurrence. Web asset provenance is weaker
than navigation or UI binding; routes and references do not imply calls.
"""
from __future__ import annotations

import posixpath
import re
from urllib.parse import urlsplit

from feature_evidence import llm_operation_reference, occurrences, sources
from registry import is_reference_only_source


_REFERENCE_FACTS = frozenset({
    'llm_protocol_route', 'llm_serialized_field', 'llm_model_property_reference',
})
_WEB_FACT_TYPES = frozenset({
    'known_llm_api_domain', 'known_llm_web_domain', 'known_llm_model_name',
    'known_llm_sdk_package', 'business_ai_endpoint', 'llm_json_schema',
    'llm_prompt_template', 'llm_special_token', 'llm_operation_context',
    'llm_http_api_usage', 'baseurl_override', 'api_key_input_prompt',
})
_AUXILIARY_PARTS = frozenset({
    'node_modules', 'vendor', 'vendors', 'third_party', 'third-party',
    'test', 'tests', '__tests__', 'example', 'examples', 'sample', 'samples',
    'doc', 'docs', 'documentation', 'licenses', 'licences',
})


def isolated_occurrence(signal: dict, instance: dict) -> dict:
    """Keep value/type identity, but never borrow another occurrence's proof."""
    return {
        **signal,
        **{key: instance.get(key) for key in (
            'role', 'scope', 'relation', 'verification', 'source_kind')},
        'components': list(instance.get('components') or []),
        'source_files': sources(instance),
        'evidence_instances': [instance],
    }


def _components(instance: dict, key: str) -> set[str]:
    prefix = key + '='
    return {str(c)[len(prefix):] for c in (instance.get('components') or [])
            if str(c).startswith(prefix) and str(c)[len(prefix):].strip()}


def _asset_path(value: str) -> str:
    path = str(value).replace('\\', '/')
    if '://' in path:
        return ''
    path = re.sub(r'#inline-\d+$', '', path)
    # Decoded assets and their APK-relative forms denote the same source.
    if not path.startswith('assets/'):
        if '/assets/' not in path:
            return ''
        path = 'assets/' + path.split('/assets/', 1)[1]
    if any(part in {'', '.', '..'} for part in path.split('/')):
        return ''
    if '?' in path or '#' in path or '://' in path:
        return ''
    return path


def web_frontend_llm_observation(signals: list[dict], metadata: dict) -> bool:
    """LLM observations in configured HTML trees, not proven web usage.

    Roots come from retained entry configuration/arguments, not directory name
    guesses. Partial parsing does not erase independent source observations.
    """
    roots = set()
    frontend = metadata.get('frontend_analysis') or {}
    for entry in frontend.get('entries', []):
        if entry.get('binding') not in {
                'framework_entry_configuration', 'local_asset_argument'}:
            continue
        asset = _asset_path(entry.get('asset', ''))
        if asset and asset.lower().endswith(('.html', '.htm')):
            roots.add(posixpath.dirname(asset) + '/')
    if not roots:
        return False
    for signal in signals:
        if signal.get('signal_type') not in _WEB_FACT_TYPES:
            continue
        if (signal.get('signal_type') == 'llm_operation_context'
                and not llm_operation_reference(signal)):
            continue
        for instance in occurrences(signal):
            if instance.get('source_kind') in {'hermes_string_literal', 'ram_string_literal'}:
                continue
            for source in sources(instance):
                path = _asset_path(source)
                parts = path.lower().split('/')
                if (not path or not any(path.startswith(root) for root in roots)
                        or not path.lower().endswith(('.html', '.htm', '.js', '.mjs', '.cjs'))
                        or _AUXILIARY_PARTS.intersection(parts)
                        or is_reference_only_source(path)
                        or re.match(r'^(?:licen[cs]e|notice|readme)(?:[._-]|$)', parts[-1])):
                    continue
                return True
    return False


def business_llm_route_observation(signal: dict) -> bool:
    """An explicit related route observation, not invocation or ownership."""
    if signal.get('signal_type') != 'business_ai_endpoint':
        return False
    for instance in occurrences(signal):
        paths = {p for p in _components(instance, 'llm_path')
                 if p.split('?', 1)[0].strip('/')}
        if (instance.get('role') != 'relation'
                or not instance.get('relation')
                or instance.get('scope') not in {'value', 'config', 'method',
                                                'local_method_graph', 'class', 'file'}
                or instance.get('verification') not in {'observed', 'verified'}
                or not _components(instance, 'route_strength') & {'strong', 'contextual'}
                or not paths
                or not sources(instance)
                or all(is_reference_only_source(s) for s in sources(instance))):
            continue
        value = str(signal.get('value') or '').strip()
        try:
            target = urlsplit(value)
        except ValueError:
            continue
        if target.scheme in {'http', 'https'} and target.hostname:
            return True
        # Relative constants retain a business-route observation without
        # inventing which base URL or request eventually uses that route.
        if (value and not target.scheme and not target.netloc
                and value in paths
                and _components(instance, 'custom_domain')):
            return True
    return False


def llm_protocol_structure_reference(signal: dict) -> bool:
    """Protocol/structure references remain separate from generic AI text."""
    for instance in occurrences(signal):
        single = isolated_occurrence(signal, instance)
        if (llm_operation_reference(single)
                and _components(instance, 'business_fact') & _REFERENCE_FACTS):
            return True
    return False
