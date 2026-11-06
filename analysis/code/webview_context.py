"""LLM-related WebView observations, without embedding-mode decisions.

Binding and relevance are separate facts. A chat-like route or a nearby LLM
identity is a candidate, even when the navigation argument itself is known.
"""
from __future__ import annotations

import re
from urllib.parse import unquote, urlsplit

from registry import extract_urls, has_strong_llm_context, match_api_service, match_web_service
from signal_types import (
    RawSignal, SignalRelation, SignalRole, SignalScope, SignalType, SignalVerification,
)

WEBVIEW_SIGNAL_TYPES = frozenset({
    SignalType.WEBVIEW_PRESENT.value, SignalType.WEBVIEW_LOAD_CALL.value,
    SignalType.WEBVIEW_LLM_TARGET.value, SignalType.WEBVIEW_CHAT_PAGE.value,
})
_AUXILIARY = frozenset({
    "privacy", "privacy-policy", "privacy_policy", "privacypolicy", "terms",
    "terms-of-service", "terms_of_service", "termsofservice", "legal", "policies",
    "cookie-policy", "cookies", "documentation", "docs", "help", "support",
    "checkout", "payment", "payments", "billing", "ads", "advertising",
})
_CHAT_ROUTES = frozenset({"chat", "assistant", "conversation", "ai", "llm", "message", "copilot"})
_ACCESS_ROUTES = frozenset({"login", "signin", "sign-in", "auth", "oauth", "account"})
_POLICY_ROUTES = frozenset({
    "privacy", "privacy-policy", "privacy_policy", "privacypolicy", "terms",
    "terms-of-service", "terms_of_service", "termsofservice", "policies",
    "cookie-policy", "cookies",
})


def page_segments(url: str) -> list[str]:
    try:
        parts = urlsplit(url)
    except ValueError:
        return []
    # SPA hash routes are pages too; arbitrary query values are not routes.
    route = parts.path + ("/" + parts.fragment if parts.fragment.startswith(("/", "!/")) else "")
    return [re.sub(r"\.(?:html?|php|aspx?)$", "", s.lower())
            for s in unquote(route).strip("/!").split("/") if s]


def is_auxiliary_page(url: str) -> bool:
    """Check explicit page paths, never words in a host or query value."""
    return bool(_AUXILIARY.intersection(page_segments(url)))


def llm_page_context(text: str) -> bool:
    # A reference link alone must not make an unrelated loaded page relevant.
    for url in extract_urls(text):
        text = text.replace(url, "")
    text = re.sub(r"(?m)^\s*(?:package|import|\.class|\.super)\b[^\n]*", "", text)
    return has_strong_llm_context(text)


def is_auxiliary_html(text: str) -> bool:
    titles = re.findall(r"<(?:title|h1)\b[^>]*>(.*?)</(?:title|h1)>", text, re.I | re.S)
    return any(re.search(r"\b(?:privacy policy|terms of (?:service|use)|cookie policy)\b", title, re.I)
               or re.fullmatch(r"\s*(?:checkout|payment)\s*", title, re.I) for title in titles)


def llm_html_context(text: str) -> bool:
    if is_auxiliary_html(text):
        return False
    # URLs inside the actual HTML argument/content are related page facts.
    # They stay candidates: a link or script reference is not navigation.
    return llm_page_context(text) or any(
        not is_auxiliary_page(url) and (match_web_service(url)
                                       or match_api_service(url, classification_only=True))
        for url in extract_urls(text)
    )


def page_observation(
    url: str, source: str, layer: str, scope: SignalScope, *,
    call: str, binding: str, source_kind: str, typed: bool = False,
    context: str = "", extra: tuple[str, ...] = (),
) -> RawSignal | None:
    segments = page_segments(url)
    if not url or _POLICY_ROUTES.intersection(segments):
        return None
    auxiliary = is_auxiliary_page(url)
    # "support"/"legal" can also host an AI assistant. Keep related
    # candidates when chat routes or actual page/code context support them.
    if auxiliary and not (_CHAT_ROUTES.intersection(segments) or llm_page_context(context)):
        return None
    known = match_web_service(url)
    if known:
        kind, relation, hint = (SignalType.WEBVIEW_LLM_TARGET,
                               SignalRelation.WEBVIEW_KNOWN_LLM_TARGET, "known_llm_website")
    elif match_api_service(url, classification_only=True):
        # Loading an API URL is not evidence of an interactive web page.
        return None
    elif (llm_html_context(context) if binding == "local_asset_argument" else llm_page_context(context)):
        kind, relation, hint = (SignalType.WEBVIEW_CHAT_PAGE,
                               SignalRelation.WEBVIEW_CUSTOM_CHAT_TARGET, "llm_page_context")
    elif _CHAT_ROUTES.intersection(segments):
        kind, relation, hint = (SignalType.WEBVIEW_CHAT_PAGE,
                               SignalRelation.WEBVIEW_CUSTOM_CHAT_TARGET, "chat_like_route")
    else:
        return None
    bound = binding in {"literal_argument", "register_argument", "local_asset_argument",
                        "constructor_field_argument"}
    verified = bool(known and typed and bound and not auxiliary and not _ACCESS_ROUTES.intersection(segments))
    return RawSignal(
        kind, url, layer, source, role=SignalRole.RELATION, scope=scope,
        relation=relation, verification=(SignalVerification.VERIFIED if verified
                                       else SignalVerification.CANDIDATE),
        source_kind=source_kind,
        components=(f"webview_call={call}", f"binding={binding}", f"relevance_hint={hint}",
                    f"{'target_url' if bound else 'candidate_url'}={url}", *extra),
    )


def related_usage(source: str, call: str, components: tuple[str, ...],
                  verification: SignalVerification, *,
                  scope: SignalScope = SignalScope.METHOD) -> list[RawSignal]:
    return [RawSignal(
        kind, value, "apktool", source, role=role, scope=scope,
        verification=verification, source_kind="typed_smali_webview",
        components=components,
    ) for kind, value, role in (
        (SignalType.WEBVIEW_PRESENT, "android.webkit.WebView", SignalRole.REFERENCE),
        (SignalType.WEBVIEW_LOAD_CALL, f"WebView.{call}", SignalRole.CALL),
    )]


def web_signal_feature_eligible(signal: dict) -> bool:
    """Require binding and LLM relevance in ONE occurrence, not a merged bag."""
    instances = signal.get('evidence_instances') or [signal]
    bindings = {'literal_argument', 'register_argument', 'local_asset_argument',
                'constructor_field_argument'}
    hints = {'known_llm_website', 'llm_page_context', 'llm_html_context'}
    for instance in instances:
        proof = set(instance.get('components') or ())
        if (instance.get('source_kind') == 'frontend_navigation'
                and instance.get('scope') == 'file'
                and {'binding=frontend_entry_navigation','relevance_hint=known_llm_website'}.issubset(proof)
                and any(c.startswith('entry_asset=assets/') for c in proof)
                and any(c in proof for c in ('entry_binding=local_asset_argument',
                                            'entry_binding=framework_entry_configuration'))):
            return True
        if (instance.get('source_kind') == 'frontend_entry_request'
                and instance.get('verification') == 'verified'
                and instance.get('scope') in {'file', 'method'}
                and {'binding=frontend_entry_request_response', 'relevance_hint=bound_text_generation',
                     'input_binding=dom_event', 'response_binding=dom_text'}.issubset(proof)
                and any(c.startswith('entry_asset=assets/') for c in proof)
                and any(c in proof for c in ('entry_binding=local_asset_argument',
                                            'entry_binding=framework_entry_configuration'))):
            return True
        if instance.get('scope') not in {'value', 'config', 'method', 'local_method_graph', 'class', 'file'}:
            continue
        components = set(instance.get('components') or ())
        if (any('binding=' + binding in components for binding in bindings)
                and any('relevance_hint=' + hint in components for hint in hints)):
            return True
    return False
