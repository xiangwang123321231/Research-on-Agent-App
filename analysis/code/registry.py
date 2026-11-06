"""Keyword / pattern registries and shared utility functions.

All constants use ``set`` / ``tuple`` / ``frozenset`` — they are never
mutated at runtime.  Extractor modules import the registries they need
and pass them to utility functions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import ipaddress
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Iterator
from urllib.parse import parse_qsl, urlsplit

from signal_types import (
    RawSignal,
    SignalRelation,
    SignalRole,
    SignalScope,
    SignalType,
    SignalVerification,
)
from source_facts import serialized_field_groups, without_comments

# ═══════════════════════════════════════════════════════════════════
# Androguard (optional — AXMLPrinter only, for manifest decoding)
# ═══════════════════════════════════════════════════════════════════

try:
    from androguard.core.axml import AXMLPrinter  # type: ignore
except Exception:  # pragma: no cover
    AXMLPrinter = None

# Silence androguard's loguru debug output (AXMLPrinter is chatty).
try:
    import loguru  # type: ignore
    loguru.logger.remove()
except Exception:
    pass
import logging
logging.getLogger("androguard").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════
# Versioned service / SDK registry
# ═══════════════════════════════════════════════════════════════════

EVIDENCE_REGISTRY_VERSION = "llm-evidence-registry-v26"
SIGNAL_SCHEMA_VERSION = "neutral-signals-v2026.10.24.2"


@dataclass(frozen=True)
class ApiServiceEntry:
    service_id: str
    provider: str
    service_class: str
    exact_hosts: tuple[str, ...] = ()
    host_suffixes: tuple[str, ...] = ()
    required_host_markers: tuple[str, ...] = ()
    required_path_markers: tuple[str, ...] = ()
    classification_signal: bool = True
    evidence_strength: str = "high"
    source_reference: str = "provider_api_documentation"


@dataclass(frozen=True)
class SdkEntry:
    package: str
    provider: str | None
    kind: str
    classification_signal: bool = True
    requires_context: bool = False
    evidence_strength: str = "high"
    source_reference: str = "published_package_namespace"


@dataclass(frozen=True)
class WebServiceEntry:
    service_id: str
    provider: str
    hosts: tuple[str, ...]
    evidence_strength: str = "high"
    source_reference: str = "provider_web_service"
    path_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeEntry:
    runtime_family: str
    markers: tuple[str, ...]
    weak_path_identity: bool = False
    source_reference: str = "runtime_project_identity"


API_SERVICE_REGISTRY: tuple[ApiServiceEntry, ...] = (
    ApiServiceEntry("openai_api", "openai", "first_party_api",
                    exact_hosts=("api.openai.com",)),
    ApiServiceEntry("anthropic_api", "anthropic", "first_party_api",
                    exact_hosts=("api.anthropic.com", "api.claude.ai")),
    ApiServiceEntry("gemini_api", "google_gemini", "first_party_api",
                    exact_hosts=("generativelanguage.googleapis.com",)),
    ApiServiceEntry("firebase_ai", "google_gemini", "cloud_platform_api",
                    exact_hosts=("firebasevertexai.googleapis.com",),
                    source_reference="firebase_ai_sdk_generative_service_endpoint"),
    # Only the text service, not Pollinations' image service or website.
    ApiServiceEntry("pollinations_text", "pollinations", "aggregator_api",
                    exact_hosts=("text.pollinations.ai",),
                    source_reference="pollinations_text_api_endpoint"),
    ApiServiceEntry("mistral_api", "mistral_ai", "first_party_api",
                    exact_hosts=("api.mistral.ai",)),
    ApiServiceEntry("groq_api", "groq", "first_party_api",
                    exact_hosts=("api.groq.com",)),
    ApiServiceEntry("cohere_api", "cohere", "first_party_api",
                    exact_hosts=("api.cohere.ai",)),
    ApiServiceEntry("perplexity_api", "perplexity", "first_party_api",
                    exact_hosts=("api.perplexity.ai",)),
    ApiServiceEntry("together_api", "together_ai", "aggregator_api",
                    exact_hosts=("api.together.xyz",)),
    ApiServiceEntry("deepseek_api", "deepseek", "first_party_api",
                    exact_hosts=("api.deepseek.com",)),
    ApiServiceEntry("xai_api", "xai", "first_party_api",
                    exact_hosts=("api.x.ai",)),
    ApiServiceEntry("azure_openai", "azure_openai", "cloud_platform_api",
                    host_suffixes=("openai.azure.com",)),
    # OpenRouter's public API is hosted under /api/v1 on the web origin.
    ApiServiceEntry("openrouter_api", "openrouter", "aggregator_api",
                    exact_hosts=("openrouter.ai",),
                    required_path_markers=("/api/v1",)),
    ApiServiceEntry("vapi_api", "vapi", "voice_agent_platform_api",
                    exact_hosts=("api.vapi.ai",)),
    # These platforms also serve non-LLM models.  Keep them in research
    # evidence, but require an LLM-specific path before producing a signal.
    ApiServiceEntry("vertex_generative_api", "google_vertex_ai",
                    "cloud_platform_api",
                    exact_hosts=("aiplatform.googleapis.com",),
                    required_path_markers=("generatecontent", "streamgeneratecontent",
                                           "/publishers/google/models/gemini")),
    ApiServiceEntry("bedrock_runtime", "aws_bedrock", "cloud_platform_api",
                    host_suffixes=("amazonaws.com",),
                    required_host_markers=("bedrock-runtime",),
                    required_path_markers=("/model/", "invoke")),
    ApiServiceEntry("replicate_api", "replicate", "aggregator_api",
                    exact_hosts=("api.replicate.com",),
                    required_path_markers=("/models/", "/predictions"),
                    classification_signal=False, evidence_strength="medium"),
    ApiServiceEntry("novita_api", "novita_ai", "aggregator_api",
                    exact_hosts=("api.novita.ai",),
                    required_path_markers=("/llm/", "/chat/completions"),
                    classification_signal=False, evidence_strength="medium"),
)


LITERT_LM_PACKAGE = "com.google.ai.edge.litertlm"

SDK_REGISTRY: tuple[SdkEntry, ...] = (
    SdkEntry("com.google.firebase.vertexai", "google_vertex_ai", "cloud_sdk"),
    SdkEntry("com.google.firebase.ai", "google_gemini", "cloud_sdk"),
    SdkEntry("software.amazon.awssdk.services.bedrockruntime", "aws_bedrock", "cloud_sdk"),
    SdkEntry("software.amazon.awssdk.services.bedrock", "aws_bedrock", "cloud_sdk"),
    SdkEntry("com.theokanning.openai", "openai_compatible", "cloud_sdk"),
    SdkEntry("com.aallam.openai", "openai_compatible", "cloud_sdk"),
    SdkEntry("com.openai.client", "openai", "cloud_sdk"),
    SdkEntry("com.azure.ai.openai", "azure_openai", "cloud_sdk"),
    SdkEntry("com.google.ai.client.generativeai", "google_gemini", "cloud_sdk"),
    SdkEntry("dev.langchain4j", None, "orchestration_sdk"),
    SdkEntry("com.anthropic", "anthropic", "cloud_sdk"),
    SdkEntry("@vapi-ai/react-native", "vapi", "cloud_sdk"),
    SdkEntry("com.google.mediapipe.tasks.genai", None, "on_device_sdk"),
    SdkEntry(LITERT_LM_PACKAGE, None, "on_device_sdk"),
    SdkEntry("android.app.aicore", None, "on_device_sdk"),
    SdkEntry("com.google.mlkit.genai.prompt", None, "on_device_sdk"),
    SdkEntry("ai.mlc.mlcllm", None, "on_device_sdk"),
    SdkEntry("org.pytorch.executorch", None, "on_device_sdk"),
    SdkEntry("chat.teracube.llamacpp", None, "on_device_sdk"),
    # These are product/runtime identities, not Java package namespaces.
    # They remain useful neutral observations but use token boundaries below.
    SdkEntry("litellm", None, "self_hosted_gateway",
             requires_context=True, evidence_strength="medium"),
    SdkEntry("localai", None, "self_hosted_gateway",
             requires_context=True, evidence_strength="medium"),
    SdkEntry("vllm", None, "self_hosted_runtime",
             requires_context=True, evidence_strength="medium"),
    SdkEntry("fastchat", None, "self_hosted_runtime",
             requires_context=True, evidence_strength="medium"),
    SdkEntry("ollama", None, "self_hosted_runtime",
             requires_context=True, evidence_strength="medium"),
    SdkEntry("xinference", None, "self_hosted_runtime",
             requires_context=True, evidence_strength="medium"),
)


WEB_SERVICE_REGISTRY: tuple[WebServiceEntry, ...] = (
    WebServiceEntry("openai_chatgpt", "openai", ("chatgpt.com", "chat.openai.com")),
    WebServiceEntry("anthropic_claude", "anthropic", ("claude.ai",)),
    WebServiceEntry("google_gemini", "google", ("gemini.google.com",)),
    WebServiceEntry("deepseek_chat", "deepseek", ("chat.deepseek.com",)),
    WebServiceEntry("poe", "quora", ("poe.com",)),
    WebServiceEntry("meta_ai", "meta", ("meta.ai",)),
    WebServiceEntry("xai_grok", "xai", ("grok.com",)),
    WebServiceEntry("perplexity", "perplexity", ("perplexity.ai",)),
    WebServiceEntry("hugging_face_chat", "hugging_face", ("huggingface.co",), path_prefixes=("/chat",)),
    WebServiceEntry("microsoft_copilot", "microsoft", ("copilot.microsoft.com",)),
    WebServiceEntry("character_ai", "character_ai", ("character.ai",)),
)


RUNTIME_REGISTRY: tuple[RuntimeEntry, ...] = (
    RuntimeEntry("llama_cpp", ("librnllama", "libllama", "libggml", "llama.cpp", "ggml")),
    RuntimeEntry("mlc_llm", ("mlc-chat-config", "mlc_llm", "mlcllm", "tvm_runtime")),
    RuntimeEntry("executorch", ("executorch",)),
    RuntimeEntry("mediapipe_genai", ("mediapipe genai", "mediapipe.tasks.genai")),
    RuntimeEntry("android_aicore", ("android.app.aicore", "bind_aicore_service")),
    RuntimeEntry("mnn", ("mnn_llm", "mnnllm", "libmnn"), True),
    RuntimeEntry("onnx_runtime", ("onnxruntime",), True),
    RuntimeEntry("tflite_litert", ("litertlm", "tflite", "litert"), True),
)


KNOWN_LLM_PACKAGES: frozenset[str] = frozenset(
    entry.package for entry in SDK_REGISTRY if entry.classification_signal
)

# Subsets used by FeatureVectorizer — grouped by what the SDK *is*,
# not what classification it implies.
CLOUD_SDK_PACKAGES: frozenset[str] = frozenset(
    entry.package for entry in SDK_REGISTRY
    if entry.kind in {"cloud_sdk", "orchestration_sdk"}
    and entry.classification_signal
)

SELF_HOSTED_SDK_PACKAGES: frozenset[str] = frozenset(
    entry.package for entry in SDK_REGISTRY
    if entry.kind in {"self_hosted_gateway", "self_hosted_runtime"}
    and entry.classification_signal
)

ON_DEVICE_SDK_PACKAGES: frozenset[str] = frozenset(
    entry.package for entry in SDK_REGISTRY
    if entry.kind == "on_device_sdk" and entry.classification_signal
) | frozenset({
    "com.google.android.play.core.assetpacks",
    "com.google.android.play.core.splitinstall",
})


_REFERENCE_ONLY_SOURCE_MARKERS = (
    "meta-inf/", "meta-inf\\", "license", "notice", "readme",
    "changelog", "third_party", "third-party", "proguard", "mapping.txt",
)

# These files can contain real AI project names and URLs, but the URLs are
# references or attribution links rather than endpoints called by the app.
# Keep this separate from ``is_reference_only_source`` so URL evidence can be
# filtered without suppressing every other neutral observation in the file.
_ENDPOINT_REFERENCE_SOURCE_MARKERS = (
    "/credits/", "\\credits\\", "thirdparty", "third_party", "third-party",
    "/terms/", "\\terms\\", "/privacy/", "\\privacy\\",
    "/legal/", "\\legal\\", "/documentation/", "\\documentation\\",
    "/docs/", "\\docs\\", "/support/", "\\support\\",
    "readme", "license", "notice", "changelog",
)


def is_reference_only_source(source: str) -> bool:
    lower = source.replace("\\", "/").lower()
    return any(marker.replace("\\", "/") in lower
               for marker in _REFERENCE_ONLY_SOURCE_MARKERS)


def is_endpoint_reference_source(source: str) -> bool:
    """Return whether *source* is primarily attribution/reference material.

    This is intentionally narrower than ``is_reference_only_source``: it is
    used only to prevent a URL in credits, terms, or documentation from being
    promoted to a business endpoint. Other neutral observations in the file
    remain available to the extractor.
    """
    lower = source.replace("\\", "/").lower()
    return any(marker.replace("\\", "/") in lower
               for marker in _ENDPOINT_REFERENCE_SOURCE_MARKERS)


def text_contains_sdk_package(text: str, package: str) -> bool:
    """Return whether *text* contains one complete SDK package identity."""
    lower = text.lower()
    token = package.lower()
    if token.startswith("@") and "/" in token:
        # Scoped NPM packages use '/' as part of their identity. A following
        # slash may introduce an exported subpath, but '-', '_' and '.' can
        # legally extend the package name and therefore are not boundaries.
        pattern = (
            rf"(?<![a-z0-9_@.\-]){re.escape(token)}"
            rf"(?![a-z0-9_.\-])"
        )
        return bool(re.search(pattern, lower))

    # Java class/package names may be rendered as slash-separated bytecode
    # paths. A following dot is allowed because registry entries commonly
    # name a package prefix rather than one concrete class.
    package_text = lower.replace("/", ".")
    normalized = token.replace("/", ".")
    pattern = rf"(?<![a-z0-9_]){re.escape(normalized)}(?![a-z0-9_])"
    return bool(re.search(pattern, package_text))


def iter_known_sdk_packages(text: str) -> Iterator[str]:
    """Yield reliable SDK/runtime identities with token boundaries."""
    for entry in SDK_REGISTRY:
        if not entry.classification_signal:
            continue
        if entry.requires_context:
            continue
        # The new LiteRT-LM identity must not be introduced by a source
        # comment. Preserve the existing treatment of all other SDKs.
        if not text_contains_sdk_package(text, entry.package):
            continue
        if entry.package == LITERT_LM_PACKAGE:
            # Only strip syntax after an identity hit, not for every binary
            # blob/source file in a large batch without LiteRT-LM.
            candidate = re.sub(r"(?m)^\s*#[^\n]*", "", without_comments(text))
            if not text_contains_sdk_package(candidate, entry.package):
                continue
        yield entry.package


def iter_contextual_sdk_identities(text: str) -> Iterator[str]:
    """Yield self-hosted identities only with protocol/config context."""
    lower = text.lower()
    common_context = any(marker in lower for marker in (
        "/v1/chat/completions", "/chat/completions", "openai_api_base",
        "api_base_url", "base_url", "setbaseurl", "endpoint",
    ))
    for entry in SDK_REGISTRY:
        if not entry.classification_signal or not entry.requires_context:
            continue
        token = entry.package.lower()
        if not re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", lower):
            continue
        if token == "ollama":
            context_ok = common_context or any(marker in lower for marker in (
                "11434", "/api/chat", "/api/generate", "ollama.com/api",
            ))
        else:
            context_ok = common_context
        if context_ok:
            yield entry.package


# ═══════════════════════════════════════════════════════════════════
# OpenAI-compatible JSON schema keys
# ═══════════════════════════════════════════════════════════════════

OPENAI_SCHEMA_KEYS: frozenset[str] = frozenset({
    "model", "messages", "role", "content", "system", "user", "assistant",
    "prompt", "input", "temperature", "max_tokens", "max_output_tokens",
    "top_p", "stream", "stop", "frequency_penalty", "presence_penalty",
    "choices", "delta", "finish_reason", "usage",
})
OPENAI_SCHEMA_STRONG_KEYS: frozenset[str] = frozenset({
    "model", "messages", "role", "assistant", "temperature", "max_tokens",
    "max_output_tokens", "top_p", "choices", "delta", "finish_reason",
})

_GENERATION_PARAMETER_KEYS: frozenset[str] = frozenset({
    "temperature", "max_tokens", "max_output_tokens", "top_p", "stream",
    "stop", "frequency_penalty", "presence_penalty",
})


def extract_llm_schema_components(text: str) -> tuple[str, ...]:
    """Return fields forming a language-model request/response schema.

    A provider URL is deliberately not required. Server-mediated apps often
    expose their own route while retaining an OpenAI-like payload. Conversely,
    common chat fields such as ``messages``/``role``/``content`` are not enough
    on their own because ordinary person-to-person messaging uses them too.
    """
    if not text:
        return ()
    for group in serialized_field_groups(text):
        fields = set(group) & OPENAI_SCHEMA_KEYS
        if _is_llm_schema(fields):
            return tuple(sorted(fields))
    return ()


def _is_llm_schema(fields: set[str]) -> bool:
    generation_fields = fields & _GENERATION_PARAMETER_KEYS

    request_shape = (
        {"model", "messages"}.issubset(fields) and bool(generation_fields)
    )
    role_request_shape = (
        {"messages", "role", "content"}.issubset(fields)
        and len(generation_fields) >= 2
    )
    assistant_role_shape = (
        {"messages", "role", "assistant"}.issubset(fields)
        and bool(generation_fields)
    )
    prompt_request_shape = (
        "prompt" in fields and len(generation_fields) >= 2
    )
    response_shape = (
        {"choices", "finish_reason"}.issubset(fields)
        and bool(fields & {"delta", "usage"})
    )
    role_vocabulary_shape = (
        {"system", "user", "assistant", "messages"}.issubset(fields)
        and ("model" in fields or bool(generation_fields))
    )
    return any((request_shape, role_request_shape, assistant_role_shape,
                prompt_request_shape, response_shape, role_vocabulary_shape))


# ═══════════════════════════════════════════════════════════════════
# Model name literals (word-bounded)
# ═══════════════════════════════════════════════════════════════════

# Family rules require an actual version or provider model variant.  Bare
# words such as "mistral", "llama" and "phi" are deliberately excluded.
# ``o1``/``o3`` are also excluded because they collide heavily in binaries.
_MODEL_ID_PATTERN_TEXTS: tuple[str, ...] = (
    r"gpt[-_.]?\d[a-z0-9._-]{0,60}",
    r"claude[-_.]?(?:\d|instant)[a-z0-9._-]{0,60}",
    r"gemini[-_.]?(?:\d|(?:pro|ultra|flash|nano)(?![a-z]))[a-z0-9._-]{0,60}",
    r"deepseek-(?:[a-z0-9][a-z0-9._-]{1,60})",
    r"qwen(?:[-_.]?(?:\d|coder|vl|max|plus|turbo))[a-z0-9._-]{0,60}",
    r"(?:meta[-_.])?llama[-_.]?\d[a-z0-9._-]{0,60}",
    r"mixtral[-_.]?\d[a-z0-9._-]{0,60}",
    r"mistral[-_.]?(?:\d|small|medium|large|nemo|codestral)[a-z0-9._-]{0,60}",
    r"gemma[-_.]?\d[a-z0-9._-]{0,60}",
    r"phi[-_.]?\d[a-z0-9._-]{0,60}",
    r"command-r(?:[-+._][a-z0-9][a-z0-9._-]{0,40})?",
    r"grok[-_.]?\d[a-z0-9._-]{0,60}",
    r"glm[-_.]?\d[a-z0-9._-]{0,60}",
)

LLM_MODEL_HINT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"(?<![A-Za-z0-9])(?:{pattern})(?![A-Za-z0-9])", re.IGNORECASE)
    for pattern in _MODEL_ID_PATTERN_TEXTS
)

_HF_MODEL_ID_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_.-])"
    rf"(?P<owner>[A-Za-z0-9][A-Za-z0-9_.-]{{0,63}})/"
    rf"(?P<model>(?:{'|'.join(_MODEL_ID_PATTERN_TEXTS)}))"
    rf"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_HF_REPOSITORY_OWNER_NOISE = frozenset({
    "assets", "res", "raw", "files", "model", "models", "weights",
})

# Kept as a compatibility/export aid. Matching is performed by the patterns
# above because a finite literal list becomes stale quickly.
LLM_MODEL_HINTS: frozenset[str] = frozenset({
    "gpt", "claude", "gemini", "deepseek", "qwen", "llama",
    "mixtral", "mistral", "gemma", "phi", "command-r", "grok", "glm",
})

_MODEL_CONFIGURATION_RE = re.compile(
    r"(?ix)(?:"
    r"(?:\bmodel(?:[_-]?(?:name|id))?\b|\bdeployment(?:[_-]?name)?\b)"
    r"[\"']?\s*(?:[:=]|\()\s*"
    r"|\b(?:put|putopt|addproperty|set)\s*\(\s*[\"']"
    r"(?:model(?:[_-]?(?:name|id))?|deployment(?:[_-]?name)?)[\"']\s*,\s*"
    r"|\b(?:setmodel|setmodelname|setmodelid)\s*\(\s*"
    r"|[\"'](?:model(?:[_-]?(?:name|id))?|deployment(?:[_-]?name)?)[\"']"
    r"\s+to\s+"
    r")[\"'](?P<model_value>[^\"'\\\r\n]{2,160})[\"']"
)

_BARE_CONFIGURED_MODEL_IDS = frozenset({
    "mistral", "qwen", "llama", "gemma", "phi", "deepseek-chat",
    "deepseek-reasoner", "command-r",
})


def normalize_llm_model_id(value: str) -> str:
    """Canonicalize known separator aliases without guessing model identity."""
    candidate = value.strip().rstrip("._-")
    if "/" in candidate:
        owner, model = candidate.split("/", 1)
        return f"{owner.lower()}/{normalize_llm_model_id(model)}"
    match = re.fullmatch(r"(?i)(gpt|claude|gemini)[-_.]?(.*)", candidate)
    if match and match.group(2):
        return f"{match.group(1).lower()}-{match.group(2).lower()}"
    return candidate.lower()


_MODEL_PROPERTY_SUFFIX = re.compile(
    r'(?:'
    r'(?:badge|info|tooltip|button|banner|dialog|popup|hint|offer|feature|toggle)'
    r'(?:_(?:auto|show|hide))*_(?:visible|enabled|disabled|hidden|shown|dismissed)'
    r'|(?:auto_)?(?:show|hide)_(?:enabled|disabled)'
    r'|(?:success|successful|failed|total|remaining|used)_requests?(?:_count)?'
    r'|(?:hourly|daily|weekly|monthly|yearly)_(?:(?:request|token)_)?(?:limit|quota|count)'
    r'(?:_(?:last_)?reset_(?:date|time|timestamp))?'
    r'|(?:request|token|usage)_(?:count|limit|quota)(?:_(?:last_)?reset_(?:date|time|timestamp))?'
    r'|(?:enabled|disabled|visible|hidden))$')


def model_property_reference(value: str) -> str | None:
    """A model-prefixed setting/counter key is a reference, not a whole ID.

    Require a complete property grammar after a separator or camel boundary.
    Return the prefix only as a reference hint, never as a newly asserted
    model identity. Model versions, dates, repository IDs, quantizations and
    arbitrary unknown suffixes are not stripped. No application identity or
    complete key list participates in this rule.
    """
    if not value or len(value) > 256 or not re.fullmatch(r'[A-Za-z0-9_.-]+', value):
        return None
    for boundary in re.finditer(r'[-_.]+|(?<=[a-z0-9])(?=[A-Z])', value):
        prefix, suffix = value[:boundary.start()], value[boundary.end():]
        words = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', suffix)
        words = re.sub(r'[-_.]+', '_', words).lower()
        if (_MODEL_PROPERTY_SUFFIX.fullmatch(words)
                and any(pattern.fullmatch(prefix) for pattern in LLM_MODEL_HINT_PATTERNS)):
            return normalize_llm_model_id(prefix)
    return None


def iter_model_configuration_values(
    text: str,
) -> Iterator[tuple[str, int, int]]:
    """Yield literal model values from one structured assignment or call."""
    for assignment in _MODEL_CONFIGURATION_RE.finditer(text):
        value = assignment.group("model_value").strip()
        yield (
            value,
            assignment.start("model_value"),
            assignment.end("model_value"),
        )


def _iter_llm_model_id_matches(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield canonical model IDs with their original spans."""
    covered_spans: list[tuple[int, int]] = []
    for match in _HF_MODEL_ID_PATTERN.finditer(text):
        if match.group("owner").lower() in _HF_REPOSITORY_OWNER_NOISE:
            continue
        covered_spans.append(match.span())
        yield normalize_llm_model_id(match.group(0)), match.start(), match.end()
    for pattern in LLM_MODEL_HINT_PATTERNS:
        for match in pattern.finditer(text):
            if any(start <= match.start() and match.end() <= end
                   for start, end in covered_spans):
                continue
            if model_property_reference(match.group(0)):
                continue
            yield normalize_llm_model_id(match.group(0)), match.start(), match.end()
    for value, start, end in iter_model_configuration_values(text):
        if value.lower() in _BARE_CONFIGURED_MODEL_IDS:
            yield normalize_llm_model_id(value), start, end


def extract_llm_model_ids(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for value, _, _ in _iter_llm_model_id_matches(text):
        if value not in seen:
            seen.add(value)
            found.append(value)
    return found


def has_model_configuration_context(text: str, source: str = "") -> bool:
    """Require configuration, API-path, or model-artifact context."""
    return bool(model_configuration_context_components(text, source))


def model_configuration_context_components(
    text: str, source: str = "",
) -> tuple[str, ...]:
    """Return the objective markers that make a model ID contextual."""
    lower = text.lower()
    source_lower = source.lower()
    if is_reference_only_source(source):
        return ()

    components: list[str] = []
    if _MODEL_CONFIGURATION_RE.search(text):
        components.append("model_context=config_assignment")
    if re.search(r"/models/[a-z0-9]", lower):
        components.append("model_context=api_model_path")
    for marker in (
        "chat/completions", "responses.create", "generatecontent",
        "from_pretrained", "load_model", "loadmodel", "model_path",
    ):
        if marker in lower:
            components.append(f"model_context={marker}")
    for marker in (
        "model", "llm", "inference", "generation_config",
        "mlc-chat-config", "tokenizer_config",
    ):
        if marker in source_lower:
            components.append(f"source_context={marker}")
    return tuple(dict.fromkeys(components))


def extract_contextual_llm_model_evidence(
    text: str,
    *,
    context_radius: int = 512,
) -> list[tuple[str, tuple[str, ...]]]:
    """Return normalized model IDs and the local markers supporting them."""
    normalized = text.replace("\x00", " ")
    found: dict[str, set[str]] = {}
    radius = max(0, context_radius)
    for model_id, start, end in _iter_llm_model_id_matches(normalized):
        window = normalized[max(0, start - radius):min(len(normalized), end + radius)]
        components = model_configuration_context_components(window)
        if components:
            found.setdefault(model_id, set()).update(components)
    return [
        (model_id, tuple(sorted(components)))
        for model_id, components in found.items()
    ]


_BINARY_LLM_IDENTITY_RE: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9])(?:chatgpt|openai|anthropic|claude|gemini|deepseek|"
    r"qwen|llama|mistral|mixtral|gemma|phi|grok|command-r)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)
_BINARY_MODEL_MARKER_RE: re.Pattern[str] = re.compile(
    r"(?:gpt|claude|gemini|deepseek|qwen|llama|mistral|mixtral|gemma|"
    r"phi|grok|command-r)",
    re.IGNORECASE,
)

# Native runtimes expose architecture dispatch names, chat-template enum
# labels and C++ source/symbol names in their string tables.  They describe
# implementation support, not a concrete model identifier.  Keep the filter
# structural and apply it only to compiled string pools; the same words in a
# real source/config literal remain available to the normal scanners.
_BINARY_MODEL_ID_NOISE_RE: re.Pattern[str] = re.compile(
    r"(?:\.cpp$|rk\d+|common_chat|templater|templates_params|"
    r"(?:^|[_-])sys(?:[-_]?(?:bos|strip))?$|"
    r"(?:^|[_-])iswa$|ilb\d+[a-z0-9]*$)",
    re.IGNORECASE,
)


def is_binary_model_identifier_noise(model_id: str) -> bool:
    """Return whether a compiled token is an implementation symbol, not an ID."""
    return bool(_BINARY_MODEL_ID_NOISE_RE.search(model_id.strip()))


def extract_binary_llm_model_evidence(
    text: str,
) -> list[tuple[str, tuple[str, ...]]]:
    """Recover explicit model IDs from compiled string pools.

    AOT/native binaries often keep a model literal and the app/provider
    identity as separate NUL-delimited strings.  The normal contextual scan
    handles nearby values; this fallback handles only two *short printable
    strings* within a bounded window.  It deliberately does not treat an
    entire binary file as one scope: compiled string pools contain unrelated
    SDK names and concatenated fragments that otherwise manufacture model IDs.
    """
    family_prefixes: dict[str, tuple[str, ...]] = {
        "chatgpt": ("gpt",),
        "openai": ("gpt",),
        "anthropic": ("claude",),
        "claude": ("claude",),
        "gemini": ("gemini",),
        "deepseek": ("deepseek",),
        "qwen": ("qwen",),
        "llama": ("llama", "meta-llama/"),
        "mistral": ("mistral", "mixtral"),
        "mixtral": ("mistral", "mixtral"),
        "gemma": ("gemma",),
        "phi": ("phi",),
        "grok": ("grok",),
        "command-r": ("command-r",),
    }
    # Keep this fallback separate from the normal contextual extractor.  A
    # printable run longer than a few KiB is usually a bundled JS/data blob,
    # not one string value with a trustworthy owner.
    printable_region = re.compile(r"[\x20-\x7e]{3,}")
    max_region_length = 4096
    context_radius = 512
    found: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for region in printable_region.finditer(text):
        if region.end() - region.start() > max_region_length:
            continue
        segment = region.group(0)
        # Most printable runs in native libraries are ordinary symbols,
        # licenses, or compressed-data artefacts.  Avoid applying the full
        # model-ID pattern set unless the run has a provider/model family
        # marker; this keeps the fallback cheap for multi-megabyte .so files.
        if not _BINARY_MODEL_MARKER_RE.search(segment):
            continue
        # C++ ABI names in a native library encode class/function names as a
        # single printable run (for example ``_ZN17llama_model_gemma17...``).
        # They are implementation symbols, not configured model IDs, even
        # though the family token is recognizable.  Do not inspect mangled
        # symbol runs in this fallback.
        if (
            re.match(r"^\s*(?:_+Z|_+R|\?)", segment)
            or re.search(
                r"(?:llama_model_|model_loader|load_arch|build_arch|"
                r"llm_graph_params)",
                segment,
                re.IGNORECASE,
            )
        ):
            continue
        for model_id, local_start, local_end in _iter_llm_model_id_matches(segment):
            if len(model_id) > 64:
                continue
            if is_binary_model_identifier_noise(model_id):
                continue
            model_lower = model_id.lower()
            # The provider identity may be the adjacent NUL-delimited string,
            # but it must still be close to this model literal.  Identities
            # occurring inside the model candidate itself are not evidence.
            window_start = max(0, region.start() + local_start - context_radius)
            window_end = min(len(text), region.start() + local_end + context_radius)
            identity_matches = []
            for identity_match in _BINARY_LLM_IDENTITY_RE.finditer(
                text[window_start:window_end]
            ):
                identity_start = window_start + identity_match.start()
                identity_end = window_start + identity_match.end()
                candidate_start = region.start() + local_start
                candidate_end = region.start() + local_end
                if candidate_start <= identity_start and identity_end <= candidate_end:
                    continue
                # A native symbol can contain both a family name and a
                # mangled function suffix (for example
                # ``llama_model_gemma17load_arch...``).  Treating that
                # prefix as an independent provider identity would
                # re-authorize the entire mangled symbol.  Provider and model
                # literals must therefore come from separate printable runs
                # (normally NUL-delimited string-table entries).
                if region.start() <= identity_start < region.end():
                    continue
                identity_matches.append(identity_match.group(0).lower())
            identities = tuple(dict.fromkeys(identity_matches))
            allowed_prefixes = tuple(
                prefix for identity in identities
                for prefix in family_prefixes.get(identity, ())
            )
            if not allowed_prefixes or not any(
                model_lower.startswith(prefix) for prefix in allowed_prefixes
            ):
                continue
            if model_id in seen:
                continue
            seen.add(model_id)
            found.append((
                model_id,
                tuple(f"binary_context=bounded_llm_identity:{identity}"
                      for identity in identities),
            ))
    return found


_BINARY_MODEL_ASSIGNMENT_RE: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9_])(?:model|model_id|model_name|model_path|modelname)"
    r"\s*[:=]",
    re.IGNORECASE,
)


def extract_bounded_binary_model_evidence(
    text: str,
    *,
    context_radius: int = 256,
) -> list[tuple[str, tuple[str, ...]]]:
    """Recover model IDs next to an explicit assignment in a binary pool.

    This retains structured literals such as ``model=\"gpt-4o\"`` while
    avoiding the APK-wide contextual scan that joins unrelated native strings.
    The assignment and model ID must occur in the same bounded normalized
    window, and oversized model candidates are rejected.
    """
    normalized = text.replace("\x00", " ")
    radius = max(0, context_radius)
    found: dict[str, set[str]] = {}
    for model_id, start, end in _iter_llm_model_id_matches(normalized):
        if len(model_id) > 64:
            continue
        if is_binary_model_identifier_noise(model_id):
            continue
        window_start = max(0, start - radius)
        window_end = min(len(normalized), end + radius)
        window = normalized[window_start:window_end]
        if not _BINARY_MODEL_ASSIGNMENT_RE.search(window):
            continue
        components = model_configuration_context_components(window)
        if not components:
            components = ("model_context=binary_assignment",)
        found.setdefault(model_id, set()).update(components)
    return [
        (model_id, tuple(sorted(components)))
        for model_id, components in found.items()
    ]


def extract_contextual_llm_model_ids(
    text: str,
    *,
    context_radius: int = 512,
) -> list[str]:
    """Return model IDs supported by nearby configuration or use context.

    Binary string pools can contain unrelated product copy, dependency
    metadata, and model names in the same file.  Check a bounded window around
    each model ID so a distant marker cannot promote an isolated mention to a
    classifier signal.  NUL separators are normalized because DEX/native
    string tables commonly place them between otherwise related tokens.
    """
    return [
        model_id
        for model_id, _ in extract_contextual_llm_model_evidence(
            text, context_radius=context_radius)
    ]


_TOKENIZER_IDENTITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("auto_tokenizer", re.compile(r"(?<![A-Za-z0-9])AutoTokenizer(?![A-Za-z0-9])")),
    ("pretrained_tokenizer", re.compile(
        r"(?<![A-Za-z0-9])PreTrainedTokenizer(?:Fast)?(?![A-Za-z0-9])")),
    ("tokenizer_json", re.compile(r"(?i)(?<![A-Za-z0-9])tokenizer\.json(?![A-Za-z0-9])")),
    ("tokenizer_model", re.compile(r"(?i)(?<![A-Za-z0-9])tokenizer\.model(?![A-Za-z0-9])")),
    ("sentencepiece", re.compile(r"(?i)(?<![A-Za-z0-9])sentencepiece(?![A-Za-z0-9])")),
    ("tiktoken", re.compile(r"(?i)(?<![A-Za-z0-9])tiktoken(?![A-Za-z0-9])")),
)


def tokenizer_llm_context_components(
    text: str,
    source: str = "",
) -> tuple[str, ...]:
    """Return tokenizer identity plus objective local-LLM anchors.

    The word ``tokenizer`` alone is deliberately insufficient: Java's
    ``StringTokenizer`` and parser/lexer tokenizers are common in ordinary
    applications.  A concrete tokenizer artifact/API and a local model or
    runtime anchor must both be present.
    """
    normalized = text.replace("\x00", " ")
    identity_hits = tuple(
        name for name, pattern in _TOKENIZER_IDENTITY_PATTERNS
        if pattern.search(normalized) or pattern.search(source)
    )
    if not identity_hits:
        return ()

    model_ids = tuple(
        model_id for model_id, _ in extract_contextual_llm_model_evidence(
            normalized
        )
    )
    lower = normalized.lower()
    runtime_hits = tuple(
        marker for marker in (
            "gguf", "ggml", "litertlm", "llama.cpp", "mlc-chat-config",
            "mlc_llm", "mnn_llm", "mediapipe genai", "executorch",
        )
        if marker in lower
    )
    if not model_ids and not runtime_hits:
        return ()
    return tuple(dict.fromkeys([
        *(f"tokenizer_identity={name}" for name in identity_hits),
        *(f"model_id={model_id}" for model_id in model_ids),
        *(f"llm_runtime={marker}" for marker in runtime_hits),
    ]))


_RESOURCE_MODEL_KEY_RE = re.compile(
    r"(?i)(?:^|[_-])(?:model|model[_-]?id|model[_-]?name|deployment|engine)"
    r"(?:$|[_-])"
)
_RESOURCE_MODEL_KEY_NOISE_RE = re.compile(
    r"(?i)(?:^|[_-])(?:about|description|marketing|message|promo|subtitle|title)"
    r"(?:$|[_-])"
)


def extract_resource_model_selector_ids(key: str, value: str) -> list[str]:
    """Return model IDs from explicit resource configuration/selector pairs.

    Android string resources frequently encode a selector as either
    ``model_option="gpt-4.1-mini"`` or an identity-preserving alias such as
    ``gpt41mini="gpt-4.1-mini"``.  Requiring one of those key relationships
    keeps unrelated marketing copy and translated product descriptions out.
    """
    candidates = extract_llm_model_ids(value)
    if not candidates:
        return []
    key_lower = key.strip().lower()
    key_identity = re.sub(r"[^a-z0-9]", "", key_lower)
    explicit_selector = (
        bool(_RESOURCE_MODEL_KEY_RE.search(key_lower))
        and not _RESOURCE_MODEL_KEY_NOISE_RE.search(key_lower)
    )
    found: list[str] = []
    for candidate in candidates:
        candidate_identity = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if explicit_selector or (
            key_identity
            and len(key_identity) >= 4
            and key_identity == candidate_identity
        ):
            if candidate not in found:
                found.append(candidate)
    return found


# ═══════════════════════════════════════════════════════════════════
# Special tokens
# ═══════════════════════════════════════════════════════════════════

SPECIAL_TOKEN_HINTS: frozenset[str] = frozenset({
    "<|im_start|>", "<|im_end|>", "<|eot_id|>",
    "<think>", "</think>",
    "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
})


PROMPT_TEMPLATE_PATTERNS = (
    re.compile(r'\byou are (?:a|an) (?:(?:helpful|AI|intelligent|conversational|language model)[ -]+){0,3}assistant\b', re.I),
    re.compile(r'\brespond in json format\b', re.I),
    re.compile(r"\blet's think step by step\b", re.I),
    re.compile(r'你是(?:一个|一個|一位)?(?:有帮助的|有幫助的|智能|人工智能|AI|语言模型|語言模型)助手', re.I),
)


def prompt_template_markers(text: str) -> Iterator[str]:
    """Retain only the matched instruction phrase, never the full prompt."""
    for pattern in PROMPT_TEMPLATE_PATTERNS:
        match = pattern.search(text)
        if match:
            yield match[0].lower()


_AICORE_PERMISSION = re.compile(r'(?<![\w.])(?:[A-Za-z_]\w*\.)+permission\.BIND_AICORE_SERVICE(?![\w.])')


def scan_aicore_permissions(text: str, source: str, layer: str, *, manifest=False) -> Iterator[RawSignal]:
    """Distinguish an actual uses-permission declaration from a code reference."""
    if manifest:
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return
        values = {node.get('{http://schemas.android.com/apk/res/android}name', '')
                  for node in root.iter()
                  if node.tag in {'uses-permission', 'uses-permission-sdk-23'}}
        values = {value for value in values if _AICORE_PERMISSION.fullmatch(value)}
    else:
        if is_reference_only_source(source):
            return
        values = set(_AICORE_PERMISSION.findall(text))
    for value in sorted(values):
        yield RawSignal(SignalType.ANDROID_PERMISSION, value, layer, source,
                        role=SignalRole.METADATA if manifest else SignalRole.REFERENCE,
                        scope=SignalScope.VALUE,
                        components=('permission_evidence=manifest_declaration' if manifest
                                    else 'permission_evidence=literal_reference',))


# ═══════════════════════════════════════════════════════════════════
# WebView LLM target domains
# ═══════════════════════════════════════════════════════════════════

WEBVIEW_TARGET_HINTS: frozenset[str] = frozenset(
    host for entry in WEB_SERVICE_REGISTRY for host in entry.hosts
)


# ═══════════════════════════════════════════════════════════════════
# Backend proxy / gateway keywords
# ═══════════════════════════════════════════════════════════════════

BACKEND_PROXY_KEYWORDS: tuple[str, ...] = (
    "api_base_url", "prompts_base_url", "survey_base_url",
    "worker", "gateway", "proxy",
)


# ═══════════════════════════════════════════════════════════════════
# Play Asset Delivery / split-install hints
# ═══════════════════════════════════════════════════════════════════

PAD_HINTS: tuple[str, ...] = (
    "play asset delivery", "assetpack", "asset pack", "assetpacks",
    "fast-follow", "on-demand", "install-time",
    "splitinstall", "split install", "dynamic feature",
)


# ═══════════════════════════════════════════════════════════════════
# Dynamic model download hints
# ═══════════════════════════════════════════════════════════════════

DYNAMIC_DOWNLOAD_HINTS: tuple[str, ...] = (
    "huggingface", "hf hub", "snapshot_download", "from_pretrained",
    "cached_file", "model repo", "weights",
    "downloadmodel", "downloadafter", "download_url", "model download",
)


# ═══════════════════════════════════════════════════════════════════
# Native JNI / engine function names
# ═══════════════════════════════════════════════════════════════════

# Keep ambiguous operation names out of NATIVE_HINTS: these names identify
# LiteRT-LM only when bound to its exact JNI class, not an arbitrary SDK.
_LITERT_LM_JNI_OWNER = "Lcom/google/ai/edge/litertlm/LiteRtLmJni;"
LITERT_LM_NATIVE_METHODS: tuple[str, ...] = (
    "nativeCreateEngine", "nativeGenerateContent", "nativeGenerateContentStream",
    "nativeSendMessage", "nativeSendMessageAsync",
)
_LITERT_LM_METHOD_NAMES = frozenset(name.lower() for name in LITERT_LM_NATIVE_METHODS)


def is_litert_lm_native_library(name: str) -> bool:
    """Recognize the LM-specific library, not generic LiteRT/TFLite/QNN."""
    return bool(re.fullmatch(
        r"lib/[^/]+/liblitertlm(?:_jni)?\.so", name.replace("\\", "/"), re.IGNORECASE))


def is_litert_lm_native_target(target: str) -> bool:
    """Require the exact bytecode owner and one supported JNI operation."""
    owner, separator, signature = target.partition("->")
    name, paren, _ = signature.partition("(")
    return bool(separator and paren
                and owner.lower() == _LITERT_LM_JNI_OWNER.lower()
                and name.lower() in _LITERT_LM_METHOD_NAMES)


def litert_lm_native_symbols(text: str) -> tuple[str, ...]:
    """Find JNI declarations/calls with explicit LiteRT-LM type identity.

    This is a bounded syntax observation, not Java receiver/data-flow
    resolution. Unresolved aliases and bare operation names remain unknown.
    Quoted examples and comments cannot establish a native operation.
    """
    if "litertlm" not in text.lower():
        return ()
    clean = without_comments(text)
    clean = re.sub(r"(?m)^\s*#[^\n]*", "", clean)
    # Remove literals before matching source syntax, retaining newlines so
    # anchored package/class/method declarations still have their boundaries.
    clean = re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                   lambda match: re.sub(r"[^\n]", " ", match[0]), clean)
    hits: set[str] = set()
    for match in re.finditer(
            r"(?m)^\s*invoke-[\w/-]+\s+\{[^}]*\},\s*(L[^;\s]+;->[^\s]+)", clean):
        if is_litert_lm_native_target(match[1]):
            hits.add(match[1].split("->", 1)[1].split("(", 1)[0].lower())
    owner = re.search(r"(?m)^\s*\.class\b[^\n]*\s+(L[^;\s]+;)\s*$", clean)
    if owner and owner[1].lower() == _LITERT_LM_JNI_OWNER.lower():
        for declaration in re.finditer(
                r"(?m)^\s*\.method\b[^\n]*\bnative\s+[^\n]*?([\w$]+)\(", clean):
            if declaration[1].lower() in _LITERT_LM_METHOD_NAMES:
                hits.add(declaration[1].lower())
    java_package = re.search(
        r"(?m)^\s*package\s+com\.google\.ai\.edge\.litertlm\s*;", clean)
    if java_package and re.search(r"\bclass\s+LiteRtLmJni\b", clean):
        for declaration in re.finditer(r"\bnative\s+[^;{}()]*?\b([\w$]+)\s*\(", clean):
            if declaration[1].lower() in _LITERT_LM_METHOD_NAMES:
                hits.add(declaration[1].lower())
    imported = bool(java_package or re.search(
        r"(?m)^\s*import\s+com\.google\.ai\.edge\.litertlm\.(?:LiteRtLmJni|\*)\s*;",
        clean))
    receivers = [r"com\.google\.ai\.edge\.litertlm\.LiteRtLmJni"]
    if imported:
        receivers.append(r"LiteRtLmJni")
    for match in re.finditer(
            rf"(?<![\w$.])(?:{'|'.join(receivers)})(?:\.INSTANCE)?\.([\w$]+)\s*\(", clean):
        if match[1].lower() in _LITERT_LM_METHOD_NAMES:
            hits.add(match[1].lower())
    return tuple(sorted(hits))


NATIVE_HINTS: tuple[str, ...] = (
    "nativeloadmodel", "nativegeneratetext", "nativetokenize", "nativedetokenize",
    # Llama.cpp
    "llama_backend_init", "llama_decode", "llama_kv_cache", "llama_model_quantize",
    # MLC-LLM
    "mlc_chat_create", "mlc_llm_chat", "tvm_runtime",
    # ExecuTorch
    "executorch_forward",
    # MNN-LLM (Alibaba)
    "mnn_llm", "mnn_chat",
    # Generic LLM ops
    "apply_chat_template", "sampler_sample", "greedy_search", "top_p_sampling",
    # Whisper.cpp (speech-to-text LLM)
    "whisper_init", "whisper_full", "whisper_full_parallel",
    "whisper_full_n_segments", "whisper_full_lang_id",
)


# ═══════════════════════════════════════════════════════════════════
# Model file extensions
# ═══════════════════════════════════════════════════════════════════

# Only formats whose identity is strongly tied to local generative-language
# inference enter MODEL_FILE_EXTENSION without additional context. Generic
# tensor containers (SafeTensors/ONNX/TFLite/PTE/Task/MNN) stay in the
# context-graded research sidecar.
MODEL_FILE_EXTENSIONS: tuple[str, ...] = (
    ".gguf", ".ggml", ".litertlm", "mlc-chat-config.json",
)

# Tensor/model containers are structured binary formats.  Treating their raw
# bytes as latin-1 text creates accidental tokens such as ``gguf`` and turns
# ordinary OCR/CV weights into LLM evidence.  Strong LLM-specific formats are
# still detected by filename above; generic formats remain research evidence.
OPAQUE_MODEL_BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".ckpt", ".fb", ".ggml", ".gguf", ".litertlm", ".mnn", ".model", ".onnx",
    ".ort", ".pb", ".pt", ".pte", ".pth", ".safetensors", ".task",
    ".tflite", ".weights",
})

_KNOWN_NON_LLM_MODEL_PATH_MARKERS: tuple[str, ...] = (
    "mlkit-google-ocr-models", "/gocr/", "line_recognition_legacy",
    "text-recognition-models", "face-detection-models",
    "object-detection-models", "pose-detection-models",
    "barcode-scanning-models", "image-segmentation-models",
)


def is_opaque_model_binary(source: str) -> bool:
    """Return whether model semantics must not be inferred from raw bytes."""
    normalized = source.replace("\\", "/").lower()
    return Path(normalized).suffix in OPAQUE_MODEL_BINARY_EXTENSIONS


def is_known_non_llm_model_artifact(source: str) -> bool:
    """Return whether a path explicitly identifies an OCR/CV ML artifact."""
    normalized = source.replace("\\", "/").lower()
    return any(marker in normalized
               for marker in _KNOWN_NON_LLM_MODEL_PATH_MARKERS)


# ═══════════════════════════════════════════════════════════════════
# .so library name substrings indicating local LLM
# ═══════════════════════════════════════════════════════════════════

STRONG_LOCAL_MODEL_HINTS: tuple[str, ...] = (
    "llmodel", "gptj", "llama", "mistral", "gemma", "whisper",
    "executorch", "mlc", "mnn", "qwen", "ggml",
)


# ═══════════════════════════════════════════════════════════════════
# Text-based local model hints
# ═══════════════════════════════════════════════════════════════════

LOCAL_MODEL_TEXT_HINTS: tuple[str, ...] = (
    "gguf", "ggml", "safetensors", "mlc-chat-config",
    "tokenizer", "sentencepiece", "vocab", "merges",
)
_STRONG_MODEL_TEXT_HINTS: tuple[str, ...] = ("gguf", "ggml", "mlc-chat-config", "sentencepiece")


# ═══════════════════════════════════════════════════════════════════
# License notice keywords
# ═══════════════════════════════════════════════════════════════════

LICENSE_HINTS: tuple[str, ...] = (
    "llama.cpp", "langchain4j", "huggingface tokenizers",
    "mediapipe genai", "vertex ai",
)


# ═══════════════════════════════════════════════════════════════════
# Hardcoded API key patterns
# ═══════════════════════════════════════════════════════════════════

_HARDCODED_API_KEY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # OpenAI project key: sk-proj-<alnum>-
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),
    # Anthropic: sk-ant-<alnum>-
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    # Google / Gemini
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    # HuggingFace
    re.compile(r"hf_[a-z0-9]{20,}"),
    # Groq
    # Provider-issued Groq keys use the stable ``gsk_`` prefix.  Keep a
    # generous minimum length and token boundaries so ordinary identifiers
    # such as ``gsk_config`` are not treated as credentials.
    re.compile(r"(?<![A-Za-z0-9])gsk_[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    # xAI / Grok
    re.compile(r"(?<![A-Za-z0-9])xai-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    # Perplexity
    re.compile(r"(?<![A-Za-z0-9])pplx-[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    # Replicate
    re.compile(r"(?<![A-Za-z0-9_])r8_[A-Za-z0-9_-]{20,}(?![A-Za-z0-9_-])"),
    # Generic sk- — at most 2 hyphens after ``sk-``, final segment
    # ≥ 4 chars, rejects Flutter locale codes (``sk-SK``, ``sk-y``).
    # Post-filter in scan_for_api_keys rejects short / all-alpha
    # matches (CSS classes, binary noise).
    re.compile(r"\bsk-(?:[A-Za-z0-9]+-){0,2}[A-Za-z0-9]{4,}(?![A-Za-z0-9-])"),
)

_CODE_TOKEN_OR_COMMENT_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`(?:\\.|[^`\\])*`|'
    r'(?<![:\\])//[^\r\n]*|/\*[\s\S]*?\*/|<!--[\s\S]*?-->',
)
_GOOGLE_LLM_KEY_CONTEXT_RE = re.compile(
    r"(?i)(?:\bgemini(?:\b|[_.$-])|generativelanguage\.googleapis\.com|"
    r"google\.ai\.client\.generativeai|firebasevertexai|"
    r"\bvertex\s*ai\b|\bgenerative\s*model\b|\bgenai[_.$-])",
)


def _code_without_comments(text: str) -> str:
    """Mask comments while preserving string literals and character offsets."""
    return _CODE_TOKEN_OR_COMMENT_RE.sub(
        lambda match: (
            " " * len(match[0])
            if match[0].startswith(("//", "/*", "<!--"))
            else match[0]
        ),
        text,
    )


def _google_key_has_llm_context(text: str, source: str, start: int, end: int) -> bool:
    """Disambiguate the Google-wide AIza prefix using bounded local context."""
    left = max(0, start - 1024)
    right = min(len(text), end + 1024)
    window = _code_without_comments(text[left:right])
    local_start = start - left
    # A credential written only in a comment is not an embedded app value.
    if not window[local_start:local_start + (end - start)].strip():
        return False
    return bool(_GOOGLE_LLM_KEY_CONTEXT_RE.search(f"{source}\n{window}"))


# ═══════════════════════════════════════════════════════════════════
# BaseURL override patterns (cloud API signal)
# ═══════════════════════════════════════════════════════════════════

_BASEURL_OVERRIDE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"setBaseUrl\s*\(", re.IGNORECASE),
    re.compile(r"setApiBase\s*\(", re.IGNORECASE),
    re.compile(r"base_url\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE),
    re.compile(r"baseURL\s*[:=]\s*[\"'][^\"']*[\"']", re.IGNORECASE),
    re.compile(r"api_base_url\s*[:=]", re.IGNORECASE),
    re.compile(r"setCustomEndpoint\s*\(", re.IGNORECASE),
)


# ═══════════════════════════════════════════════════════════════════
# Serverless / cloud-function proxy domains (cloud API signal)
# ═══════════════════════════════════════════════════════════════════

_SERVERLESS_DOMAINS: tuple[str, ...] = (
    "cloudfunctions.net",
    "vercel.app",
    "workers.dev",
    "fly.dev",
    "railway.app",
    "supabase.co",
    "netlify.app",
    "herokuapp.com",
    "onrender.com",
)


# ═══════════════════════════════════════════════════════════════════
# HTTP client API patterns (used by callgraph analyser)
# ═══════════════════════════════════════════════════════════════════

_HTTP_CLIENT_INDICATORS: tuple[str, ...] = (
    "okhttpclient",
    "okhttp3",
    "httpurlconnection",
    "retrofit",
    "ktorclient",
    "httpclient",
    "openconnection(",
    ".execute(",
    ".enqueue(",
    "request.builder",
)

_WEBVIEW_API_INDICATORS: tuple[str, ...] = (
    "loadurl(",
    "loadurl ",
    "webview.loadurl",
    "evaluatejavascript(",
    "addjavascriptinterface(",
)

_JNI_API_INDICATORS: tuple[str, ...] = (
    "system.loadlibrary(",
    "system.load(",
    "native ",
)


# ═══════════════════════════════════════════════════════════════════
# URL extraction patterns
# ═══════════════════════════════════════════════════════════════════

_URL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE),
    re.compile(r"(?:api\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s'\"<>]*)?", re.IGNORECASE),
)


# ═══════════════════════════════════════════════════════════════════
# URL noise filters
# ═══════════════════════════════════════════════════════════════════

_CODE_SYNTAX_RE: re.Pattern[str] = re.compile(r"[{}()\[\];|]")

_SOURCE_FILE_PATH_RE: re.Pattern[str] = re.compile(
    r"/[\w.-]+\.(?:c|cpp|cc|cxx|h|hpp|rs|java|kt|py|js|ts|go|swift"
    r"|o|obj|a|so|dylib|dll|wasm)" + r"(\x00.*|$)",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════
# Known LLM API domain fragments (used in blob scanning)
# ═══════════════════════════════════════════════════════════════════

_LLM_API_DOMAINS: tuple[str, ...] = tuple(sorted({
    host
    for entry in API_SERVICE_REGISTRY
    if entry.classification_signal and not entry.required_path_markers
    for host in (*entry.exact_hosts, *entry.host_suffixes)
}))

_BINARY_REGISTERED_HOST_MARKERS: tuple[bytes, ...] = tuple(sorted({
    host.encode("ascii")
    for entry in API_SERVICE_REGISTRY
    if entry.classification_signal
    for host in (*entry.exact_hosts, *entry.host_suffixes)
}, key=len, reverse=True))

_BINARY_URL_TAIL_LIMIT = 1024
_BINARY_URL_STOP_BYTES = frozenset(b"\"'<>`,;()[]{}|\\")
_BINARY_HOST_BYTES = frozenset(b"abcdefghijklmnopqrstuvwxyz0123456789.-")


def _has_bounded_http_scheme(data: bytes, host_start: int) -> bool:
    """Return whether *host_start* follows an independent HTTP(S) scheme."""
    for scheme in (b"https://", b"http://"):
        scheme_start = host_start - len(scheme)
        if scheme_start < 0 or data[scheme_start:host_start] != scheme:
            continue
        if scheme_start == 0 or not chr(data[scheme_start - 1]).isalnum():
            return True
    return False


def _host_matches_entry(host: str, entry: ApiServiceEntry) -> bool:
    normalized = host.lower().strip(".")
    if normalized in entry.exact_hosts:
        return True
    return any(
        normalized == suffix or normalized.endswith("." + suffix)
        for suffix in entry.host_suffixes
    )


def match_api_service(url_or_host: str, classification_only: bool = False) -> ApiServiceEntry | None:
    """Return the registry entry for a real host/path match.

    Host boundaries are parsed rather than checked as arbitrary substrings, so
    values such as ``api.openai.com.attacker.example`` never match OpenAI.
    Contextual cloud/aggregator entries additionally require their documented
    path markers.
    """
    candidate = url_or_host.strip().strip("\"'")
    parsed_value = candidate if re.match(r"^[a-z][a-z0-9+.-]*://", candidate, re.I) \
        else "https://" + candidate
    try:
        parsed = urlsplit(parsed_value)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    path_context = (parsed.path + "?" + parsed.query).lower()
    # A suffix literal such as '.openai.azure.com' is not a hostname.
    # Do not normalize away malformed leading dots or empty labels.
    if not host or any(not re.fullmatch(
        r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label
    ) for label in host.split(".")):
        return None
    for entry in API_SERVICE_REGISTRY:
        if classification_only and not entry.classification_signal:
            continue
        if not _host_matches_entry(host, entry):
            continue
        if entry.required_host_markers and not all(
            marker in host for marker in entry.required_host_markers
        ):
            continue
        if entry.required_path_markers and not any(
            marker in path_context for marker in entry.required_path_markers
        ):
            continue
        return entry
    return None


def scan_binary_for_known_api_domains(
    data: bytes, source: str, layer: str = "zip"
) -> Iterator[RawSignal]:
    """Find registered LLM API hosts in binary code/config containers.

    Unlike the general URL extractor, this scanner searches only strict hosts
    from ``API_SERVICE_REGISTRY``.  A small bounded tail preserves the path
    requirements of contextual services without parsing an entire concatenated
    Unity/DEX string table as one URL.
    """
    seen: set[tuple[str, str]] = set()
    lower_data = data.lower()
    host_ranges: set[tuple[int, int]] = set()
    for marker in _BINARY_REGISTERED_HOST_MARKERS:
        offset = 0
        while True:
            marker_start = lower_data.find(marker, offset)
            if marker_start < 0:
                break
            marker_end = marker_start + len(marker)
            offset = marker_end
            host_start = marker_start
            while (host_start > 0
                   and lower_data[host_start - 1] in _BINARY_HOST_BYTES):
                host_start -= 1
            host_end = marker_end
            while (host_end < len(lower_data)
                   and lower_data[host_end] in _BINARY_HOST_BYTES):
                host_end += 1
            host_ranges.add((host_start, host_end))

    for host_start, host_end in sorted(host_ranges):
        host = lower_data[host_start:host_end].decode("ascii")
        tail = data[host_end:host_end + _BINARY_URL_TAIL_LIMIT]
        stop = len(tail)
        lower_tail = tail.lower()
        for scheme in (b"http://", b"https://"):
            pos = lower_tail.find(scheme)
            if pos >= 0:
                stop = min(stop, pos)
        for pos, byte in enumerate(tail[:stop]):
            if byte <= 0x20 or byte in _BINARY_URL_STOP_BYTES:
                stop = pos
                break
        tail = tail[:stop]
        if tail and not tail.startswith((b"/", b"?", b"#", b":")):
            tail = b""

        path_context = tail.decode("ascii", errors="ignore")
        candidate = f"https://{host}{path_context}"
        service = match_api_service(candidate, classification_only=True)
        if service is None:
            continue
        key = (service.service_id, host)
        if key in seen:
            continue
        seen.add(key)
        # A registered host can be recovered safely without trusting a broken
        # string-pool scheme.  Preserve its path only when the scheme itself
        # has a real left boundary; otherwise the tail may belong to the next
        # Hermes/DEX string and only the host is an objective observation.
        signal_value = (
            candidate
            if _has_bounded_http_scheme(lower_data, host_start)
            else f"https://{host}"
        )
        yield RawSignal(
            SignalType.KNOWN_LLM_API_DOMAIN,
            signal_value,
            layer,
            source,
            detail="binary_registered_host_scan",
        )


def match_web_service(url_or_host: str) -> WebServiceEntry | None:
    """Return a Web LLM service only for an exact host or real subdomain."""
    candidate = url_or_host.strip().strip("\"'")
    parsed_value = candidate if re.match(r"^[a-z][a-z0-9+.-]*://", candidate, re.I) \
        else "https://" + candidate
    try:
        parsed = urlsplit(parsed_value)
        host = (parsed.hostname or "").lower().strip(".")
    except ValueError:
        return None
    for entry in WEB_SERVICE_REGISTRY:
        if any(host == marker or host.endswith("." + marker)
               for marker in entry.hosts):
            if entry.path_prefixes and not any(parsed.path == p or parsed.path.startswith(p + "/") for p in entry.path_prefixes):
                continue
            return entry
    return None


def text_contains_known_api(text: str, classification_only: bool = True) -> bool:
    """Recognize a registered API URL/domain in source or binary text."""
    for match in _URL_PATTERNS:
        for candidate in match.finditer(text):
            if match_api_service(candidate.group(0), classification_only):
                return True
    # Smali strings sometimes split URL punctuation, but unconditional hosts
    # remain safe when guarded by hostname token boundaries.
    lower = text.lower()
    for domain in _LLM_API_DOMAINS:
        if re.search(rf"(?<![a-z0-9.-]){re.escape(domain)}(?![a-z0-9.-])", lower):
            return True
    return False


_STRONG_LLM_CONTEXT_MARKERS: tuple[str, ...] = (
    "chat/completions",
    "chat/completion",
    "/v1/responses",
    "responses.create",
    "completions.create",
    "generatecontent",
    "generative_ai",
    "generativeai",
    "openai_api_base",
    "mlc-chat-config",
)

_STRONG_LLM_IDENTITY_RE: re.Pattern[str] = re.compile(
    r"(?<![a-z0-9])(?:openai|anthropic|claude|gemini|chatgpt|llama|llm|aigc)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)

_CONFIGURED_GPT_RE: re.Pattern[str] = re.compile(
    r"(?:"
    r"(?<![a-z0-9])gpt[-_. ]?(?:[345](?:[.-]\d+)?|4o|oss)(?![a-z0-9])"
    r"|(?<![a-z0-9])gpt(?=[_-](?:model|provider|api|base|endpoint|url)(?![a-z0-9]))"
    r")",
    re.IGNORECASE,
)


def has_strong_llm_context(text: str) -> bool:
    """Return whether *text* contains reproducible LLM-specific context.

    Generic words such as ``completion`` and ``tokenize`` are intentionally
    excluded.  They occur in protocol documentation and ordinary libraries
    without describing an LLM request.
    """
    lower = text.lower()
    return (
        any(marker in lower for marker in _STRONG_LLM_CONTEXT_MARKERS)
        or bool(_STRONG_LLM_IDENTITY_RE.search(text))
        or bool(_CONFIGURED_GPT_RE.search(text))
        or bool(extract_llm_schema_components(text))
        or text_contains_known_api(text, classification_only=True)
    )


# ═══════════════════════════════════════════════════════════════════
# Backend marker patterns (word-bounded)
# ═══════════════════════════════════════════════════════════════════

_BACKEND_MARKER_PATS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(rf"(?<![a-zA-Z]){re.escape(m)}(?![a-zA-Z])", re.IGNORECASE)
    for m in ("worker", "gateway", "proxy", "backend", "relay")
)




# ═══════════════════════════════════════════════════════════════════
# Shared utility functions
# ═══════════════════════════════════════════════════════════════════

def decode_bytes(data: bytes) -> str:
    """Decode *data* as text, or return '' for binary blobs."""
    if not data:
        return ""
    if len(data) > 2 * 1024 * 1024:
        return ""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ("utf-16",)
    elif data.startswith(b"\xef\xbb\xbf"):
        encodings = ("utf-8-sig",)
    else:
        # BOM-less UTF-16 is admitted only with an alternating NUL pattern.
        sample = data[:4096]
        odd = sample[1::2]
        even = sample[::2]
        encodings = ("utf-8",)
        if odd and even:
            if odd.count(0) / len(odd) > .6 and even.count(0) / len(even) < .2:
                encodings = ("utf-16-le",)
            elif even.count(0) / len(even) > .6 and odd.count(0) / len(odd) < .2:
                encodings = ("utf-16-be",)
    for enc in encodings:
        try:
            text = data.decode(enc, errors="strict")
        except UnicodeError:
            continue
        dec_printable = sum(ch.isprintable() or ch in ('\t', '\n', '\r') for ch in text)
        if dec_printable / max(len(text), 1) < .95 or "\x00" in text:
            continue
        return text
    return ""


def _clean_extracted_url(candidate: str) -> str:
    """Remove surrounding prose/markup without damaging IPv6 hosts."""
    value = candidate.strip().strip('"').strip("'")

    # A broad URL regex can consume the boundary between a URL used as
    # Markdown text and the following Markdown target: ``url](url)``.
    markdown_boundary = value.find("](")
    if markdown_boundary >= 0:
        value = value[:markdown_boundary]

    value = value.rstrip(".,;:)}")
    # ``]`` is punctuation after an ordinary URL, but it is structural in an
    # IPv6 literal such as ``http://[::1]``.  Remove only unmatched closers.
    while value.endswith("]") and value.count("]") > value.count("["):
        value = value[:-1].rstrip(".,;:)}")
    return value


def extract_urls(text: str) -> list[str]:
    """Extract unique URL-like strings from *text* (max 200)."""
    seen: dict[str, None] = {}
    explicit_spans: list[tuple[int, int]] = []
    _MAX = 200
    for pattern_index, pat in enumerate(_URL_PATTERNS):
        for m in pat.finditer(text):
            if pattern_index > 0 and any(
                m.start() < end and m.end() > start
                for start, end in explicit_spans
            ):
                continue
            c = _clean_extracted_url(m.group(0))
            if c:
                seen.setdefault(c, None)
                if pattern_index == 0:
                    explicit_spans.append((m.start(), m.end()))
            if len(seen) >= _MAX:
                return list(seen)
    return list(seen)


def classify_url_signal(source: str, url: str, layer: str) -> Iterator[RawSignal]:
    """Yield zero or more neutral signals for a candidate URL."""
    lower = url.lower()

    if _CODE_SYNTAX_RE.search(url):
        return
    if _SOURCE_FILE_PATH_RE.search(url):
        return

    if match_api_service(url, classification_only=True):
        yield RawSignal(SignalType.KNOWN_LLM_API_DOMAIN, url, layer, source)
        return

    if match_web_service(url):
        yield RawSignal(SignalType.KNOWN_LLM_WEB_DOMAIN, url, layer, source)
        return

    if not ("/" in url or url.startswith("http")):
        return
    if any(p.search(lower) for p in _BACKEND_MARKER_PATS):
        yield RawSignal(SignalType.BACKEND_PROXY_KEYWORD, url, layer, source)


def scan_for_api_keys(text: str, source: str, layer: str) -> Iterator[RawSignal]:
    """Scan *text* for hardcoded API key patterns."""
    for pat in _HARDCODED_API_KEY_PATTERNS:
        for m in pat.finditer(text):
            matched = m.group(0)
            if matched.lower().startswith("aiza") and not _google_key_has_llm_context(
                text, source, m.start(), m.end()
            ):
                continue
            # ── Post-filter the generic sk-* pattern ──────────────
            # Provider-specific prefixes (including gsk_, xai-, pplx- and
            # r8_) pass through as-is; the patterns already enforce length
            # and token boundaries.
            # The generic sk-* pattern fires on CSS classes like
            # "sk-overlay" and binary-noise fragments like "sk-6Otm"
            # in latin-1-decoded PNG/TTF files.  Real LLM API keys
            # are ≥ 20 chars and contain at least one digit.
            if matched.lower().startswith("sk-") and not (
                matched.lower().startswith(("sk-proj-", "sk-ant-"))):
                if len(matched) < 20:
                    continue
                if not any(ch.isdigit() for ch in matched):
                    continue
            lower_matched = matched.lower()
            prefix = next((candidate for candidate in (
                "sk-proj-", "sk-ant-", "gsk_", "xai-", "pplx-",
                "r8_", "hf_", "AIza", "sk-",
            ) if lower_matched.startswith(candidate.lower())), "key-")
            yield RawSignal(
                SignalType.HARDCODED_API_KEY,
                f"{prefix}[REDACTED]",
                layer,
                source,
                detail="credential_value_redacted",
                dedupe_key=hashlib.sha256(matched.encode("utf-8")).hexdigest(),
            )


def scan_for_baseurl_override(text: str, source: str, layer: str) -> Iterator[RawSignal]:
    """Scan *text* for BaseURL dynamic-configuration patterns."""
    for pat in _BASEURL_OVERRIDE_PATTERNS:
        m = pat.search(text)
        if m:
            matched = m.group(0)
            quoted = re.search(r'''["']([^"']*)["']''', matched)
            if quoted and (not quoted[1].strip() or quoted[1].strip().lower() in {"null", "undefined"}):
                continue
            # ── Value-level guard: reject JS literals and non-URLs ────
            # mathjaxscript.js contains `baseURL:"undefined"` — a JS
            # literal, not a real base URL.  Similarly, reject `null`,
            # empty strings, and matches that contain neither a dot nor
            # a scheme (e.g. `baseURL = "foo"`).
            _val_lower = matched.lower()
            if ':"undefined"' in _val_lower or ':"null"' in _val_lower:
                return
            if ':"' in _val_lower or ":'" in _val_lower:
                # Extract the quoted value
                _v = re.search(r"""["']([^"']*)["']""", matched)
                if _v:
                    _inner = _v.group(1)
                    if not _inner or ("." not in _inner and "://" not in _inner):
                        return
            yield RawSignal(
                SignalType.BASEURL_OVERRIDE,
                matched,
                layer,
                source,
                role=SignalRole.CONFIG,
                scope=SignalScope.FILE,
                verification=SignalVerification.CANDIDATE,
                source_kind="config_marker",
            )
            return  # one hit per file is enough


def scan_for_serverless_url(
    text: str,
    source: str,
    layer: str,
    context_text: str | None = None,
) -> Iterator[RawSignal]:
    """Scan for serverless domains only in explicit LLM context."""
    # Public suffix lists contain every known domain suffix — a single
    # file can hit all 9 serverless domains.  Skip these files.
    _NOISE_FILES = {
        "publicsuffix", "public_suffix", "suffix-list",
        "publicsuffixdatabase",
    }
    lower_src = source.lower()
    if any(nf in lower_src for nf in _NOISE_FILES):
        return

    if not has_strong_llm_context(context_text or text):
        return

    lower = text.lower()
    for domain in _SERVERLESS_DOMAINS:
        if domain in lower:
            context = context_text or text
            locations = re.finditer(re.escape(domain), context.lower())
            if not any(has_strong_llm_context(context[max(0, loc.start()-512):loc.end()+512]) for loc in locations):
                continue
            yield RawSignal(SignalType.SERVERLESS_URL,
                            domain, layer, source)


# ═══════════════════════════════════════════════════════════════════
# Custom business LLM endpoint detection
# ═══════════════════════════════════════════════════════════════════

# Strong routes are specific to language-model protocols or model identities.
# They can support a relation from one complete custom URL without borrowing
# context from the rest of a file or binary string pool.
_STRONG_LLM_PATH_RE: re.Pattern[str] = re.compile(
    r"(?:^|/)(?:v\d+/)?chat/completions?(?:[/\\?#]|$)|"
    r"(?:^|/)v\d+/(?:responses|completions?)(?:[/\\?#]|$)|"
    r"(?:^|[/.:])generatecontent(?:[/\\?#]|$)|"
    r"(?:^|/)(?:llm|gpt|claude|gemini)(?:[/\\?#_.:-]|$)|"
    r"gpt/sendgpt|sendgpt",
    re.IGNORECASE,
)

# A provider namespace is an observed service reference, not a proven text
# inference request (a provider can also serve images/audio). Keep candidates
# separate from the strong protocol routes above.
_LLM_PROVIDER_PATH_RE = re.compile(r'(?:^|/)open[-_]?ai(?:/|$)', re.I)
_NON_LANGUAGE_SERVICE_PATH_RE = re.compile(
    r'(?:^|/)(?:images?|audio|speech|transcriptions?|embeddings?)(?:/|$)', re.I)
_EXECUTORCH_IDENTITY_RE = re.compile(r'(?<![a-z0-9])(?:lib)?executorch(?![a-z0-9])', re.I)


def framework_keyword_matches(token: str, lowered_text: str) -> bool:
    """Preserve existing framework rules while bounding ambiguous ExecuTorch."""
    return bool(_EXECUTORCH_IDENTITY_RE.search(lowered_text)) if token == 'executorch' else token in lowered_text


def llm_config_label_marker(value: str) -> str | None:
    """Identify explicit text-model labels, including CamelCase product labels."""
    if not value or len(value) > 256 or '://' in value:
        return None
    label = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', value)
    if re.search(r'\b(?:no|not|without|disable[ds]?)\b|不支持|不使用|禁用',label,re.I):
        return None
    # An image-only feature can name an AI provider; it is not a text LLM link.
    if re.search(r'\b(?:images?|photos?|pictures?|videos?|audio|speech|embeddings?)\b', label, re.I):
        return None
    marker = re.search(r'(?<![a-z0-9])(?:chatgpt|gpt|llm)(?![a-z0-9])|large[-_ ]language[-_ ]model', label, re.I)
    return marker[0].lower() if marker else None

# These routes are common in LLM backends but also in ordinary messaging,
# media generation and business APIs. They require a local LLM schema or
# identity and therefore never qualify from an isolated URL alone.
_CONTEXTUAL_LLM_PATH_RE: re.Pattern[str] = re.compile(
    r"(?:^|[/&?=_-])(?:ai|assistant|chat|conversation|generate|generation|"
    r"message|prompt|ask|query|inference|infer)"
    r"(?:$|[/&?=_.-])",
    re.IGNORECASE,
)

_POLICY_OR_LEGAL_URL_RE: re.Pattern[str] = re.compile(
    r"(?:^|[./?&#=_-])(?:privacy|legal|terms?|polic(?:y|ies)|gdpr|ccpa)"
    r"(?:$|[./?&#=_-])",
    re.IGNORECASE,
)

_REFERENCE_URL_RE: re.Pattern[str] = re.compile(
    r"(?:^|[./?&#=_-])(?:docs?|documentation|reference|developer|"
    r"account|api[-_]?keys?|support|help|credits|about|pricing|blog)"
    r"(?:$|[./?&#=_-])",
    re.IGNORECASE,
)

# Public project/help origins are not app-controlled inference endpoints.
# They are kept out of ``business_ai_endpoint`` while their other URL/domain
# observations can still be emitted through the dedicated signal types.
_REFERENCE_ENDPOINT_HOSTS: frozenset[str] = frozenset({
    "openai.com", "www.openai.com", "platform.openai.com",
    "ggml.ai", "www.ggml.ai",
    # Provider consoles, authentication origins and model/project repositories
    # are related resources, but they are not language-model inference APIs.
    "console.anthropic.com", "aistudio.google.com", "auth0.openai.com",
    "huggingface.co", "www.huggingface.co",
    "github.com", "www.github.com", "api.github.com",
    "gitee.com", "www.gitee.com", "gitlab.com", "www.gitlab.com",
    # Stable public documentation/service sites cannot become an inference
    # endpoint merely because an LLM URL occurs elsewhere in the same file.
    "android.com", "www.android.com", "apache.org", "www.apache.org",
    "whatismybrowser.com", "www.whatismybrowser.com",
    "keepandroidopen.org", "www.keepandroidopen.org",
})

# Exact service roles, never substring/product-name heuristics.  These hosts
# expose image search/vision, translation widgets or a UI dependency rather
# than LLM text inference.  Google Cloud Translation is handled separately
# because it can explicitly select Translation LLM.
_NON_LLM_SERVICE_HOSTS: frozenset[str] = frozenset({
    "vision.googleapis.com", "api.pexels.com",
    "ssl.microsofttranslator.com", "www.microsofttranslator.com",
    "swiperjs.com", "www.swiperjs.com",
})
_GOOGLE_TRANSLATION_HOSTS: frozenset[str] = frozenset({
    "translation.googleapis.com",
})


def _endpoint_local_context(url: str, context_text: str, radius: int = 768) -> str:
    """Return comment-free context around this URL, never the whole file."""
    if not context_text:
        return ""
    code = _code_without_comments(context_text)
    candidates = (url, url.replace("/", r"\/"))
    lower = code.lower()
    for candidate in candidates:
        position = lower.find(candidate.lower())
        if position >= 0:
            return code[
                max(0, position - radius):
                min(len(code), position + len(candidate) + radius)
            ]
    return ""


def translation_llm_endpoint_evidence(
    url: str,
    context_text: str = "",
    *,
    service_evidence: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return objective evidence that Cloud Translation selects Translation LLM."""
    try:
        parsed = urlsplit(url if "://" in url else "https://" + url)
    except ValueError:
        return ()
    if (parsed.hostname or "").lower().rstrip(".") not in _GOOGLE_TRANSLATION_HOSTS:
        return ()
    evidence: list[str] = []
    if parsed.path.lower().rstrip("/").endswith(":adaptivemttranslate"):
        evidence.append("translation_llm_adaptive_route")
    try:
        query = parse_qsl(parsed.query, keep_blank_values=False, max_num_fields=64)
    except ValueError:
        query = []
    for key, value in query:
        normalized = value.strip().lower().strip("/")
        if key.lower() == "model" and re.fullmatch(
            r"(?:llm|(?:[^/\s]+/)*translation-llm(?:-adaptive)?)", normalized
        ):
            evidence.append("translation_llm_model_query")
    if any(marker == "translation_llm" or marker.startswith("translation_llm_")
           for marker in service_evidence):
        evidence.append("translation_llm_service_marker")
    local = _endpoint_local_context(url, context_text)
    if local and re.search(
        r'''(?ix)(?<![A-Za-z0-9_$])(?:model|model[_-]?name|model[_-]?id)'''
        r'''["']?\s*(?:[:=]|,\s*)\s*["'](?:llm|(?:[^"'\s]+/)*translation-llm(?:-adaptive)?)["']''',
        local,
    ):
        evidence.append("translation_llm_model_assignment")
    return tuple(dict.fromkeys(evidence))


def endpoint_noise_reason(
    url: str,
    context_text: str = "",
    *,
    service_evidence: tuple[str, ...] = (),
) -> str | None:
    """Classify only deterministic non-inference endpoint roles."""
    try:
        parsed = urlsplit(url if "://" in url else "https://" + url)
    except ValueError:
        return "malformed_endpoint"
    host = (parsed.hostname or "").lower().rstrip(".")
    if host in _NON_LLM_SERVICE_HOSTS:
        return "dedicated_non_llm_service"
    if host in _REFERENCE_ENDPOINT_HOSTS:
        return "non_inference_reference_host"
    if host in _GOOGLE_TRANSLATION_HOSTS and not translation_llm_endpoint_evidence(
        url, context_text, service_evidence=service_evidence
    ):
        return "translation_without_llm_selection"
    if context_text:
        code = _code_without_comments(context_text)
        originals = (url, url.replace("/", r"\/"))
        if (any(item.lower() in context_text.lower() for item in originals)
                and not any(item.lower() in code.lower() for item in originals)):
            return "comment_only_endpoint"
    return None

_IMAGE_GENERATION_CONTEXT_RE: re.Pattern[str] = re.compile(
    r"(?:stable[-_ ]?diffusion|txt2img|img2img|controlnet|cfg_scale|"
    r"image[-_ ]?(?:generation|generator)|generate[-_ ]?image|"
    r"stability\.ai)",
    re.IGNORECASE,
)


def has_language_generation_context(text: str) -> bool:
    """Return whether context specifically describes language generation."""
    lower = text.lower()
    return bool(
        extract_llm_schema_components(text)
        or any(marker in lower for marker in (
            "chat/completions", "chat/completion", "/v1/responses",
            "responses.create", "completions.create",
        ))
        or re.search(
            r"(?<![a-z0-9])(?:chatgpt|claude|gemini|llama|mistral|llm)"
            r"(?![a-z0-9])",
            text,
            re.IGNORECASE,
        )
        or _CONFIGURED_GPT_RE.search(text)
    )


def is_policy_or_legal_url(url: str) -> bool:
    """Return whether a URL identifies policy/legal/reference content."""
    return bool(
        _POLICY_OR_LEGAL_URL_RE.search(url) or _REFERENCE_URL_RE.search(url)
    )


_DOCUMENT_NAMESPACE_PATH_RE = re.compile(r'(?:^|/)(?:pdfa|pdfua)/ns(?:/|$)', re.I)
_HELP_MESSAGE_PREFIX_RE = re.compile(
    r'\b(?:report\s+(?:this|(?:an?|the)\s+(?:issue|bug|problem|error))\s+(?:on|at|to|via)|'
    r'(?:file|open|submit)\s+(?:an?|the)\s+(?:issue|bug\s+report)\s+(?:on|at|via)|'
    r'(?:ask|seek|get)\s+(?:for\s+)?(?:help|support)\s+(?:on|at|via)|'
    r'for\s+(?:help|support)\s*(?::|,|visit\b))[^"\'\r\n]{0,120}$', re.I)


def _is_help_message_url(url: str, context: str) -> bool:
    """Reject only occurrences explicitly introduced as help/issue links.

    Quotes bound the preceding prose. The same address in a separate endpoint
    assignment remains usable; nearby error words cannot taint it.
    """
    if not context:
        return False
    found = False
    start = 0
    while True:
        pos = context.find(url, start)
        if pos < 0:
            return found
        prefix = context[max(0, pos-256):pos].rsplit('\x00', 1)[-1]
        if not _HELP_MESSAGE_PREFIX_RE.search(prefix):
            return False
        found = True
        start = pos + len(url)


def is_reference_endpoint_url(url: str, source: str = "", context_text: str = "") -> bool:
    """Return whether a URL/source pair is reference material, not an API.

    Host filtering is deliberately exact. A custom host such as
    ``openai.example`` must not be excluded merely because it contains a
    provider name.
    """
    if is_endpoint_reference_source(source) or is_policy_or_legal_url(url):
        return True
    try:
        candidate = url if re.match(r"^[a-z][a-z0-9+.-]*://", url, re.I) \
            else "https://" + url
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return False
    return (host in _REFERENCE_ENDPOINT_HOSTS
            or bool(_DOCUMENT_NAMESPACE_PATH_RE.search(parsed.path))
            or _is_help_message_url(url, context_text))


def is_custom_service_url(url: str) -> bool:
    """Return whether a URL can objectively identify an app-controlled API."""
    return _parse_custom_service_url(url) is not None


_ENDPOINT_CONFIG_VALUE_RE = re.compile(
    r'''\b(?:[A-Za-z_$][\w$]*?(?:url|host|endpoint)|url|host|endpoint)'''
    r'''["']?\s*[:=]\s*["']([^"'\r\n]+)["']''', re.IGNORECASE,
)


def _parse_custom_service_url(value: str, context: str = ""):
    """Require URL provenance; a Java qualified name is not a bare host.

    Bare hosts remain usable in explicit URL/host/endpoint assignments. Known
    provider and website identities are emitted by their own scanners.
    """
    value = value.strip()
    explicit = value.lower().startswith(("http://", "https://"))
    if not explicit and not any(
        match[1] == value for match in _ENDPOINT_CONFIG_VALUE_RE.finditer(context)
    ):
        return None
    if is_reference_endpoint_url(value) or re.search(r'''[\s<>"'\\]''', value):
        return None
    try:
        parsed = urlsplit(value if explicit else "https://" + value)
        host = (parsed.hostname or "").lower().rstrip(".")
        parsed.port  # Reject malformed port syntax as well as malformed hosts.
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not host:
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        if len(host) > 253 or any(not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label
        ) for label in host.split(".")):
            return None
    if (host in _BLOB_DOMAIN_NOISE
            or match_api_service(value, classification_only=False)
            or match_web_service(value)):
        return None
    return parsed


# A root URL has no route to classify, so it is accepted only when the
# surrounding value identifies it as an application endpoint/base URL.  This
# recovers configurations such as ``API_URL=https://api.example.ai`` without
# treating a bare product/home-page URL as an inference endpoint.
_ENDPOINT_CONFIG_CONTEXT_RE: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9])(?:api|base|auth|service|server|backend|endpoint|"
    r"origin|host|url)[A-Za-z0-9_$.-]*(?:url|host|endpoint|base|origin|"
    r"server)?\s*(?:=|:)\s*",
    re.IGNORECASE,
)


def _has_endpoint_config_context(context: str) -> bool:
    """Return whether *context* contains an endpoint-like assignment key."""
    return bool(context and _ENDPOINT_CONFIG_CONTEXT_RE.search(context))


def _has_binary_ai_host_identity(host: str) -> bool:
    """Recognize explicit AI service host labels in compiled string pools.

    Binary pools have no field or method ownership.  Only strong host labels
    (or an ``ai-*`` service label) qualify for a root URL; generic ``.ai``
    origins still need source/config context.
    """
    normalized = host.lower().strip(".")
    if normalized.endswith(".firebaseio.com"):
        return False
    raw_labels = normalized.split(".")
    if any(label.startswith("ai-") and len(label) > 3
           for label in raw_labels):
        return True
    labels = re.split(r"[.-]+", normalized)
    return any(
        label in {"gpt", "chat", "chatgpt", "llm", "claude", "gemini",
                  "openai", "anthropic", "deepseek", "qwen", "assistant",
                  "brainshop"}
        or label.startswith("ai") and len(label) > 2
        or label.endswith("gpt")
        for label in labels
    )


def classify_llm_route(path_context: str, context_text: str = "") -> str | None:
    """Classify an endpoint route as ``strong`` or ``contextual``.

    Contextual routes are accepted only when the same bounded program scope
    contains an explicit LLM identity, registered provider, or a distinctive
    language-model request/response schema.
    """
    # Product-prefixed GPT namespaces with explicit text/chat operations.
    # A suffix alone, /sendPrompt alone, or generateImage does not qualify.
    if re.search(
        r"(?:^|/)[a-z0-9_-]*gpt/(?:send[-_]?prompt|send[-_]?message|"
        r"chat|chat[-_]?completion[s]?)(?:[/?#]|$)", path_context, re.IGNORECASE,
    ):
        return "strong"
    if _STRONG_LLM_PATH_RE.search(path_context):
        return "strong"
    if (_CONTEXTUAL_LLM_PATH_RE.search(path_context)
            and context_text
            and has_strong_llm_context(context_text)):
        if (_IMAGE_GENERATION_CONTEXT_RE.search(context_text)
                and not has_language_generation_context(context_text)):
            return None
        return "contextual"
    return None


def _domain_is_explicit_llm(domain: str) -> bool:
    """Return true only for host labels that explicitly name an LLM service."""
    labels = re.split(r"[.-]+", domain.lower().strip("."))
    strong = {"gpt", "chatgpt", "llm", "claude", "gemini", "openai",
              "anthropic", "deepseek", "qwen"}
    return any(label in strong or (len(label) > 3 and label.endswith("gpt"))
               for label in labels)


_TEXT_AI_HOST_LABEL_RE = re.compile(
    r'(?:(?:ai|llm|text|chat)[-_]?(?:completions?|generation|generate|prompts?|assistant)(?:api|service)?|'
    r'ai\d*[-_]?chat)', re.I)
_AUXILIARY_HOST_LABELS = frozenset({
    'auth', 'login', 'accounts', 'invite', 'pics', 'img', 'images',
    'cdn', 'static', 'assets', 'docs', 'support', 'status', 'help',
    'billing', 'payments', 'checkout', 'analytics', 'telemetry', 'ads',
})


_NAMED_ENDPOINT_CONFIG_RE = re.compile(
    r'''\b(?P<key>[A-Za-z_$][\w$]*(?:url|host|endpoint))["']?\s*[:=]\s*'''
    r'''["'](?P<value>[^"'\r\n]+)["']''', re.I)


def llm_endpoint_config_keys(url: str, context: str) -> tuple[str, ...]:
    """Retain only keys that assign this exact related address, not neighbours."""
    if not context:
        return ()
    keys = tuple(m['key'] for m in _NAMED_ENDPOINT_CONFIG_RE.finditer(context)
                 if m['value'].strip() == url.strip() and len(m['key']) <= 128)
    return tuple(key for key in keys if llm_host_candidate_evidence(url, config_keys=(key,)))


def llm_host_candidate_evidence(url: str, paths=(), *, endpoint_config: bool = False,
                                config_keys=()) -> tuple[str, ...]:
    """Neutral relevance hints for a host candidate, without an execution chain.

    A generic .ai suffix/chat host alone is insufficient. Text-model host
    labels, inference routes, or an AI-named configured endpoint can enter
    address features, while remaining candidates with unknown ownership/use.
    This function does not read labels, APK files or unrelated observations.
    """
    parsed = _parse_custom_service_url(url)
    if parsed is None:
        return ()
    host = (parsed.hostname or '').lower().rstrip('.')
    labels = host.split('.')
    if host.endswith('.firebaseio.com') or labels[0] in _AUXILIARY_HOST_LABELS:
        return ()
    route_paths = [path for path in (parsed.path, *paths) if path]
    usable_paths = [path for path in route_paths if not is_reference_endpoint_url(
        url.split('?', 1)[0].rstrip('/') + '/' + path.lstrip('/'))]
    if route_paths and not usable_paths:
        return ()
    reasons = []
    if _domain_is_explicit_llm(host) or any(_TEXT_AI_HOST_LABEL_RE.fullmatch(label) for label in labels):
        reasons.append('llm_host_identity')
    if any(classify_llm_route(path) == 'strong' for path in usable_paths):
        reasons.append('llm_inference_route')
    if any(_LLM_PROVIDER_PATH_RE.search(path)
           and not _NON_LANGUAGE_SERVICE_PATH_RE.search(path) for path in usable_paths):
        reasons.append('llm_provider_path')
    # Assignment provenance is weaker than invocation; never use a bare .ai
    # suffix or broad compiled-pool AI prefixes as a configuration proof.
    if endpoint_config and any(label.startswith('ai') and len(label)>2
                               and not label.startswith(('air', 'aid', 'ail', 'ain', 'ait'))
                               for label in labels):
        reasons.append('ai_endpoint_config')
    for key in config_keys:
        if not re.fullmatch(r'[A-Za-z_$][\w$]{0,127}', key):
            continue
        normalized = re.sub(r'([a-z])([A-Z])', r'\1_\2', key).lower()
        tokens = set(re.split(r'[_$]+', normalized))
        if not tokens.intersection({'url', 'host', 'endpoint'}):
            continue
        if tokens.intersection(_AUXILIARY_HOST_LABELS) or tokens.intersection({'web', 'webview', 'image', 'photo', 'vision'}):
            continue
        if (tokens.intersection({'llm', 'gpt', 'chatgpt', 'genai'})
                or {'text', 'generation'} <= tokens):
            reasons.append('llm_endpoint_config')
        elif (host.endswith('.ai') and tokens.intersection({'lang', 'language'})
              and 'call' in tokens and 'socket' in tokens):
            # Configured language/AI service address, not an observed model
            # request. The same URL must occur in this exact assignment.
            reasons.append('language_ai_endpoint_config')
    return tuple(dict.fromkeys(reasons))


def json_llm_link_feature_eligible(signal: dict) -> bool:
    """An object-local link is a related address, never an inferred API call.

    Check one retained occurrence at a time so merging unrelated observations
    cannot assemble the configuration proof from different sources.
    """
    url = str(signal.get('value') or '')
    if not is_custom_service_url(url) or endpoint_noise_reason(url):
        return False

    def eligible(instance):
        components = instance.get('components') or []
        fields = dict(item.split('=', 1) for item in components
                      if isinstance(item, str) and '=' in item)
        return (instance.get('source_kind') == 'json_llm_link_config'
                and instance.get('role') == SignalRole.CONFIG.value
                and instance.get('scope') == SignalScope.CONFIG.value
                and fields.get('binding') == 'same_json_object'
                and fields.get('candidate_evidence') == 'llm_link_config'
                and fields.get('route_strength') == 'config_identity'
                and fields.get('config_llm_marker') in {
                    'gpt', 'chatgpt', 'llm', 'large language model',
                    'large-language-model', 'large_language_model'}
                and bool(fields.get('config_url_field'))
                and bool(fields.get('config_label_field'))
                and not is_reference_only_source(str(instance.get('source_file') or '')))

    instances = signal.get('evidence_instances')
    return any(eligible(item) for item in instances) if instances else eligible(signal)


_TWA_ENTRYPOINT_CONTEXT_RE: re.Pattern[str] = re.compile(
    r"(?:launchurl|fullscopeurl|webmanifesturl|assetstatements|"
    r"trusted[._ -]?web[._ -]?activity|bubblewrap|manifest\.webmanifest)",
    re.IGNORECASE,
)


def _is_twa_entrypoint(url: str, source: str, context_text: str,
                       path: str) -> bool:
    """Return whether a URL is a Web/TWA page entrypoint, not an API route.

    TWA-generated resources routinely contain a branded ``.ai`` origin in
    ``launchUrl``/``webManifestUrl`` and should not become a business endpoint
    solely because of that hostname.
    """
    normalized_path = (path or "").rstrip("/").lower()
    if normalized_path.endswith("/manifest.webmanifest"):
        return True
    if normalized_path:
        return False
    combined = f"{source}\n{context_text}"
    return bool(_TWA_ENTRYPOINT_CONTEXT_RE.search(combined))


def is_plausible_api_route_literal(value: str) -> bool:
    """Return whether a standalone string has the shape of an API route.

    LLM-related identifiers also occur in enum values, database columns and
    local file paths. A split base-URL relation may use only a bounded,
    whitespace-free path containing a real segment separator; complete URLs
    are handled by ``scan_url_for_business_ai_endpoint`` instead.
    """
    candidate = value.strip().replace(r"\/", "/")
    if not candidate or len(candidate) > 512:
        return False
    if candidate.lower().startswith(("http://", "https://")):
        return False
    if "/" not in candidate or any(char.isspace() for char in candidate):
        return False
    if any(ord(char) < 0x20 for char in candidate):
        return False
    return bool(re.fullmatch(
        r"/?[A-Za-z0-9][A-Za-z0-9._~{}:$@!+,%=-]*"
        r"(?:/[A-Za-z0-9][A-Za-z0-9._~{}:$@!+,%=-]*)*/?"
        r"(?:\?[^\s#]*)?(?:#[^\s]*)?",
        candidate,
    ))


def scan_url_for_business_ai_endpoint(
    url: str,
    source: str,
    layer: str,
    context_text: str = "",
    context_scope: SignalScope = SignalScope.VALUE,
    *, bound_request: bool = False,
) -> Iterator[RawSignal]:
    """Yield *BUSINESS_AI_ENDPOINT* if *url* is a business AI endpoint
    on a custom domain (not a known LLM vendor domain).

    Must be called on individual extracted URLs, not raw text.
    """
    service_evidence = translation_llm_endpoint_evidence(url, context_text)
    if endpoint_noise_reason(
        url, context_text, service_evidence=service_evidence
    ):
        return
    # Callers already provide bounded context, but a decoded source file may
    # still hold several unrelated addresses.  Bind every contextual decision
    # to this URL's comment-free local window.
    context_text = _endpoint_local_context(url, context_text)
    if is_reference_endpoint_url(url, source, context_text):
        return
    parsed = _parse_custom_service_url(url, context_text)
    if parsed is None:
        return
    host = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path or ""
    path_context = path
    if parsed.query:
        path_context += f"?{parsed.query}"
    # A nearby model name is not this URL's request. Contextual routes need
    # an object/argument binding established by the caller's structural flow.
    route_strength = classify_llm_route(path_context, context_text if bound_request else "")
    if service_evidence:
        route_strength = route_strength or "strong"
    config_keys = (() if is_reference_only_source(source)
                   else llm_endpoint_config_keys(url, context_text))
    if not route_strength and config_keys:
        route_strength = 'host_identity'
    provider_path = (not parsed.username and not parsed.password
                     and _LLM_PROVIDER_PATH_RE.search(path)
                     and not _NON_LANGUAGE_SERVICE_PATH_RE.search(path))
    if not route_strength and provider_path:
        route_strength = 'host_identity'
    # Some proxy services expose the LLM identity only in the hostname and
    # use a product-specific route (for example ``/get``, ``/v1/`` or
    # ``/api/prompt``) rather than a canonical ``/chat/completions`` path.
    # Preserve that explicit observation regardless of the route.  This is
    # still bounded to host labels that explicitly identify an LLM service;
    # generic ``.ai``/serverless hosts continue to require a contextual route.
    # Normalize a root path before the TWA check so ``https://host/`` and
    # ``https://host`` follow the same policy.
    normalized_path = path.rstrip("/")
    if _is_twa_entrypoint(url, source, context_text, normalized_path):
        return
    if _domain_is_explicit_llm(host):
        route_strength = route_strength or "strong"
    elif not route_strength and not normalized_path:
        # A root custom origin is not an API route by itself. In a structured
        # app endpoint/base-url assignment, an AI-looking host is still a
        # useful neutral observation even when the route is in another class
        # or only assembled at runtime. Compiled pools have no assignment
        # boundary, so they use the narrower explicit-host fallback.
        # A different endpoint assignment is not this URL's LLM context.
        # Also keep new semantic config keys from bypassing exclusions via
        # the older contextual-root fallback.
        root_context = _NAMED_ENDPOINT_CONFIG_RE.sub(
            lambda m: m[0] if m['value'].strip() == url.strip() else '', context_text)
        if (is_reference_only_source(source)
                or host.split('.')[0] in _AUXILIARY_HOST_LABELS):
            root_context = ''
        if bound_request and root_context and has_strong_llm_context(root_context):
            route_strength = "contextual_config"
        elif context_text and _domain_looks_ai(host) and _has_endpoint_config_context(context_text):
            route_strength = "host_identity"
        elif layer.lower() == "zip" and _has_binary_ai_host_identity(host):
            route_strength = "host_identity"
    if not route_strength and layer.lower() == "zip" and (
            _has_binary_ai_host_identity(host)
            or 'llm_host_identity' in llm_host_candidate_evidence(url)):
        route_strength = "host_identity"
    try:
        ipaddress.ip_address(host)
        is_ip = True
    except ValueError:
        is_ip = False
    if "." not in host and not is_ip and not _has_endpoint_config_context(context_text):
        # Local single-label hosts require actual configuration provenance.
        return
    if host and route_strength:
        # Endpoint queries commonly carry API keys, user IDs or prompts.
        # They are not needed for the neutral endpoint feature. For an
        # explicitly LLM-named host, use one canonical origin so a base URL
        # and its request route do not inflate the endpoint count; the route
        # remains available in the bounded ``llm_path`` component below.
        if _domain_is_explicit_llm(host) or (route_strength == "host_identity" and not provider_path):
            netloc = host
            if parsed.port:
                netloc += f":{parsed.port}"
            signal_url = f"{parsed.scheme.lower()}://{netloc}"
        else:
            signal_url = url.split("?", 1)[0].split("#", 1)[0]
        candidate_evidence = llm_host_candidate_evidence(
            url, endpoint_config=any(match[1] == url for match in _ENDPOINT_CONFIG_VALUE_RE.finditer(context_text)),
            config_keys=config_keys,
        ) if route_strength == 'host_identity' else ()
        yield RawSignal(
            SignalType.BUSINESS_AI_ENDPOINT,
            signal_url,
            layer,
            source,
            role=(SignalRole.REFERENCE if provider_path and route_strength == 'host_identity'
                  else SignalRole.RELATION),
            scope=(SignalScope.VALUE
                   if route_strength == "strong" or provider_path else context_scope),
            relation=(None if provider_path and route_strength == 'host_identity'
                      else SignalRelation.CUSTOM_DOMAIN_LLM_PATH),
            components=(f"custom_domain={host}", f"llm_path={path}",
                        f"route_strength={route_strength}",
                        *(f"service_evidence={item}" for item in service_evidence),
                        *(f'endpoint_config_key={key}' for key in config_keys),
                        *(f'candidate_evidence={reason}' for reason in candidate_evidence)),
            verification=(SignalVerification.CANDIDATE if route_strength == "host_identity"
                          else SignalVerification.VERIFIED),
            source_kind="binary_url" if layer == "zip" and not context_text else "endpoint_value",
        )


# ═══════════════════════════════════════════════════════════════════
# JADX source scanning — LLM path constants on custom domains
# ═══════════════════════════════════════════════════════════════════

# Matches string constants whose value contains an LLM path keyword.
# Examples observed in real APKs:
#   public static final String POST_SIMPLE_MESSAGE = "GPT/SendGPT";
#   String ENDPOINT = "api/chat/completions";
#   const val SEND = "GPT/SaveAudio"
# The constant must include a ``/`` AND an LLM keyword so we don't fire
# on generic strings like "GPT model" or "chat with us".
_JADX_LLM_PATH_RE: re.Pattern[str] = re.compile(
    r'(?:=\s*|@(?:GET|POST|PUT|PATCH|DELETE)\s*\(\s*)"([^"]*('
    r'GPT/|/chat|/completion|/generate|/inference|/assistant|/llm|/responses'
    r'|/conversation|/message|/prompt|/ask|SendGPT|sendMessage'
    r')[^"]*)"',
    re.IGNORECASE,
)


def scan_for_retrofit_ai_paths(
    text: str, source: str, layer: str
) -> Iterator[RawSignal]:
    """Yield *BUSINESS_AI_ENDPOINT* for LLM path constants on custom domains.

    Scans JADX-decompiled Java/Kotlin source for string constants that
    look like API endpoint paths (e.g. ``"GPT/SendGPT"``,
    ``"api/chat/completions"``) AND confirms the same file references a
    custom (non-LLM-vendor) domain URL.  This catches the configuration
    pattern where ``baseUrl`` and endpoint paths are defined as separate
    constants in the same file — a pattern that URL-level scanning alone
    cannot see because the path never appears as a complete URL.

    Only intended for JADX-layer use (scoped to app package).
    """
    if is_endpoint_reference_source(source):
        return

    # ── 1. Check for a custom-domain URL in the same file ─────────
    # Known provider and web-service URLs use their dedicated signal types.
    # Without a custom URL the split endpoint relation cannot be established.
    urls = extract_urls(text)
    custom_domains: set[str] = set()
    for url in urls:
        lower = url.lower()
        if not lower.startswith(("http://", "https://")):
            continue
        if is_reference_endpoint_url(url, source, text):
            continue
        parsed = _parse_custom_service_url(url)
        if parsed is None:
            continue
        host = (parsed.hostname or "").lower().rstrip(".")
        if host and host not in _BLOB_DOMAIN_NOISE:
            custom_domains.add(host)

    if not custom_domains:
        return

    # ── 2. Find LLM path constants ──────────────────────────────
    seen: set[str] = set()
    for m in _JADX_LLM_PATH_RE.finditer(text):
        value = m.group(1).strip()
        # A complete URL is not a split route constant. It is classified on
        # its own host/path by ``scan_url_for_business_ai_endpoint``.
        if value.lower().startswith(("http://", "https://")):
            continue
        if not is_plausible_api_route_literal(value):
            continue
        route_strength = classify_llm_route(value, text)
        if value and value not in seen and route_strength:
            seen.add(value)
            yield RawSignal(
                SignalType.BUSINESS_AI_ENDPOINT,
                value,
                layer,
                source,
                detail="jadx_constant",
                role=SignalRole.RELATION,
                scope=SignalScope.CLASS,
                relation=SignalRelation.CUSTOM_BASE_URL_LLM_PATH,
                components=tuple(
                    [*(f"custom_domain={domain}" for domain in sorted(custom_domains)),
                     f"llm_path={value}",
                     f"route_strength={route_strength}"]
                ),
            )


# A bounded fallback for JavaScript/config bundles where a base URL and route
# are separate string literals rather than an assigned Retrofit annotation.
# It deliberately operates within one decoded source file only; DEX/SO/Unity
# pools remain excluded because they do not preserve ownership boundaries.
_LOCAL_SPLIT_ROUTE_RE: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9])(?P<route>/(?:api/)?(?:v\d+/)?(?:chat/completions?|"
    r"responses|completions?|generate|generation|inference|assistant|"
    r"llm|prompt|message|ask)(?:[/?:#&=._A-Za-z0-9{}%-]*)?)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_LOCAL_SPLIT_RELATION_RADIUS = 16 * 1024


def scan_for_local_split_ai_endpoint(
    text: str,
    source: str,
    layer: str,
    radius: int = _LOCAL_SPLIT_RELATION_RADIUS,
) -> Iterator[RawSignal]:
    """Recover explicit local base/route compositions, without guessing pairs."""
    if not text or is_endpoint_reference_source(source):
        return

    if not _LOCAL_SPLIT_ROUTE_RE.search(text):
        return
    from endpoint_literals import split_endpoint_literal_pairs
    seen: set[tuple[str, str]] = set()
    for url, route, route_pos, binding in split_endpoint_literal_pairs(text, radius):
        if is_reference_endpoint_url(url, source, text):
            continue
        parsed = _parse_custom_service_url(url)
        if parsed is None:
            continue
        host = (parsed.hostname or "").lower().rstrip(".")
        if not host:
            continue
        local_context = text[max(0,route_pos-2048):route_pos+len(route)+2048]
        route_strength = classify_llm_route(route, local_context)
        if not route_strength:
            continue
        base = url.split("?", 1)[0].split("#", 1)[0].rstrip("/")
        key = (base, route.lower())
        if key in seen:
            continue
        seen.add(key)
        signal_url = base + route
        yield RawSignal(
            SignalType.BUSINESS_AI_ENDPOINT,
            signal_url,
            layer,
            source,
            detail="bounded_split_base_route",
            role=SignalRole.RELATION,
            scope=SignalScope.FILE,
            relation=SignalRelation.CUSTOM_BASE_URL_LLM_PATH,
            components=(
                f"custom_domain={host}",
                f"llm_path={route}",
                f"route_strength={route_strength}",
                f"relation_radius={radius}",
                f"binding={binding}",
            ),
        )


# ═══════════════════════════════════════════════════════════════════
# Blob scan — complete custom LLM endpoint URLs only
# ═══════════════════════════════════════════════════════════════════

_BLOB_URL_RE: re.Pattern[str] = re.compile(
    r"https?://([a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,})",
)

# Common non-AI domains that appear in framework / build-tool metadata
# and would produce false positives if treated as custom AI endpoints.
_BLOB_DOMAIN_NOISE: frozenset[str] = frozenset({
    # Build & dependency toolchain
    "github.com", "android.com", "goo.gle", "google.com",
    "play.google.com", "developer.android.com", "schemas.android.com",
    "gradle.org", "apache.org", "maven.org", "jitpack.io",
    "bintray.com", "sonatype.org", "w3.org", "xml.org", "xmlpull.org",
    # Google infra (general — LLM-specific endpoints are in _LLM_API_DOMAINS)
    "googleapis.com",
    "issuetracker.google.com",
    "firebase.google.com",
    "crashlytics.com",
    "plus.google.com",
    # GitHub / social / CDN
    "avatars.githubusercontent.com",
    "raw.githubusercontent.com",
    "twitter.com", "facebook.com", "linkedin.com", "instagram.com",
    # Adobe / misc infra
    "adobe.com", "ns.adobe.com",
    # AI vendor developer / reference sites — never actual API proxy endpoints
    "ai.google.dev",
    "gpt4all.io",
    # Domains that contain the letters ``ai`` but are not AI services.
    "aiim.org",
})

_BINARY_HTTP_SCHEME_RE: re.Pattern[str] = re.compile(r"https?://", re.IGNORECASE)
_BINARY_URL_ALLOWED_CHARS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ".-_~:/?#[]@!$&()*+,;=%"
)
_BINARY_CUSTOM_URL_MAX_LENGTH = 2048


def _iter_binary_http_urls(blob_text: str) -> Iterator[str]:
    """Yield bounded HTTP URLs without merging adjacent binary strings."""
    seen: set[str] = set()
    accepted_ends: set[int] = set()
    matches = list(_BINARY_HTTP_SCHEME_RE.finditer(blob_text))
    for index, match in enumerate(matches):
        # A binary string pool can contain ``Right`` next to ``tps://...``.
        # Searching without a left boundary would borrow the final ``h`` and
        # invent an ``https://`` URL.  A scheme directly following another
        # complete URL remains recoverable because the preceding candidate
        # establishes the otherwise missing string boundary.
        previous = blob_text[match.start() - 1] if match.start() > 0 else ""
        # ``latin-1`` decoding makes arbitrary binary bytes look like
        # Unicode letters (for example ``\xBC`` is ``¼`` and ``isalnum()``),
        # which incorrectly suppresses a real URL after a DEX/SO string.
        # Only ASCII alphanumerics can be part of the synthetic ``https``
        # boundary check.
        if (match.start() > 0
                and previous.isascii() and previous.isalnum()
                and match.start() not in accepted_ends):
            continue

        hard_end = min(
            len(blob_text), match.start() + _BINARY_CUSTOM_URL_MAX_LENGTH)
        if index + 1 < len(matches):
            hard_end = min(hard_end, matches[index + 1].start())

        end = match.end()
        while end < hard_end and blob_text[end] in _BINARY_URL_ALLOWED_CHARS:
            end += 1
        candidate = blob_text[match.start():end].rstrip(".,;:)]}")
        try:
            parsed_candidate = urlsplit(candidate)
        except ValueError:
            continue
        if not parsed_candidate.hostname:
            continue

        accepted_ends.add(match.start() + len(candidate))
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def scan_blob_for_business_ai_endpoint(
    blob_text: str | bytes, filename: str
) -> Iterator[RawSignal]:
    """Yield business endpoints proved by one complete binary URL value.

    DEX/SO/Unity string tables do not preserve method or configuration
    ownership, so proximity between separate strings is not treated as a
    classifier relation.
    """
    layer = "zip"
    if isinstance(blob_text, bytes):
        # Preserve byte boundaries: UTF-8 errors='ignore' can concatenate
        # unrelated binary strings. URLs themselves use ASCII delimiters.
        blob_text = blob_text.decode("latin-1")
    for url in _iter_binary_http_urls(blob_text):
        # Context can exclude an explicit help URL within its own string,
        # including oversized JS fallback. It must never provide positive
        # schema/config associations between compiled strings.
        if is_reference_endpoint_url(url, filename, blob_text):
            continue
        yield from scan_url_for_business_ai_endpoint(url, filename, layer)


# ═══════════════════════════════════════════════════════════════════
# Conservative AI semantics in custom domain names
# ═══════════════════════════════════════════════════════════════════

_AI_DOMAIN_SEGMENT_KW: tuple[str, ...] = (
    "gpt", "chat", "llm", "claude", "gemini", "openai",
    "assistant", "completion", "generate", "infer",
)


def _domain_looks_ai(domain: str) -> bool:
    """Check whether a domain name itself suggests AI usage, using
    conservative segment-level matching.  ``daily.com`` does NOT pass
    (``daily`` embeds ``ai`` in a longer word); ``aicomic-dev`` passes
    (segment starts with ``ai``).

    ``api`` is intentionally excluded — it starts with ``ai`` but is
    the most common API subdomain prefix and has nothing to do with AI.
    """
    # Common English / product prefixes that happen to start with
    # "ai" but have nothing to do with AI — air(pods|table|tel|bnb),
    # aid, aim, etc.  Excluding them prevents noise from non-AI apps
    # whose domains happen to start with these letters.
    _AI_START_NOISE = {"air", "aid", "ail", "aim", "ain", "ais", "ait"}

    for seg in domain.lower().split("."):
        if seg == "api":
            continue
        if seg.startswith("ai") and seg[:3] not in _AI_START_NOISE:
            return True
        if seg.endswith("ai"):
            return True
        if any(kw in seg for kw in _AI_DOMAIN_SEGMENT_KW):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════

_API_KEY_INPUT_PATS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE) for p in (
        r"enter\s+(?:your\s+)?(?:openai|api|openai\s+api)\s+key",
        r"(?:openai|api)\s+key\s*(?::|is)",
        r"set\s+(?:your\s+)?(?:openai|api)\s+key",
        r"api[_ ]?key\s*(?::|is|—|–)",
        r"openai\s+api\s+key",
    )
)


def scan_for_api_key_input_prompt(
    text: str, source: str, layer: str
) -> Iterator[RawSignal]:
    """Yield *API_KEY_INPUT_PROMPT* if *text* contains UI strings that
    prompt the user to enter an API key.

    The observation is category-neutral: it records a user-facing key input
    prompt and leaves its architectural interpretation to the classifier.
    """
    for pat in _API_KEY_INPUT_PATS:
        m = pat.search(text)
        if m:
            directive = bool(re.search(r"\b(?:enter|set)\b", m[0], re.I))
            provider = bool(_STRONG_LLM_IDENTITY_RE.search(m[0]))
            yield RawSignal(SignalType.API_KEY_INPUT_PROMPT,
                            m.group(0), layer, source,
                            verification=SignalVerification.OBSERVED if directive and provider else SignalVerification.CANDIDATE,
                            source_kind="ui_prompt" if directive else "text_marker")


def decode_manifest_xml(data: bytes) -> str:
    """Decode binary AndroidManifest.xml to a best-effort string."""
    if AXMLPrinter is not None:
        try:
            xml = AXMLPrinter(data).get_xml()
            if xml:
                return xml.decode("utf-8", errors="ignore") if isinstance(xml, bytes) else str(xml)
        except Exception:
            pass
    try:
        candidate = data.decode("utf-16-le", errors="ignore")
        if "manifest" in candidate.lower() and "android" in candidate.lower():
            candidate = re.sub(r"\x00+", "", candidate)
            candidate = re.sub(r"(?<=[a-zA-Z])\.(?=[a-zA-Z])", ".", candidate)
            return candidate
    except Exception:
        pass
    return data.decode("utf-8", errors="ignore")


def normalize_binary_manifest_text(text: str) -> str:
    """Collapse control characters and whitespace in manifest text."""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text)
    return re.sub(r"\s+", " ", text)


def extract_attr(text: str, pattern: str) -> str | None:
    """Return the first capture group of *pattern* in *text*, or None."""
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1) if m else None


def dedupe_sequence(values: list[str]) -> list[str]:
    """Preserve first-seen order, drop duplicates."""
    out: list[str] = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


# ── Re-exports ────────────────────────────────────────────────────
LLM_API_DOMAINS: tuple[str, ...] = _LLM_API_DOMAINS
BACKEND_MARKER_PATS: tuple[re.Pattern[str], ...] = _BACKEND_MARKER_PATS
STRONG_MODEL_TEXT_HINTS: tuple[str, ...] = _STRONG_MODEL_TEXT_HINTS
SERVERLESS_DOMAINS: tuple[str, ...] = _SERVERLESS_DOMAINS
HTTP_CLIENT_INDICATORS: tuple[str, ...] = _HTTP_CLIENT_INDICATORS
WEBVIEW_API_INDICATORS: tuple[str, ...] = _WEBVIEW_API_INDICATORS
JNI_API_INDICATORS: tuple[str, ...] = _JNI_API_INDICATORS
