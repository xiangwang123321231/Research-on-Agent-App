"""Neutral data model for APK LLM signal extraction.

This module defines *what* was observed, never *why*.  The five-category
classification is the responsibility of downstream ML, not of the
extraction layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re


# ═══════════════════════════════════════════════════════════════════
# Signal taxonomy
# ═══════════════════════════════════════════════════════════════════

class SignalType(str, Enum):
    """Neutral signal taxonomy — describes *what* was observed."""

    # ── URL / endpoint signals ──────────────────────────────────────
    KNOWN_LLM_API_DOMAIN = "known_llm_api_domain"          # api.openai.com, api.anthropic.com, …
    KNOWN_LLM_WEB_DOMAIN = "known_llm_web_domain"          # chatgpt.com, claude.ai, …

    BACKEND_PROXY_KEYWORD = "backend_proxy_keyword"         # worker / gateway / proxy / backend / relay

    # ── SDK / package identity signals ─────────────────────────────
    KNOWN_LLM_SDK_PACKAGE = "known_llm_sdk_package"        # com.theokanning.openai, dev.langchain4j, …

    # ── Model identity signals ─────────────────────────────────────
    KNOWN_LLM_MODEL_NAME = "known_llm_model_name"          # gpt-4, claude-3.7, deepseek-v3, …
    MODEL_FILE_EXTENSION = "model_file_extension"           # .gguf, mlc-chat-config.json
    MODEL_FILE_REFERENCE = "model_file_reference"           # content mentions .gguf / mlc-chat-config

    # ── Token / prompt signals ─────────────────────────────────────
    LLM_SPECIAL_TOKEN = "llm_special_token"                # <|im_start|>, <think>, <|eot_id|>, …
    LLM_PROMPT_TEMPLATE = "llm_prompt_template"            # "you are a helpful assistant", system prompts
    LLM_JSON_SCHEMA = "llm_json_schema"                    # OpenAI-style JSON keys (messages, role, …)
    LLM_HTTP_API_USAGE = "llm_http_api_usage"              # invoked HTTP contract with local LLM request evidence
    LLM_OPERATION_CONTEXT = "llm_operation_context"        # related text/assistant operation reference, no call claim

    # ── Native / engine signals ────────────────────────────────────
    NATIVE_LLM_FUNCTION = "native_llm_function"            # llama_decode, executorch_forward, …
    NATIVE_LLM_LIBRARY = "native_llm_library"              # libllama.so, libexecutorch.so, …
    LLM_FRAMEWORK_KEYWORD = "llm_framework_keyword"        # llama.cpp, mediapipe genai, executorch, …
    ONNX_TFLITE_RUNTIME = "onnx_tflite_runtime"            # onnxruntime, tflite, qnn, snpe
    LOCAL_GENERATION_CONFIG = "local_generation_config"  # structured decoder config, not weights
    LOCAL_INFERENCE_RUNTIME = "local_inference_runtime"  # generic runtime, not standalone LLM evidence
    ON_DEVICE_GENERATION_USAGE = "on_device_generation_usage"  # typed interface reference / static call

    # ── WebView signals ────────────────────────────────────────────

    WEBVIEW_PRESENT = "webview_present"                  # application WebView usage
    WEBVIEW_LOAD_CALL = "webview_load_call"              # loadUrl/loadData invocation

    WEBVIEW_LLM_TARGET = "webview_llm_target"              # WebView + known LLM web domain
    WEBVIEW_CHAT_PAGE = "webview_chat_page"                # WebView + chat-like path on custom domain
    BROWSER_LLM_NAVIGATION = "browser_llm_navigation"      # bound Custom Tabs/TWA target; not WebView

    # ── Asset / delivery signals ───────────────────────────────────
    ASSET_DELIVERY_HINT = "asset_delivery_hint"            # PAD / split-install / dynamic-feature
    DYNAMIC_DOWNLOAD_HINT = "dynamic_download_hint"        # huggingface.co / snapshot_download / weights
    LICENSE_KEYWORD = "license_keyword"                    # llama.cpp / mediapipe genai / vertex ai in licenses

    # ── Manifest signals ───────────────────────────────────────────
    ANDROID_PERMISSION = "android_permission"              # INTERNET, BIND_AICORE_SERVICE, …

    # ── Credential / configuration signals ──────────────────────────
    HARDCODED_API_KEY = "hardcoded_api_key"                # redacted provider-prefix credential observation
    BASEURL_OVERRIDE = "baseurl_override"                  # setBaseUrl(), api_base_url, base_url
    SERVERLESS_URL = "serverless_url"                      # cloudfunctions.net, vercel.app, fly.dev, …
    BUSINESS_AI_ENDPOINT = "business_ai_endpoint"          # /api/chat, /v1/assistant, /ai/generate, … on custom domains
    API_KEY_INPUT_PROMPT = "api_key_input_prompt"           # "Enter your OpenAI key", "API Key" — user-facing UI prompt


class SignalRole(str, Enum):
    """How an observation is represented in the static artifact."""

    LITERAL = "literal"
    CONFIG = "config"
    CALL = "call"
    ARTIFACT = "artifact"
    METADATA = "metadata"
    REFERENCE = "reference"
    RELATION = "relation"


class SignalVerification(str, Enum):
    """Fact validation, never a category or classifier confidence."""
    OBSERVED = "observed"
    CANDIDATE = "candidate"
    VERIFIED = "verified"


class SignalScope(str, Enum):
    """Smallest scope in which all parts of an observation were verified."""

    VALUE = "value"
    CONFIG = "config"
    METHOD = "method"
    LOCAL_METHOD_GRAPH = "local_method_graph"
    CLASS = "class"
    FILE = "file"
    APK = "apk"


class SignalRelation(str, Enum):
    """Stable names for category-neutral, reproducible relations."""

    CUSTOM_DOMAIN_LLM_PATH = "custom_domain_with_llm_path"
    CUSTOM_BASE_URL_LLM_PATH = "custom_base_url_with_llm_path_constant"
    BINARY_LOCAL_LLM_ENDPOINT = "binary_local_llm_endpoint_cooccurrence"
    APK_CROSS_FILE_LLM_ENDPOINT = "apk_cross_file_llm_endpoint_cooccurrence"
    WEBVIEW_KNOWN_LLM_TARGET = "webview_with_known_llm_target"
    WEBVIEW_CUSTOM_CHAT_TARGET = "webview_with_custom_chat_target"
    BROWSER_LLM_TARGET = "browser_with_llm_target"
    JSON_FIELDS_LLM_API = "json_fields_with_llm_api_shape"
    HTTP_API_LLM_CONTEXT = "http_api_with_llm_request_context"
    MODEL_ID_USAGE_CONTEXT = "model_id_with_usage_context"
    RUNTIME_LLM_CONTEXT = "runtime_with_llm_context"
    TOKENIZER_LLM_CONTEXT = "tokenizer_with_llm_context"
    MODEL_DOWNLOAD_CONTEXT = "download_hint_with_model_context"
    ASSET_DELIVERY_MODEL_CONTEXT = "asset_delivery_with_model_context"


# ═══════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RawSignal:
    """A single neutral observation extracted from an APK.

    No field references any embedding-mode category.
    The ``signal_type`` describes the literal finding; downstream
    feature-engineering and ML determine what it implies.
    """
    signal_type: SignalType
    value: str                              # the matched content
    layer: str                              # "zip" | "apktool" | "jadx"
    source_file: str = ""                   # file path within the APK / decoded tree
    detail: str | None = None               # optional extra context
    role: SignalRole = SignalRole.LITERAL
    scope: SignalScope = SignalScope.VALUE
    relation: SignalRelation | None = None
    # Atomic observations participating in a relation.  Values are bounded,
    # non-secret strings such as ``custom_domain=api.example.com``.
    components: tuple[str, ...] = ()
    # Internal-only identity for values that must be redacted before output.
    # It is deliberately excluded from repr/equality and is never serialized.
    dedupe_key: str | None = field(default=None, repr=False, compare=False)
    verification: SignalVerification = SignalVerification.OBSERVED
    source_kind: str = ""


@dataclass
class AnalysisResult:
    """Top-level result for one APK: metadata + neutral signals."""
    apk_path: str
    sha256: str
    metadata: dict[str, object]
    signals: list[RawSignal] = field(default_factory=list)
    # Separate from ``signals`` so research enrichment never changes f01/f02
    # or any other feature consumed by the versioned classifier.
    research_evidence: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "apk_path": self.apk_path,
            "sha256": self.sha256,
            "metadata": self.metadata,
            "research_evidence": self.research_evidence,
            "signals": [
                {
                    "signal_type": s.signal_type.value,
                    "value": s.value,
                    "layer": s.layer,
                    "source_file": s.source_file,
                    "detail": s.detail,
                    "role": s.role.value,
                    "scope": s.scope.value,
                    "relation": s.relation.value if s.relation else None,
                    "components": list(s.components),
                    "verification": s.verification.value,
                    "source_kind": s.source_kind,
                }
                for s in self.signals
            ],
        }


# ═══════════════════════════════════════════════════════════════════
# Signal aggregation (cross-layer dedup)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SignalAggregator:
    """Merge signals across the extraction layers.

    Deduplication key normally includes ``(signal_type, normalized_value,
    role, scope, relation)``. Model identities are semantic facts, so the same
    normalized model is represented once while all literal/config/file
    evidence remains in ``evidence_instances``. Redacted observations may
    provide a non-serialized
    ``dedupe_key`` so distinct secrets do not collapse to the same displayed
    value.
    Two layers reporting the same ``api.openai.com`` URL become one
    merged signal whose ``layers`` set reflects cross-layer agreement.
    """

    @staticmethod
    def _normalize_value(signal_type: SignalType, value: str) -> str:
        """Normalize *value* for dedup.

        For domain-type signals, keep only host[:port] so that
        ``api.openai.com``, ``https://api.openai.com/``, and
        ``https://api.openai.com/v1/chat`` all dedup to the same key.
        """
        v = value.strip().lower()
        if signal_type in (SignalType.KNOWN_LLM_API_DOMAIN,
                           SignalType.KNOWN_LLM_WEB_DOMAIN):
            v = re.sub(r"^https?://", "", v)
            v = v.split("/")[0].split("?")[0].split("#")[0]
        return v

    def merge(self, signals: list[RawSignal]) -> list[dict[str, object]]:
        """Deduplicate and merge *signals* into a list of dicts.

        Merged output fields
        --------------------
        signal_type, value (canonical), layers (sorted list),
        layer_count, source_files (deduped list).
        """
        groups: dict[tuple, list[RawSignal]] = {}
        for s in signals:
            role = s.role.value
            scope = s.scope.value
            relation = s.relation.value if s.relation else ""
            if (s.signal_type == SignalType.KNOWN_LLM_MODEL_NAME
                    and s.dedupe_key is None):
                key = (
                    s.signal_type.value,
                    "model_identity",
                    self._normalize_value(s.signal_type, s.value),
                    s.verification.value,
                )
            elif s.dedupe_key is not None:
                key = (
                    s.signal_type.value, "private", s.dedupe_key,
                    role, scope, relation,
                )
            else:
                key = (
                    s.signal_type.value,
                    "value",
                    self._normalize_value(s.signal_type, s.value),
                    role,
                    scope,
                    relation,
                )
            # Candidates cannot inherit validation from another occurrence.
            # Model identities already carry verification in their semantic
            # key and deliberately merge source kinds as provenance.
            if s.signal_type != SignalType.KNOWN_LLM_MODEL_NAME:
                key = (*key, s.verification.value, s.source_kind)
            groups.setdefault(key, []).append(s)

        merged: list[dict[str, object]] = []
        scope_rank = {
            SignalScope.APK: 0,
            SignalScope.FILE: 1,
            SignalScope.VALUE: 2,
            SignalScope.CLASS: 3,
            SignalScope.CONFIG: 4,
            SignalScope.METHOD: 5,
            SignalScope.LOCAL_METHOD_GRAPH: 6,
        }
        role_rank = {
            SignalRole.REFERENCE: 0,
            SignalRole.LITERAL: 1,
            SignalRole.RELATION: 2,
            SignalRole.CONFIG: 3,
            SignalRole.CALL: 4,
        }
        source_kind_rank = {
            "": 0,
            "binary_file_identity": 1,
            "source_string_literal": 2,
            "hermes_string_literal": 3,
        }
        for group in groups.values():
            representative = max(group, key=lambda signal: (
                scope_rank.get(signal.scope, 0),
                role_rank.get(signal.role, 0),
                source_kind_rank.get(signal.source_kind, 1),
            ))
            stype = representative.signal_type.value
            role = representative.role.value
            scope = representative.scope.value
            relation = (representative.relation.value
                        if representative.relation else "")
            verification = representative.verification.value
            source_kind = representative.source_kind
            best_value = max((s.value for s in group), key=len)
            layers = sorted(set(s.layer for s in group))
            source_files = sorted(set(s.source_file for s in group if s.source_file))
            details = list(dict.fromkeys(s.detail for s in group if s.detail))
            components = sorted({
                component
                for signal in group
                for component in signal.components
            })

            merged.append({
                "signal_type": stype,
                "value": best_value,
                "role": role,
                "scope": scope,
                "relation": relation or None,
                "components": components,
                "verification": verification,
                "source_kind": source_kind,
                "evidence_instances": list({
                    (
                        s.layer,
                        s.source_file,
                        s.role.value,
                        s.scope.value,
                        s.relation.value if s.relation else None,
                        s.verification.value,
                        s.source_kind,
                        tuple(s.components),
                    ): {
                        "layer": s.layer,
                        "source_file": s.source_file,
                        "role": s.role.value,
                        "scope": s.scope.value,
                        "relation": s.relation.value if s.relation else None,
                        "verification": s.verification.value,
                        "source_kind": s.source_kind,
                        "components": list(s.components),
                    }
                    for s in group
                }.values()),
                "layers": layers,
                "layer_count": len(layers),
                "source_files": source_files,
                "source_file_count": len(source_files),
                "detail": "; ".join(details) if details else None,
            })

        merged.sort(key=lambda m: (m["layer_count"], len(m.get("source_files", []))),
                    reverse=True)
        return merged
