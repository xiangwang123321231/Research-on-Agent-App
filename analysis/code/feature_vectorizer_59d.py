"""Frozen 2026-10-25 vectorizer for the historical 59d ablation protocol.

Production extraction/export uses feature_vectorizer.py. Do not extend this
historical mapping with new feature semantics.

Every feature is computed from observed signals and metadata only.
No embedding-mode category is used in feature computation.
Semantic facts remain separate from embedding-mode labels; the classifier
learns their relationships. Historical composite columns retain their meaning.

Usage::

    from feature_vectorizer import FeatureVectorizer

    fv = FeatureVectorizer()
    X, names = fv.transform(records)   # list[dict] → list[list[float]]
"""

from __future__ import annotations

import math
from typing import Sequence

from signal_types import SignalRelation, SignalRole, SignalScope, SignalType
from registry import (CLOUD_SDK_PACKAGES, SELF_HOSTED_SDK_PACKAGES,
                      endpoint_noise_reason, is_reference_only_source,
                      llm_host_candidate_evidence, json_llm_link_feature_eligible)
from webview_context import WEBVIEW_SIGNAL_TYPES, web_signal_feature_eligible
from feature_evidence import (related_web_observation, typed_web_observation,
                              observed_web_hosts, related_configuration,
                              collapse_identity_facts, source_categories,
                              host_identity, programmatic_llm_usage,
                              llm_operation_reference)
from device_feature_evidence import (generation_configuration, model_binary_artifact,
                                     generation_interface)


FEATURE_SCHEMA_VERSION = "features-59d-v2026.10.25.1"


def _collapse_model_feature_facts(signals: list[dict]) -> list[dict]:
    """Count one model identity once while retaining all source provenance."""
    result: list[dict] = []
    positions: dict[str, int] = {}
    source_kind_priority = {
        "": 0,
        "binary_url": 1,
        "source_string_literal": 2,
        "hermes_string_literal": 3,
        "ram_string_literal": 3,
    }
    for signal in signals:
        if signal.get("signal_type") != SignalType.KNOWN_LLM_MODEL_NAME.value:
            result.append(signal)
            continue
        value = str(signal.get("value", "")).strip().lower()
        if not value or value not in positions:
            clone = dict(signal)
            clone["source_files"] = list(signal.get("source_files", []) or [])
            clone["layers"] = list(signal.get("layers", []) or [])
            clone["components"] = list(signal.get("components", []) or [])
            clone["evidence_instances"] = list(
                signal.get("evidence_instances", []) or [])
            positions[value] = len(result)
            result.append(clone)
            continue
        merged = result[positions[value]]
        for field in ("source_files", "layers", "components"):
            merged[field] = list(dict.fromkeys([
                *(merged.get(field, []) or []),
                *(signal.get(field, []) or []),
            ]))
        merged["evidence_instances"] = [
            *(merged.get("evidence_instances", []) or []),
            *(signal.get("evidence_instances", []) or []),
        ]
        current_kind = str(merged.get("source_kind", "") or "")
        incoming_kind = str(signal.get("source_kind", "") or "")
        if source_kind_priority.get(incoming_kind, 1) > source_kind_priority.get(
            current_kind, 1
        ):
            merged["source_kind"] = incoming_kind
    return result


class FeatureVectorizer:
    """59-dimensional feature vector from neutral APK signals.

    All features are deterministic — no ``fit()`` step required.
    """

    _names: list[str] | None = None

    @property
    def feature_names(self) -> list[str]:
        if self._names is None:
            self._names = list(self._iter_names())
        return self._names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    # ── public API ─────────────────────────────────────────────────

    def vectorize_one(self, merged_signals: list[dict[str, object]],
                       metadata: dict[str, object]) -> list[float]:
        v: list[float] = []

        # A relation must have a reproducible local scope before it can enter
        # training.  Legacy rows without role/scope remain readable, while
        # explicitly APK-wide cross-file co-occurrences are excluded.
        accepted_relation_scopes = {
            SignalScope.VALUE.value,
            SignalScope.CONFIG.value,
            SignalScope.METHOD.value,
            SignalScope.LOCAL_METHOD_GRAPH.value,
            SignalScope.CLASS.value,
            SignalScope.FILE.value,
        }

        def feature_eligible(signal: dict[str, object]) -> bool:
            components = signal.get('components') or []
            if signal.get('signal_type') == SignalType.LOCAL_INFERENCE_RUNTIME.value:
                # A generic TVM runtime is audit material, not LLM relevance.
                return False
            if signal.get('signal_type') == SignalType.LOCAL_GENERATION_CONFIG.value:
                return generation_configuration(signal)
            if signal.get('signal_type') == SignalType.ON_DEVICE_GENERATION_USAGE.value:
                return generation_interface(signal)
            if (signal.get('signal_type') in WEBVIEW_SIGNAL_TYPES
                    and not related_web_observation(signal)):
                return False
            configuration = signal.get('signal_type') in {
                SignalType.BASEURL_OVERRIDE.value, SignalType.API_KEY_INPUT_PROMPT.value}
            related_config = configuration and related_configuration(signal)
            if configuration and not related_config:
                return False
            browser_navigation = signal.get('signal_type') == SignalType.BROWSER_LLM_NAVIGATION.value
            if browser_navigation:
                # No bare container, unknown argument, or arbitrary candidate
                # enters through the new browser path. Configuration is not a call.
                binding_ok = ((signal.get('source_kind') == 'smali_browser_navigation'
                               and 'navigation_evidence=invocation' in components
                               and any('binding=' + binding in components for binding in
                                       ('uri_argument', 'constructor_field_uri_argument')))
                              or (signal.get('source_kind') == 'twa_manifest_entry'
                                  and 'container=twa' in components
                                  and 'binding=launcher_default_url' in components
                                  and 'navigation_evidence=entry_configuration' in components))
                if not (binding_ok and signal.get('role') == SignalRole.RELATION.value
                        and signal.get('relation') == SignalRelation.BROWSER_LLM_TARGET.value
                        and any('container=' + kind in components for kind in ('custom_tabs', 'twa'))
                        and any('relevance_hint=' + hint in components for hint in
                                ('known_llm_website', 'explicit_llm_context'))):
                    return False
            if signal.get('signal_type') == SignalType.BUSINESS_AI_ENDPOINT.value:
                service_evidence = tuple(
                    item.split('=', 1)[1]
                    for item in components
                    if str(item).startswith('service_evidence=')
                )
                if endpoint_noise_reason(
                    str(signal.get('value') or ''),
                    service_evidence=service_evidence,
                ):
                    return False
            host_candidate = (signal.get('signal_type') == SignalType.BUSINESS_AI_ENDPOINT.value
                              and 'route_strength=host_identity' in components)
            config_candidate = (signal.get('signal_type') == SignalType.BUSINESS_AI_ENDPOINT.value
                                and 'route_strength=config_identity' in components)
            related_config_candidate = config_candidate and json_llm_link_feature_eligible(signal)
            related_host_candidate = host_candidate and bool(llm_host_candidate_evidence(
                str(signal.get('value') or ''),
                tuple(item.split('=', 1)[1] for item in components if item.startswith('llm_path=')),
                # A source kind is provenance, not evidence of an actual
                # assignment to this URL. Require retained config evidence.
                endpoint_config=('candidate_evidence=ai_endpoint_config' in components),
                config_keys=(tuple(item.split('=', 1)[1] for item in components
                                   if item.startswith('endpoint_config_key='))
                             if signal.get('source_kind') == 'endpoint_value' else ()),
            ))
            if related_host_candidate and signal.get('source_files'):
                related_host_candidate = any(not is_reference_only_source(str(source))
                                             for source in signal['source_files'])
            related_http_candidate = (
                signal.get('signal_type') == SignalType.LLM_HTTP_API_USAGE.value
                and signal.get('source_kind') == 'smali_http_contract'
                and signal.get('scope') == SignalScope.METHOD.value
                and signal.get('relation') == SignalRelation.HTTP_API_LLM_CONTEXT.value
                and 'binding=interface_invocation' in components
                and any(item in components for item in (
                    'protocol_evidence=inference_route_with_text_input',
                    'protocol_evidence=inference_route_with_stream',
                    'protocol_evidence=typed_message_structure',
                ))
            )
            # Relevant WebView and custom-host candidates are model inputs.
            # Host candidates mean related addresses, not observed execution.
            # Referenced HTTP candidates also contribute general signal counts,
            # never schema/endpoint presence or a proven client binding.
            if (signal.get("verification") == "candidate"
                    and signal.get("signal_type") not in WEBVIEW_SIGNAL_TYPES
                    and not browser_navigation
                    and not related_host_candidate
                    and not related_config_candidate
                    and not related_config
                    and not related_http_candidate):
                return False
            # Validate relevance even for legacy host rows marked observed.
            if host_candidate and not related_host_candidate:
                return False
            if config_candidate and not related_config_candidate:
                return False
            # Google uses the same AIza prefix for Gemini and for ubiquitous
            # Firebase/Maps/Ads credentials.  Preserve it in neutral signal
            # output for audit, but do not let an ambiguous Android credential
            # affect density, type, source-layer, or credential features.
            if (
                str(signal.get("signal_type", ""))
                == SignalType.HARDCODED_API_KEY.value
                and str(signal.get("value", "")).strip().lower().startswith("aiza")
            ):
                return False
            role = str(signal.get("role", "") or "")
            if not role:
                return True
            if role != SignalRole.RELATION.value:
                return True
            return str(signal.get("scope", "") or "") in accepted_relation_scopes

        feature_signals = collapse_identity_facts(_collapse_model_feature_facts([
            s for s in merged_signals if feature_eligible(s)
        ]))
        by_type: dict[str, list[dict]] = {}
        for s in feature_signals:
            st = str(s.get("signal_type", ""))
            by_type.setdefault(st, []).append(s)

        def has(st: SignalType) -> bool:
            return st.value in by_type

        def count(st: SignalType) -> int:
            return len(by_type.get(st.value, []))

        def unique_value_count(st: SignalType) -> int:
            return len({
                str(signal.get("value", "")).strip().lower()
                for signal in by_type.get(st.value, [])
                if str(signal.get("value", "")).strip()
            })

        cg = metadata.get("callgraph", {}) or {}
        lc = metadata.get("layer_coverage", {}) or {}

        # ═══════════════════════════════════════════════════════════
        # Group 1 — Signal density (3 dims)
        # ═══════════════════════════════════════════════════════════

        v.append(float(len(feature_signals)))                         # f01  total_signal_count
        all_types = set(str(s.get("signal_type", "")) for s in feature_signals)
        v.append(float(len(all_types)))                               # f02  unique_signal_types
        v.append(float(count(SignalType.LLM_SPECIAL_TOKEN)))           # f03  special_token_count

        # ═══════════════════════════════════════════════════════════
        # Group 2 — SDK signals, split by SDK category (4 dims)
        # ═══════════════════════════════════════════════════════════

        sdk_signals = by_type.get(SignalType.KNOWN_LLM_SDK_PACKAGE.value, [])
        sdk_official: set[str] = set()
        sdk_selfhosted: set[str] = set()
        for s in sdk_signals:
            val = str(s.get("value", "")).strip().lower()
            if not val:
                continue
            for pkg in CLOUD_SDK_PACKAGES:
                if pkg in val:
                    sdk_official.add(pkg)
                    break
            for pkg in SELF_HOSTED_SDK_PACKAGES:
                if pkg in val:
                    sdk_selfhosted.add(pkg)
                    break

        v.append(1.0 if sdk_official else 0.0)     # f04  official_sdk_present
        v.append(float(len(sdk_official)))          # f05  official_sdk_count
        v.append(1.0 if sdk_selfhosted else 0.0)    # f06  self_hosted_sdk_present
        v.append(float(len(sdk_selfhosted)))         # f07  self_hosted_sdk_count

        # ═══════════════════════════════════════════════════════════
        # Group 3 — Domains & endpoints (10 dims)
        # ═══════════════════════════════════════════════════════════

        v.append(1.0 if has(SignalType.KNOWN_LLM_API_DOMAIN) else 0.0)   # f08
        v.append(float(count(SignalType.KNOWN_LLM_API_DOMAIN)))          # f09
        web_hosts = {host_identity(str(s.get('value', '')))
                     for s in by_type.get(SignalType.KNOWN_LLM_WEB_DOMAIN.value, [])}
        for s in feature_signals:
            if s.get('signal_type') in WEBVIEW_SIGNAL_TYPES:
                web_hosts.update(observed_web_hosts(s))
        web_hosts.discard('')
        v.append(float(bool(web_hosts)))                               # f10
        v.append(float(len(web_hosts)))                                # f11
        v.append(1.0 if has(SignalType.BACKEND_PROXY_KEYWORD) else 0.0)  # f12
        v.append(float(count(SignalType.BACKEND_PROXY_KEYWORD)))         # f13
        v.append(1.0 if has(SignalType.SERVERLESS_URL) else 0.0)         # f14
        v.append(float(count(SignalType.SERVERLESS_URL)))                # f15
        # f16/f17 include relevant host candidates as neutral address hints.
        # Only f44-f48 describe recovered method/endpoint bindings.
        v.append(1.0 if has(SignalType.BUSINESS_AI_ENDPOINT) else 0.0)  # f16
        v.append(float(
            unique_value_count(SignalType.BUSINESS_AI_ENDPOINT)))          # f17

        # ═══════════════════════════════════════════════════════════
        # Group 4 — Credentials & config (7 dims)
        # ═══════════════════════════════════════════════════════════

        # f18: LLM-specific API key only (provider-specific prefixes such as
        # sk-*, hf_*, gsk_, xai-, pplx- and r8_). AIza signals have already
        # been excluded from every classifier feature by feature_eligible().
        # A matched key is still evidence of an embedded credential, not proof
        # that the runtime actually uses the provider.
        llm_apikey = has(SignalType.HARDCODED_API_KEY)
        v.append(1.0 if llm_apikey else 0.0)                              # f18
        # Historical column name: concrete LLM endpoint configuration, not
        # proof that a user can override it or that a request was executed.
        v.append(1.0 if has(SignalType.BASEURL_OVERRIDE) else 0.0)       # f19
        v.append(1.0 if has(SignalType.LLM_JSON_SCHEMA) else 0.0)        # f20
        v.append(1.0 if has(SignalType.LLM_PROMPT_TEMPLATE) else 0.0)    # f21
        has_api = has(SignalType.KNOWN_LLM_API_DOMAIN)
        v.append(1.0 if (has_api and not has(SignalType.LLM_JSON_SCHEMA)) else 0.0)  # f22  official_api_no_json_schema
        v.append(1.0 if has(SignalType.KNOWN_LLM_MODEL_NAME) else 0.0)   # f23
        v.append(float(
            unique_value_count(SignalType.KNOWN_LLM_MODEL_NAME)))          # f24
        v.append(1.0 if has(SignalType.API_KEY_INPUT_PROMPT) else 0.0)   # f25

        # ═══════════════════════════════════════════════════════════
        # Group 5 — Local engine signals (10 dims)
        # ═══════════════════════════════════════════════════════════

        model_file_present = (has(SignalType.MODEL_FILE_EXTENSION)
                              or has(SignalType.MODEL_FILE_REFERENCE))
        model_file_count = (count(SignalType.MODEL_FILE_EXTENSION)
                            + count(SignalType.MODEL_FILE_REFERENCE))
        v.append(1.0 if model_file_present else 0.0)                     # f26
        v.append(float(model_file_count))                                # f27

        v.append(1.0 if has(SignalType.NATIVE_LLM_LIBRARY) else 0.0)    # f28
        v.append(float(count(SignalType.NATIVE_LLM_LIBRARY)))            # f29
        v.append(1.0 if has(SignalType.NATIVE_LLM_FUNCTION) else 0.0)   # f30
        v.append(float(count(SignalType.NATIVE_LLM_FUNCTION)))           # f31
        v.append(1.0 if has(SignalType.LLM_FRAMEWORK_KEYWORD) else 0.0) # f32
        v.append(float(count(SignalType.LLM_FRAMEWORK_KEYWORD)))         # f33
        v.append(1.0 if has(SignalType.ONNX_TFLITE_RUNTIME) else 0.0)   # f34
        v.append(float(count(SignalType.ONNX_TFLITE_RUNTIME)))           # f35

        # ═══════════════════════════════════════════════════════════
        # Group 6 — Asset & delivery signals (2 dims)
        # ═══════════════════════════════════════════════════════════

        dyn_dl = (has(SignalType.ASSET_DELIVERY_HINT)
                  or has(SignalType.DYNAMIC_DOWNLOAD_HINT))
        v.append(1.0 if dyn_dl else 0.0)                                 # f36
        v.append(1.0 if has(SignalType.LICENSE_KEYWORD) else 0.0)        # f37

        # Group 7 — Web interaction (3 dims); TWA configuration is not a call.

        browser = by_type.get(SignalType.BROWSER_LLM_NAVIGATION.value, [])
        browser_call = any('navigation_evidence=invocation' in (s.get('components') or []) for s in browser)
        web_container = any(typed_web_observation(s) for s in by_type.get(SignalType.WEBVIEW_PRESENT.value, []))
        web_call = any(typed_web_observation(s) for s in by_type.get(SignalType.WEBVIEW_LOAD_CALL.value, []))
        web_bound = any(web_signal_feature_eligible(s) for st in (
            SignalType.WEBVIEW_LLM_TARGET, SignalType.WEBVIEW_CHAT_PAGE)
            for s in by_type.get(st.value, []))
        v.append(float(web_container or bool(browser)))                 # f38
        v.append(float(web_call or browser_call))                       # f39
        v.append(float(web_bound or bool(browser)))                     # f40

        # Group 8 — Manifest signal (1 dim)

        aicore = False
        for s in by_type.get(SignalType.ANDROID_PERMISSION.value, []):
            if (s.get('role') == SignalRole.METADATA.value
                    and 'permission_evidence=manifest_declaration' in (s.get('components') or [])
                    and 'aicore' in str(s.get('value', '')).lower()):
                aicore = True
                break
        v.append(1.0 if aicore else 0.0)                                 # f41

        # Group 9 — Signal source layer (2 dims)

        code_count = 0
        resource_count = 0
        for m in feature_signals:
            code_source, resource_source = source_categories(m)
            code_count += int(code_source)
            resource_count += int(resource_source)
        v.append(float(code_count))                                      # f42
        v.append(float(resource_count))                                  # f43

        # ═══════════════════════════════════════════════════════════
        # Group 12 — Method-level LLM reachability (5 dims)
        # ═══════════════════════════════════════════════════════════

        # In method_reachability_v9, f44-f47 count distinct reachable
        # source-sink method pairs. f48 counts distinct LLM route/schema
        # source methods whose local graph also reaches a custom base URL and
        # an HTTP sink (including bounded concrete request string flows).
        # These method-level values must not be mixed with the
        # removed file-level implementation.
        # Reachability counts can grow combinatorially around a shared HTTP
        # helper in a large app.  Preserve ordering while preventing one APK
        # from dominating training solely because its helper graph is large.
        v.append(math.log1p(int(cg.get("api_domain_near_http", 0) or 0)))       # f44
        v.append(math.log1p(int(cg.get("webview_near_llm_domain", 0) or 0)))    # f45
        v.append(math.log1p(int(cg.get("native_near_jni", 0) or 0)))            # f46
        v.append(math.log1p(int(cg.get("json_schema_near_http", 0) or 0)))      # f47
        v.append(math.log1p(int(cg.get("custom_domain_llm_near_http", 0) or 0))) # f48

        # ═══════════════════════════════════════════════════════════
        # Group 13 — Metadata & degradation (4 dims)
        # ═══════════════════════════════════════════════════════════

        v.append(math.log(max(int(metadata.get("size_bytes", 1)), 1)))   # f49
        entries = metadata.get("entries", 0) or 0
        v.append(math.log(max(int(entries), 1)))                         # f50
        app_java_files = lc.get("jadx_app_source_files")
        if app_java_files is None:
            # Backward compatibility for JSONL generated before the scoped
            # count was added.  New records always use the app-scoped value.
            app_java_files = lc.get("jadx_source_files", 0)
        v.append(float(int(app_java_files or 0)))                        # f51
        jadx_status = str(metadata.get("jadx_status") or "").lower()
        if jadx_status:
            jadx_ok = jadx_status in {"success", "partial"}
        else:
            # Backward compatibility for JSONL generated before status was
            # recorded.  New records use the explicit status above.
            jadx_ok = bool(metadata.get("jadx_available"))
        v.append(0.0 if jadx_ok else 1.0)                                # f52

        # Independent facts, not category rules or APK-wide conjunctions.
        # Configuration and reference never imply that weights are bundled.
        v.append(float(any(generation_configuration(s) for s in feature_signals))) # f53
        v.append(float(any(model_binary_artifact(s) for s in feature_signals)))    # f54
        v.append(float(has(SignalType.MODEL_FILE_REFERENCE)))                     # f55
        v.append(float(any(generation_interface(s) for s in feature_signals)))    # f56
        v.append(float(any(generation_interface(s, call=True) for s in feature_signals))) # f57

        # Preserve reference vs. interface-use semantics without adding raw
        # matches or requiring a complete application call chain.
        v.append(float(any(programmatic_llm_usage(s) for s in feature_signals)))  # f58
        v.append(float(any(llm_operation_reference(s) for s in feature_signals))) # f59

        return v

    def transform(self,
                  records: Sequence[dict[str, object]],
                  ) -> tuple[list[list[float]], list[str]]:
        X = [self.vectorize_one(
            r.get("signals", []),
            r.get("metadata", {}),
        ) for r in records]
        return X, self.feature_names

    def transform_to_numpy(self,
                           records: Sequence[dict[str, object]],
                           ) -> tuple[object, list[str]]:
        import numpy as np  # type: ignore
        X, names = self.transform(records)
        return np.array(X, dtype=np.float32), names

    # ── feature name generation ────────────────────────────────────

    def _iter_names(self):
        yield from (
            # Group 1 — Signal density
            "f01_total_signal_count",
            "f02_unique_signal_types",
            "f03_special_token_count",
            # Group 2 — SDK (split by category)
            "f04_official_sdk_present",
            "f05_official_sdk_count",
            "f06_self_hosted_sdk_present",
            "f07_self_hosted_sdk_count",
            # Group 3 — Domains & endpoints
            "f08_api_domain_present",
            "f09_api_domain_count",
            "f10_web_domain_present",
            "f11_web_domain_count",
            "f12_backend_proxy_present",
            "f13_backend_proxy_count",
            "f14_serverless_url_present",
            "f15_serverless_url_count",
            "f16_business_ai_endpoint_present",
            "f17_business_ai_endpoint_count",
            # Group 4 — Credentials & config
            "f18_llm_apikey_present",
            "f19_baseurl_override_present",
            "f20_json_schema_present",
            "f21_prompt_template_present",
            "f22_official_api_no_json_schema",
            "f23_model_name_present",
            "f24_model_name_count",
            "f25_api_key_input_prompt_present",
            # Group 5 — Local engine
            "f26_model_file_present",
            "f27_model_file_count",
            "f28_native_library_present",
            "f29_native_library_count",
            "f30_native_function_present",
            "f31_native_function_count",
            "f32_framework_keyword_present",
            "f33_framework_keyword_count",
            "f34_weak_runtime_present",
            "f35_onnx_tflite_count",
            # Group 6 — Assets & delivery
            "f36_dynamic_download_present",
            "f37_license_present",
            # Group 7 — WebView
            "f38_web_container_present",
            "f39_web_navigation_call_present",
            "f40_web_llm_relation_present",
            # Group 8 — Manifest
            "f41_has_aicore",
            # Group 9 — Signal source layer
            "f42_code_sourced_signal_count",
            "f43_resource_sourced_signal_count",
            # Group 10 — Method-level LLM reachability
            "f44_reachable_api_http_pairs",
            "f45_reachable_web_navigation_llm_pairs",
            "f46_reachable_native_jni_pairs",
            "f47_reachable_schema_http_pairs",
            "f48_reachable_custom_http_pairs",
            # Group 11 — Metadata & degradation
            "f49_apk_size_log",
            "f50_zip_entries_log",
            "f51_jadx_source_file_count",
            "f52_jadx_failed",
            "f53_generation_config_present",
            "f54_model_binary_artifact_present",
            "f55_model_file_reference_present",
            "f56_on_device_generation_interface_present",
            "f57_on_device_generation_call_present",
            "f58_programmatic_llm_usage_present",
            "f59_llm_operation_reference_present",
        )
