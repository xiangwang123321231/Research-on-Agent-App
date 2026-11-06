"""Collect research-oriented LLM technology evidence outside feature input.

The versioned ``RawSignal`` stream is the input to the versioned
classifier. This module records additional, structured observations for
post-classification empirical analysis. Its output must never be appended to
``signals`` or consumed by ``FeatureVectorizer``.
"""

from __future__ import annotations

import re
import struct
import zipfile
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import BinaryIO, Iterable
from urllib.parse import urlsplit, urlunsplit

from signal_types import RawSignal, SignalType
from registry import (
    API_SERVICE_REGISTRY,
    SDK_REGISTRY,
    RUNTIME_REGISTRY,
    EVIDENCE_REGISTRY_VERSION,
    text_contains_sdk_package,
    extract_llm_model_ids,
    iter_model_configuration_values,
    normalize_llm_model_id,
    is_reference_only_source,
    match_api_service,
    match_web_service,
)


SCHEMA_VERSION = "research-evidence-v3"

_MAX_LOCATIONS = 20
_MAX_REFERENCED_TEXT_FILES = 250
_MAX_TEXT_FILE_BYTES = 2 * 1024 * 1024
_MAX_CONFIG_BYTES = 2 * 1024 * 1024


# The first matching entry wins.  Keep service-specific domains before more
# generic cloud domains.
_API_DOMAIN_MARKERS: tuple[str, ...] = tuple(sorted({
    marker
    for entry in API_SERVICE_REGISTRY
    for marker in (*entry.exact_hosts, *entry.host_suffixes)
}))

_SDK_INFO: tuple[tuple[str, str | None, str], ...] = tuple(
    (entry.package, entry.provider, entry.kind) for entry in SDK_REGISTRY
)

_GENERIC_SDK_TOKENS: dict[str, tuple[str | None, str]] = {
    "openai": ("openai_compatible", "code_token"),
    "anthropic": ("anthropic", "code_token"),
    "gemini": ("google_gemini", "code_token"),
    "vertexai": ("google_vertex_ai", "code_token"),
}

_RUNTIME_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (entry.runtime_family, entry.markers) for entry in RUNTIME_REGISTRY
)

_NON_GENERATIVE_MODEL_ID_RE = re.compile(
    r"(?ix)(?<![a-z0-9])(?:"
    r"whisper-(?:[a-z0-9][a-z0-9._-]{1,60})|"
    r"text-embedding-(?:[a-z0-9][a-z0-9._-]{1,60})|"
    r"(?:bge|e5)[-_.](?:reranker|embed)[a-z0-9._-]{0,60}"
    r")(?![a-z0-9])"
)

_GOOGLE_MODEL_PATH_RE = re.compile(
    r"(?i)/models/([a-z0-9][a-z0-9._-]{1,100})(?::[a-z]+)?"
)

_HF_URL_RE = re.compile(
    r"(?i)https?://(?:www\.)?huggingface\.co/"
    r"([a-z0-9_.-]+)/([a-z0-9_.-]+)"
    r"(?:/resolve/([^/\s\"'?#]+)/([^\s\"'?#]+))?"
)

_URL_CANDIDATE_RE = re.compile(
    r"(?i)(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}"
    r"(?::\d+)?(?:/[^\s\x00\"'<>]{0,400})?"
)

_TEXT_SUFFIXES = {
    ".java", ".kt", ".json", ".xml", ".txt", ".properties",
    ".yaml", ".yml", ".js", ".dart", ".html", ".htm", ".env",
}

_STRONG_MODEL_ARTIFACT_SUFFIXES = (".gguf", ".ggml", ".litertlm")
_CANDIDATE_MODEL_ARTIFACT_SUFFIXES = (".safetensors", ".pte", ".task", ".mnn")
_CONTEXTUAL_MODEL_ARTIFACT_SUFFIXES = (".onnx", ".ort", ".tflite")
_ARTIFACT_CONTEXT_HINTS = (
    "llm", "language_model", "generative", "llama", "qwen", "gemma",
    "mistral", "deepseek", "phi", "gpt", "chat", "tokenizer",
)

_GGUF_KEEP_KEYS = {
    "general.name",
    "general.architecture",
    "general.basename",
    "general.finetune",
    "general.size_label",
    "general.file_type",
    "general.quantization_version",
}


class ResearchEvidenceCollector:
    """Collect bounded, structured observations for later research analysis."""

    def collect(
        self,
        apk_path: str | Path,
        signals: Iterable[RawSignal | dict[str, object]],
        apktool_dir: Path | None = None,
        jadx_dir: Path | None = None,
    ) -> dict[str, object]:
        self._apk_path = Path(apk_path)
        self._roots = [
            ("apktool", apktool_dir),
            ("jadx", jadx_dir),
        ]
        self._indexes: dict[str, dict[str, dict[str, object]]] = {
            "remote_endpoints": {},
            "sdk_references": {},
            "model_candidates": {},
            "webview_targets": {},
            "local_runtime_candidates": {},
            "local_model_artifacts": {},
            "model_download_sources": {},
            "credential_indicators": {},
            "configuration_hints": {},
        }
        self._referenced_files: set[Path] = set()
        self._errors: list[str] = []
        self._stats = {
            "referenced_text_files_scanned": 0,
            "referenced_text_files_skipped": 0,
            "apk_config_files_scanned": 0,
            "apk_entries_inspected": 0,
        }

        for signal in signals:
            self._collect_signal(signal)

        self._collect_apk_entries()
        self._scan_referenced_files()

        result: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "registry_version": EVIDENCE_REGISTRY_VERSION,
            "status": "partial" if self._errors else "success",
        }
        for name, index in self._indexes.items():
            result[name] = self._finalize_index(index)
        result["collection_stats"] = self._stats
        if self._errors:
            result["errors"] = self._errors[:10]
        return result

    # ------------------------------------------------------------------
    # Existing neutral signals
    # ------------------------------------------------------------------

    def _collect_signal(self, signal: RawSignal | dict[str, object]) -> None:
        stype, value, layers, sources, role, scope = self._signal_parts(signal)
        locations = self._locations(layers, sources)
        text = value or ""

        for source in sources:
            self._remember_text_source(source)

        if stype == SignalType.KNOWN_LLM_API_DOMAIN.value:
            for provider, domain, endpoint in self._extract_api_endpoints(text):
                for location in locations:
                    self._add_endpoint(provider, domain, endpoint, "api_domain_signal", location)

        if stype == SignalType.KNOWN_LLM_SDK_PACKAGE.value:
            self._collect_sdk_reference(text, locations)

        if stype == SignalType.KNOWN_LLM_MODEL_NAME.value:
            evidence_role = (
                "configured" if role == "config" or scope == "config"
                else "referenced"
            )
            for model_id in self._extract_model_ids(text):
                for location in locations:
                    self._add_model(model_id, evidence_role, location)

        if stype in {
            SignalType.KNOWN_LLM_WEB_DOMAIN.value,
            SignalType.WEBVIEW_LLM_TARGET.value,
            SignalType.WEBVIEW_CHAT_PAGE.value,
        }:
            self._collect_web_targets(text, locations, stype)

        if stype in {
            SignalType.NATIVE_LLM_LIBRARY.value,
            SignalType.NATIVE_LLM_FUNCTION.value,
            SignalType.LLM_FRAMEWORK_KEYWORD.value,
            SignalType.ONNX_TFLITE_RUNTIME.value,
        }:
            runtime_text = " ".join([text, *sources])
            self._collect_runtime(runtime_text, locations, stype)

        if stype == SignalType.MODEL_FILE_EXTENSION.value:
            path, size = self._split_model_file_signal(text)
            if path:
                self._add_artifact(
                    path=path,
                    model_format=self._artifact_format(path),
                    uncompressed_size=size,
                    compressed_size=None,
                    crc32=None,
                    evidence_strength="high",
                    metadata=None,
                )

        if stype == SignalType.DYNAMIC_DOWNLOAD_HINT.value:
            for location in locations:
                self._add_configuration_hint("runtime_model_download", location)
            self._collect_hugging_face_urls(text, locations)

        if stype == SignalType.ASSET_DELIVERY_HINT.value:
            for location in locations:
                self._add_configuration_hint("play_asset_delivery", location)

        if stype == SignalType.BASEURL_OVERRIDE.value:
            for location in locations:
                self._add_configuration_hint("custom_base_url", location)

        if stype == SignalType.API_KEY_INPUT_PROMPT.value:
            for location in locations:
                self._add_configuration_hint("user_supplied_api_key", location)

        if stype == SignalType.HARDCODED_API_KEY.value:
            provider = self._credential_provider(text)
            for location in locations:
                self._add_credential(provider, location)

    @staticmethod
    def _signal_parts(
        signal: RawSignal | dict[str, object],
    ) -> tuple[str, str, list[str], list[str], str, str]:
        if isinstance(signal, RawSignal):
            return (
                signal.signal_type.value,
                str(signal.value or ""),
                [str(signal.layer or "unknown")],
                [str(signal.source_file)] if signal.source_file else [],
                signal.role.value,
                signal.scope.value,
            )
        stype = str(signal.get("signal_type", ""))
        value = str(signal.get("value", "") or "")
        layers = [str(v) for v in (signal.get("layers", []) or [])]
        if not layers and signal.get("layer"):
            layers = [str(signal["layer"])]
        sources = [str(v) for v in (signal.get("source_files", []) or [])]
        if not sources and signal.get("source_file"):
            sources = [str(signal["source_file"])]
        return (
            stype,
            value,
            layers or ["unknown"],
            sources,
            str(signal.get("role", "") or ""),
            str(signal.get("scope", "") or ""),
        )

    # ------------------------------------------------------------------
    # APK artifacts and small configuration files
    # ------------------------------------------------------------------

    def _collect_apk_entries(self) -> None:
        try:
            with zipfile.ZipFile(self._apk_path) as zf:
                infos = [info for info in zf.infolist() if not info.is_dir()]
                context_parts = [info.filename.lower() for info in infos]
                directory_parts: dict[str, list[str]] = {}
                for info in infos:
                    directory = info.filename.replace("\\", "/").rpartition("/")[0].lower()
                    directory_parts.setdefault(directory, []).append(info.filename.lower())
                context_budget = 2 * 1024 * 1024
                for info in infos:
                    if context_budget <= 0 or not self._is_likely_config(
                        info.filename, info.file_size
                    ):
                        continue
                    try:
                        raw = zf.read(info)
                    except Exception:
                        continue
                    raw = raw[:context_budget]
                    context_budget -= len(raw)
                    decoded = self._decode_text(raw)
                    if decoded:
                        context_parts.append(decoded.lower())
                        directory = info.filename.replace("\\", "/").rpartition("/")[0].lower()
                        directory_parts.setdefault(directory, []).append(decoded.lower())
                archive_context = "\n".join(context_parts)

                for info in infos:
                    if info.is_dir():
                        continue
                    self._stats["apk_entries_inspected"] += 1
                    path = info.filename.replace("\\", "/")
                    lower = path.lower()

                    for runtime in self._runtimes_from_text(lower, strong_only=True):
                        self._add_runtime(
                            runtime,
                            "native_library_path",
                            {"layer": "zip", "source_file": f"apk/{path}"},
                        )

                    directory = path.rpartition("/")[0].lower()
                    sibling_context = "\n".join(directory_parts.get(directory, []))
                    artifact = self._artifact_candidate(
                        path, archive_context, sibling_context
                    )
                    if artifact:
                        model_format, evidence_strength = artifact
                        metadata = None
                        if model_format == "gguf":
                            try:
                                with zf.open(info) as stream:
                                    metadata = _read_gguf_metadata(stream)
                            except Exception:
                                metadata = None
                        elif model_format == "safetensors":
                            try:
                                with zf.open(info) as stream:
                                    metadata = _read_safetensors_metadata(stream)
                            except Exception:
                                metadata = None
                        self._add_artifact(
                            path=path,
                            model_format=model_format,
                            uncompressed_size=info.file_size,
                            compressed_size=info.compress_size,
                            crc32=f"{info.CRC:08x}",
                            evidence_strength=evidence_strength,
                            metadata=metadata,
                        )
                        if metadata and metadata.get("general.name"):
                            self._add_model(
                                str(metadata["general.name"]),
                                "artifact_metadata",
                                {"layer": "zip", "source_file": f"apk/{path}"},
                                allow_unrecognized=True,
                            )
                        for model_id in self._extract_model_ids(path):
                            self._add_model(
                                model_id,
                                "artifact_filename",
                                {"layer": "zip", "source_file": f"apk/{path}"},
                            )

                    if self._is_likely_config(path, info.file_size):
                        try:
                            raw = zf.read(info)
                            text = self._decode_text(raw)
                        except Exception:
                            continue
                        if text and self._has_research_context(path + "\n" + text):
                            self._stats["apk_config_files_scanned"] += 1
                            self._scan_source_content(
                                text,
                                {"layer": "zip", "source_file": f"apk/{path}"},
                            )
        except (OSError, zipfile.BadZipFile) as exc:
            self._errors.append(f"apk_zip_unavailable:{type(exc).__name__}")

    def _scan_referenced_files(self) -> None:
        files = sorted(self._referenced_files, key=lambda p: str(p).lower())
        for fp in files[:_MAX_REFERENCED_TEXT_FILES]:
            try:
                if not fp.is_file() or fp.stat().st_size > _MAX_TEXT_FILE_BYTES:
                    self._stats["referenced_text_files_skipped"] += 1
                    continue
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                self._stats["referenced_text_files_skipped"] += 1
                continue
            self._stats["referenced_text_files_scanned"] += 1
            self._scan_source_content(
                text,
                {"layer": self._source_layer(fp), "source_file": self._portable_source(fp)},
            )
        if len(files) > _MAX_REFERENCED_TEXT_FILES:
            self._stats["referenced_text_files_skipped"] += (
                len(files) - _MAX_REFERENCED_TEXT_FILES
            )

    def _scan_source_content(self, text: str, location: dict[str, str]) -> None:
        """Scan structured XML values without treating resource keys as data."""
        source = location.get("source_file", "")
        if Path(source).suffix.lower() != ".xml":
            self._scan_text_content(text, location)
            return

        values: list[str] = []
        try:
            root = ET.fromstring(text)
            for element in root.iter():
                if element.text and element.text.strip():
                    values.append(element.text.strip())
                if element.tail and element.tail.strip():
                    values.append(element.tail.strip())
                for raw_name, raw_value in element.attrib.items():
                    name = raw_name.rpartition("}")[2].lower()
                    if name not in {"name", "id"} and raw_value.strip():
                        values.append(raw_value.strip())
        except ET.ParseError:
            # APKTool output can contain malformed fragments.  Recover text
            # nodes only; scanning the raw tag would reintroduce resource-key
            # false positives such as ``name=\"gpt41mini\"``.
            values.extend(
                value.strip()
                for value in re.findall(r">([^<]+)<", text, re.DOTALL)
                if value.strip()
            )

        for value in dict.fromkeys(values):
            self._scan_text_content(value, location)

    def _scan_text_content(self, text: str, location: dict[str, str]) -> None:
        for provider, domain, endpoint in self._extract_api_endpoints(text):
            self._add_endpoint(provider, domain, endpoint, "text_reference", location)

        assigned: set[str] = set()
        for raw_assignment, start, end in iter_model_configuration_values(text):
            model_ids = self._extract_model_ids(raw_assignment)
            if not model_ids:
                model_ids = self._extract_model_ids(f'model="{raw_assignment}"')
            for model_id in model_ids:
                assigned.add(model_id.lower())
                self._add_model(
                    model_id,
                    "configured",
                    location,
                    context=self._span_context(text, start, end),
                )
        for match in _GOOGLE_MODEL_PATH_RE.finditer(text):
            model_id = match.group(1)
            if self._extract_model_ids(model_id):
                assigned.add(model_id.lower())
                self._add_model(
                    model_id,
                    "api_path",
                    location,
                    context=self._span_context(text, match.start(), match.end()),
                )

        for model_id in self._extract_model_ids(text):
            if model_id.lower() not in assigned:
                self._add_model(model_id, "referenced", location)

        self._collect_hugging_face_urls(text, [location])
        if not is_reference_only_source(location.get("source_file", "")):
            self._collect_sdk_reference(text, [location])

        lower = text.lower()
        if "webview" in lower or "loadurl" in lower:
            self._collect_web_targets(text, [location], "webview_text_context")

    # ------------------------------------------------------------------
    # Structured evidence aggregation
    # ------------------------------------------------------------------

    def _add_endpoint(
        self,
        provider: str,
        domain: str,
        endpoint: str,
        role: str,
        location: dict[str, str],
    ) -> None:
        service = match_api_service(endpoint, classification_only=False)
        key = f"{provider}|{domain}|{endpoint}"
        self._record(
            "remote_endpoints",
            key,
            {
                "provider": provider,
                "service_id": service.service_id if service else None,
                "service_class": service.service_class if service else None,
                "source_reference": service.source_reference if service else None,
                "domain": domain,
                "endpoint": endpoint,
                "evidence_strength": service.evidence_strength if service else "medium",
            },
            role,
            location,
        )

    def _collect_sdk_reference(
        self, text: str, locations: list[dict[str, str]],
    ) -> None:
        lower = text.lower()
        found = False
        for package, provider, kind in _SDK_INFO:
            if not text_contains_sdk_package(lower, package):
                continue
            entry = next(item for item in SDK_REGISTRY if item.package == package)
            if package in lower:
                found = True
                for location in locations:
                    self._record(
                        "sdk_references",
                        package,
                        {
                            "package_or_token": package,
                            "provider": provider,
                            "kind": kind,
                            "source_reference": entry.source_reference,
                            "evidence_strength": entry.evidence_strength,
                        },
                        "package_reference",
                        location,
                    )
                if kind == "on_device_sdk":
                    self._collect_runtime(package, locations, "on_device_sdk")
        if found:
            return
        token = lower.strip()
        if token in _GENERIC_SDK_TOKENS:
            provider, kind = _GENERIC_SDK_TOKENS[token]
            for location in locations:
                self._record(
                    "sdk_references",
                    token,
                    {"package_or_token": token, "provider": provider, "kind": kind},
                    "generic_code_token",
                    location,
                )

    def _add_model(
        self,
        raw_model_id: str,
        role: str,
        location: dict[str, str],
        allow_unrecognized: bool = False,
        context: str = "",
    ) -> None:
        raw = re.sub(r"\s+", " ", raw_model_id).strip()[:200]
        model_ids = self._extract_model_ids(raw)
        if model_ids:
            normalized = normalize_llm_model_id(model_ids[0])
            raw = model_ids[0]
        elif allow_unrecognized and raw:
            normalized = normalize_llm_model_id(raw)
        else:
            return
        self._record(
            "model_candidates",
            normalized,
            {
                "model_id": normalized,
                "normalized_model_id": normalized,
                "raw_value": raw,
                "model_family": self._model_family(normalized),
                "model_developer": self._model_developer(normalized),
                "capability_type": self._capability_type(normalized, context),
                "parameter_size": self._parameter_size(normalized),
                "quantization": self._quantization(normalized),
                "deployment_context": self._deployment_context(role),
                "evidence_strength": self._model_evidence_strength(role),
            },
            role,
            location,
        )

    def _collect_web_targets(
        self,
        text: str,
        locations: list[dict[str, str]],
        role: str,
    ) -> None:
        for match in _URL_CANDIDATE_RE.finditer(text):
            raw = match.group(0).rstrip(".,;:)]}")
            service = match_web_service(raw)
            if not service:
                continue
            parse_value = raw if raw.lower().startswith(("http://", "https://")) \
                else "https://" + raw
            host = (urlsplit(parse_value).hostname or "").lower()
            for location in locations:
                self._record(
                    "webview_targets",
                    f"{service.service_id}|{host}",
                    {
                        "service": service.service_id,
                        "provider": service.provider,
                        "domain": host,
                        "source_reference": service.source_reference,
                        "evidence_strength": service.evidence_strength,
                    },
                    role,
                    location,
                )

    def _collect_runtime(
        self,
        text: str,
        locations: list[dict[str, str]],
        role: str,
    ) -> None:
        for runtime in self._runtimes_from_text(text):
            for location in locations:
                self._add_runtime(runtime, role, location)

    def _add_runtime(
        self, runtime: str, role: str, location: dict[str, str],
    ) -> None:
        self._record(
            "local_runtime_candidates",
            runtime,
            {"runtime_family": runtime},
            role,
            location,
        )

    def _add_artifact(
        self,
        path: str,
        model_format: str,
        uncompressed_size: int | None,
        compressed_size: int | None,
        crc32: str | None,
        evidence_strength: str,
        metadata: dict[str, object] | None,
    ) -> None:
        normalized_path = path.replace("\\", "/")
        item = {
            "path": normalized_path,
            "format": model_format,
            "uncompressed_size_bytes": uncompressed_size,
            "compressed_size_bytes": compressed_size,
            "crc32": crc32,
            "evidence_strength": evidence_strength,
            "distribution": "bundled",
        }
        if metadata:
            item["format_metadata"] = metadata
        location = {"layer": "zip", "source_file": f"apk/{normalized_path}"}
        self._record(
            "local_model_artifacts",
            normalized_path.lower(),
            item,
            "bundled_artifact",
            location,
        )
        self._add_configuration_hint("bundled_model_artifact", location)

    def _collect_hugging_face_urls(
        self, text: str, locations: list[dict[str, str]],
    ) -> None:
        for match in _HF_URL_RE.finditer(text):
            owner, repo, revision, filename = match.groups()
            url = match.group(0).rstrip(".,;:)]}")
            key = "|".join([owner.lower(), repo.lower(), revision or "", filename or ""])
            for location in locations:
                self._record(
                    "model_download_sources",
                    key,
                    {
                        "host": "huggingface.co",
                        "repository": f"{owner}/{repo}",
                        "revision": revision,
                        "filename": filename,
                        "url": self._sanitize_url(url),
                    },
                    "model_repository_url",
                    location,
                )

    def _add_credential(self, provider: str, location: dict[str, str]) -> None:
        # Never persist the matched secret.  Public output needs only presence
        # and a best-effort provider family.
        self._record(
            "credential_indicators",
            provider,
            {"provider": provider, "credential_present": True, "value_saved": False},
            "hardcoded_credential_pattern",
            location,
        )

    def _add_configuration_hint(
        self, hint: str, location: dict[str, str],
    ) -> None:
        self._record(
            "configuration_hints",
            hint,
            {"hint": hint},
            "observed",
            location,
        )

    def _record(
        self,
        collection: str,
        key: str,
        base: dict[str, object],
        role: str,
        location: dict[str, str],
    ) -> None:
        index = self._indexes[collection]
        item = index.get(key)
        if item is None:
            item = dict(base)
            item.setdefault("canonical_id", key)
            item.setdefault("registry_version", EVIDENCE_REGISTRY_VERSION)
            item.setdefault("evidence_strength", self._role_evidence_strength(role))
            item["_roles"] = set()
            item["_location_keys"] = set()
            item["locations"] = []
            index[key] = item
        else:
            # The same fact can arrive first from an existing signal and then
            # from direct APK inspection.  Preserve the fact identity while
            # filling richer fields (for example ZIP CRC/size and GGUF header
            # metadata) that were unavailable to the signal extractor.
            for field, value in base.items():
                if value is None:
                    continue
                current = item.get(field)
                if field == "evidence_strength":
                    rank = {"context_required": 0, "candidate": 1,
                            "medium": 2, "high": 3}
                    if rank.get(str(value), -1) > rank.get(str(current), -1):
                        item[field] = value
                elif (current is None or current == "" or current == {}
                      or current == [] or current == "unknown"):
                    item[field] = value
                elif field == "format_metadata" and isinstance(current, dict) \
                        and isinstance(value, dict):
                    current.update({k: v for k, v in value.items() if v is not None})
        item["_roles"].add(role)  # type: ignore[union-attr]
        location_key = (location.get("layer", ""), location.get("source_file", ""))
        if location_key not in item["_location_keys"]:  # type: ignore[operator]
            item["_location_keys"].add(location_key)  # type: ignore[union-attr]
            if len(item["locations"]) < _MAX_LOCATIONS:  # type: ignore[arg-type]
                item["locations"].append(location)  # type: ignore[union-attr]

    @staticmethod
    def _role_evidence_strength(role: str) -> str:
        if role in {
            "configured", "api_path", "runtime_call", "artifact_metadata",
            "native_library_path", "hardcoded_credential_pattern",
            "package_reference", "bundled_artifact",
        }:
            return "high"
        if role in {
            "api_domain_signal", "on_device_sdk", "model_repository_url",
            "artifact_filename",
        }:
            return "medium"
        if role in {"text_reference", "generic_code_token"}:
            return "candidate"
        return "medium"

    @staticmethod
    def _finalize_index(index: dict[str, dict[str, object]]) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for key in sorted(index):
            item = dict(index[key])
            roles = item.pop("_roles")
            locations = item.pop("_location_keys")
            item["evidence_roles"] = sorted(roles)  # type: ignore[arg-type]
            item["location_count"] = len(locations)  # type: ignore[arg-type]
            result.append(item)
        return result

    # ------------------------------------------------------------------
    # Parsing and normalization helpers
    # ------------------------------------------------------------------

    def _locations(
        self, layers: list[str], sources: list[str],
    ) -> list[dict[str, str]]:
        if not sources:
            return [{"layer": layers[0] if layers else "unknown", "source_file": ""}]
        layer = layers[0] if len(layers) == 1 else "+".join(sorted(set(layers)))
        return [
            {"layer": layer, "source_file": self._portable_source(Path(source))}
            for source in sources[:_MAX_LOCATIONS]
        ]

    def _remember_text_source(self, source: str) -> None:
        if not source:
            return
        fp = Path(source)
        if fp.suffix.lower() not in _TEXT_SUFFIXES or not fp.is_absolute():
            return
        resolved = self._within_analysis_roots(fp)
        if resolved is not None:
            self._referenced_files.add(resolved)

    def _within_analysis_roots(self, fp: Path) -> Path | None:
        try:
            resolved = fp.resolve()
        except OSError:
            return None
        for _, root in self._roots:
            if root is None:
                continue
            try:
                resolved.relative_to(root.resolve())
                return resolved
            except (OSError, ValueError):
                continue
        return None

    def _portable_source(self, fp: Path) -> str:
        if not fp.is_absolute():
            return f"apk/{str(fp).replace(chr(92), '/')}"
        try:
            resolved = fp.resolve()
        except OSError:
            return fp.name
        for label, root in self._roots:
            if root is None:
                continue
            try:
                relative = resolved.relative_to(root.resolve())
                return f"{label}/{relative.as_posix()}"
            except (OSError, ValueError):
                continue
        return fp.name

    def _source_layer(self, fp: Path) -> str:
        try:
            resolved = fp.resolve()
        except OSError:
            return "unknown"
        for label, root in self._roots:
            if root is None:
                continue
            try:
                resolved.relative_to(root.resolve())
                return label
            except (OSError, ValueError):
                continue
        return "unknown"

    @staticmethod
    def _extract_api_endpoints(text: str) -> list[tuple[str, str, str]]:
        found: dict[tuple[str, str, str], None] = {}
        for match in _URL_CANDIDATE_RE.finditer(text):
            raw = match.group(0).rstrip(".,;:)]}")
            parse_value = raw if raw.lower().startswith(("http://", "https://")) else "https://" + raw
            try:
                parsed = urlsplit(parse_value)
            except ValueError:
                continue
            host = (parsed.hostname or "").lower().strip(".")
            service = match_api_service(raw, classification_only=False)
            if not service:
                continue
            endpoint = ResearchEvidenceCollector._sanitize_url(raw)
            found[(service.provider, host, endpoint)] = None
        return list(found)

    @staticmethod
    def _provider_for_host(host: str) -> str | None:
        service = match_api_service(host, classification_only=False)
        return service.provider if service else None

    @staticmethod
    def _sanitize_url(raw: str) -> str:
        has_scheme = raw.lower().startswith(("http://", "https://"))
        candidate = raw if has_scheme else "https://" + raw
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            return raw.split("?", 1)[0].split("#", 1)[0][:500]
        safe = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        return safe if has_scheme else safe.removeprefix("https://")

    @staticmethod
    def _extract_model_ids(text: str) -> list[str]:
        candidates = extract_llm_model_ids(text)
        candidates.extend(match.group(0) for match in _NON_GENERATIVE_MODEL_ID_RE.finditer(text))
        found: list[str] = []
        for candidate in candidates:
            value = candidate.rstrip("._-")
            # File names are useful evidence locations, but an artifact suffix
            # is not part of the model identifier itself.
            for suffix in (
                ".safetensors", ".litertlm", ".tflite", ".gguf", ".ggml",
                ".onnx", ".ort", ".task", ".mnn", ".pte",
            ):
                if value.lower().endswith(suffix):
                    value = value[:-len(suffix)].rstrip("._-")
                    break
            normalized = normalize_llm_model_id(value)
            if normalized and normalized not in found:
                found.append(normalized)
        return found

    @staticmethod
    def _capability_type(model_id: str, context: str = "") -> str:
        lower = model_id.lower()
        context_lower = context.lower()
        if "whisper" in lower:
            return "asr"
        if "embedding" in lower or "embed" in lower:
            return "embedding"
        if "rerank" in lower:
            return "reranker"
        if any(marker in lower for marker in ("vision", "vl", "multimodal")):
            return "multimodal_generation"
        if any(marker in lower for marker in ("tts", "speech")):
            return "tts"
        if any(marker in lower for marker in ("dall-e", "stable-diffusion", "flux-")):
            return "image_generation"
        if any(marker in context_lower for marker in (
            "image_url", "input_image", "inline_data", "pixel_values",
        )):
            return "multimodal_generation"
        if any(marker in context_lower for marker in (
            "/embeddings", "getembedding", "embedding.create", "embedcontent",
        )):
            return "embedding"
        if any(marker in context_lower for marker in ("/rerank", "rerank(")):
            return "reranker"
        if any(marker in context_lower for marker in (
            "/audio/transcriptions", "speech_to_text", "transcribe(",
        )):
            return "asr"
        if any(marker in context_lower for marker in (
            "/audio/speech", "text_to_speech", "synthesize(",
        )):
            return "tts"
        if any(marker in context_lower for marker in (
            "/images/generations", "image_generation", "generateimage",
        )):
            return "image_generation"
        if any(marker in context_lower for marker in (
            "chat/completions", "responses.create", "/v1/responses",
            "generatecontent", "chat_template", "max_tokens", "temperature",
        )):
            return "text_generation"
        return "unknown"

    @staticmethod
    def _parameter_size(model_id: str) -> str | None:
        match = re.search(r"(?i)(?<![a-z0-9])(\d+(?:\.\d+)?)\s*b(?![a-z])", model_id)
        return f"{match.group(1)}B" if match else None

    @staticmethod
    def _quantization(model_id: str) -> str | None:
        match = re.search(
            r"(?i)(?<![a-z0-9])(?:q[2-8](?:_[kms]+)?|int[248]|fp(?:16|8)|bf16)(?![a-z0-9])",
            model_id,
        )
        return match.group(0).upper() if match else None

    @staticmethod
    def _deployment_context(role: str) -> str:
        if role in {"artifact_metadata", "artifact_filename"}:
            return "bundled"
        if role == "api_path":
            return "remote_api"
        return "unknown"

    @staticmethod
    def _model_evidence_strength(role: str) -> str:
        if role in {"configured", "api_path", "artifact_metadata"}:
            return "high"
        if role == "artifact_filename":
            return "medium"
        return "candidate"

    @staticmethod
    def _model_family(model_id: str) -> str | None:
        lower = model_id.lower().rpartition("/")[2]
        for family, prefixes in (
            ("gpt", ("gpt-", "gpt_")),
            ("claude", ("claude-", "claude_")),
            ("gemini", ("gemini-", "gemini_")),
            ("deepseek", ("deepseek-",)),
            ("qwen", ("qwen",)),
            ("llama", ("llama", "meta-llama", "meta.llama")),
            ("mistral", ("mistral", "mixtral")),
            ("gemma", ("gemma",)),
            ("phi", ("phi",)),
            ("command-r", ("command-r",)),
            ("grok", ("grok-", "grok_")),
            ("glm", ("glm-", "glm_")),
        ):
            if lower.startswith(prefixes):
                return family
        return None

    @staticmethod
    def _model_developer(model_id: str) -> str | None:
        family = ResearchEvidenceCollector._model_family(model_id)
        return {
            "gpt": "openai",
            "claude": "anthropic",
            "gemini": "google",
            "deepseek": "deepseek",
            "qwen": "alibaba",
            "llama": "meta",
            "mistral": "mistral_ai",
            "gemma": "google",
            "phi": "microsoft",
            "command-r": "cohere",
            "grok": "xai",
            "glm": "zhipu_ai",
        }.get(family)

    @staticmethod
    def _runtimes_from_text(text: str, strong_only: bool = False) -> list[str]:
        lower = text.lower()
        found: list[str] = []
        for entry in RUNTIME_REGISTRY:
            if strong_only and entry.weak_path_identity:
                continue
            if any(marker in lower for marker in entry.markers):
                found.append(entry.runtime_family)
        return found

    @staticmethod
    def _runtime_from_text(text: str, strong_only: bool = False) -> str | None:
        """Compatibility helper returning the first matched runtime."""
        runtimes = ResearchEvidenceCollector._runtimes_from_text(text, strong_only)
        return runtimes[0] if runtimes else None

    @staticmethod
    def _span_context(text: str, start: int, end: int, radius: int = 512) -> str:
        return text[max(0, start - radius):min(len(text), end + radius)]

    @staticmethod
    def _credential_provider(value: str) -> str:
        lower = value.lower()
        if lower.startswith("sk-ant-"):
            return "anthropic"
        if lower.startswith("gsk_"):
            return "groq"
        if lower.startswith("xai-"):
            return "xai"
        if lower.startswith("pplx-"):
            return "perplexity"
        if lower.startswith("r8_"):
            return "replicate"
        if lower.startswith("hf_"):
            return "hugging_face"
        if lower.startswith("aiza"):
            return "google_api_unspecified"
        if lower.startswith("sk-"):
            return "openai_compatible"
        return "unknown"

    @staticmethod
    def _split_model_file_signal(value: str) -> tuple[str | None, int | None]:
        path, separator, size_text = value.rpartition("::")
        if not separator:
            return (value or None), None
        try:
            size = int(size_text)
        except ValueError:
            size = None
        return path or None, size

    @staticmethod
    def _artifact_format(path: str) -> str:
        lower = path.lower()
        for suffix in (
            ".safetensors", ".litertlm", ".tflite", ".gguf", ".ggml",
            ".onnx", ".ort", ".task", ".mnn", ".pte",
        ):
            if lower.endswith(suffix):
                return suffix[1:]
        return "unknown"

    @staticmethod
    def _artifact_candidate(
        path: str,
        archive_context: str = "",
        sibling_context: str = "",
    ) -> tuple[str, str] | None:
        lower = path.lower()
        if lower.endswith(_STRONG_MODEL_ARTIFACT_SUFFIXES):
            return ResearchEvidenceCollector._artifact_format(path), "high"
        local_context = (lower + "\n" + sibling_context).lower()
        global_context = (local_context + "\n" + archive_context).lower()
        has_language_context = (
            bool(extract_llm_model_ids(local_context))
            or any(hint in local_context for hint in (
                "tokenizer.json", "tokenizer_config.json", "chat_template",
                "generation_config.json", "mlc-chat-config.json",
                "sentencepiece", "litertlm",
            ))
        )
        if lower.endswith(".safetensors") and has_language_context:
            return "safetensors", "candidate"
        if lower.endswith(".pte") and "executorch" in global_context and has_language_context:
            return "pte", "context_required"
        if lower.endswith(".task") and any(
            marker in global_context for marker in ("mediapipe", "litertlm", "genai")
        ) and has_language_context:
            return "task", "context_required"
        if lower.endswith(".mnn") and any(
            marker in global_context for marker in ("mnn_llm", "mnnllm", "libmnn")
        ) and has_language_context:
            return "mnn", "context_required"
        if lower.endswith(_CONTEXTUAL_MODEL_ARTIFACT_SUFFIXES) and has_language_context:
            runtime_ok = (
                lower.endswith((".onnx", ".ort")) and "onnxruntime" in global_context
            ) or (
                lower.endswith(".tflite")
                and any(marker in global_context for marker in ("litertlm", "mediapipe", "genai"))
            )
            if runtime_ok:
                return ResearchEvidenceCollector._artifact_format(path), "context_required"
        return None

    @staticmethod
    def _is_likely_config(path: str, size: int) -> bool:
        if size <= 0 or size > _MAX_CONFIG_BYTES:
            return False
        lower = path.lower()
        name = Path(lower).name
        if Path(name).suffix not in _TEXT_SUFFIXES:
            return False
        explicit = {
            "config.json", "mlc-chat-config.json", "tokenizer.json",
            "tokenizer_config.json", "generation_config.json", "model_market.json",
        }
        return name in explicit or any(
            hint in name for hint in ("model", "llm", "chat", "generative")
        )

    @staticmethod
    def _has_research_context(text: str) -> bool:
        lower = text.lower()
        return (
            any(marker in lower for marker in _API_DOMAIN_MARKERS)
            or bool(ResearchEvidenceCollector._extract_model_ids(lower))
            or bool(_HF_URL_RE.search(lower))
            or any(hint in lower for hint in _ARTIFACT_CONTEXT_HINTS)
        )

    @staticmethod
    def _decode_text(raw: bytes) -> str:
        for encoding in ("utf-8", "utf-16-le", "latin-1"):
            try:
                text = raw.decode(encoding)
            except UnicodeError:
                continue
            printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text)
            if printable / max(len(text), 1) >= 0.7:
                return text
        return ""


class _GgufBudgetExceeded(Exception):
    pass


def _read_safetensors_metadata(stream: BinaryIO) -> dict[str, object] | None:
    """Read only the bounded SafeTensors JSON header, never tensor bytes."""
    try:
        raw_size = stream.read(8)
        if len(raw_size) != 8:
            return None
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size <= 1 or header_size > 2 * 1024 * 1024:
            return {"parse_status": "header_size_rejected"}
        raw_header = stream.read(header_size)
        if len(raw_header) != header_size:
            return None
        header = json.loads(raw_header.decode("utf-8"))
        if not isinstance(header, dict):
            return None
        tensors = [value for key, value in header.items()
                   if key != "__metadata__" and isinstance(value, dict)]
        dtypes = sorted({str(value.get("dtype")) for value in tensors
                         if value.get("dtype")})
        result: dict[str, object] = {
            "parse_status": "success",
            "tensor_count": len(tensors),
            "dtypes": dtypes[:20],
        }
        metadata = header.get("__metadata__")
        if isinstance(metadata, dict):
            result["metadata"] = {
                str(key)[:100]: str(value)[:500]
                for key, value in list(metadata.items())[:20]
            }
        return result
    except (OSError, UnicodeError, json.JSONDecodeError, struct.error, ValueError):
        return None


class _BudgetReader:
    def __init__(self, stream: BinaryIO, budget: int = 4 * 1024 * 1024) -> None:
        self.stream = stream
        self.remaining = budget

    def read(self, size: int) -> bytes:
        if size < 0 or size > self.remaining:
            raise _GgufBudgetExceeded
        data = self.stream.read(size)
        if len(data) != size:
            raise EOFError
        self.remaining -= size
        return data

    def unpack(self, fmt: str) -> object:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]

    def string(self) -> str:
        size = int(self.unpack("<Q"))
        if size > 1024 * 1024:
            raise _GgufBudgetExceeded
        return self.read(size).decode("utf-8", errors="replace")


_GGUF_SCALAR_FORMATS: dict[int, str] = {
    0: "<B",   # uint8
    1: "<b",   # int8
    2: "<H",   # uint16
    3: "<h",   # int16
    4: "<I",   # uint32
    5: "<i",   # int32
    6: "<f",   # float32
    7: "<?",   # bool
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}


def _read_gguf_value(reader: _BudgetReader, value_type: int) -> object:
    if value_type in _GGUF_SCALAR_FORMATS:
        return reader.unpack(_GGUF_SCALAR_FORMATS[value_type])
    if value_type == 8:
        return reader.string()
    if value_type == 9:
        item_type = int(reader.unpack("<I"))
        count = int(reader.unpack("<Q"))
        if item_type in _GGUF_SCALAR_FORMATS:
            size = struct.calcsize(_GGUF_SCALAR_FORMATS[item_type]) * count
            reader.read(size)
            return None
        if item_type == 8 and count <= 1000:
            for _ in range(count):
                reader.string()
            return None
        raise _GgufBudgetExceeded
    raise _GgufBudgetExceeded


def _read_gguf_metadata(stream: BinaryIO) -> dict[str, object] | None:
    """Read selected GGUF header fields without loading model tensors."""
    reader = _BudgetReader(stream)
    try:
        if reader.read(4) != b"GGUF":
            return None
        version = int(reader.unpack("<I"))
        if version not in {2, 3}:
            return {"gguf_version": version, "parse_status": "unsupported_version"}
        tensor_count = int(reader.unpack("<Q"))
        metadata_count = int(reader.unpack("<Q"))
        result: dict[str, object] = {
            "gguf_version": version,
            "tensor_count": tensor_count,
            "metadata_count": metadata_count,
            "parse_status": "success",
        }
        for _ in range(metadata_count):
            key = reader.string()
            value_type = int(reader.unpack("<I"))
            value = _read_gguf_value(reader, value_type)
            if key in _GGUF_KEEP_KEYS and value is not None:
                result[key] = value
        return result
    except _GgufBudgetExceeded:
        if "result" in locals():
            result["parse_status"] = "partial_budget_limit"
            return result
        return None
    except (EOFError, UnicodeError, struct.error, ValueError):
        return None
