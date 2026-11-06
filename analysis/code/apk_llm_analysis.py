"""APK LLM capability evidence extraction pipeline.

Orchestrates three independent extraction layers (ZIP / APKTool / JADX)
and aggregates their neutral signals via ``SignalAggregator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

from app_scope import AppCodeScope, expand_shared_app_scope, resolve_app_scope
from signal_types import (
    AnalysisResult,
    RawSignal,
    SignalAggregator,
    SignalRelation,
    SignalRole,
    SignalScope,
    SignalType,
    SignalVerification,
)
from registry import (
    EVIDENCE_REGISTRY_VERSION,
    SIGNAL_SCHEMA_VERSION,
    decode_manifest_xml,
    normalize_binary_manifest_text,
    extract_attr,
    dedupe_sequence,
)
from extractors import ZIPExtractor, APKToolExtractor, JADXExtractor
from webview_evidence import extract_webview_usage
from browser_navigation import extract_browser_navigation
from on_device_generation import extract_on_device_generation
from callgraph_analyzer import CallGraphAnalyzer
from research_evidence import ResearchEvidenceCollector, SCHEMA_VERSION
from feature_vectorizer import FEATURE_SCHEMA_VERSION
from tool_cache import fingerprint, read_manifest, write_manifest


@dataclass(frozen=True)
class _JadxOutcome:
    """Result of the JADX stage, including partial and unscoped outcomes.

    ``f52_jadx_failed`` is derived from ``usable`` downstream.  A partial
    decompilation with at least one application-owned source file remains
    usable for signal extraction and is therefore not treated as a hard
    failure.
    """

    output_dir: Path | None
    status: str
    app_java_files: int = 0
    total_java_files: int = 0
    return_code: int | None = None
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return (
            self.output_dir is not None
            and self.app_java_files > 0
            and self.status in {"success", "partial"}
        )


# This is a quality band, not a hard failure rule.  Apps with 1-9 source
# files are retained as partial JADX results instead of losing usable signals.
_JADX_HIGH_CONFIDENCE_APP_JAVA_FILES = 10
_JADX_TIMEOUT_SECONDS = 600
EXTRACTOR_VERSION = "apk-static-extractor-v2026.10.24.2"


@dataclass
class APKAnalyzer:
    """Orchestrate evidence extraction across three layers (ZIP / APKTool / JADX).

    Parameters
    ----------
    apktool_bin:  Path to apktool jar (or ``java -jar …``).
    jadx_bin:     Path to jadx executable.
    work_dir:     Root directory for extracted artefacts and caches.
    """

    apktool_bin: str = field(
        default_factory=lambda: os.environ.get(
            "APKTOOL_BIN", r"D:\work\analysis\tool\apktool_3.0.2.jar"))
    jadx_bin: str = field(
        default_factory=lambda: os.environ.get(
            "JADX_BIN", r"D:\work\analysis\tool\jadx\bin\jadx.bat"))
    work_dir: Path = field(
        default_factory=lambda: Path(r"D:\work\analysis\_analysis_work"))
    jadx_output_dir: Path | None = None
    _apktool_outcome: dict = field(default_factory=dict, init=False, repr=False)

    # ── main entry point ────────────────────────────────────────────

    def analyze(self, apk_path: str | os.PathLike[str]) -> AnalysisResult:
        path = Path(apk_path)
        if not path.exists():
            raise FileNotFoundError(f"APK not found: {path}")
        if path.suffix.lower() != ".apk":
            raise ValueError(f"Expected an .apk file, got: {path}")

        sha256 = self._sha256(path)
        metadata = self._collect_metadata(path)
        metadata.update({
            "extractor_version": EXTRACTOR_VERSION,
            "signal_schema_version": SIGNAL_SCHEMA_VERSION,
            "signal_registry_version": EVIDENCE_REGISTRY_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        })
        self.work_dir.mkdir(parents=True, exist_ok=True)

        layer_counts: dict[str, int] = {}
        all_signals: list = []

        # ── toolchain ──────────────────────────────────────────────
        extracted_dir = self._extract_with_apktool(path)
        metadata["apktool_analysis"] = dict(self._apktool_outcome) or {"status": "unavailable"}
        if not metadata.get("manifest_package") and extracted_dir:
            # Binary AXML decoding can fail in a long-running batch even when
            # APKTool has decoded the same manifest successfully. Recover the
            # app scope before deciding whether JADX output can be used.
            try:
                manifest_text = (extracted_dir / "AndroidManifest.xml").read_text(
                    encoding="utf-8", errors="ignore"
                )
                recovered = self._parse_manifest_text(manifest_text)
            except OSError:
                recovered = {}
            if recovered.get("manifest_package"):
                metadata.update(recovered)
                metadata["manifest_source"] = "apktool_decoded"
        app_scope = resolve_app_scope(
            metadata.get("manifest_package"),
            metadata.get("manifest_components"),
            metadata.get("manifest_launcher_components"),
            metadata.get("manifest_application_class"),
        )
        metadata["app_code_scope"] = app_scope.to_metadata()
        jadx_outcome = self._decompile_with_jadx(path, app_scope)
        jadx_dir = jadx_outcome.output_dir
        app_scope = expand_shared_app_scope(app_scope, jadx_dir)
        cg = CallGraphAnalyzer(extracted_dir)
        app_scope = cg.prepare_scope(app_scope, jadx_dir)
        metadata["app_code_scope"] = app_scope.to_metadata()
        if jadx_dir and (app_scope.shared_classes or app_scope.referenced_obfuscated_classes):
            jadx_outcome = replace(jadx_outcome, app_java_files=sum(
                1 for _ in app_scope.iter_jadx_files(jadx_dir, frozenset({".java"}))
            ))

        # ── Layer 1: ZIP ──────────────────────────────────────────
        zip_extractor = ZIPExtractor()
        zip_signals, zip_entries, zip_text_decoded = zip_extractor.extract(path)
        metadata["zip_analysis"] = zip_extractor.diagnostics
        all_signals.extend(zip_signals)
        layer_counts.update({
            "zip_entries": zip_entries,
            "zip_text_decoded": zip_text_decoded,
            "zip_signals": len(zip_signals),
        })

        if extracted_dir and extracted_dir.exists():
            metadata["apktool_output_dir"] = str(extracted_dir)

        # ── Layer 2: APKTool ──────────────────────────────────────
        apktool_signals = (
            APKToolExtractor().extract(extracted_dir) if extracted_dir else [])
        if extracted_dir:
            field_diagnostics: dict = {}
            apktool_signals.extend(extract_webview_usage(
                extracted_dir, app_scope, field_diagnostics=field_diagnostics))
            metadata["webview_field_analysis"] = field_diagnostics
            browser_diagnostics: dict = {}
            apktool_signals.extend(extract_browser_navigation(
                extracted_dir, app_scope, browser_diagnostics))
            metadata["browser_navigation_analysis"] = browser_diagnostics
        all_signals.extend(apktool_signals)
        layer_counts["apktool_signals"] = len(apktool_signals)

        from frontend_requests import extract_frontend_requests
        frontend_diagnostics: dict = {}
        frontend_signals = extract_frontend_requests(
            path, extracted_dir, app_scope, metadata, frontend_diagnostics)
        all_signals.extend(frontend_signals)
        layer_counts['frontend_signals'] = len(frontend_signals)
        layer_counts['zip_signals'] += sum(s.layer == 'zip' for s in frontend_signals)
        metadata['frontend_analysis'] = frontend_diagnostics

        # ── Layer 3: JADX ─────────────────────────────────────────
        layer_counts.update({
            # Keep the old key for compatibility, but define it explicitly as
            # all Java output.  f51 uses the app-scoped key below.
            "jadx_source_files": jadx_outcome.total_java_files,
            "jadx_app_source_files": jadx_outcome.app_java_files,
        })
        metadata.update({
            "jadx_available": jadx_outcome.usable,
            "jadx_status": jadx_outcome.status,
            "jadx_app_java_files": jadx_outcome.app_java_files,
            "jadx_total_java_files": jadx_outcome.total_java_files,
            "jadx_return_code": jadx_outcome.return_code,
        })
        if jadx_outcome.reason:
            metadata["jadx_status_reason"] = jadx_outcome.reason
            if jadx_outcome.status == "failed":
                metadata["jadx_failure_reason"] = jadx_outcome.reason

        if jadx_outcome.usable and jadx_dir and jadx_dir.exists():
            jadx_extractor = JADXExtractor()
            jadx_signals = jadx_extractor.extract(
                jadx_dir,
                app_scope,
                bool(metadata.get("has_assets")),
            )
            all_signals.extend(jadx_signals)
            metadata['jadx_signal_analysis'] = getattr(jadx_extractor, 'diagnostics', {})
        else:
            jadx_signals = []
        layer_counts['jadx_signals'] = len(jadx_signals)

        # Method-level local reachability is independent of JADX source
        # quality.  APKTool smali contains the original invoke-* edges and
        # remains usable when JADX only produces partial Java output.
        if extracted_dir and extracted_dir.exists():
            device_diagnostics: dict = {}
            all_signals.extend(extract_on_device_generation(extracted_dir, app_scope, device_diagnostics))
            metadata['on_device_generation'] = device_diagnostics
            callgraph_result = cg.analyze(app_scope=app_scope)
            independent = callgraph_result.pop('independent_code_signals', [])
            all_signals.extend(independent)
            layer_counts['smali_independent_signals'] = len(independent)
            all_signals.extend(callgraph_result.pop('http_protocol_signals', []))
            response_signals = callgraph_result.pop('response_protocol_signals', [])
            all_signals.extend(response_signals)
            layer_counts['smali_response_signals'] = len(response_signals)
            for relation in callgraph_result.pop("verified_custom_relations", []):
                all_signals.append(RawSignal(
                    SignalType.BUSINESS_AI_ENDPOINT,
                    str(relation.get("value") or "custom_llm_endpoint"),
                    "apktool",
                    str(relation.get("source_method") or "smali_method_graph"),
                    detail="verified_local_method_graph",
                    role=SignalRole.RELATION,
                    scope=SignalScope.LOCAL_METHOD_GRAPH,
                    relation=SignalRelation.CUSTOM_BASE_URL_LLM_PATH,
                    components=tuple(
                        str(item) for item in relation.get("components", [])
                    ),
                    verification=SignalVerification.VERIFIED,
                    source_kind="smali_value_flow",
                ))
            metadata["callgraph"] = callgraph_result
        else:
            metadata["callgraph"] = {
                "version": CallGraphAnalyzer.VERSION,
                "status": "unavailable",
                "api_domain_near_http": 0,
                "webview_near_llm_domain": 0,
                "native_near_jni": 0,
                "json_schema_near_http": 0,
                "custom_domain_llm_near_http": 0,
                "coverage": 0.0,
            }

        metadata["layer_coverage"] = layer_counts

        # Collect research-oriented facts while APKTool/JADX artifacts are
        # still available. This remains separate from all_signals so sidecar
        # enrichment cannot silently alter the versioned feature input.
        try:
            research_evidence = ResearchEvidenceCollector().collect(
                path,
                all_signals,
                apktool_dir=extracted_dir,
                jadx_dir=jadx_dir,
            )
        except Exception as exc:
            # Enrichment must never turn a successful classification-signal
            # extraction into a failed APK record.
            research_evidence = {
                "schema_version": SCHEMA_VERSION,
                "registry_version": EVIDENCE_REGISTRY_VERSION,
                "status": "failed",
                "error": type(exc).__name__,
            }

        return AnalysisResult(
            str(path), sha256, metadata, all_signals, research_evidence)

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    # ═══════════════════════════════════════════════════════════════
    # Metadata collection
    # ═══════════════════════════════════════════════════════════════

    def _collect_metadata(self, path: Path) -> dict[str, object]:
        metadata: dict[str, object] = {"size_bytes": path.stat().st_size}
        try:
            with zipfile.ZipFile(path) as zf:
                names = [i.filename for i in zf.infolist()]
                metadata.update({
                    "entries": len(names),
                    "has_manifest": any(n == "AndroidManifest.xml" for n in names),
                    "has_meta_inf": any(n.lower().startswith("meta-inf/") for n in names),
                    "has_assets": any(n.lower().startswith("assets/") for n in names),
                    "has_libs": any(n.lower().startswith("lib/") for n in names),
                })
                metadata.update(self._parse_manifest(zf))
        except zipfile.BadZipFile:
            metadata.update({
                "entries": None, "has_manifest": False, "has_meta_inf": False,
                "has_assets": False, "has_libs": False,
                "manifest_permissions": [], "manifest_components": [],
                "manifest_launcher_components": [],
                "manifest_application_class": None,
                "manifest_package": None, "manifest_network_security_config": None,
                "warning": "not_a_valid_zip_archive",
            })
        return metadata

    def _parse_manifest(self, zf: zipfile.ZipFile) -> dict[str, object]:
        try:
            raw = zf.read("AndroidManifest.xml")
        except KeyError:
            return self._parse_manifest_text("")
        return self._parse_manifest_text(decode_manifest_xml(raw))

    def _parse_manifest_text(self, xml_text: str) -> dict[str, object]:
        result: dict[str, object] = {
            "manifest_permissions": [], "manifest_components": [],
            "manifest_launcher_components": [],
            "manifest_application_class": None,
            "manifest_package": None, "manifest_network_security_config": None,
        }
        if not xml_text:
            return result
        xml_text = normalize_binary_manifest_text(xml_text)
        result["manifest_package"] = extract_attr(
            xml_text, r"package\s*=\s*['\"]([^'\"]+)['\"]")
        result["manifest_permissions"] = dedupe_sequence(
            self._extract_manifest_permissions(xml_text))
        result["manifest_components"] = dedupe_sequence(
            self._extract_manifest_components(xml_text))
        application_class, launchers = self._extract_manifest_entrypoints(xml_text)
        result["manifest_application_class"] = application_class
        result["manifest_launcher_components"] = dedupe_sequence(launchers)
        network_cfg = (
            extract_attr(xml_text, r"android:networkSecurityConfig\s*=\s*['\"]([^'\"]+)['\"]")
            or extract_attr(xml_text, r"networkSecurityConfig\s*=\s*['\"]([^'\"]+)['\"]")
        )
        result["manifest_network_security_config"] = network_cfg
        return result

    def _extract_manifest_permissions(self, text: str) -> list[str]:
        permissions: list[str] = []
        for tag in ("uses-permission", "uses-permission-sdk-23",
                      "permission", "uses-feature"):
            for block in re.finditer(rf"<{tag}\b[^>]*>", text, re.IGNORECASE):
                chunk = block.group(0)
                value = (extract_attr(chunk, r"android:name\s*=\s*['\"]([^'\"]+)['\"]")
                         or extract_attr(chunk, r"name\s*=\s*['\"]([^'\"]+)['\"]"))
                if value:
                    permissions.append(value)
        if "bind_aicore_service" in text.lower():
            permissions.append("android.permission.BIND_AICORE_SERVICE")
        return permissions

    @staticmethod
    def _extract_manifest_components(text: str) -> list[str]:
        components: list[str] = []
        for tag in ("activity", "activity-alias", "service", "receiver", "provider"):
            for block in re.finditer(rf"<{tag}\b[^>]*>", text, re.IGNORECASE):
                value = extract_attr(block.group(0),
                                     r"android:name\s*=\s*['\"]([^'\"]+)['\"]")
                if value:
                    components.append(value)
        return components

    @staticmethod
    def _extract_manifest_entrypoints(
        text: str,
    ) -> tuple[str | None, list[str]]:
        """Extract manifest anchors without accepting every bundled component."""
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return None, []

        android_ns = "{http://schemas.android.com/apk/res/android}"

        def local_name(tag: str) -> str:
            return tag.rsplit("}", 1)[-1]

        def android_attr(element: ET.Element, name: str) -> str | None:
            return (
                element.attrib.get(android_ns + name)
                or element.attrib.get("android:" + name)
                or element.attrib.get(name)
            )

        application = next(
            (element for element in root if local_name(element.tag) == "application"),
            None,
        )
        if application is None:
            return None, []

        application_class = android_attr(application, "name")
        launchers: list[str] = []
        for component in application:
            if local_name(component.tag) not in {"activity", "activity-alias"}:
                continue
            is_main = False
            is_launcher = False
            for intent_filter in component:
                if local_name(intent_filter.tag) != "intent-filter":
                    continue
                for item in intent_filter:
                    item_name = android_attr(item, "name") or ""
                    if (local_name(item.tag) == "action"
                            and item_name == "android.intent.action.MAIN"):
                        is_main = True
                    if (local_name(item.tag) == "category"
                            and item_name == "android.intent.category.LAUNCHER"):
                        is_launcher = True
            if is_main and is_launcher:
                value = (
                    android_attr(component, "targetActivity")
                    or android_attr(component, "name")
                )
                if value:
                    launchers.append(value)
        return application_class, launchers

    # ═══════════════════════════════════════════════════════════════
    # Toolchain (APKTool + JADX launchers + cache management)
    # ═══════════════════════════════════════════════════════════════

    def _extract_with_apktool(self, path: Path) -> Path | None:
        output_dir = self.work_dir / "apktool" / path.stem
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if (output_dir / "AndroidManifest.xml").is_file():
            self._apktool_outcome = {
                "status": "success" if (output_dir / "apktool.yml").is_file() else "partial",
                "return_code": None,
                "reason": "reused_existing_directory",
            }
            return output_dir
        key = fingerprint(self._sha256(path), "apktool", self.apktool_bin, ("decode", 600))
        cmd = ["java", "-jar", self.apktool_bin, "d", "-f",
               str(path), "-o", str(output_dir)]
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True, errors='replace', timeout=600)
            status = "success" if result.returncode == 0 else "partial"
            reason = None if result.returncode == 0 else "nonzero_exit"
            code = result.returncode
        except subprocess.TimeoutExpired:
            status, reason, code = "partial", "timeout", None
        except Exception as exc:
            status, reason, code = "failed", type(exc).__name__, None
        usable = (output_dir / "AndroidManifest.xml").is_file()
        if not usable:
            status = "failed"
        self._apktool_outcome = {"status": status, "return_code": code, "reason": reason}
        write_manifest(output_dir, key, self._apktool_outcome)
        return output_dir if usable else None

    def _decompile_with_jadx(
        self,
        path: Path,
        app_scope: AppCodeScope,
    ) -> _JadxOutcome:
        key = fingerprint(self._sha256(path), "jadx", self.jadx_bin, ("decompile", _JADX_TIMEOUT_SECONDS))
        output_dir = self.jadx_output_dir or (self.work_dir / "jadx" / path.stem)
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        # Version the marker so results produced by the old 10-file hard
        # threshold are re-evaluated under the new partial-result policy.
        _SKIP_MARKER = output_dir / "_jadx_skip_v3"
        # Failure markers never permanently suppress a retry.

        # JADXExtractor cannot safely scope source files without the package
        # name.  Avoid spending minutes decompiling an APK whose output could
        # not be used by the app-scoped extractor.
        if not app_scope.usable:
            return _JadxOutcome(
                None, "unscoped", reason="missing_app_code_scope")

        def _count_app_java(root: Path) -> int:
            return sum(
                1 for _ in app_scope.iter_jadx_files(
                    root, frozenset({".java"})
                )
            )

        def _count_total_java(root: Path) -> int:
            sources = root / "sources"
            if not sources.is_dir():
                return 0
            return sum(1 for _ in sources.rglob("*.java"))

        cached = read_manifest(output_dir, key)
        if cached and cached.get("status") in {"success", "partial"}:
            try:
                cached_java = _count_app_java(output_dir)
                cached_total = _count_total_java(output_dir)
            except Exception:
                cached_java = 0
                cached_total = 0
            if cached_java > 0:
                status = str(cached["status"])
                return _JadxOutcome(
                    output_dir, status, cached_java, cached_total,
                    return_code=cached.get("return_code"), reason=cached.get("reason"))

        # A stable directory name must not mix sources from different APK or
        # JADX versions when its cache manifest no longer matches.
        for name in ("sources", "resources"):
            generated = output_dir / name
            if generated.is_symlink():
                generated.unlink()
            elif generated.is_dir():
                if generated.resolve().parent != output_dir.resolve():
                    raise RuntimeError(f"JADX output escapes its cache directory: {generated}")
                shutil.rmtree(generated)

        # ── run jadx ──────────────────────────────────────────────
        cmd = [self.jadx_bin, "-d", str(output_dir), str(path)]
        jadx_stdout = ""
        jadx_stderr = ""
        use_shell = self.jadx_bin.lower().endswith((".bat", ".cmd"))
        try:
            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, errors='replace',
                timeout=_JADX_TIMEOUT_SECONDS,
                shell=use_shell,
            )
            jadx_stdout = result.stdout or ""
            jadx_stderr = result.stderr or ""
        except subprocess.TimeoutExpired:
            reason = f"jadx timed out after {_JADX_TIMEOUT_SECONDS} s"
            self._write_jadx_failure_log(output_dir, reason, "", "")
            app_java_count = _count_app_java(output_dir)
            total_java = _count_total_java(output_dir)
            if app_java_count:
                write_manifest(output_dir, key, {"status": "partial", "return_code": None, "reason": reason})
                return _JadxOutcome(output_dir, "partial", app_java_count, total_java, reason=reason)
            try: _SKIP_MARKER.touch()
            except Exception: pass
            return _JadxOutcome(None, "failed", reason=reason)
        except FileNotFoundError:
            reason = f"jadx binary not found: {self.jadx_bin}"
            self._write_jadx_failure_log(output_dir, reason, "", "")
            try: _SKIP_MARKER.touch()
            except Exception: pass
            return _JadxOutcome(None, "failed", reason=reason)
        except Exception as exc:
            reason = f"jadx unexpected error: {exc}"
            self._write_jadx_failure_log(output_dir, reason, "", "")
            try: _SKIP_MARKER.touch()
            except Exception: pass
            return _JadxOutcome(None, "failed", reason=reason)

        app_java_count = _count_app_java(output_dir)
        total_java = _count_total_java(output_dir)
        if app_java_count == 0:
            reason = (
                f"jadx produced no app-own .java files"
                f" ({total_java} total)"
            )
            if result.returncode != 0:
                reason += f"; exit code: {result.returncode}"
            self._write_jadx_failure_log(
                output_dir,
                reason,
                jadx_stdout, jadx_stderr)
            try: _SKIP_MARKER.touch()
            except Exception: pass
            return _JadxOutcome(
                None, "failed", 0, total_java, result.returncode, reason)

        status = (
            "success"
            if (
                result.returncode == 0
                and app_java_count >= _JADX_HIGH_CONFIDENCE_APP_JAVA_FILES
            )
            else "partial"
        )
        reason = None
        if result.returncode != 0:
            reason = f"jadx returned exit code {result.returncode}"
        elif app_java_count < _JADX_HIGH_CONFIDENCE_APP_JAVA_FILES:
            reason = (
                f"limited app-own source output: {app_java_count} files"
            )
        write_manifest(output_dir, key, {"status": status, "return_code": result.returncode, "reason": reason})
        return _JadxOutcome(
            output_dir, status, app_java_count, total_java,
            result.returncode, reason)
    @staticmethod
    def _write_jadx_failure_log(output_dir: Path, message: str,
                                 stdout: str, stderr: str) -> None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "_jadx_failure.log"
            lines = [message, "", "--- stdout ---", stdout, "--- stderr ---", stderr]
            log_path.write_text("\n".join(lines), encoding="utf-8", errors="ignore")
        except Exception:
            pass
