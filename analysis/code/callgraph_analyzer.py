"""Targeted, method-level call analysis for LLM source-to-sink evidence.

The primary analyzer reads APKTool smali output and builds a conservative
method graph from ``invoke-*`` instructions. It deliberately does not try to
reconstruct the complete Android call graph: only LLM-related source and
execution sink reachability is measured.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from pathlib import Path
import re
from typing import Iterator
from signal_types import SignalVerification
from browser_navigation import scan_browser_method

from app_scope import AppCodeScope, is_obfuscated_root_name, resolve_app_scope
from registry import (
    NATIVE_HINTS,
    is_litert_lm_native_target,
    litert_lm_native_symbols,
    classify_llm_route,
    extract_llm_schema_components,
    is_custom_service_url,
    is_plausible_api_route_literal,
    match_web_service,
    scan_url_for_business_ai_endpoint,
    text_contains_known_api,
)


_CLASS_RE = re.compile(r"^\s*\.class\b.*?\s+(L[^;\s]+;)\s*$")
_INVOKE_RE = re.compile(
    r"\binvoke-[\w/-]+\s+\{[^}]*\},\s*(L[^;\s]+;->[\S]+)",
    re.IGNORECASE,
)
_INVOKE_ANY_RE = re.compile(r"\binvoke-[\w/-]+\b", re.IGNORECASE)
_URL_RE = re.compile(
    r"https?://[^\s\"'<>\\]+|"
    r"[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?::\d+)?(?:[/][^\s\"']*)?",
    re.IGNORECASE,
)
_SMALI_CONST_STRING_RE = re.compile(
    r'\bconst-string(?:/jumbo)?\s+[^,]+,\s*"((?:\\.|[^"\\])*)"',
    re.IGNORECASE,
)
@dataclass(frozen=True)
class _Method:
    key: str
    owner: str
    name: str
    body: str
    app_owned: bool
    native: bool = False
    static: bool = False


@dataclass
class _MethodTags:
    api_source: bool = False
    web_source: bool = False
    native_source: bool = False
    schema_source: bool = False
    custom_source: bool = False
    custom_base: bool = False
    custom_route: bool = False
    custom_base_values: tuple[str, ...] = ()
    custom_route_values: tuple[str, ...] = ()
    schema_components: tuple[str, ...] = ()
    http_sink: bool = False
    webview_sink: bool = False
    native_sink: bool = False


class CallGraphAnalyzer:
    """Build an app-scoped smali method graph and search local paths."""

    VERSION = "method_reachability_v17"
    _PLATFORM_ROOTS = (
        "android/", "androidx/", "java/", "javax/", "kotlin/", "kotlinx/",
    )

    def __init__(self, apktool_dir: Path | None = None,
                 max_depth: int = 5) -> None:
        self.apktool_dir = Path(apktool_dir) if apktool_dir else None
        self.max_depth = max(1, int(max_depth))
        self._prepared: tuple[AppCodeScope, list[Path], list[Path], frozenset[str]] | None = None
        self._scope_limited = False

    def prepare_scope(self, scope: AppCodeScope, jadx_dir: Path | None = None) -> AppCodeScope:
        """Share the existing exact invoke-derived class selection with scanners.

        Cache selection so analysis does not repeat class discovery or expand
        another five hops from the already-expanded scope.
        """
        if not self.apktool_dir or not self.apktool_dir.is_dir() or not scope.usable:
            return scope
        files = list(self._iter_smali_files(self.apktool_dir))
        seeds = [p for p in files if scope.matches_smali_path(p, self.apktool_dir)]
        selected, owners = self._expand_obfuscated_app_files(seeds, files)
        classes = frozenset(owner[1:-1].replace("/", ".") for owner in owners)
        mapped: set[str] = set()
        if jadx_dir:
            sources = jadx_dir / "sources"
            for owner in sorted(classes):
                # JADX normally nests $ classes in the enclosing Java file.
                for name in (owner, owner.split("$", 1)[0]):
                    for suffix in (".java", ".kt"):
                        relative = name.replace(".", "/") + suffix
                        if (sources / relative).is_file():
                            mapped.add(relative)
                            break
                    else:
                        continue
                    break
        scope = replace(scope, referenced_obfuscated_classes=classes,
                        referenced_obfuscated_files=tuple(sorted(mapped)))
        self._prepared = (scope, files, selected, owners)
        return scope

    def analyze(
        self,
        app_package: str | None = None,
        manifest_components: object = None,
        *,
        app_scope: AppCodeScope | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "version": self.VERSION,
            "status": "unavailable",
            "graph_built": False,
            "skip_reason": None,
            "max_depth": self.max_depth,
            "api_domain_near_http": 0,
            "webview_near_llm_domain": 0,
            "native_near_jni": 0,
            "json_schema_near_http": 0,
            "custom_domain_llm_near_http": 0,
            "smali_files": 0,
            "graph_smali_files": 0,
            "graph_smali_files_limited": False,
            "app_smali_files": 0,
            "reachable_obfuscated_smali_files": 0,
            "method_count": 0,
            "method_count_complete": False,
            "app_method_count": 0,
            "edge_count": 0,
            "invoke_sites": 0,
            "resolved_invoke_sites": 0,
            "unresolved_invoke_sites": 0,
            "source_method_count": 0,
            "sink_method_count": 0,
            "candidate_source_method_count": 0,
            "candidate_sink_method_count": 0,
            "candidate_custom_base_method_count": 0,
            "candidate_custom_route_method_count": 0,
            "verified_custom_relations": [],
            "coverage": 0.0,
        }
        if not self.apktool_dir or not self.apktool_dir.exists():
            return result
        scope = app_scope or resolve_app_scope(app_package, manifest_components)
        if not scope.usable:
            result["status"] = "unscoped"
            return result

        methods: dict[str, _Method] = {}
        if self._prepared is None or self._prepared[0] != scope:
            scope = self.prepare_scope(scope)
        assert self._prepared is not None
        _, smali_files, app_smali_files, associated_owners = self._prepared
        result["smali_files"] = len(smali_files)
        if not smali_files:
            return result
        result["app_smali_files"] = len(app_smali_files)
        result["reachable_obfuscated_smali_files"] = len(associated_owners)
        if not app_smali_files:
            result["status"] = "skipped_no_app_scope"
            result["skip_reason"] = "no_app_owned_smali"
            return result

        # A large APK can contain hundreds of thousands of methods that are
        # unrelated to LLM execution.  First do a lightweight method-scoped
        # candidate scan and avoid retaining method bodies or call edges when
        # no source exists.  The exact graph pass below is unchanged for APKs
        # that do contain a candidate source.
        candidates = self._scan_candidates(
            app_smali_files, scope, associated_owners)
        result['independent_code_signals'] = candidates.pop('independent_code_signals', [])
        result['independent_code_analysis'] = candidates.pop('independent_code_analysis', {})
        from static_string_fields import recover_static_strings, FIELD_READ
        from webview_fields import without_comments
        static_diagnostics: dict = {}
        static_fields = recover_static_strings(
            self.apktool_dir, {}, static_diagnostics,
            wanted_fields=candidates.get("static_read_fields", ()))
        result["static_string_fields"] = static_diagnostics
        # Audited fields are candidate seeds only. Their actual receiver/request
        # binding must still be recovered by the concrete value interpreter.
        static_llm_fields = {key for key, value in static_fields.items() if any(
            signal.verification != SignalVerification.CANDIDATE
            for signal in scan_url_for_business_ai_endpoint(value, key, "smali"))}
        from smali_protocols import ProtocolIndex
        protocols = ProtocolIndex.build(app_smali_files, None, available_paths=smali_files)
        result["protocol_candidates"] = len(protocols.api_methods)
        result["http_body_protocol_candidates"] = sum(api.request_kind == 'body' for api in protocols.all_api_methods.values())
        result["related_http_protocol_candidates"] = len(protocols.candidate_api_methods)
        result["protocol_type_recoveries"] = protocols.recovered_type_count
        result["protocol_missing_type_count"] = len(protocols.missing_types)
        anchored_scope = replace(scope, referenced_obfuscated_classes=frozenset())
        protocol_sources = protocols.referenced_sources(anchored_scope.matches_owner, self.max_depth)
        result["http_protocol_signals"] = protocols.usage_signals(protocol_sources)
        result['response_protocol_signals'] = protocols.response_signals(protocol_sources)
        result["protocol_index_limited"] = protocols.limited
        result["scope_limited"] = self._scope_limited
        result.update({
            "method_count": candidates["method_count"],
            "app_method_count": candidates["app_method_count"],
            "source_method_count": candidates["source_method_count"],
            "sink_method_count": candidates["sink_method_count"],
            "candidate_source_method_count": candidates["source_method_count"],
            "candidate_sink_method_count": candidates["sink_method_count"],
            "candidate_custom_base_method_count": candidates["custom_base_method_count"],
            "candidate_custom_route_method_count": candidates["custom_route_method_count"],
            "parse_errors": candidates["parse_errors"],
        })
        if not candidates["method_count"]:
            result["status"] = "failed"
            return result
        # Do not skip after an incomplete candidate scan: a parse error could
        # have hidden a source method, so retain the conservative exact path.
        has_split_custom_candidate = bool(
            candidates["custom_base_method_count"]
            and (candidates["custom_route_method_count"]
                 or candidates["schema_method_count"])
            and candidates["http_sink_method_count"]
        )
        if (not candidates["source_method_count"]
                and not has_split_custom_candidate
                and not static_llm_fields
                and static_diagnostics.get("status") != "partial"
                and not result["http_protocol_signals"]
                and not protocols.limited and not protocols.parse_errors and not self._scope_limited
                and not candidates["parse_errors"]):
            result["status"] = "skipped_no_source"
            result["skip_reason"] = "no_llm_source_method"
            return result

        # Candidate discovery is intentionally app-scoped.  Once a source is
        # found, resolve only APK-internal owners reached from those app
        # methods instead of retaining every bundled dependency.  This keeps
        # method-level reachability useful on large Flutter/Android bundles
        # without changing the source/sink tagging rules.
        request_seeds = [*candidates.get('source_files', []),
                         *(signal.source_file.split('->', 1)[0]
                           for signal in result['http_protocol_signals'])]
        if not request_seeds:
            request_seeds = app_smali_files  # e.g. proven static field readers
        from request_code_index import select_request_files
        graph_files, selection_errors = select_request_files(
            smali_files, request_seeds, scope, associated_owners)
        graph_limited = bool(selection_errors)
        result['graph_selection'] = 'request_reference_closure'
        result['graph_selection_errors'] = selection_errors
        result["graph_smali_files"] = len(graph_files)
        result["graph_smali_files_limited"] = graph_limited

        parse_errors = 0

        for fp in graph_files:
            try:
                for method in self._parse_file(fp, scope, associated_owners):
                    methods[method.key] = method
            except (OSError, UnicodeError, ValueError):
                parse_errors += 1

        result["method_count"] = len(methods)
        result["method_count_complete"] = len(graph_files) == len(smali_files)
        result["app_method_count"] = sum(1 for m in methods.values() if m.app_owned)
        if not methods:
            result["status"] = "failed"
            return result

        edges: dict[str, set[str]] = defaultdict(set)
        invoke_sites = 0
        resolved_sites = 0
        tags: dict[str, _MethodTags] = {}
        for method in methods.values():
            body_lower = method.body.lower()
            targets = _INVOKE_RE.findall(method.body)
            invoke_sites += len(_INVOKE_ANY_RE.findall(method.body))
            for target in targets:
                edges[method.key].add(target)
                if target in methods:
                    resolved_sites += 1
            tags[method.key] = self._tag_method(method, body_lower, targets)
            if method.app_owned and static_llm_fields.intersection(
                    FIELD_READ.findall(without_comments(method.body))):
                tags[method.key].custom_source = True

        edge_count = sum(len(v) for v in edges.values())
        result["edge_count"] = edge_count
        result["invoke_sites"] = invoke_sites
        result["resolved_invoke_sites"] = resolved_sites
        result["unresolved_invoke_sites"] = max(invoke_sites - resolved_sites, 0)
        result["coverage"] = (resolved_sites / invoke_sites) if invoke_sites else 0.0

        source_keys = {
            key for key, tag in tags.items()
            if tag.api_source or tag.web_source or tag.native_source
            or tag.schema_source or tag.custom_source
        }
        sink_keys = {
            key for key, tag in tags.items()
            if tag.http_sink or tag.webview_sink or tag.native_sink
        }
        result["source_method_count"] = len(source_keys)
        result["sink_method_count"] = len(sink_keys)
        result["candidate_source_method_count"] = len(source_keys)
        result["candidate_sink_method_count"] = len(sink_keys)

        result["api_domain_near_http"] = self._count_reachable(
            methods, edges, tags, "api_source", "http_sink")
        result["webview_near_llm_domain"] = self._count_reachable(
            methods, edges, tags, "web_source", "webview_sink")
        result["native_near_jni"] = self._count_reachable(
            methods, edges, tags, "native_source", "native_sink")
        result["json_schema_near_http"] = self._count_reachable(
            methods, edges, tags, "schema_source", "http_sink")
        custom_relations = self._find_split_custom_relations(
            methods, edges, tags)
        # Reachability cannot prove which client received a payload. Preserve
        # these hypotheses for diagnostics; only value-bound flows enter f48.
        result["candidate_custom_relations"] = custom_relations
        from smali_endpoint_flow import find_endpoint_flows
        flow_relations, flow_limited = find_endpoint_flows(methods, tags, self.max_depth, protocols, static_fields)
        # Prefer an actual string/argument trace over a reachability-only value.
        custom_relations = flow_relations
        result["endpoint_flow_limited"] = flow_limited
        result["custom_domain_llm_near_http"] = len({r["source_method"] for r in custom_relations})
        result["verified_custom_relations"] = custom_relations
        result["status"] = "partial" if (parse_errors or graph_limited or flow_limited or protocols.limited or protocols.parse_errors or self._scope_limited or static_diagnostics.get("status") == "partial") else "success"
        result["parse_errors"] = parse_errors
        result["graph_built"] = True
        return result

    @classmethod
    def _scan_candidates(
            cls,
            smali_files: list[Path],
            app_scope: AppCodeScope,
            associated_owners: frozenset[str] = frozenset(),
    ) -> dict[str, object]:
        """Find method-level source/sink candidates without building a graph.

        This pass deliberately keeps only counters.  Method bodies are owned by
        the generator returned from ``_parse_file`` and become collectible as
        soon as each method has been tagged.  A second pass is used only when
        at least one application-owned source is present.
        """
        method_count = 0
        app_method_count = 0
        source_keys: set[str] = set()
        sink_keys: set[str] = set()
        custom_base_keys: set[str] = set()
        custom_route_keys: set[str] = set()
        static_read_fields: set[str] = set()
        schema_keys: set[str] = set()
        http_sink_keys: set[str] = set()
        source_files: set[Path] = set()
        parse_errors = 0

        from smali_facts import scan_smali_facts
        independent_signals = []
        independent_info = {'files_read': 0, 'read_errors': 0, 'constants_read': 0,
                            'invalid_constants': 0, 'signals': 0}

        # Most APKs have no LLM source at all.  Avoid parsing every method in
        # those files merely to discover that fact.  This preflight is only a
        # performance gate; a positive marker falls through to the exact
        # method parser below, and no signal is emitted from this pass.
        may_contain_source = False
        for fp in smali_files:
            try:
                text = fp.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                parse_errors += 1
                independent_info['read_errors'] += 1
                continue
            independent_info['files_read'] += 1
            independent_signals.extend(scan_smali_facts(text, str(fp), independent_info))
            method_count += len(re.findall(r"^\s*\.method\b", text, re.MULTILINE))
            lower = text.lower()
            urls = tuple(_URL_RE.findall(text))
            # Avoid running URL/schema regexes over every obfuscated method
            # body.  These cheap anchors are only a gate; the exact checks
            # below still decide whether this file enters the graph pass.
            if not any(marker in lower for marker in (
                "http://", "https://", "openai", "anthropic", "claude",
                "chatgpt", "gemini", "deepseek", "qwen", "llama",
                "mistral", "mixtral", "gemma", "grok", "messages",
                "max_tokens", "chat/completions", "generatecontent",
                "inference", "webview", "loadurl", "productgpt",
                "sendgpt", "sendprompt", "sendmessage", "gguf", "ggml", "litertlm",
            )):
                continue
            if (
                text_contains_known_api(text, classification_only=True)
                or any(match_web_service(url) for url in urls)
                # Keep this preflight in sync with the exact method tagger.
                # This catches explicit LLM hosts using product-specific
                # routes such as ``/get`` without accepting arbitrary
                # custom-domain strings or rebuilding an APK-wide relation.
                or any(
                    any(scan_url_for_business_ai_endpoint(
                        url, str(fp), "smali"))
                    for url in urls
                )
                or any(h.lower() in lower for h in NATIVE_HINTS)
                or bool(litert_lm_native_symbols(text))
                or bool(extract_llm_schema_components(text))
                or bool(re.search(
                    r"(?:https?://|const-string[^\n]*)(?:[^\n]*"
                    r"(?:chat/completions?|responses|generatecontent|"
                    r"/chat(?:/|$)|/inference(?:/|$)|sendgpt))",
                    text,
                    re.IGNORECASE,
                ))
                or bool(re.search(
                    r"(?:[a-z0-9_-]*gpt|gpt)[/_.-]"
                    r"(?:send|chat|complete|completion|message|prompt)",
                    lower,
                    re.IGNORECASE,
                ))
            ):
                may_contain_source = True

        if not may_contain_source and not parse_errors:
            # All files passed here are already application-owned or an exact
            # obfuscated dependency selected from application invokes.
            app_method_count = method_count
            return {
                "method_count": method_count,
                "app_method_count": app_method_count,
                "source_method_count": 0,
                "sink_method_count": 0,
                "custom_base_method_count": 0,
                "custom_route_method_count": 0,
                "schema_method_count": 0,
                "http_sink_method_count": 0,
                "parse_errors": 0,
                'independent_code_signals': independent_signals,
                'independent_code_analysis': independent_info,
            }

        # A marker was found (or a read failed), so preserve the precise
        # method-level candidate scan for the conservative path.
        method_count = 0
        app_method_count = 0
        parse_errors = 0
        for fp in smali_files:
            try:
                for method in cls._parse_file(
                        fp, app_scope, associated_owners):
                    method_count += 1
                    if method.app_owned:
                        app_method_count += 1
                        from static_string_fields import FIELD_READ
                        from webview_fields import without_comments
                        static_read_fields.update(FIELD_READ.findall(without_comments(method.body)))
                    targets = _INVOKE_RE.findall(method.body)
                    tag = cls._tag_method(method, method.body.lower(), targets)
                    if (tag.api_source or tag.web_source or tag.native_source
                            or tag.schema_source or tag.custom_source):
                        source_keys.add(method.key)
                        source_files.add(fp)
                    if tag.custom_route:
                        source_files.add(fp)
                    if tag.custom_base:
                        custom_base_keys.add(method.key)
                    if tag.custom_route:
                        custom_route_keys.add(method.key)
                    if tag.schema_source:
                        schema_keys.add(method.key)
                    if tag.http_sink:
                        http_sink_keys.add(method.key)
                    if tag.http_sink or tag.webview_sink or tag.native_sink:
                        sink_keys.add(method.key)
            except (OSError, UnicodeError, ValueError):
                parse_errors += 1
        return {
            "method_count": method_count,
            "app_method_count": app_method_count,
            "source_method_count": len(source_keys),
            "sink_method_count": len(sink_keys),
            "custom_base_method_count": len(custom_base_keys),
            "custom_route_method_count": len(custom_route_keys),
            "static_read_fields": sorted(static_read_fields),
            "source_files": sorted(source_files),
            "schema_method_count": len(schema_keys),
            "http_sink_method_count": len(http_sink_keys),
            "parse_errors": parse_errors,
            'independent_code_signals': independent_signals,
            'independent_code_analysis': independent_info,
        }

    @staticmethod
    def _iter_smali_files(root: Path) -> Iterator[Path]:
        for fp in root.rglob("*.smali"):
            if fp.is_file():
                yield fp

    def _expand_obfuscated_app_files(
        self,
        seed_files: list[Path],
        all_files: list[Path],
    ) -> tuple[list[Path], frozenset[str]]:
        """Add exact dependency classes reached from manifest-owned code.

        R8 can move app implementations into roots such as ``a/`` or ``A0/``.
        Scanning every short root also includes large unrelated dependencies,
        so only concrete owners reached by calls and type references are admitted.
        """
        if not self.apktool_dir:
            return seed_files, frozenset()

        owner_to_path: dict[str, Path] = {}
        for path in all_files:
            try:
                parts = path.relative_to(self.apktool_dir).parts
            except ValueError:
                continue
            if len(parts) < 2 or not parts[0].lower().startswith("smali"):
                continue
            tail = "/".join(parts[1:])
            if not tail.lower().endswith(".smali"):
                continue
            owner_to_path[f"L{tail[:-6]};"] = path
            # APKTool may suffix aliases to preserve classes on case-insensitive
            # filesystems. Declarations, not these aliases, identify symbols.
            if any(re.search(r"\.\d+$", part) for part in parts[1:]):
                try:
                    with path.open(encoding="utf-8") as stream:
                        for _ in range(8):
                            match = _CLASS_RE.match(stream.readline())
                            if match:
                                owner_to_path[match[1]] = path
                                break
                except (OSError, UnicodeError):
                    self._scope_limited = True

        selected = set(seed_files)
        associated_owners: set[str] = set()
        from smali_protocols import contract_type_references
        # Summarize only selected files, once each. Semantic dependency order
        # replaces alphabetic fan-out, without another whole-APK content scan
        # or an increased file/depth budget. Hints do not establish app usage.
        summaries = {}
        expanded = {}
        pending = {}
        queue = []

        def references(path):
            if path in summaries:
                return summaries[path]
            try:
                text = path.read_text(encoding='utf-8')
            except (OSError, UnicodeError):
                self._scope_limited = True
                summaries[path] = (set(), set(), set(), set())
                return summaries[path]
            calls, interfaces, types = set(), set(), set()
            in_signature = False
            for line in text.splitlines():
                stripped = line.strip()
                call = re.match(r'invoke-([\w/-]+)\s+\{[^}]*\},\s*(L[^;\s]+;)->\S+', stripped)
                if call:
                    calls.add(call[2])
                    if call[1].startswith('interface'):
                        interfaces.add(call[2])
                if stripped.startswith('.annotation system Ldalvik/annotation/Signature;'):
                    in_signature = True
                elif stripped.startswith('.end annotation'):
                    in_signature = False
                elif in_signature:
                    types.update(re.findall(r'L[^;\s<>"{}]+;', line))
                if stripped.startswith(('.field', '.method', '.annotation', '.super', '.implements',
                                        'new-instance', 'const-class', 'check-cast', 'instance-of', 'iget', 'iput')):
                    types.update(re.findall(r'L[^;\s<>]+;', line))
            summaries[path] = (interfaces, calls, types, contract_type_references(text))
            return summaries[path]

        def expand(path, depth, contract=False):
            previous = expanded.get((path, contract))
            if previous is not None and previous <= depth:
                return
            expanded[path, contract] = depth
            if depth >= self.max_depth:
                return
            interfaces, calls, types, contracts = references(path)
            for owner in sorted(interfaces | calls | types | contracts):
                if any(owner[1:].lower().startswith(prefix) for prefix in self._PLATFORM_ROOTS):
                    continue
                target = owner_to_path.get(owner)
                if target is None or target == path:
                    continue
                is_contract = owner in contracts or (contract and owner in types)
                # Close declared payload dependencies first. Other references
                # retain locality: a distant SDK interface must not displace
                # a directly held/injected business class.
                priority = 0 if is_contract else depth+1 + int(owner not in interfaces)
                state = (target, is_contract)
                score = (priority, depth+1)
                old = pending.get(state)
                if old is not None:
                    score = (min(old[0], score[0]), min(old[1], score[1]))
                    if old == score:
                        continue
                pending[state] = score
                heappush(queue, (*score, target, owner, is_contract))

        for path in sorted(selected):
            expand(path, 0)
        while queue:
            priority, depth, path, owner, contract = heappop(queue)
            if pending.get((path, contract)) != (priority, depth):
                continue
            if path not in selected:
                selected.add(path)
                associated_owners.add(owner)
            expand(path, depth, contract)
        return sorted(selected), frozenset(associated_owners)

    @classmethod
    def _parse_file(
        cls,
        fp: Path,
        app_scope: AppCodeScope,
        associated_owners: frozenset[str] = frozenset(),
    ) -> Iterator[_Method]:
        text = fp.read_text(encoding="utf-8")
        lines = text.splitlines()
        owner: str | None = None
        for line in lines:
            match = _CLASS_RE.match(line)
            if match:
                owner = match.group(1)
                break
        if not owner:
            return

        in_method = False
        method_lines: list[str] = []
        method_decl = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(".method"):
                in_method = True
                method_lines = []
                method_decl = stripped[len(".method"):].strip()
                continue
            if not in_method:
                continue
            if stripped == ".end method":
                parsed = cls._parse_method_decl(owner, method_decl, method_lines,
                                                app_scope, associated_owners)
                if parsed:
                    yield parsed
                in_method = False
                method_lines = []
                continue
            method_lines.append(line)

    @staticmethod
    def _parse_method_decl(owner: str, declaration: str,
                           body_lines: list[str],
                           app_scope: AppCodeScope,
                           associated_owners: frozenset[str] = frozenset(),
                           ) -> _Method | None:
        open_paren = declaration.find("(")
        if open_paren <= 0:
            return None
        prefix = declaration[:open_paren].strip()
        name = prefix.rsplit(None, 1)[-1]
        signature = declaration[open_paren:].strip()
        if not name or not signature:
            return None
        key = f"{owner}->{name}{signature}"
        app_owned = app_scope.matches_owner(owner) or owner in associated_owners
        native = " native " in f" {declaration.lower()} "
        # Full-line smali comments can contain words such as ``invoke``;
        # remove them before extracting call sites and signal literals.
        body = "\n".join(
            line for line in body_lines if not line.strip().startswith("#")
        )
        return _Method(key, owner, name, body, app_owned, native,
                       " static " in f" {declaration.lower()} ")

    @staticmethod
    def _tag_method(method: _Method, body_lower: str,
                    targets: list[str]) -> _MethodTags:
        tag = _MethodTags()
        source_allowed = method.app_owned
        has_api_domain = text_contains_known_api(method.body, classification_only=True)
        urls = _URL_RE.findall(method.body)
        has_web_domain = any(match_web_service(url) for url in urls)
        has_http = any(CallGraphAnalyzer._is_http_target(t) for t in targets)
        has_webview = any(CallGraphAnalyzer._is_webview_target(t) for t in targets)
        # Browser sinks require a bound target, not a method-name/SDK match.
        # Manifest-only TWA entries never create method-graph edges or pairs.
        browser = scan_browser_method(method.body, method.owner,
                                      ("static " if method.static else "")
                                      + method.key.split("->", 1)[-1]) if source_allowed else []
        has_webview = has_webview or bool(browser)
        has_web_domain = has_web_domain or any('relevance_hint=known_llm_website' in s.components for s in browser)
        litert_native_hits = litert_lm_native_symbols(method.body)
        has_jni = any(
            CallGraphAnalyzer._is_native_target(t)
            and (not is_litert_lm_native_target(t) or bool(litert_native_hits))
            for t in targets
        )

        schema_components = extract_llm_schema_components(method.body)
        custom_url = any(
            any(s.verification != SignalVerification.CANDIDATE for s in scan_url_for_business_ai_endpoint(
                url, method.key, "smali", method.body))
            for url in urls
        )
        string_literals = _SMALI_CONST_STRING_RE.findall(method.body)
        route_literals = tuple(
            value.replace(r'\/', '/') for value in string_literals
        )
        custom_base_values = tuple(sorted({
            url for url in urls if is_custom_service_url(url)
        }))
        custom_route_values = tuple(sorted({
            value for value in route_literals
            if (is_plausible_api_route_literal(value)
                and classify_llm_route(value, method.body) is not None)
        }))

        if source_allowed:
            tag.api_source = has_api_domain
            tag.web_source = has_web_domain
            tag.native_source = any(h.lower() in body_lower for h in NATIVE_HINTS)
            tag.native_source = tag.native_source or any(
                h.lower() in method.name.lower() for h in NATIVE_HINTS)
            tag.native_source = tag.native_source or bool(litert_native_hits)
            tag.native_source = tag.native_source or (
                method.native and is_litert_lm_native_target(method.key))
            tag.schema_source = bool(schema_components)
            tag.custom_base = bool(custom_base_values)
            tag.custom_route = bool(custom_route_values)
            tag.custom_base_values = custom_base_values
            tag.custom_route_values = custom_route_values
            tag.schema_components = schema_components
            tag.custom_source = custom_url or (
                tag.custom_base and (tag.custom_route or tag.schema_source)
            )

        tag.http_sink = has_http
        tag.webview_sink = has_webview
        tag.native_sink = has_jni or method.native
        return tag

    @staticmethod
    def _is_http_target(target: str) -> bool:
        lower = target.lower()
        owner = lower.split("->", 1)[0]
        if owner == "ljava/net/url;":
            return lower.startswith("ljava/net/url;->openconnection(")
        owner_hit = any(x in owner for x in (
            "okhttp", "retrofit", "urlconnection", "ktor",
            "org/apache/http", "java/net/http", "volley",
        ))
        return owner_hit

    @staticmethod
    def _is_webview_target(target: str) -> bool:
        lower = target.lower()
        return (
            "android/webkit/webview;->loadurl(" in lower
            or "android/webkit/webview;->evaluatejavascript(" in lower
            or "android/webkit/webview;->addjavascriptinterface(" in lower
            or ("webview" in lower and any(x in lower for x in (
                "->loadurl(", "->evaluatejavascript(",
                "->addjavascriptinterface(",
            )))
        )

    @staticmethod
    def _is_native_target(target: str) -> bool:
        lower = target.lower()
        if "java/lang/system;->loadlibrary(" in lower or \
                "java/lang/system;->load(" in lower:
            return True
        return (any(h.lower() in lower for h in NATIVE_HINTS)
                or is_litert_lm_native_target(target))

    def _count_reachable(self, methods: dict[str, _Method],
                         edges: dict[str, set[str]],
                         tags: dict[str, _MethodTags],
                         source_attr: str, sink_attr: str) -> int:
        pairs: set[tuple[str, str]] = set()
        sink_keys = {
            key for key, tag in tags.items() if getattr(tag, sink_attr)
        }
        source_keys = {
            key for key, tag in tags.items() if getattr(tag, source_attr)
        }
        for source in source_keys:
            queue: deque[tuple[str, int]] = deque([(source, 0)])
            seen: set[str] = {source}
            while queue:
                node, depth = queue.popleft()
                if node in sink_keys:
                    pairs.add((source, node))
                if depth >= self.max_depth:
                    continue
                for target in edges.get(node, ()):
                    if target in methods and target not in seen:
                        seen.add(target)
                        queue.append((target, depth + 1))
        return len(pairs)

    def _find_split_custom_relations(
        self,
        methods: dict[str, _Method],
        edges: dict[str, set[str]],
        tags: dict[str, _MethodTags],
    ) -> list[dict[str, object]]:
        """Return app methods whose custom LLM relation is locally connected.

        Base URLs, routes and payload construction commonly live in separate
        methods. Follow outgoing calls from the LLM route/schema source so
        both the base URL and HTTP sink occur on its execution corridor.
        Reversing edges through shared callers would connect unrelated feature
        branches in large applications. Each source is counted at most once.
        """
        relations: list[dict[str, object]] = []
        llm_keys = {
            key for key, tag in tags.items()
            if methods[key].app_owned and (
                tag.custom_source or tag.custom_route or tag.schema_source
            )
        }
        for source in llm_keys:
            queue: deque[tuple[str, int]] = deque([(source, 0)])
            seen = {source}
            base_distances: dict[str, int] = {}
            has_http_sink = False
            while queue:
                node, depth = queue.popleft()
                node_tag = tags[node]
                for base_value in node_tag.custom_base_values:
                    base_distances.setdefault(base_value, depth)
                has_http_sink = has_http_sink or node_tag.http_sink
                if depth >= self.max_depth:
                    continue
                for target in edges.get(node, ()):
                    if target in methods and target not in seen:
                        seen.add(target)
                        queue.append((target, depth + 1))
            if not base_distances or not has_http_sink:
                continue
            source_tag = tags[source]
            route = source_tag.custom_route_values[0] \
                if source_tag.custom_route_values else ""
            min_base_distance = min(base_distances.values())
            nearest_bases = sorted(
                value for value, distance in base_distances.items()
                if distance == min_base_distance
            )
            # Equally close bases cannot be paired deterministically. If no
            # route/schema is present, there is no relation to emit: the
            # concrete URL scanner already records each base independently.
            # In particular, never use the method descriptor as an endpoint
            # value when several base URLs are visible in one method.
            if (not route and not source_tag.schema_source
                    and len(nearest_bases) != 1):
                continue
            if route:
                value = route if len(nearest_bases) != 1 else (
                    nearest_bases[0].rstrip("/") + "/" + route.lstrip("/")
                )
            else:
                value = nearest_bases[0]
            components = [
                *(f"custom_base_url={item}" for item in nearest_bases),
                *(f"llm_route={item}" for item in source_tag.custom_route_values),
                *(f"json_field={item}" for item in source_tag.schema_components),
                f"llm_source_method={source}",
                f"base_distance={min_base_distance}",
                "http_sink=reachable",
            ]
            relations.append({
                "value": value,
                "source_method": source,
                "components": components,
            })
        return relations
