"""LLM-related application WebView facts from typed Smali.

Never infer a load from a method name alone or turn a loadData base origin
into a navigation target. No full call graph or JADX output is required.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote, urlsplit

from app_scope import AppCodeScope
from smali_literals import decode_smali_string
from registry import decode_bytes, extract_urls
from signal_types import RawSignal, SignalRole, SignalScope, SignalType, SignalVerification
from webview_context import is_auxiliary_html, llm_html_context, llm_page_context, page_observation, related_usage
from webview_fields import (ConstructorField, FIELD_READ, recover_constructor_fields,
                            register_normalizer, stable_this, written_registers)

_BASE = "Landroid/webkit/WebView;"
_SDK_PREFIXES = (
    "android/", "androidx/", "java/", "javax/", "kotlin/",
    "com/stripe/", "com/shopify/", "com/hcaptcha/", "com/adjust/",
    "com/google/android/gms/", "com/google/firebase/", "com/facebook/",
    "com/microsoft/identity/", "com/microsoft/authentication/",
    "com/microsoft/clarity/", "com/microsoft/appcenter/",
)
_INVOKE = re.compile(
    r"^invoke-(?:virtual|super)(?:/range)?\s+\{([^}]*)\},\s*"
    r"(L[^;]+;)->(\w+)(\([^)]*\)V)$"
)
_LOAD_SIGNATURES = {
    ("loadUrl", "(Ljava/lang/String;)V"),
    ("loadUrl", "(Ljava/lang/String;Ljava/util/Map;)V"),
    ("loadData", "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V"),
    ("loadDataWithBaseURL", "(" + "Ljava/lang/String;" * 5 + ")V"),
}
_METHOD = re.compile(r"(?ms)^\s*\.method\s+([^\n]+)\n(.*?)^\s*\.end method")
_STRING = re.compile(r'const-string(?:/jumbo)?\s+([vp]\d+),\s*("(?:\\.|[^"\\])*")$')
_MOVE = re.compile(r"move-object(?:/from16|/16)?\s+([vp]\d+),\s*([vp]\d+)$")
_COMMENT = re.compile(r'"(?:\\.|[^"\\])*"|#[^\n]*')


def _registers(spec: str) -> list[str]:
    if ".." not in spec:
        return [s.strip() for s in spec.split(",") if s.strip()]
    first, last = (s.strip() for s in spec.split("..", 1))
    if not re.fullmatch(r"[vp]\d+", first) or not re.fullmatch(r"[vp]\d+", last) or first[0] != last[0]:
        return []
    start, end = int(first[1:]), int(last[1:])
    return [first[0] + str(i) for i in range(start, min(end + 1, start + 256))]


def _asset_text(root: Path, url: str) -> str:
    prefix = "file:///android_asset/"
    if not url.startswith(prefix):
        return ""
    asset_root = (root / "assets").resolve()
    path = (asset_root / unquote(urlsplit(url).path[len("/android_asset/"):])).resolve()
    if not path.is_relative_to(asset_root) or path.suffix.lower() not in {".html", ".htm"}:
        return ""
    try:
        if path.stat().st_size <= 2 * 1024 * 1024:
            return decode_bytes(path.read_bytes())
    except OSError:
        pass
    return ""


def _method_signals(root: Path, path: Path, header: str, body: str,
                    types: set[str], *, owner: str = "",
                    field_constants: dict[str, ConstructorField] | None = None) -> list[RawSignal]:
    values: dict[str, str] = {}
    origins: dict[str, ConstructorField] = {}
    signals: list[RawSignal] = []
    body = _COMMENT.sub(lambda m: "" if m[0].startswith("#") else m[0], body)
    reg = register_normalizer(header, body)
    # Constructor summaries describe post-initialization values, not reads
    # from a partially initialized receiver inside <init> itself.
    fields = field_constants or {}
    if re.search(r"(?:^|\s)<init>\(", header):
        fields = {}
    base_this = {"p0"} if fields and stable_this(header, body) else set()
    aliases = set(base_this)

    method_fact = (f"source_method={header.strip()}",)
    for raw in body.splitlines():
        line = raw.strip()
        if line.startswith((":", "if-", "goto", "packed-switch", "sparse-switch", "return", "throw")):
            values.clear()
            origins.clear()
            aliases = set(base_this)
        literal = _STRING.fullmatch(line)
        move = _MOVE.fullmatch(line)
        call = _INVOKE.fullmatch(line)
        read = FIELD_READ.fullmatch(line)
        if literal:
            origins.pop(reg(literal[1]), None)
            aliases.discard(reg(literal[1]))
            try:
                values[reg(literal[1])] = decode_smali_string(literal[2])
            except ValueError:
                values.pop(reg(literal[1]), None)
        elif move:
            target = reg(move[1])
            source = reg(move[2])
            value, origin, is_this = values.get(source), origins.get(source), source in aliases
            values.pop(target, None)
            origins.pop(target, None)
            aliases.discard(target)
            if value is not None:
                values[target] = value
                if origin:
                    origins[target] = origin
            if is_this:
                aliases.add(target)
        elif read:
            target, receiver = reg(read[1]), reg(read[2])
            field = fields.get(read[3])
            is_this = receiver in aliases
            values.pop(target, None)
            origins.pop(target, None)
            aliases.discard(target)
            if field and field.owner == owner and is_this:
                values[target] = field.value
                origins[target] = field
        elif call and call[2] in types and (call[3], call[4]) in _LOAD_SIGNATURES:
            method = call[3]
            registers = _registers(call[1])
            # loadDataWithBaseURL's first parameter is an origin, not HTML.
            index = 2 if method == "loadDataWithBaseURL" else 1
            value = values.get(reg(registers[index])) if len(registers) > index else None
            origin = origins.get(reg(registers[index])) if len(registers) > index else None
            facts = method_fact + (origin.components if origin else ())
            binding = "constructor_field_argument" if origin else "register_argument"
            relation_scope = SignalScope.CLASS if origin else SignalScope.METHOD
            related: list[RawSignal] = []
            if method == "loadUrl" and value is not None:
                asset = _asset_text(root, value)
                if asset and is_auxiliary_html(asset):
                    continue
                observation = page_observation(
                    value, str(path), "apktool", relation_scope, call=method,
                    binding="local_asset_argument" if asset else binding,
                    source_kind="typed_smali_webview", typed=True,
                    context=asset or body, extra=facts,
                )
                if observation:
                    if asset:
                        observation = replace(observation, components=observation.components + ("page_content=local_html",))
                    related.append(observation)
            elif value is not None:
                if llm_html_context(value):
                    related.append(RawSignal(
                        SignalType.WEBVIEW_CHAT_PAGE, "inline_html", "apktool", str(path),
                        role=SignalRole.RELATION, scope=relation_scope,
                        verification=SignalVerification.CANDIDATE, source_kind="typed_smali_inline_html",
                        components=(f"webview_call={method}", f"binding={binding}",
                                    "relevance_hint=llm_html_context", *facts),
                    ))
            else:
                # Unknown arguments stay unknown. Same-method hints retain
                # candidates, never a proven target. Base origins do not
                # qualify unknown HTML as an LLM page.
                if method == "loadUrl":
                    for url in extract_urls(body):
                        observation = page_observation(
                            url, str(path), "apktool", SignalScope.METHOD, call=method,
                            binding="unresolved_method_context", source_kind="smali_method_cooccurrence",
                            extra=method_fact,
                        )
                        if observation:
                            related.append(observation)
                if not related and llm_page_context(body):
                    signals.extend(related_usage(
                        str(path), method, ("binding=unresolved_argument", "relevance_hint=llm_method_context",
                                           *method_fact), SignalVerification.CANDIDATE))
            if related:
                for observation in related:
                    signals.extend(related_usage(str(path), method, observation.components,
                                                 observation.verification, scope=observation.scope))
                signals.extend(related)
        else:
            # Every unhandled write invalidates a value; a previous URL
            # cannot survive iget, move-result, const, arithmetic, etc.
            for dest in written_registers(line):
                values.pop(reg(dest), None)
                origins.pop(reg(dest), None)
                aliases.discard(reg(dest))
    return signals


def extract_webview_usage(root: Path, scope: AppCodeScope, *,
                          field_diagnostics: dict | None = None) -> list[RawSignal]:
    """Inspect only manifest-resolved app roots; unknown ownership stays unknown.

    Known SDK namespaces are excluded even if a broad app prefix encloses
    them. App subclasses are resolved across DEX files. Unresolved third-party
    subclasses and reflection are intentionally not guessed.
    """
    if not root.is_dir() or not scope.usable:
        if field_diagnostics is not None:
            field_diagnostics.update(status="skipped", reason="missing_root_or_app_scope")
        return []
    files: set[Path] = set()
    prefixes = [p.replace(".", "/") for p in scope.package_prefixes]
    prefixes.extend(scope.obfuscated_roots)
    for dex_root in root.glob("smali*"):
        if dex_root.is_dir():
            for prefix in prefixes:
                app_root = dex_root.joinpath(*prefix.split("/"))
                if app_root.is_dir():
                    files.update(app_root.rglob("*.smali"))
            for owner in scope.shared_classes | scope.referenced_obfuscated_classes:
                shared_path = dex_root.joinpath(*owner.split(".")).with_suffix(".smali")
                if shared_path.is_file():
                    files.add(shared_path)
                if shared_path.parent.is_dir():
                    files.update(shared_path.parent.glob(shared_path.stem + "$*.smali"))

    classes: dict[str, tuple[Path, str, list[tuple[str, str]]]] = {}
    field_sources: dict[str, tuple[Path, str]] = {}
    duplicate_owners: set[str] = set()
    for path in sorted(files):
        if not scope.matches_smali_path(path, root):
            continue
        owner_path = "/".join(path.relative_to(root).parts[1:])
        if owner_path.startswith(_SDK_PREFIXES):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        owner = re.search(r"(?m)^\s*\.class\s+[^\n]*?(L[^;\s]+;)\s*$", text)
        parent = re.search(r"(?m)^\s*\.super\s+(L[^;\s]+;)\s*$", text)
        if owner and parent and scope.matches_owner(owner[1]):
            methods = [(m[1], m[2]) for m in _METHOD.finditer(text)
                       if any(marker in m[2] for marker in ("->loadUrl(", "->loadData(", "->loadDataWithBaseURL("))]
            if owner[1] in classes:
                duplicate_owners.add(owner[1])
            classes[owner[1]] = (path, parent[1], methods)
            if methods and any(FIELD_READ.search(line.strip()) for _, body in methods
                               for line in body.splitlines()):
                field_sources[owner[1]] = (path, text)

    for owner in duplicate_owners:
        field_sources.pop(owner, None)
    field_constants = recover_constructor_fields(root, field_sources, field_diagnostics)

    types = {_BASE}
    children: dict[str, list[str]] = {}
    for owner, (_, parent, _) in classes.items():
        children.setdefault(parent, []).append(owner)
    pending = [_BASE]
    while pending:
        for child in children.get(pending.pop(), []):
            if child not in types:
                types.add(child)
                pending.append(child)

    signals: list[RawSignal] = []
    for owner, (path, _, methods) in classes.items():
        for header, body in methods:
            signals.extend(_method_signals(root, path, header, body, types,
                                           owner=owner, field_constants=field_constants))
    return signals
