"""Bound browser navigation, distinct from Android WebView observations.

Supported: AndroidX/legacy CustomTabsIntent.launchUrl; AndroidX TWA builders
and intents; android-browser-helper TwaLauncher; passive manifest launchers.
No SESSION-string heuristics, arbitrary ACTION_VIEW, prefetch co-occurrence,
or unbounded interprocedural inference. Configurations never become calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit, urlunsplit
import xml.etree.ElementTree as ET

from app_scope import AppCodeScope, normalize_component_name
from registry import match_api_service, match_web_service
from signal_types import RawSignal, SignalRelation, SignalRole, SignalScope, SignalType, SignalVerification
from webview_context import is_auxiliary_page, llm_page_context, page_segments
from webview_evidence import _registers, _SDK_PREFIXES
from webview_fields import (METHOD, STRING, MOVE, FIELD_READ, ConstructorField,
                            recover_constructor_fields, register_normalizer,
                            stable_this, without_comments, written_registers)

_CT = ("Landroidx/browser/customtabs/CustomTabsIntent;", "Landroid/support/customtabs/CustomTabsIntent;")
_TWA_BUILDER = "Landroidx/browser/trusted/TrustedWebActivityIntentBuilder;"
_TWA_INTENT = "Landroidx/browser/trusted/TrustedWebActivityIntent;"
_TWA_LAUNCHER = "Lcom/google/androidbrowserhelper/trusted/TwaLauncher;"
_ENTRY = "com.google.androidbrowserhelper.trusted.LauncherActivity"
_DEFAULT_URL = "android.support.customtabs.trusted.DEFAULT_URL"
_URI = "Landroid/net/Uri;"
_INVOKE = re.compile(r"invoke-(virtual|direct|static|interface|super)(?:/range)?\s+\{([^}]*)\},\s*(L[^;\s]+;)->([^\s(]+)(\([^)]*\)\S+)$")
_NEW = re.compile(r"new-instance\s+([vp]\d+),\s*(L[^;\s]+;)$")
_ACCESS = {"login", "signin", "sign-in", "auth", "oauth", "oauth2", "authorize",
           "callback", "sso", "logout", "register", "signup", "consent", "account"}
_MARKERS = ("->launchUrl(", "->launchTrustedWebActivity(", _TWA_LAUNCHER + "->launch(",
            "CustomTabsIntent;->intent:", _TWA_INTENT + "->getIntent(")
_A = "{http://schemas.android.com/apk/res/android}"
_TWA_STYLE_METHODS = {
    "setDisplayMode(Landroidx/browser/trusted/TrustedWebActivityDisplayMode;)" + _TWA_BUILDER,
    "setAdditionalTrustedOrigins(Ljava/util/List;)" + _TWA_BUILDER,
    "setSplashScreenParams(Landroid/os/Bundle;)" + _TWA_BUILDER,
    "setScreenOrientation(I)" + _TWA_BUILDER,
    "setDefaultColorSchemeParams(Landroidx/browser/customtabs/CustomTabColorSchemeParams;)" + _TWA_BUILDER,
}
MAX_APP_FILES = 8192
MAX_APP_BYTES = 64 * 1024 * 1024
MAX_RESOURCE_FILES = 512


@dataclass(frozen=True)
class _Value:
    kind: str
    text: str
    origin: ConstructorField | None = None


@dataclass
class _Object:
    kind: str
    url: _Value | None = None
    valid: bool = True
    intent: _Object | None = None


def _observation(value: _Value | None, container: str, source: str, header: str,
                 *, config: bool = False, extra: tuple[str, ...] = ()) -> RawSignal | None:
    if value is None:
        return None
    if any(char.isspace() or ord(char) < 32 for char in value.text):
        return None
    try:
        parsed = urlsplit(value.text)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password):
            return None
        parsed.port  # Reject malformed authorities rather than repair them.
    except ValueError:
        return None
    if (is_auxiliary_page(value.text) or _ACCESS.intersection(page_segments(value.text))
            or {"client_id", "redirect_uri", "response_type", "code_challenge"}.intersection(
                key.lower() for key in parse_qs(parsed.query, keep_blank_values=True))):
        return None
    known = bool(match_web_service(value.text))
    if not known and match_api_service(value.text, classification_only=True):
        return None
    if not known and (config or not llm_page_context(header)):
        return None
    # Do not serialize user/session query values. SPA route fragments remain
    # relevant for filtering above, but are not necessary for service identity.
    url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    origin = value.origin
    binding = "launcher_default_url" if config else "constructor_field_uri_argument" if origin else "uri_argument"
    components = (f"container={container}", f"binding={binding}",
                  f"navigation_evidence={'entry_configuration' if config else 'invocation'}",
                  f"target_url={url}", f"relevance_hint={'known_llm_website' if known else 'explicit_llm_context'}",
                  "runtime_container=not_verified", *extra, *(origin.components if origin else ()))
    if container == "twa":
        components += ("twa_origin_verification=not_evaluated",)
    return RawSignal(
        SignalType.BROWSER_LLM_NAVIGATION, url, "apktool", source,
        role=SignalRole.RELATION,
        scope=SignalScope.CONFIG if config else SignalScope.CLASS if origin else SignalScope.METHOD,
        relation=SignalRelation.BROWSER_LLM_TARGET, components=components,
        verification=SignalVerification.VERIFIED if known and not config else SignalVerification.CANDIDATE,
        source_kind="twa_manifest_entry" if config else "smali_browser_navigation")


def scan_browser_method(body: str, owner: str, header: str, source: str = "", *,
                        field_constants: dict[str, ConstructorField] | None = None) -> list[RawSignal]:
    if not any(marker in body for marker in _MARKERS):
        return []
    body = without_comments(body)
    normalize = register_normalizer(header, body)
    fields = field_constants or {}
    if "<init>(" in header:
        fields = {}
    base_this = {"p0"} if fields and stable_this(header, body) else set()
    this_aliases = set(base_this)
    values: dict[str, _Value | _Object] = {}
    pending = None
    result = []

    def get(reg):
        return values.get(normalize(reg))

    def emit(value, container, call):
        observation = _observation(value if isinstance(value, _Value) and value.kind == "uri" else None,
                                   container, source, body,
                                   extra=(f"source_method={header}", f"navigation_call={call}"))
        if observation:
            result.append(observation)

    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("."):
            continue
        if line.startswith((":", "if-", "goto", "packed-switch", "sparse-switch", "return", "throw")):
            values.clear()
            this_aliases = set(base_this)
            pending = None
            continue
        move_result = re.fullmatch(r"move-result-object\s+([vp]\d+)", line)
        old_pending, pending = pending, None
        if move_result:
            dest = normalize(move_result[1])
            values.pop(dest, None)
            this_aliases.discard(dest)
            if old_pending is not None:
                values[dest] = old_pending
            continue
        literal, move, read, new, call = (STRING.fullmatch(line), MOVE.fullmatch(line),
                                          FIELD_READ.fullmatch(line), _NEW.fullmatch(line), _INVOKE.fullmatch(line))
        if literal:
            dest = normalize(literal[1])
            values.pop(dest, None)
            this_aliases.discard(dest)
            try:
                values[dest] = _Value("string", json.loads(literal[2]))
            except ValueError:
                pass
        elif move:
            dest, src = normalize(move[1]), normalize(move[2])
            value, is_this = values.get(src), src in this_aliases
            values.pop(dest, None)
            this_aliases.discard(dest)
            if value is not None:
                values[dest] = value
            if is_this:
                this_aliases.add(dest)
        elif read:
            dest, receiver = normalize(read[1]), normalize(read[2])
            field = fields.get(read[3])
            is_this = receiver in this_aliases
            values.pop(dest, None)
            this_aliases.discard(dest)
            if field and field.owner == owner and is_this:
                values[dest] = _Value("string", field.value, field)
        elif new:
            dest, typ = normalize(new[1]), new[2]
            values.pop(dest, None)
            this_aliases.discard(dest)
            if typ == _TWA_BUILDER or typ in [c[:-1] + "$Builder;" for c in _CT]:
                values[dest] = _Object(typ)
        elif call:
            mode, spec, typ, method, signature = call.groups()
            regs = _registers(spec)
            args = [get(r) for r in regs]
            target = typ + "->" + method + signature
            receiver = args[0] if args else None
            recognized = False
            if mode == "static" and target == _URI + "->parse(Ljava/lang/String;)Landroid/net/Uri;" and len(args) == 1:
                recognized = True
                if isinstance(args[0], _Value) and args[0].kind == "string":
                    pending = _Value("uri", args[0].text, args[0].origin)
            elif typ in _CT and method == "launchUrl" and signature == "(Landroid/content/Context;Landroid/net/Uri;)V" and mode == "virtual" and len(args) == 3:
                recognized = True
                emit(args[2], "custom_tabs", target)
            elif typ == _TWA_LAUNCHER and method == "launch" and mode == "virtual":
                if signature == "(Landroid/net/Uri;)V" and len(args) == 2:
                    recognized = True
                    emit(args[1], "twa", target)
                elif signature in _TWA_BUILDER_LAUNCHES and len(args) in {5, 6}:
                    recognized = True
                    builder = args[1]
                    if isinstance(builder, _Object) and builder.kind == _TWA_BUILDER and builder.valid:
                        emit(builder.url, "twa", target)
            elif isinstance(receiver, _Object) and receiver.valid:
                if method == "<init>" and mode == "direct" and typ == receiver.kind:
                    if typ == _TWA_BUILDER and signature == "(Landroid/net/Uri;)V" and len(args) == 2:
                        recognized = True
                        receiver.url = args[1] if isinstance(args[1], _Value) and args[1].kind == "uri" else None
                    elif typ in [c[:-1] + "$Builder;" for c in _CT] and signature in {"()V", "(Landroidx/browser/customtabs/CustomTabsSession;)V", "(Landroid/support/customtabs/CustomTabsSession;)V"}:
                        recognized = True
                elif typ == receiver.kind and method == "build" and mode == "virtual":
                    if typ == _TWA_BUILDER and signature == "(Landroidx/browser/customtabs/CustomTabsSession;)Landroidx/browser/trusted/TrustedWebActivityIntent;":
                        recognized = True
                        pending = _Object(_TWA_INTENT, receiver.url)
                    elif typ in [c[:-1] + "$Builder;" for c in _CT] and signature == "()" + typ.replace("$Builder", ""):
                        recognized = True
                        pending = _Object(typ.replace("$Builder", ""), intent=_Object("custom_tabs_android_intent"))
                elif typ == _TWA_BUILDER and receiver.kind == _TWA_BUILDER and mode == "virtual" and method + signature in _TWA_STYLE_METHODS:
                    # Only enumerated non-navigation setters preserve the URL.
                    recognized = True
                    pending = receiver
                elif typ == _TWA_INTENT and receiver.kind == _TWA_INTENT and method == "launchTrustedWebActivity" and signature == "(Landroid/content/Context;)V" and mode == "virtual":
                    recognized = True
                    if receiver.intent is None or receiver.intent.valid:
                        emit(receiver.intent.url if receiver.intent else receiver.url, "twa", target)
                elif typ == _TWA_INTENT and receiver.kind == _TWA_INTENT and method == "getIntent" and signature == "()Landroid/content/Intent;" and mode == "virtual":
                    recognized = True
                    if receiver.intent is None:
                        receiver.intent = _Object("twa_android_intent", receiver.url)
                    pending = receiver.intent
                elif typ == "Landroid/content/Intent;" and receiver.kind in {"custom_tabs_android_intent", "twa_android_intent"} and method == "setData" and signature == "(Landroid/net/Uri;)Landroid/content/Intent;" and len(args) == 2 and mode == "virtual":
                    recognized = True
                    receiver.url = args[1] if isinstance(args[1], _Value) and args[1].kind == "uri" else None
                    pending = receiver
            intent = None
            if target in {"Landroid/app/Activity;->startActivity(Landroid/content/Intent;)V", "Landroid/content/Context;->startActivity(Landroid/content/Intent;)V"} and mode == "virtual" and len(args) == 2:
                intent = args[1]
            elif target == "Landroidx/core/content/ContextCompat;->startActivity(Landroid/content/Context;Landroid/content/Intent;Landroid/os/Bundle;)V" and mode == "static" and len(args) == 3:
                intent = args[1]
            if isinstance(intent, _Object) and intent.valid and intent.kind in {"custom_tabs_android_intent", "twa_android_intent"}:
                recognized = True
                emit(intent.url, "twa" if intent.kind == "twa_android_intent" else "custom_tabs", target)
            if not recognized:
                for value in args:
                    if isinstance(value, _Object):
                        value.valid = False
                        if value.intent:
                            value.intent.valid = False
        else:
            ct_field = re.fullmatch(r"iget-object\s+([vp]\d+),\s*([vp]\d+),\s*(L[^;]+;)->intent:Landroid/content/Intent;", line)
            value = get(ct_field[2]) if ct_field else None
            escape = re.match(r"(?:iput-object|sput-object|aput-object)\s+([vp]\d+)(?:,|$)", line)
            escaped = get(escape[1]) if escape else None
            if isinstance(escaped, _Object):
                escaped.valid = False
                if escaped.intent:
                    escaped.intent.valid = False
            if line.startswith("invoke-"):
                # Unsupported invoke formats must not preserve mutable aliases.
                for value in values.values():
                    if isinstance(value, _Object):
                        value.valid = False
                        if value.intent:
                            value.intent.valid = False
            for dest in written_registers(line):
                values.pop(normalize(dest), None)
                this_aliases.discard(normalize(dest))
            if (ct_field and isinstance(value, _Object) and value.valid
                    and value.kind == ct_field[3] and value.kind in _CT):
                values[normalize(ct_field[1])] = value.intent
    return result


_TWA_BUILDER_LAUNCHES = {
    "(" + _TWA_BUILDER + "Landroidx/browser/customtabs/CustomTabsCallback;Lcom/google/androidbrowserhelper/trusted/splashscreens/SplashScreenStrategy;Ljava/lang/Runnable;" + tail + ")V"
    for tail in ("", _TWA_LAUNCHER[:-1] + "$FallbackStrategy;")
}


def _app_files(root: Path, scope: AppCodeScope):
    files = set()
    for dex in root.glob("smali*"):
        if not dex.is_dir():
            continue
        for prefix in (*scope.package_prefixes, *scope.obfuscated_roots):
            directory = dex.joinpath(*prefix.replace(".", "/").split("/"))
            if directory.is_dir():
                files.update(directory.rglob("*.smali"))
        for owner in scope.shared_classes | scope.referenced_obfuscated_classes:
            path = dex.joinpath(*owner.split(".")).with_suffix(".smali")
            if path.is_file():
                files.add(path)
                files.update(path.parent.glob(path.stem + "$*.smali"))
    for path in sorted(files):
        relative = "/".join(path.relative_to(root).parts[1:])
        if scope.matches_smali_path(path, root) and not relative.startswith((*_SDK_PREFIXES, "com/google/androidbrowserhelper/")):
            yield path


def _resource_url(root: Path, value: str) -> str | None:
    if not value.startswith("@"):
        return value
    seen = set()
    for _ in range(5):
        if not re.fullmatch(r"@string/[\w]+", value) or value in seen:
            return None
        seen.add(value)
        name, found, default_found = value.split("/", 1)[1], set(), False
        paths = sorted((root / "res").glob("values*/*.xml"))
        if len(paths) > MAX_RESOURCE_FILES:
            return None
        try:
            for path in paths:
                if path.stat().st_size > 2 * 1024 * 1024:
                    return None
                for node in ET.parse(path).getroot():
                    if node.tag == "string" and node.get("name") == name:
                        if list(node):
                            return None
                        found.add((node.text or "").strip())
                        default_found |= path.parent.name == "values"
        except (OSError, ET.ParseError):
            return None
        if len(found) != 1 or not default_found:
            return None
        value = found.pop()
        if not value.startswith("@"):
            return value
    return None


def _manifest_entries(root: Path, classes: dict, diagnostics: dict) -> list[RawSignal]:
    path = root / "AndroidManifest.xml"
    try:
        if path.stat().st_size > 4 * 1024 * 1024:
            diagnostics["manifest_status"] = "size_budget"
            return []
        manifest = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        diagnostics["manifest_status"] = "unavailable_or_unparsed"
        return []
    app = manifest.find("application")
    if app is None or app.get(_A + "enabled", "true") != "true":
        return []
    package = manifest.get("package", "")
    activities = {normalize_component_name(node.get(_A + "name"), package): node for node in app.findall("activity")}
    result = []
    # Alias metadata resolution is not assumed equivalent to target metadata.
    # Keep this first adapter to concrete default launcher activities.
    diagnostics["manifest_aliases_not_resolved"] = len(app.findall("activity-alias"))
    for entry in app.findall("activity"):
        if (entry.get(_A + "enabled", "true") != "true"
                or entry.get(_A + "exported", "true") != "true"):
            continue
        if not any(any(a.get(_A + "name") == "android.intent.action.MAIN" for a in f.findall("action"))
                   and any(c.get(_A + "name") == "android.intent.category.LAUNCHER" for c in f.findall("category"))
                   for f in entry.findall("intent-filter")):
            continue
        name = normalize_component_name(entry.get(_A + "name"), package)
        activity = activities.get(name)
        if activity is None or activity.get(_A + "enabled", "true") != "true":
            continue
        current, visited = name, set()
        while current != _ENTRY and current not in visited and len(visited) < 8:
            visited.add(current)
            cls = classes.get("L" + current.replace(".", "/") + ";")
            class_text = without_comments(cls[2]) if cls else ""
            if (cls is None or len(list(METHOD.finditer(class_text))) != len(re.findall(r"(?m)^\s*\.method\s", class_text))
                    or any("<init>(" not in m[1] for m in METHOD.finditer(class_text))):
                break
            constructors = list(METHOD.finditer(class_text))
            if not constructors or any(
                not re.search(r"(?:^|\s)<init>\(\)V$", m[1].strip())
                or [line.strip() for line in m[2].splitlines() if line.strip() and not line.strip().startswith(".")]
                != ["invoke-direct {p0}, " + cls[1] + "-><init>()V", "return-void"]
                for m in constructors):
                break
            current = cls[1][1:-1].replace("/", ".")
        if current != _ENTRY:
            continue
        # Require a packaged named launcher, not merely a manifest class name.
        launcher_present = False
        for dex in root.glob("smali*"):
            candidate = dex / (_ENTRY.replace(".", "/") + ".smali")
            try:
                text = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            launcher_present |= bool(re.search(r"(?m)^\s*\.class\s+[^\n]*Lcom/google/androidbrowserhelper/trusted/LauncherActivity;\s*$", text)
                                     and re.search(r"(?m)^\s*\.method\s+[^\n]*\bonCreate\(Landroid/os/Bundle;\)V", text))
        if not launcher_present:
            continue
        urls = [node.get(_A + "value", "") for node in activity.findall("meta-data") if node.get(_A + "name") == _DEFAULT_URL]
        if len(urls) != 1:
            continue
        value = _resource_url(root, urls[0])
        observation = _observation(_Value("uri", value) if value else None, "twa", str(path), "", config=True,
                                   extra=(f"entry_activity={name}", f"entry_metadata={_DEFAULT_URL}", "entry_policy=default_launcher_only"))
        if observation:
            result.append(observation)
    diagnostics["manifest_status"] = "parsed"
    return result


def extract_browser_navigation(root: Path, scope: AppCodeScope, diagnostics: dict | None = None) -> list[RawSignal]:
    info = diagnostics if diagnostics is not None else {}
    info.update(status="complete", files_checked=0, bytes_checked=0)
    if not root.is_dir() or not scope.usable:
        info.update(status="skipped", reason="missing_root_or_app_scope")
        return []
    classes, sources, duplicates = {}, {}, set()
    for path in _app_files(root, scope):
        try:
            size = path.stat().st_size
            if info["files_checked"] >= MAX_APP_FILES or info["bytes_checked"] + size > MAX_APP_BYTES or size > 8 * 1024 * 1024:
                info.update(status="partial", reason="app_scan_budget")
                break
            # Class headers are anchored Smali directives. Strip comments only
            # inside relevant methods/constructor summaries, not every SDK file.
            text = path.read_text(encoding="utf-8")
            info["files_checked"] += 1
            info["bytes_checked"] += size
        except (OSError, UnicodeError):
            info.update(status="partial", reason="app_scan_unreadable")
            continue
        owner = re.search(r"(?m)^\s*\.class\s+[^\n]*?(L[^;\s]+;)\s*$", text)
        parent = re.search(r"(?m)^\s*\.super\s+(L[^;\s]+;)\s*$", text)
        if owner and parent and scope.matches_owner(owner[1]):
            if owner[1] in classes:
                duplicates.add(owner[1])
            classes[owner[1]] = (path, parent[1], text)
            if any(marker in text for marker in _MARKERS):
                sources[owner[1]] = (path, text)
    for owner in duplicates:
        classes.pop(owner, None)
        sources.pop(owner, None)
    field_info = {}
    fields = recover_constructor_fields(root, sources, field_info)
    info["constructor_fields"] = field_info
    signals = []
    for owner, (path, text) in sources.items():
        for method in METHOD.finditer(text):
            signals.extend(scan_browser_method(method[2], owner, method[1], str(path), field_constants=fields))
    signals.extend(_manifest_entries(root, classes, info))
    info["navigation_observations"] = len(signals)
    return signals
