"""Resolve a conservative, shared application-code scope.

Android's manifest package (the application id) does not always match the
Java/Kotlin namespace that owns the implementation.  Conversely, bundled
libraries often use very short package names after obfuscation.  This module
uses manifest entry points to recover alternate or short application roots
without treating every short directory in JADX/APKTool output as app code.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
from typing import Iterator


# R8/ProGuard may repackage classes below roots such as ``a/``, ``A0/`` or
# ``C3/``.  Do not classify ordinary two-letter namespace prefixes (``io``,
# ``ai``, ``co``...) as obfuscated roots merely because they are short.
_SHORT_ROOT_RE = re.compile(
    r"^(?:[a-z](?:\d{1,2})?|[A-Z][A-Za-z]?\d{0,2})$"
)
_COMPONENT_LAYER_NAMES = frozenset({
    "activity", "activities", "app", "provider", "providers",
    "receiver", "receivers", "service", "services", "ui", "view", "views",
})
_SOURCE_EXTENSIONS = frozenset({".java", ".json", ".kt", ".txt"})
_PLATFORM_COMPONENT_PREFIXES = (
    "android.", "androidx.", "java.", "javax.", "kotlin.",
)


def normalize_component_name(name: object, manifest_package: str | None) -> str:
    """Return a fully qualified manifest component name when possible."""
    value = str(name or "").strip()
    package = str(manifest_package or "").strip().strip(".")
    if not value:
        return ""
    if value.startswith("."):
        return f"{package}{value}" if package else value[1:]
    if "." not in value and package:
        return f"{package}.{value}"
    return value.strip(".")


def _class_package(component: str) -> str:
    return component.rpartition(".")[0]


def _is_short_root(value: str) -> bool:
    return value == "defpackage" or bool(_SHORT_ROOT_RE.fullmatch(value))


def is_obfuscated_root_name(value: str) -> bool:
    """Return whether a top-level class root has an obfuscated shape."""
    return _is_short_root(str(value or ""))


def _is_platform_component(value: str) -> bool:
    lower = value.lower()
    return lower.startswith(_PLATFORM_COMPONENT_PREFIXES)


def _starts_with_package(value: str, prefix: str) -> bool:
    value_lower = value.lower()
    prefix_lower = prefix.lower()
    return value_lower == prefix_lower or value_lower.startswith(prefix_lower + ".")


@dataclass(frozen=True)
class AppCodeScope:
    """Manifest-derived roots accepted as application-owned code."""

    package_prefixes: tuple[str, ...] = ()
    obfuscated_roots: frozenset[str] = frozenset()
    basis: tuple[str, ...] = ()
    shared_classes: frozenset[str] = frozenset()
    shared_jadx_files: tuple[str, ...] = ()
    shared_reference_links: tuple[tuple[str, str], ...] = ()
    shared_scope_status: str = "pending"
    shared_scope_read_errors: int = 0
    referenced_obfuscated_classes: frozenset[str] = frozenset()
    referenced_obfuscated_files: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return bool(self.package_prefixes or self.obfuscated_roots or self.shared_classes)

    def matches_owner(self, owner: str) -> bool:
        class_path = str(owner or "").strip()
        if class_path.startswith("L"):
            class_path = class_path[1:]
        class_path = class_path.rstrip(";").replace(".", "/").strip("/")
        lower = class_path.lower()
        dotted = class_path.replace("/", ".")
        if dotted in self.referenced_obfuscated_classes:
            return True
        if dotted in self.shared_classes or dotted.split("$", 1)[0] in self.shared_classes:
            return True
        for prefix in self.package_prefixes:
            package_path = prefix.replace(".", "/").lower().strip("/")
            if lower == package_path or lower.startswith(package_path + "/"):
                return True
        top = lower.partition("/")[0]
        return top in {root.lower() for root in self.obfuscated_roots}

    def matches_smali_path(self, path: Path, apktool_root: Path) -> bool:
        try:
            parts = path.relative_to(apktool_root).parts
        except ValueError:
            return False
        if len(parts) < 2 or not parts[0].lower().startswith("smali"):
            return False
        owner_path = "/".join(parts[1:])
        if owner_path.lower().endswith(".smali"):
            owner_path = owner_path[:-6]
        return self.matches_owner(owner_path)

    def iter_jadx_files(
        self,
        output_dir: Path,
        extensions: frozenset[str] = _SOURCE_EXTENSIONS,
    ) -> Iterator[Path]:
        """Yield only files below resolved application source roots."""
        sources_dir = output_dir / "sources"
        if not sources_dir.is_dir():
            return
        roots: list[Path] = []
        roots.extend(
            sources_dir.joinpath(*prefix.split("."))
            for prefix in self.package_prefixes
        )
        roots.extend(sources_dir / root for root in sorted(self.obfuscated_roots))

        seen: set[str] = set()
        for root in roots:
            if not root.is_dir():
                continue
            for current, _, filenames in os.walk(root):
                for filename in filenames:
                    if Path(filename).suffix.lower() not in extensions:
                        continue
                    path = Path(current) / filename
                    key = str(path.resolve()).lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    yield path
        for relative in (*self.shared_jadx_files, *self.referenced_obfuscated_files):
            path = sources_dir / relative
            key = str(path.resolve()).lower()
            if path.is_file() and path.suffix.lower() in extensions and key not in seen:
                seen.add(key)
                yield path

    def to_metadata(self) -> dict[str, object]:
        return {
            "status": "resolved" if self.usable else "unscoped",
            "package_prefixes": list(self.package_prefixes),
            "obfuscated_roots": sorted(self.obfuscated_roots),
            "basis": list(self.basis),
            "referenced_obfuscated_code": {
                "basis": "bounded_smali_invoke_and_type_references",
                "class_count": len(self.referenced_obfuscated_classes),
                "classes": sorted(self.referenced_obfuscated_classes),
                "mapped_jadx_files": len(self.referenced_obfuscated_files),
            },
            "shared_code": {
                "status": self.shared_scope_status,
                "class_count": len(self.shared_classes),
                "classes": sorted(self.shared_classes),
                "references": [
                    {"source_file": source, "target_class": target}
                    for source, target in self.shared_reference_links
                ],
                "read_errors": self.shared_scope_read_errors,
            },
        }


_JAVA_NON_CODE = re.compile(
    r'//[^\r\n]*|/\*[\s\S]*?\*/|"""[\s\S]*?"""'
    r'|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
)
_JAVA_IMPORT = re.compile(
    r"\bimport\s+(?:static\s+)?([\w$]+(?:\.[\w$]+)*(?:\.\*)?)\s*;?"
)
_JAVA_PACKAGE = re.compile(r"\bpackage\s+([\w$]+(?:\.[\w$]+)*)\s*;?")
_QUALIFIED_REFERENCE = re.compile(r"\b[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)+")
_IDENTIFIER = re.compile(r"\b[A-Za-z_$][\w$]*\b")
_SHARED_EXCLUDED_PREFIXES = (
    *_PLATFORM_COMPONENT_PREFIXES, "kotlinx.", "com.google.android.gms.",
    "com.google.firebase.", "com.facebook.", "com.microsoft.identity.",
    "com.microsoft.authentication.", "com.microsoft.clarity.",
    "com.microsoft.appcenter.",
)


def expand_shared_app_scope(
    scope: AppCodeScope, output_dir: Path | None, *, max_depth: int = 5,
    max_classes: int = 2048,
) -> AppCodeScope:
    """Follow exact source references inside an anchored application family.

    Only one parent of a manifest root (at least three namespace segments)
    is eligible. Indexing filenames does not admit their contents. Unused
    imports, comments, metadata strings and unreferenced siblings cannot grow
    the scope. This is a bounded ownership heuristic, not a call graph.
    """
    if scope.shared_scope_status != "pending":
        return scope
    sources = output_dir / "sources" if output_dir else None
    if not scope.usable or sources is None or not sources.is_dir():
        return replace(scope, shared_scope_status="unavailable")
    families = sorted({
        prefix.rpartition(".")[0] for prefix in scope.package_prefixes
        if len(prefix.split(".")) >= 4
        and not prefix.startswith(_SHARED_EXCLUDED_PREFIXES)
    })
    if not families:
        return replace(scope, shared_scope_status="no_eligible_namespace")

    index: dict[str, Path] = {}
    for family in families:
        root = sources.joinpath(*family.split("."))
        if not root.is_dir():
            continue
        for current, directories, filenames in os.walk(root):
            namespace = ".".join(Path(current).relative_to(sources).parts) + "."
            if namespace.startswith(_SHARED_EXCLUDED_PREFIXES):
                directories[:] = []
                continue
            for filename in sorted(filenames):
                path = Path(current) / filename
                if path.suffix.lower() in {".java", ".kt"}:
                    index[namespace + path.stem] = path

    def resolve(reference: str) -> str | None:
        # A qualified static member / nested class belongs to its source file.
        while "." in reference:
            if reference in index:
                return reference
            outer = reference.split("$", 1)[0]
            if outer in index:
                return outer
            reference = reference.rpartition(".")[0]
        return None

    seeds = sorted(scope.iter_jadx_files(output_dir, frozenset({".java", ".kt"})))
    queue = deque((path, 0) for path in seeds)
    visited = set(seeds)
    added: dict[str, str] = {}
    links: list[tuple[str, str]] = []
    limited = False
    errors = 0
    while queue:
        path, depth = queue.popleft()
        try:
            code = _JAVA_NON_CODE.sub(" ", path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            errors += 1
            continue
        imports = _JAVA_IMPORT.findall(code)
        package = _JAVA_PACKAGE.search(code)
        code = _JAVA_IMPORT.sub(" ", _JAVA_PACKAGE.sub(" ", code))
        identifiers = set(_IDENTIFIER.findall(code))
        references = set(_QUALIFIED_REFERENCE.findall(code))
        # Only names actually used outside import declarations are followed.
        for imported in imports:
            if imported.endswith(".*"):
                references.update(imported[:-1] + name for name in identifiers)
            elif imported.rsplit(".", 1)[-1] in identifiers:
                references.add(imported)
        if package:
            references.update(package[1] + "." + name for name in identifiers)
        for target in sorted({owner for ref in references if (owner := resolve(ref))}):
            target_path = index[target]
            if target_path in visited:
                continue
            if depth >= max_depth or len(added) >= max_classes:
                limited = True
                continue
            visited.add(target_path)
            queue.append((target_path, depth + 1))
            if not scope.matches_owner(target):
                added[target] = target_path.relative_to(sources).as_posix()
                links.append((path.relative_to(sources).as_posix(), target))
    return replace(
        scope, shared_classes=frozenset(added),
        shared_jadx_files=tuple(added.values()), shared_reference_links=tuple(links),
        shared_scope_status="limited" if limited else ("partial" if errors else "complete"),
        shared_scope_read_errors=errors,
    )


def _anchor_namespace(
    component: str,
    component_packages: tuple[str, ...],
) -> str:
    """Choose a specific namespace around a manifest entry point."""
    package = _class_package(component)
    parts = package.split(".") if package else []
    if not parts or _is_short_root(parts[0]):
        return ""

    # Entry points commonly live in a final ``ui``/``app``/``views`` package.
    # Move one level up in that case, but never below three namespace parts.
    if len(parts) >= 4 and parts[-1].lower() in _COMPONENT_LAYER_NAMES:
        parts = parts[:-1]
    root = ".".join(parts)
    coverage = sum(_starts_with_package(item, root) for item in component_packages)

    # Broaden only when doing so captures an additional manifest component.
    # Stop before a broader prefix that adds no evidence.
    while len(parts) > 3:
        parent_parts = parts[:-1]
        parent = ".".join(parent_parts)
        parent_coverage = sum(
            _starts_with_package(item, parent) for item in component_packages
        )
        if parent_coverage <= coverage:
            break
        parts = parent_parts
        root = parent
        coverage = parent_coverage
    return root


def resolve_app_scope(
    manifest_package: str | None,
    manifest_components: object = None,
    launcher_components: object = None,
    application_class: object = None,
) -> AppCodeScope:
    """Resolve application roots from category-neutral manifest structure."""
    package = str(manifest_package or "").strip().strip(".")
    raw_components = (
        manifest_components
        if isinstance(manifest_components, (list, tuple, set))
        else ()
    )
    components = tuple(dict.fromkeys(
        normalized
        for raw in raw_components
        if (normalized := normalize_component_name(raw, package))
    ))
    component_packages = tuple(
        pkg for component in components if (pkg := _class_package(component))
    )

    raw_launchers = (
        launcher_components
        if isinstance(launcher_components, (list, tuple, set))
        else ()
    )
    launchers = tuple(dict.fromkeys(
        normalized
        for raw in raw_launchers
        if (normalized := normalize_component_name(raw, package))
    ))
    app_class = normalize_component_name(application_class, package)

    prefixes: dict[str, str] = {}
    short_roots: set[str] = set()
    basis: list[str] = []

    def add_prefix(prefix: str, reason: str) -> None:
        value = prefix.strip().strip(".")
        if len(value.split(".")) < 2:
            return
        key = value.lower()
        prefixes.setdefault(key, value)
        basis.append(f"{reason}:{value}")

    if package:
        add_prefix(package, "manifest_package")

    # Both are application-owned entry points.  A custom Application class
    # may live outside the launcher's namespace, so retaining only one of the
    # two can silently discard real app code.
    anchors = list(dict.fromkeys(
        anchor for anchor in [*launchers, *([app_class] if app_class else [])]
        if not _is_platform_component(anchor)
    ))
    for anchor in anchors:
        first = anchor.split(".", 1)[0]
        if _is_short_root(first):
            short_roots.add(first)
            basis.append(f"manifest_entrypoint_short_root:{first}")
            continue
        namespace = _anchor_namespace(anchor, component_packages)
        if namespace:
            add_prefix(namespace, "manifest_entrypoint")

    # Preserve the established sibling-package case only when a manifest
    # component actually confirms code outside the application-id namespace.
    package_parts = package.split(".") if package else []
    if len(package_parts) >= 3:
        parent = ".".join(package_parts[:-1])
        if any(
            _starts_with_package(component, parent)
            and not _starts_with_package(component, package)
            for component in components
        ):
            add_prefix(parent, "manifest_sibling_namespace")

    # A short root is accepted only when an actual manifest component names it.
    for component in components:
        first = component.split(".", 1)[0]
        if _is_short_root(first):
            short_roots.add(first)
            basis.append(f"manifest_component_short_root:{first}")

    # Avoid traversing the same subtree twice when a broader confirmed root
    # already contains a narrower root.
    ordered = sorted(prefixes.values(), key=lambda value: (len(value.split(".")), value.lower()))
    compact: list[str] = []
    for candidate in ordered:
        if any(_starts_with_package(candidate, existing) for existing in compact):
            continue
        compact.append(candidate)

    return AppCodeScope(
        package_prefixes=tuple(compact),
        obfuscated_roots=frozenset(short_roots),
        basis=tuple(dict.fromkeys(basis)),
    )
