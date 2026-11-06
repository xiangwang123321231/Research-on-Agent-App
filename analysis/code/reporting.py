from __future__ import annotations

from pathlib import Path
import json


CODE_SCOPE_SCHEMA_VERSION = "app-code-scope-details-v1"


def split_code_scope_details(
    payload: dict[str, object], details_path: str | Path,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Separate diagnostic class lists without changing analysis or signals.

    The sidecar preserves the complete original scope for lossless inspection.
    Only copies of the output dictionaries are edited; the analyzer's in-memory
    scope and all per-signal evidence remain intact.
    """
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return payload, None
    scope = metadata.get("app_code_scope")
    if not isinstance(scope, dict):
        return payload, None

    compact_scope = dict(scope)
    moved = False
    for section, fields in (
        ("referenced_obfuscated_code", ("classes",)),
        ("shared_code", ("classes", "references")),
    ):
        original = scope.get(section)
        if not isinstance(original, dict):
            continue
        compact_section = dict(original)
        for field in fields:
            if field in original:
                value = compact_section.pop(field)
                if isinstance(value, list):
                    count_key = "class_count" if field == "classes" else "reference_count"
                    compact_section.setdefault(count_key, len(value))
                moved = True
        compact_scope[section] = compact_section
    if not moved:
        return payload, None

    compact_scope["details_file"] = Path(details_path).name
    details: dict[str, object] = {
        "schema_version": CODE_SCOPE_SCHEMA_VERSION,
        "apk_path": payload.get("apk_path"),
        "sha256": payload.get("sha256"),
        "extractor_version": metadata.get("extractor_version"),
        "app_code_scope": scope,
    }
    compact = {
        **payload,
        "metadata": {**metadata, "app_code_scope": compact_scope},
    }
    return compact, details


class ReportWriter:
    """Write per-APK and summary reports to disk."""

    def write_json(self, path: str | Path, payload: dict[str, object]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
