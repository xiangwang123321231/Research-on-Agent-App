"""Guard same-column feature semantic changes during export/train/predict."""
from __future__ import annotations
import json
from pathlib import Path
from feature_vectorizer import FEATURE_SCHEMA_VERSION
from registry import SIGNAL_SCHEMA_VERSION, EVIDENCE_REGISTRY_VERSION


def validate_records(records: list[dict], *, allow_feature_refresh: bool = False,
                     feature_schema_version: str = FEATURE_SCHEMA_VERSION) -> dict:
    """Validate signal contracts; frozen studies explicitly supply their schema."""
    from apk_llm_analysis import EXTRACTOR_VERSION
    expected = {"extractor_version": EXTRACTOR_VERSION, "signal_schema_version": SIGNAL_SCHEMA_VERSION,
                "signal_registry_version": EVIDENCE_REGISTRY_VERSION, "feature_schema_version": feature_schema_version}
    source_feature_versions = set()
    checked = expected if not allow_feature_refresh else {
        k: v for k, v in expected.items() if k != 'feature_schema_version'}
    for record in records:
        metadata = record.get("metadata") or {}
        feature_version = metadata.get('feature_schema_version')
        if allow_feature_refresh:
            if not isinstance(feature_version, str) or not feature_version.startswith('features-'):
                raise ValueError("Feature refresh requires an explicit source feature version.")
            source_feature_versions.add(feature_version)
        if (not allow_feature_refresh
                and isinstance(feature_version, str)
                and feature_version != feature_schema_version
                and all(metadata.get(k) == v for k, v in expected.items()
                        if k != 'feature_schema_version')):
            raise ValueError("Only the feature schema differs; use export_features.py --revectorize to recompute saved signals without APK re-extraction.")
        if any(metadata.get(k) != v for k, v in checked.items()):
            raise ValueError("Input signals have an old, missing or incompatible version; re-extract into a new output directory.")
    if allow_feature_refresh and len(source_feature_versions) != 1:
        raise ValueError("Feature refresh requires one explicit source feature version; missing/mixed manifests are not accepted.")
    return expected


def manifest_path(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.name + ".manifest.json")


def read_contract(csv_path: Path) -> dict | None:
    try:
        manifest = json.loads(manifest_path(csv_path).read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and isinstance(manifest.get("contract"), dict):
            return manifest["contract"]
    except (OSError, ValueError):
        pass
    return None


def check_model_contract(model_contract: dict | None, data_contract: dict | None, allow_legacy: bool = False) -> None:
    if model_contract is None or data_contract is None:
        if allow_legacy and model_contract is None and data_contract is None:
            return
        raise ValueError("Model/features have no compatible provenance manifest. Legacy experiments require --allow-legacy.")
    if model_contract != data_contract:
        raise ValueError("Model and feature extraction contracts differ; retrain the model with these features.")
