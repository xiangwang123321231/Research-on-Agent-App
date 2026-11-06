"""Export feature vectors and label templates from the JSONL analysis output.

Reads ``apks.jsonl`` produced by ``main.py``, runs ``FeatureVectorizer``
on every record, and writes:

    output/features.csv   —  versioned feature matrix (ready for ML training)
    output/labels.csv     —  empty single-label template (5 one-hot columns)

Usage::

    python export_features.py                          # use defaults
    python export_features.py --jsonl output/apks.jsonl
    python export_features.py --preview                # inspect old/mixed signals
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import sys
from pathlib import Path

from feature_vectorizer import FeatureVectorizer, FEATURE_SCHEMA_VERSION
from feature_contract import validate_records, read_contract, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export versioned feature vectors and label templates as CSV")
    parser.add_argument(
        "--jsonl", default=r"D:\work\analysis\output\apks.jsonl",
        help="Path to the JSONL file produced by main.py")
    parser.add_argument(
        "--features-out", default=None,
        help="Output path for features.csv (default: same dir as jsonl)")
    parser.add_argument(
        "--labels-out", default=None,
        help="Output path for labels.csv (default: same dir as jsonl)")
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing CSV files instead of appending")
    parser.add_argument(
        "--features-only", action="store_true",
        help="Write features.csv without modifying the existing labels.csv")
    parser.add_argument(
        "--revectorize", action="store_true",
        help="Recompute features from compatible signal versions with an older feature schema; write a separate feature file and preserve labels")
    parser.add_argument(
        "--preview", action="store_true",
        help="Refresh features_preview.csv using current feature rules, allowing old/mixed signals; leave labels unchanged and omit a validated training contract")
    args = parser.parse_args()
    if args.revectorize and args.preview:
        parser.error('--revectorize and --preview are mutually exclusive')
    if args.revectorize:
        args.features_only = True
        # A refresh is a complete snapshot of these signals. Append/resume
        # could silently retain stale vectors under a fresh input manifest.
        args.force = True
    if args.preview:
        args.features_only = True
        args.force = True  # A preview is a snapshot, never an append/resume export.

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: JSONL not found: {jsonl_path}", file=sys.stderr)
        return 1

    out_dir = jsonl_path.parent
    feature_filename = ("features_preview.csv" if args.preview else
                        "features_revectorized.csv" if args.revectorize else "features.csv")
    features_path = Path(args.features_out) if args.features_out else out_dir / feature_filename
    labels_path = Path(args.labels_out) if args.labels_out else out_dir / "labels.csv"

    fv = FeatureVectorizer()
    feature_names = fv.feature_names

    label_columns = [
        "is_direct_client",
        "is_server_proxy",
        "is_webview",
        "is_local",
        "is_non_ai",
    ]

    records: list[dict] = []
    error_count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                error_count += 1
                continue
            if rec.get("error"):
                error_count += 1
                continue
            records.append(rec)

    if not records:
        print("Error: no valid records found in JSONL", file=sys.stderr)
        return 1

    # Validate before opening any output, especially an existing labels file.
    try:
        if args.preview:
            protected = {jsonl_path.resolve(), (out_dir / "features.csv").resolve(), labels_path.resolve()}
            if features_path.resolve() in protected or read_contract(features_path) is not None:
                raise ValueError("Preview requires a separate CSV path; use features_preview.csv.")
            contract = None
        else:
            if args.revectorize and features_path.resolve() in {
                    jsonl_path.resolve(), (out_dir / 'features.csv').resolve(), labels_path.resolve()}:
                raise ValueError('Revectorization requires a separate feature CSV; source files and labels are protected.')
            contract = validate_records(records, allow_feature_refresh=True) if args.revectorize else validate_records(records)
            if features_path.exists() and not args.force and read_contract(features_path) != contract:
                raise ValueError("Existing feature CSV has an incompatible/missing manifest; use a new output directory or rebuild features with --force --features-only.")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # ── Build the dataframe as list-of-dicts ──────────────────────────

    feature_rows: list[dict] = []
    label_rows: list[dict] = []
    skipped = 0

    for rec in records:
        apk_path = str(rec.get("apk_path", ""))
        if not apk_path:
            skipped += 1
            continue

        signals = rec.get("signals", [])
        metadata = rec.get("metadata", {})

        vec = fv.vectorize_one(signals, metadata)
        if len(vec) != fv.n_features:
            print(f"  Warning: {Path(apk_path).name} produced {len(vec)} dims "
                  f"(expected {fv.n_features}), skipping", file=sys.stderr)
            skipped += 1
            continue

        row = {"apk_path": apk_path}
        for name, val in zip(feature_names, vec):
            row[name] = val
        feature_rows.append(row)

        label_rows.append({"apk_path": apk_path, **{col: "" for col in label_columns}})

    # ── Determine which APKs are already in the existing CSVs ─────────

    existing_features: set[str] = set()
    existing_labels: set[str] = set()
    features_exists = features_path.exists()
    labels_exists = labels_path.exists()

    if not args.force:
        if features_exists:
            with open(features_path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    existing_features.add(row.get("apk_path", ""))
        if labels_exists and not args.features_only:
            with open(labels_path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    existing_labels.add(row.get("apk_path", ""))

    new_features = [r for r in feature_rows if r["apk_path"] not in existing_features]
    new_labels = [r for r in label_rows if r["apk_path"] not in existing_labels]

    # ── Write features.csv ──────────────────────────────────────────

    write_mode = "w" if args.force or not features_exists else "a"
    with open(features_path, write_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["apk_path"] + feature_names)
        if write_mode == "w":
            writer.writeheader()
        writer.writerows(new_features if write_mode == "a" else feature_rows)

    manifest = {"contract": contract, "feature_names": feature_names}
    if args.revectorize:
        manifest['derivation'] = {
            'operation': 'revectorize', 'source_jsonl': str(jsonl_path.resolve()),
            'source_sha256': hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
            'source_record_count': len(records),
            'source_feature_schema_version': records[0]['metadata']['feature_schema_version'],
            'target_feature_schema_version': FEATURE_SCHEMA_VERSION,
        }
    if args.preview:
        version_keys = ("extractor_version", "signal_schema_version", "signal_registry_version", "feature_schema_version")
        versions = Counter(json.dumps({k: (rec.get("metadata") or {}).get(k) for k in version_keys},
                                      sort_keys=True) for rec in records)
        manifest.update({"mode": "preview", "feature_schema_version": FEATURE_SCHEMA_VERSION,
                         "source_versions": [{"versions": json.loads(v), "record_count": n}
                                             for v, n in sorted(versions.items())]})
    manifest_path(features_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ── Write labels.csv ────────────────────────────────────────────

    if not args.features_only:
        write_mode = "w" if args.force or not labels_exists else "a"
        with open(labels_path, write_mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["apk_path"] + label_columns)
            if write_mode == "w":
                writer.writeheader()
            writer.writerows(new_labels if write_mode == "a" else label_rows)

    # ── Report ──────────────────────────────────────────────────────

    skipped_existing = len(feature_rows) - len(new_features)
    features_total = (len(existing_features) + len(new_features)
                      if not args.force else len(feature_rows))
    labels_total = (len(existing_labels) + len(new_labels)
                    if not args.force else len(label_rows))

    print(f"JSONL:          {jsonl_path}")
    print(f"Total records:  {len(records)} ({error_count} errors, {skipped} skipped)")
    if skipped_existing and not args.force:
        print(f"  Already in CSV: {skipped_existing} — skipped")
    if args.preview:
        print("  Preview: current feature rules; source versions not validated for training")
    elif args.force:
        print(f"  --force: overwriting existing CSVs")
    print(f"Features:       {features_path}")
    print(f"  Rows:          {features_total} x {fv.n_features} cols")
    if args.features_only:
        reason = "preview" if args.preview else "--features-only"
        print(f"Labels:         {labels_path} (unchanged; {reason})")
    else:
        print(f"Labels:         {labels_path}")
        print(f"  Rows:          {labels_total} ({len(label_columns)} columns)")
    if not new_features and not args.force:
        print("\nNo new records — everything is already in the CSVs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
