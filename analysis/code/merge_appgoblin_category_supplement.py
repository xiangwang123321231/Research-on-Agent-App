#!/usr/bin/env python3
"""Add historical AppGoblin categories to a separate research-ready view.

Direct Google Play observations remain unchanged. The new resolved_* columns
make third-party historical supplements explicit rather than recasting them
as current Google Play observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

from fetch_google_play_categories import load_mapping


ROOT = Path(__file__).resolve().parent.parent
CATEGORY_DIR = ROOT.parent / "google_play_dataset" / "data1" / "category"
NEW_FIELDS = [
    "resolved_category_id", "resolved_research_category_id", "resolved_research_category_zh",
    "resolved_category_source", "resolved_category_as_of", "resolved_category_evidence_type",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or any(None in row for row in rows):
        raise ValueError(f"Malformed CSV: {path}")
    return fields, rows


def atomic_write(path: Path, callback, *, encoding: str = "utf-8") -> None:
    descriptor, temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as stream:
            callback(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def merge(category_dir: Path, mapping_path: Path) -> dict[str, int | str]:
    direct_path = category_dir / "categories_us_gb.csv"
    gb_path = category_dir / "gb" / "categories.csv"
    matched_path = category_dir / "appgoblin" / "matched_categories.csv"
    source_summary_path = category_dir / "appgoblin" / "summary.json"
    output_path = category_dir / "appgoblin" / "categories_with_appgoblin.csv"
    remaining_path = category_dir / "appgoblin" / "remaining_gb_404.txt"

    fields, direct = read_csv(direct_path)
    _, gb = read_csv(gb_path)
    _, supplements = read_csv(matched_path)
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    mapping, mapping_version = load_mapping(mapping_path)
    if source_summary.get("mapping_version") != mapping_version:
        raise ValueError("Supplement and current category mapping versions differ.")

    direct_by_package = {r["package_name"]: r for r in direct}
    gb_by_package = {r["package_name"]: r for r in gb}
    supplement_by_package = {r["package_name"]: r for r in supplements}
    if (len(direct_by_package) != len(direct)
            or len(gb_by_package) != len(gb)
            or len(supplement_by_package) != len(supplements)):
        raise ValueError("Duplicate package name in an input table.")
    if not set(gb_by_package).issubset(direct_by_package):
        raise ValueError("GB packages are not all in the direct category table.")

    gb_404 = {p for p, r in gb_by_package.items()
              if r["status"] == "not_available_in_country" and not direct_by_package[p]["store_category_id"]}
    if len(gb_404) != source_summary.get("gb_404_eligible"):
        raise ValueError("GB 404 population changed since the AppGoblin match.")
    if len(supplements) != source_summary.get("eligible_matched_with_category"):
        raise ValueError("AppGoblin match count differs from its summary.")
    if not set(supplement_by_package).issubset(gb_404):
        raise ValueError("AppGoblin supplement includes a package with a direct category or non-404 GB status.")

    for package, row in supplement_by_package.items():
        category = row["store_category_id"]
        mapped = mapping.get(category)
        if (not mapped or mapped != (row["research_category_id"], row["research_category_zh"])
                or row["mapping_status"] != "mapped"
                or row["category_source"] != "appgoblin_free_live_store_apps_snapshot"
                or row["last_crawl_result"] != "success"
                or not row["appgoblin_updated_at"]):
            raise ValueError(f"Invalid AppGoblin category or provenance for {package}")

    output = []
    for direct_row in direct:
        package = direct_row["package_name"]
        row = direct_row.copy()
        if direct_row["store_category_id"]:
            row.update({
                "resolved_category_id": direct_row["store_category_id"],
                "resolved_research_category_id": direct_row["research_category_id"],
                "resolved_research_category_zh": direct_row["research_category_zh"],
                "resolved_category_source": f"google_play_web_{direct_row['country']}",
                "resolved_category_as_of": direct_row["category_observed_at_utc"],
                "resolved_category_evidence_type": "direct_google_play_observation",
            })
        elif package in supplement_by_package:
            supplement = supplement_by_package[package]
            row.update({
                "resolved_category_id": supplement["store_category_id"],
                "resolved_research_category_id": supplement["research_category_id"],
                "resolved_research_category_zh": supplement["research_category_zh"],
                "resolved_category_source": supplement["category_source"],
                "resolved_category_as_of": supplement["appgoblin_updated_at"],
                "resolved_category_evidence_type": "third_party_historical_snapshot",
            })
        else:
            row.update({field: "" for field in NEW_FIELDS})
        output.append(row)

    remaining_404 = [r["package_name"] for r in direct if r["package_name"] in gb_404 - set(supplement_by_package)]
    technical = [r["package_name"] for r in direct
                 if not r["store_category_id"] and r["package_name"] not in gb_404]
    if len(output) != len(direct) or len(remaining_404) + len(supplements) != len(gb_404):
        raise ValueError("Combined category counts do not reconcile.")
    if sum(bool(r["resolved_category_id"]) for r in output) != \
            sum(bool(r["store_category_id"]) for r in direct) + len(supplements):
        raise ValueError("Resolved category count does not reconcile.")

    def write_csv(stream):
        writer = csv.DictWriter(stream, fieldnames=fields + NEW_FIELDS)
        writer.writeheader()
        writer.writerows(output)

    def write_remaining(stream):
        stream.write("# GB 404 packages still without a direct or AppGoblin historical category\n")
        stream.write(f"# Count: {len(remaining_404)}\n")
        stream.write("# Four GB parse/network errors are excluded from this list.\n")
        stream.writelines(package + "\n" for package in remaining_404)

    atomic_write(output_path, write_csv, encoding="utf-8-sig")
    atomic_write(remaining_path, write_remaining)
    return {
        "total_packages": len(output),
        "direct_categories": sum(bool(r["store_category_id"]) for r in direct),
        "appgoblin_historical_categories": len(supplements),
        "resolved_categories": sum(bool(r["resolved_category_id"]) for r in output),
        "remaining_gb_404": len(remaining_404),
        "gb_technical_errors_not_in_remaining_list": len(technical),
        "all_unresolved_including_technical_errors": len(remaining_404) + len(technical),
        "output_csv": str(output_path.resolve()),
        "output_csv_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "remaining_list": str(remaining_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category-dir", type=Path, default=CATEGORY_DIR)
    parser.add_argument("--mapping", type=Path, default=ROOT / "code" / "google_play_category_mapping.json")
    args = parser.parse_args()
    print(json.dumps(merge(args.category_dir, args.mapping), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
