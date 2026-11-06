#!/usr/bin/env python3
"""Add verified Wayback Google Play categories to the AppGoblin-resolved view.

The direct and AppGoblin observations are preserved. Wayback categories fill
only still-unresolved packages, with the archive capture time as their date.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fetch_google_play_categories import load_mapping
from fetch_wayback_category_pilot import unwrap_play_url
from merge_appgoblin_category_supplement import NEW_FIELDS, atomic_write, read_csv


ROOT = Path(__file__).resolve().parent.parent
CATEGORY_DIR = ROOT.parent / "google_play_dataset" / "data1" / "category"
MAPPING = ROOT / "code" / "google_play_category_mapping.json"
EVIDENCE_TYPE = "archived_google_play_snapshot"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_package_list(path: Path) -> list[str]:
    packages = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    if len(packages) != len(set(packages)):
        raise ValueError(f"Duplicate package in {path}")
    return packages


def capture_as_of(stamp: str) -> str:
    if len(stamp) != 14 or not stamp.isdigit():
        raise ValueError(f"Invalid Wayback capture timestamp: {stamp}")
    return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def validate_found(package: str, row: dict[str, str], mapping: dict[str, tuple[str, str]]) -> None:
    mapped = mapping.get(row["store_category_id"])
    if (row["status"] != "found" or not mapped
            or mapped != (row["research_category_id"], row["research_category_zh"])
            or row["mapping_status"] != "mapped"
            or row["category_source"] != "wayback_google_play_page"
            or len(row["response_sha256"]) != 64
            or row["parser_method"] not in {
                "wayback_raw_jsonld_identity", "wayback_rewritten_jsonld_identity"}):
        raise ValueError(f"Invalid Wayback category or provenance for {package}")
    original, stamp = unwrap_play_url(row["archive_url"])
    original_parts = urlsplit(original)
    if (stamp != row["snapshot_timestamp"] or original != row["original_play_url"]
            or parse_qs(original_parts.query).get("id") != [package]):
        raise ValueError(f"Wayback snapshot does not match package {package}")
    capture_as_of(stamp)


def merge(category_dir: Path, mapping_path: Path, output_dir: Path | None = None) -> dict:
    output_dir = output_dir or category_dir / "wayback"
    base_path = category_dir / "appgoblin" / "categories_with_appgoblin.csv"
    remaining_before_path = category_dir / "appgoblin" / "remaining_gb_404.txt"
    wayback_path = category_dir / "wayback" / "results.csv"
    wayback_summary_path = category_dir / "wayback" / "summary.json"
    fields, base = read_csv(base_path)
    wayback_fields, wayback_rows = read_csv(wayback_path)
    remaining_before = read_package_list(remaining_before_path)
    wayback_summary = json.loads(wayback_summary_path.read_text(encoding="utf-8-sig"))
    mapping, mapping_version = load_mapping(mapping_path)
    if not set(NEW_FIELDS).issubset(fields):
        raise ValueError("Base table lacks resolved category columns.")
    if (wayback_summary.get("result_csv_sha256") != sha256(wayback_path)
            or wayback_summary.get("mapping_sha256") != sha256(mapping_path)
            or wayback_summary.get("mapping_version") != mapping_version
            or wayback_summary.get("input_packages") != len(remaining_before)
            or wayback_summary.get("processed_packages") != len(wayback_rows)
            or wayback_summary.get("unprocessed_packages") != 0):
        raise ValueError("Wayback results, input population, or mapping do not match the summary.")

    base_by_package = {row["package_name"]: row for row in base}
    wayback_by_package = {row["package_name"]: row for row in wayback_rows}
    if len(base_by_package) != len(base) or len(wayback_by_package) != len(wayback_rows):
        raise ValueError("Duplicate package name in an input table.")
    if set(wayback_by_package) != set(remaining_before):
        raise ValueError("Wayback package set differs from the previous unresolved GB-404 set.")
    if not set(wayback_by_package).issubset(base_by_package):
        raise ValueError("Wayback includes a package outside the base table.")

    found = {package: row for package, row in wayback_by_package.items()
             if row["status"] == "found"}
    if (len(found) != wayback_summary.get("found_categories")
            or len(found) != wayback_summary.get("status_counts", {}).get("found")):
        raise ValueError("Wayback found count differs from the summary.")
    for package, row in found.items():
        if base_by_package[package]["resolved_category_id"]:
            raise ValueError(f"Wayback category would overwrite an existing category: {package}")
        validate_found(package, row, mapping)

    output = []
    for original in base:
        row = original.copy()
        package = row["package_name"]
        if package in found:
            supplement = found[package]
            row.update({
                "resolved_category_id": supplement["store_category_id"],
                "resolved_research_category_id": supplement["research_category_id"],
                "resolved_research_category_zh": supplement["research_category_zh"],
                "resolved_category_source": supplement["category_source"],
                "resolved_category_as_of": capture_as_of(supplement["snapshot_timestamp"]),
                "resolved_category_evidence_type": EVIDENCE_TYPE,
            })
        output.append(row)

    remaining_gb = [row["package_name"] for row in output
                    if row["package_name"] in wayback_by_package
                    and not row["resolved_category_id"]]
    remaining_all = [row["package_name"] for row in output
                     if not row["resolved_category_id"]]
    previous_resolved = sum(bool(row["resolved_category_id"]) for row in base)
    resolved = sum(bool(row["resolved_category_id"]) for row in output)
    if (len(output) != len(base) or resolved != previous_resolved + len(found)
            or len(remaining_gb) != len(remaining_before) - len(found)
            or len(remaining_all) != len(output) - resolved):
        raise ValueError("Merged category counts do not reconcile.")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "categories_with_wayback.csv"
    gb_remaining_path = output_dir / "remaining_gb_404_after_wayback.txt"
    all_remaining_path = output_dir / "remaining_all_without_category.txt"
    summary_path = output_dir / "merge_summary.json"

    def write_csv(stream):
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    def write_list(path: Path, packages: list[str], description: str):
        def callback(stream):
            stream.write(f"# {description}\n# Count: {len(packages)}\n")
            stream.writelines(package + "\n" for package in packages)
        atomic_write(path, callback)

    atomic_write(merged_path, write_csv, encoding="utf-8-sig")
    write_list(gb_remaining_path, remaining_gb,
               "GB 404 packages without direct, AppGoblin, or Wayback category")
    write_list(all_remaining_path, remaining_all,
               "All packages without direct, AppGoblin, or Wayback category")
    evidence_counts = {kind: sum(row["resolved_category_evidence_type"] == kind for row in output)
                       for kind in ("direct_google_play_observation",
                                    "third_party_historical_snapshot", EVIDENCE_TYPE)}
    summary = {
        "total_packages": len(output),
        "previous_resolved_categories": previous_resolved,
        "new_wayback_categories": len(found),
        "resolved_categories": resolved,
        "remaining_gb_404": len(remaining_gb),
        "remaining_all_without_category": len(remaining_all),
        "evidence_type_counts": evidence_counts,
        "mapping_version": mapping_version,
        "mapping_sha256": sha256(mapping_path),
        "base_file": str(base_path.resolve()),
        "base_sha256": sha256(base_path),
        "wayback_results_file": str(wayback_path.resolve()),
        "wayback_results_sha256": sha256(wayback_path),
        "merged_file": str(merged_path.resolve()),
        "merged_sha256": sha256(merged_path),
        "remaining_gb_file": str(gb_remaining_path.resolve()),
        "remaining_all_file": str(all_remaining_path.resolve()),
        "note": "Wayback categories are historical Google Play snapshots dated by capture time; original direct observations remain unchanged.",
    }

    def write_json(stream):
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    atomic_write(summary_path, write_json)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category-dir", type=Path, default=CATEGORY_DIR)
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(args.category_dir, args.mapping, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
