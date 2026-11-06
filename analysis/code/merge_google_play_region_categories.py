#!/usr/bin/env python3
"""Combine US Google Play categories with successful GB fallbacks.

Keep the country-specific source tables intact. The combined CSV retains each
row's original input identity and the country of the category observation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent.parent
CATEGORY_DIR = ROOT.parent / "google_play_dataset" / "data1" / "category"
IDENTITY_FIELDS = frozenset({"source_line", "package_name", "version_code", "sha256", "metadata_date"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(csv_path: Path, summary_path: Path, country: str) -> tuple[list[str], list[dict[str, str]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("country") != country or summary.get("snapshot_csv_sha256") != sha256(csv_path):
        raise ValueError(f"{country.upper()} source summary or CSV hash does not match.")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if len(rows) != summary.get("input_rows"):
        raise ValueError(f"{country.upper()} source row count does not match its summary.")
    if not fields or any(None in row for row in rows):
        raise ValueError(f"{country.upper()} source has an invalid CSV header or row.")
    packages = [row.get("package_name", "") for row in rows]
    if not all(packages) or len(packages) != len(set(packages)):
        raise ValueError(f"{country.upper()} source contains missing or duplicate package names.")
    return fields, rows


def atomic_text(path: Path, encoding: str, write) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge(us_csv: Path, us_summary: Path, gb_csv: Path, gb_summary: Path,
          missing_list: Path, output_csv: Path) -> dict:
    paths = (us_csv, us_summary, gb_csv, gb_summary, missing_list, output_csv)
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Input, output and summary paths must be distinct.")

    us_fields, us_rows = read_source(us_csv, us_summary, "us")
    gb_fields, gb_rows = read_source(gb_csv, gb_summary, "gb")
    if us_fields != gb_fields:
        raise ValueError("US and GB CSV schemas differ.")

    us_missing = [row["package_name"] for row in us_rows if not row["store_category_id"]]
    gb_by_package = {row["package_name"]: row for row in gb_rows}
    if set(gb_by_package) != set(us_missing):
        raise ValueError("GB result packages do not match the US missing-category set.")
    if any(row["country"] != "gb" for row in gb_rows):
        raise ValueError("GB CSV contains observations from another country.")

    recovered = {}
    for package, row in gb_by_package.items():
        if row["status"] == "success":
            if not all(row[key] for key in ("store_category_id", "research_category_id", "research_category_zh")) \
                    or row["mapping_status"] != "mapped":
                raise ValueError(f"GB success has an incomplete category: {package}")
            recovered[package] = row
        elif row["store_category_id"]:
            raise ValueError(f"GB failure unexpectedly has a category: {package}")

    remaining = [package for package in us_missing if package not in recovered]
    current_missing = [line for line in missing_list.read_text(encoding="utf-8-sig").splitlines()
                       if line and not line.startswith("#")]
    if current_missing not in (us_missing, remaining):
        raise ValueError("Missing-package list differs from both the original and merged sets.")

    merged = []
    for original in us_rows:
        row = original.copy()
        fallback = recovered.get(row["package_name"])
        if fallback:
            if row["store_category_id"]:
                raise ValueError(f"Refusing to overwrite a US category: {row['package_name']}")
            row.update({key: fallback[key] for key in us_fields if key not in IDENTITY_FIELDS})
        merged.append(row)

    if len(merged) != len(us_rows) or sum(bool(r["store_category_id"]) for r in merged) != \
            sum(bool(r["store_category_id"]) for r in us_rows) + len(recovered):
        raise ValueError("Combined category count failed reconciliation.")
    if [r["package_name"] for r in merged if not r["store_category_id"]] != remaining:
        raise ValueError("Combined missing-package set failed reconciliation.")

    # Write the full result first. A rerun safely completes the list if interrupted.
    def write_csv(stream):
        writer = csv.DictWriter(stream, fieldnames=us_fields)
        writer.writeheader()
        writer.writerows(merged)

    atomic_text(output_csv, "utf-8-sig", write_csv)

    def write_missing(stream):
        stream.write("# Packages without a Google Play category after US and GB lookup\n")
        stream.write(f"# Source: {output_csv.name}\n")
        stream.write(f"# Count: {len(remaining)}\n")
        for package in remaining:
            stream.write(package + "\n")

    atomic_text(missing_list, "utf-8", write_missing)
    return {
        "input_rows": len(us_rows),
        "us_success": sum(bool(r["store_category_id"]) for r in us_rows),
        "gb_recovered": len(recovered),
        "combined_success": sum(bool(r["store_category_id"]) for r in merged),
        "remaining_without_category": len(remaining),
        "gb_unresolved_statuses": dict(Counter(gb_by_package[p]["status"] for p in remaining)),
        "output_csv": str(output_csv.resolve()),
        "output_csv_sha256": sha256(output_csv),
        "missing_list": str(missing_list.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category-dir", type=Path, default=CATEGORY_DIR)
    args = parser.parse_args()
    directory = args.category_dir
    result = merge(directory / "categories.csv", directory / "summary.json",
                   directory / "gb" / "categories.csv", directory / "gb" / "summary.json",
                   directory / "not_fetched_packages.txt", directory / "categories_us_gb.csv")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
