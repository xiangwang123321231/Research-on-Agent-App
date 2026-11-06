#!/usr/bin/env python3
"""Join AppGoblin's free Google app-category dump to missing Play packages.

The source is a third-party historical snapshot, not a live Google Play query.
Python 3.10+; standard library only. The compressed dump is streamed and not
retained locally; its SHA-256 is recorded for reproducibility.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import lzma
from pathlib import Path
import urllib.request

from fetch_google_play_categories import load_mapping


ROOT = Path(__file__).resolve().parent.parent
CATEGORY_DIR = ROOT.parent / "google_play_dataset" / "data1" / "category"
SOURCE_URL = "https://raw.githubusercontent.com/appgoblin-dev/appgoblin-data/main/data/live_store_apps.tsv.xz"
FIELDS = [
    "package_name", "store_category_id", "store_category_name_raw",
    "research_category_id", "research_category_zh", "mapping_status",
    "app_name", "appgoblin_updated_at", "last_crawl_result", "category_source",
]


class HashedReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()

    def read(self, count: int = -1) -> bytes:
        chunk = self.source.read(count)
        self.digest.update(chunk)
        return chunk

    def seekable(self) -> bool:
        return False


def load_csv_by_package(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_package = {row["package_name"]: row for row in rows}
    if len(by_package) != len(rows):
        raise ValueError(f"Duplicate package names in {path}")
    return by_package


def collect(source, packages: set[str]) -> tuple[dict[str, list[dict[str, str]]], str]:
    hashed = HashedReader(source)
    found: dict[str, list[dict[str, str]]] = {}
    with lzma.open(hashed, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"store", "store_id", "app_category", "app_name", "appgoblin_updated_at", "last_crawl_result"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Unexpected AppGoblin columns: {reader.fieldnames}")
        for row in reader:
            if row["store"].lower() != "google" or row["store_id"] not in packages:
                continue
            found.setdefault(row["store_id"], []).append(row)
    return found, hashed.digest.hexdigest()


def atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--source-file", type=Path, help="Use an already-downloaded .tsv.xz file.")
    parser.add_argument("--output-dir", type=Path, default=CATEGORY_DIR / "appgoblin")
    parser.add_argument("--merged", type=Path, default=CATEGORY_DIR / "categories_us_gb.csv")
    parser.add_argument("--gb", type=Path, default=CATEGORY_DIR / "gb" / "categories.csv")
    parser.add_argument("--mapping", type=Path, default=ROOT / "code" / "google_play_category_mapping.json")
    args = parser.parse_args(argv)

    current = load_csv_by_package(args.merged)
    gb = load_csv_by_package(args.gb)
    if not set(gb).issubset(current):
        raise ValueError("GB results contain packages outside merged data.")
    eligible = {p for p, row in gb.items()
                if row["status"] == "not_available_in_country" and not current[p]["store_category_id"]}
    known = {p: row["store_category_id"] for p, row in current.items() if row["store_category_id"]}
    mapping, mapping_version = load_mapping(args.mapping)

    if args.source_file:
        with args.source_file.open("rb") as stream:
            records, source_sha = collect(stream, set(current))
        source_name = str(args.source_file.resolve())
    else:
        request = urllib.request.Request(args.source_url, headers={"User-Agent": "research-category-join/1.0"})
        with urllib.request.urlopen(request, timeout=60) as stream:
            records, source_sha = collect(stream, set(current))
        source_name = args.source_url

    conflicts = []
    missing_rows = []
    comparable = 0
    exact_agree = 0
    broad_agree = 0
    for package, candidates in sorted(records.items()):
        categories = {r["app_category"].strip().upper() for r in candidates if r["app_category"].strip()}
        if len(categories) > 1:
            conflicts.append(package)
            continue
        if not categories:
            continue
        category = next(iter(categories))
        app = max(candidates, key=lambda r: r["appgoblin_updated_at"])
        if package in known:
            comparable += 1
            exact_agree += category == known[package]
            broad_agree += (mapping.get(category, (None,))[0] is not None
                            and mapping.get(category, (None,))[0] == mapping.get(known[package], (None,))[0])
        if package not in eligible:
            continue
        group = mapping.get(category)
        missing_rows.append({
            "package_name": package, "store_category_id": category,
            "store_category_name_raw": app["app_category"],
            "research_category_id": group[0] if group else "",
            "research_category_zh": group[1] if group else "",
            "mapping_status": "mapped" if group else "unmapped",
            "app_name": app["app_name"],
            "appgoblin_updated_at": app["appgoblin_updated_at"],
            "last_crawl_result": app["last_crawl_result"],
            "category_source": "appgoblin_free_live_store_apps_snapshot",
        })

    payload = {
        "source_url_or_file": source_name, "source_file_sha256": source_sha,
        "source_license": "CC BY 4.0; attribution to AppGoblin required",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping_version": mapping_version,
        "gb_404_eligible": len(eligible),
        "eligible_matched_with_category": len(missing_rows),
        "eligible_match_rate": round(len(missing_rows) / len(eligible), 4),
        "eligible_unmapped": sum(row["mapping_status"] == "unmapped" for row in missing_rows),
        "eligible_category_counts": dict(sorted(Counter(r["store_category_id"] for r in missing_rows).items())),
        "validation_known_compared": comparable,
        "validation_exact_category_agree": exact_agree,
        "validation_exact_agreement_rate": round(exact_agree / comparable, 4) if comparable else None,
        "validation_broad_category_agree": broad_agree,
        "validation_broad_agreement_rate": round(broad_agree / comparable, 4) if comparable else None,
        "conflicting_source_packages": conflicts,
        "note": "Historical third-party category; do not describe as a current direct Google Play observation.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "matched_categories.csv"
    atomic_csv(result_path, missing_rows)
    atomic_json(args.output_dir / "summary.json", payload)
    print(f"Matched {len(missing_rows)}/{len(eligible)} GB 404 packages; "
          f"validation exact {exact_agree}/{comparable}; results: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
