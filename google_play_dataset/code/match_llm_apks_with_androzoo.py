#!/usr/bin/env python3
"""Match LLM app metadata to *all* Google Play APK hashes in AndroZoo.

Example:
  python match_llm_apks_with_androzoo.py --index /data/yzhou/APK/data/latest.csv.gz \
      --input /data/yzhou/APK/data/google_play_apps_llm_candidates_full_records.jsonl \
      --output-dir /data/yzhou/APK/data

The index may also be latest_with-added-date.csv.gz. Its ``added`` field is
AndroZoo ingestion time, not an APK release date. No hash is designated latest.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    SCRIPT_ROOT / "data1" / "google_play_apps_llm_candidates_full_records.jsonl"
)
OUTPUT_NAMES = {
    "matches": "google_play_apps_llm_sha256_matches.jsonl",
    "hashes": "google_play_apps_llm_sha256s.txt",
    "summary": "google_play_apps_llm_sha256_summary.json",
}
REQUIRED_COLUMNS = {"sha256", "pkg_name", "vercode", "markets"}
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


def parse_version(value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError("versionCode is missing or boolean")
    if isinstance(value, int):
        version = value
    elif isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        version = int(value.strip())
    else:
        raise ValueError(f"invalid versionCode: {value!r}")
    if version < 0:
        raise ValueError(f"negative versionCode: {version}")
    return version


def load_targets(path: Path, expected_count: int) -> tuple[list[dict], str]:
    targets = []
    seen_docids = set()
    seen_pairs = set()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                raise ValueError(f"Blank input line {line_number}")
            try:
                record = json.loads(raw_line)
                app = record["details"]["appDetails"]
                docid = record["docid"]
                package = app["packageName"]
                version = parse_version(app["versionCode"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid input line {line_number}: {error}") from error
            if not isinstance(docid, str) or not docid.strip():
                raise ValueError(f"Invalid docid on input line {line_number}")
            if not isinstance(package, str) or not package.strip():
                raise ValueError(f"Invalid packageName on input line {line_number}")
            if package != docid:
                raise ValueError(
                    f"Package/docid mismatch on input line {line_number}: "
                    f"{package!r} != {docid!r}"
                )
            pair = (package, version)
            if docid in seen_docids or pair in seen_pairs:
                raise ValueError(f"Duplicate input docid or package/version: {pair!r}")
            seen_docids.add(docid)
            seen_pairs.add(pair)
            targets.append({
                "docid": docid,
                "packageName": package,
                "versionCode": version,
                "uploadDate": app.get("uploadDate"),
                "az_metadata_date": record.get("az_metadata_date"),
            })
    if not targets:
        raise ValueError("Input contains no application records")
    if expected_count and len(targets) != expected_count:
        raise ValueError(
            f"Expected {expected_count:,} input records, found {len(targets):,}"
        )
    return targets, digest.hexdigest()


def contains_google_play(markets: str) -> bool:
    return "play.google.com" in {
        market.strip().lower() for market in markets.split("|")
    }


def scan_index(index_path: Path, targets: list[dict]) -> tuple[dict, Counter]:
    wanted = {(app["packageName"], app["versionCode"]) for app in targets}
    wanted_packages = {package for package, _ in wanted}
    matches: dict[tuple[str, int], dict[str, list[dict]]] = {}
    hash_owner: dict[str, tuple[str, int]] = {}
    stats = Counter()
    open_index = gzip.open if index_path.suffix.lower() == ".gz" else open
    with open_index(index_path, "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(f"AndroZoo index lacks columns: {sorted(missing_columns)}")
        for row in reader:
            stats["index_rows_scanned"] += 1
            if stats["index_rows_scanned"] % 2_000_000 == 0:
                print(
                    f"Scanned {stats['index_rows_scanned']:,} index rows; "
                    f"matched {len(matches):,}/{len(wanted):,} app versions",
                    flush=True,
                )
            markets = row.get("markets") or ""
            if not contains_google_play(markets):
                continue
            stats["google_play_index_rows"] += 1
            package = (row.get("pkg_name") or "").strip()
            if package not in wanted_packages:
                continue
            stats["target_package_index_rows"] += 1
            try:
                version = parse_version(row.get("vercode"))
            except ValueError:
                stats["target_package_rows_with_invalid_version"] += 1
                continue
            pair = (package, version)
            if pair not in wanted:
                continue
            stats["target_pair_index_rows"] += 1
            if None in row:
                raise ValueError(f"Malformed target CSV row near line {reader.line_num}")
            sha = (row.get("sha256") or "").strip().upper()
            if not SHA256_PATTERN.fullmatch(sha):
                raise ValueError(
                    f"Invalid SHA-256 for {pair!r} near CSV line {reader.line_num}: {sha!r}"
                )
            previous_pair = hash_owner.setdefault(sha, pair)
            if previous_pair != pair:
                raise ValueError(
                    f"SHA-256 {sha} maps to both {previous_pair!r} and {pair!r}"
                )
            by_sha = matches.setdefault(pair, {})
            if sha in by_sha:
                stats["repeated_target_sha256_rows"] += 1
            # Keep every index row for this hash; never silently discard conflicts.
            by_sha.setdefault(sha, []).append(dict(row))
    return matches, stats


def build_summary(
    targets: list[dict], matches: dict, stats: Counter,
    input_path: Path, index_path: Path, input_hash: str,
) -> dict:
    matched = sum(bool(matches.get((app["packageName"], app["versionCode"]))) for app in targets)
    ambiguous = sum(len(matches.get((app["packageName"], app["versionCode"]), {})) > 1 for app in targets)
    all_hashes = {sha for by_sha in matches.values() for sha in by_sha}
    return {
        "policy": "exact-package-and-version; Google-Play-market-token; retain-all-distinct-SHA256",
        "latest_sha256_selected": False,
        "input": str(input_path),
        "input_sha256": input_hash,
        "androzoo_index": str(index_path),
        "androzoo_index_bytes": index_path.stat().st_size,
        "counts": {
            "input_app_versions": len(targets),
            "matched_app_versions": matched,
            "unmatched_app_versions": len(targets) - matched,
            "one_sha256_app_versions": matched - ambiguous,
            "multiple_sha256_app_versions": ambiguous,
            "unique_sha256s": len(all_hashes),
            **dict(stats),
        },
        "notes": [
            "A package/version pair may have multiple distinct APK hashes; none is automatically designated newest.",
            "The optional AndroZoo added field is ingestion time, not APK release time; dex_date is unreliable.",
            "markets is a pipe-separated list; any exact play.google.com token qualifies.",
        ],
    }


def write_outputs(
    output_dir: Path, targets: list[dict], matches: dict, summary: dict,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {key: output_dir / name for key, name in OUTPUT_NAMES.items()}
    existing = [str(path) for path in destinations.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing output: {existing}")
    all_hashes = sorted({sha for by_sha in matches.values() for sha in by_sha})

    with tempfile.TemporaryDirectory(prefix="sha256_match_", dir=output_dir) as temp_name:
        temp_dir = Path(temp_name)
        staged = {key: temp_dir / name for key, name in OUTPUT_NAMES.items()}
        with staged["matches"].open("w", encoding="utf-8", newline="\n") as output:
            for target in targets:
                pair = (target["packageName"], target["versionCode"])
                by_sha = matches.get(pair, {})
                record = {
                    **target,
                    "status": "unmatched" if not by_sha else (
                        "unique" if len(by_sha) == 1 else "multiple"
                    ),
                    "sha256_count": len(by_sha),
                    "matches": [
                        {"sha256": sha, "index_rows": by_sha[sha]}
                        for sha in sorted(by_sha)
                    ],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        with staged["hashes"].open("w", encoding="ascii", newline="\n") as output:
            for sha in all_hashes:
                output.write(sha + "\n")
        with staged["summary"].open("w", encoding="utf-8", newline="\n") as output:
            json.dump(summary, output, ensure_ascii=False, indent=2)
            output.write("\n")

        # Independently re-read staged outputs before publishing them.
        verified_pairs = set()
        verified_hashes = set()
        with staged["matches"].open("r", encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                pair = (record["packageName"], record["versionCode"])
                if pair in verified_pairs:
                    raise ValueError(f"Duplicate pair in output: {pair!r}")
                verified_pairs.add(pair)
                hashes = {item["sha256"] for item in record["matches"]}
                if len(hashes) != record["sha256_count"]:
                    raise ValueError(f"Output SHA-256 count mismatch: {pair!r}")
                verified_hashes.update(hashes)
        if len(verified_pairs) != len(targets) or verified_hashes != set(all_hashes):
            raise ValueError("Staged output does not match input targets or hash list")
        with staged["hashes"].open("r", encoding="ascii") as source:
            if [line.strip() for line in source] != all_hashes:
                raise ValueError("Text SHA-256 list differs from JSONL results")
        for path in destinations.values():
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing output: {path}")
        for key, path in destinations.items():
            os.replace(staged[key], path)
    return destinations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Full LLM candidate metadata JSONL (default: {DEFAULT_INPUT})")
    parser.add_argument("--index", type=Path, required=True,
                        help="AndroZoo latest.csv.gz or latest_with-added-date.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: input file's directory)")
    parser.add_argument("--expected-count", type=int, default=8699,
                        help="Required number of input apps; use 0 to disable (default: 8699)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_count < 0:
        raise ValueError("--expected-count cannot be negative")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input not found: {args.input}")
    if not args.index.is_file():
        raise FileNotFoundError(f"AndroZoo index not found: {args.index}")
    output_dir = args.output_dir or args.input.parent
    targets, input_hash = load_targets(args.input, args.expected_count)
    print(f"Validated {len(targets):,} candidate apps; input SHA-256: {input_hash}", flush=True)
    matches, stats = scan_index(args.index, targets)
    summary = build_summary(targets, matches, stats, args.input, args.index, input_hash)
    outputs = write_outputs(output_dir, targets, matches, summary)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2), flush=True)
    for path in outputs.values():
        print(f"Saved: {path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, csv.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
