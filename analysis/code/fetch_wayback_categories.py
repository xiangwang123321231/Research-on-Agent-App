#!/usr/bin/env python3
"""Fetch archived Google Play categories for all remaining GB-404 packages.

Run the same command again to resume. Results are historical observations and
must be kept separate from current Google Play and AppGoblin source tables.
HTTP 429 is treated as a service-wide pause: stop without recording a new
package as complete, so the next run can resume it after access recovers.
Python 3.10+; standard library only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

from fetch_google_play_categories import load_mapping
from fetch_wayback_category_pilot import (
    CATEGORY_DIR, DEFAULT_INPUT, FIELDS, MAPPING, Pacer, lookup, read_population, utc_now,
)


DEFAULT_OUTPUT = CATEGORY_DIR / "wayback"
DEFAULT_PILOT = CATEGORY_DIR / "wayback_pilot" / "results.csv"
IMPORTABLE_FROM_PILOT = {"found", "no_snapshot"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_results(path: Path, population: list[str]) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected result columns in {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed result row in {path}")
    result = {row["package_name"]: row for row in rows}
    if len(result) != len(rows) or not set(result).issubset(population):
        raise ValueError(f"Duplicate or out-of-scope package in {path}")
    return result


def import_pilot(rows: dict[str, dict[str, str]], pilot_path: Path,
                 population: list[str]) -> int:
    if not pilot_path.exists():
        return 0
    pilot = read_results(pilot_path, population)
    imported = 0
    for package, row in pilot.items():
        if package not in rows and row["status"] in IMPORTABLE_FROM_PILOT:
            rows[package] = row
            imported += 1
    return imported


def atomic_write(path: Path, callback, *, encoding: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding, newline="") as stream:
            callback(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_state(output: Path, population: list[str], rows: dict[str, dict[str, str]],
                input_path: Path, input_sha: str, mapping_sha: str,
                mapping_version: str) -> None:
    def write_csv(stream):
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[package] for package in population if package in rows)

    result_path = output / "results.csv"
    atomic_write(result_path, write_csv, encoding="utf-8-sig")
    counts = Counter(row["status"] for row in rows.values())
    summary = {
        "input_file": str(input_path.resolve()), "input_sha256": input_sha,
        "input_packages": len(population), "mapping_version": mapping_version,
        "mapping_sha256": mapping_sha,
        "processed_packages": len(rows), "unprocessed_packages": len(population) - len(rows),
        "status_counts": dict(sorted(counts.items())),
        "found_categories": counts["found"],
        "found_unmapped": sum(r["status"] == "found" and r["mapping_status"] == "unmapped"
                              for r in rows.values()),
        "updated_at_utc": utc_now(),
        "result_csv_sha256": sha256(result_path),
        "interpretation": "Historical archived Google Play pages; snapshot dates are not current category dates.",
    }

    def write_json(stream):
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    atomic_write(output / "summary.json", write_json, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, int | str]:
    if args.request_interval < 1:
        raise ValueError("Use --request-interval >=1.")
    if args.max_new is not None and args.max_new < 1:
        raise ValueError("--max-new must be positive.")
    input_sha = sha256(args.input)
    population = read_population(args.input)
    if sha256(args.input) != input_sha:
        raise ValueError("Input package list changed while reading.")
    mapping_sha = sha256(args.mapping)
    mapping, mapping_version = load_mapping(args.mapping)
    if sha256(args.mapping) != mapping_sha:
        raise ValueError("Mapping changed while reading.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_summary = args.output_dir / "summary.json"
    if existing_summary.exists():
        if not (args.output_dir / "results.csv").exists():
            raise ValueError("Summary exists but results.csv is missing; restore the output pair.")
        previous = json.loads(existing_summary.read_text(encoding="utf-8-sig"))
        if (previous.get("input_sha256") != input_sha
                or previous.get("mapping_sha256") != mapping_sha):
            raise ValueError("Input list or category mapping changed; choose a new output directory.")
    rows = read_results(args.output_dir / "results.csv", population)
    imported = import_pilot(rows, args.import_pilot, population) if args.import_pilot else 0
    # A resume with no new imports must leave completed result files untouched.
    if imported or not existing_summary.exists():
        write_state(args.output_dir, population, rows, args.input,
                    input_sha, mapping_sha, mapping_version)
    print(f"Input {len(population)} packages; reused {imported} verified pilot results; "
          f"already recorded {len(rows)}.", flush=True)
    print("Each unrecorded package is queried once; HTTP 429 stops the run without saving that package.",
          flush=True)
    if args.prepare_only:
        return {"exit_code": 0, "processed": len(rows),
                "found": sum(r["status"] == "found" for r in rows.values()),
                "output_dir": str(args.output_dir.resolve())}

    pacer = Pacer(args.request_interval)
    new_count = 0
    for index, package in enumerate(population, 1):
        if package in rows and not (args.retry_rate_limited and rows[package]["status"] == "rate_limited"):
            continue
        row = lookup(package, mapping, pacer)
        if row["status"] == "rate_limited":
            print(f"{index}/{len(population)} {package}: HTTP 429; Wayback is limiting access. "
                  "Stopped without changing this package's saved result. "
                  "Rerun after access recovers.", file=sys.stderr, flush=True)
            return {"exit_code": 3, "processed": len(rows),
                    "found": sum(r["status"] == "found" for r in rows.values()),
                    "paused_package": package}
        rows[package] = row
        write_state(args.output_dir, population, rows, args.input,
                    input_sha, mapping_sha, mapping_version)
        print(f"{index}/{len(population)} {package}: {row['status']} "
              f"{row['store_category_id']}", flush=True)
        new_count += 1
        if args.max_new is not None and new_count >= args.max_new:
            break
    return {"exit_code": 0, "processed": len(rows),
            "found": sum(r["status"] == "found" for r in rows.values()),
            "output_dir": str(args.output_dir.resolve())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--import-pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--prepare-only", action="store_true", help="Validate inputs and import pilot results without network requests.")
    parser.add_argument("--retry-rate-limited", action="store_true",
                        help="Explicitly revisit previously saved 429 rows after Wayback access recovers.")
    parser.add_argument("--request-interval", type=float, default=10.0,
                        help="Minimum seconds between any two Wayback requests (default: 10).")
    parser.add_argument("--rate-limit-wait", type=float, default=180.0,
                        help="Deprecated; accepted for older commands but ignored (429 stops the run).")
    parser.add_argument("--max-rate-limit-retries", type=int, default=2,
                        help="Deprecated; accepted for older commands but ignored (no retries).")
    parser.add_argument("--max-new", type=int, help="Stop after this many new package results; useful for smoke tests.")
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except KeyboardInterrupt:
        print("Interrupted; completed results are saved. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Configuration or data error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
