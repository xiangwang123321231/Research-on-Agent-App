#!/usr/bin/env python3
"""Verify downloaded APK content against a SHA-256 list, without changing APKs.

Defaults match download_androzoo_apks_copy.py:
  input:  code/google_play_apps_llm_sha256s_new_only_848.txt
  APKs:   apks/ beside code/
  report: code/apk_sha256_verification_report.json

Only Python's standard library is required. Exit codes: 0 = all verified,
1 = APK verification problems, 2 = invalid input or report-writing error.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SHA_PATTERN = re.compile(r"[0-9A-Fa-f]{64}\Z")
STATUSES = ("ok", "missing", "hash_mismatch", "read_error", "duplicate_filenames", "changed_during_check")


def load_hashes(path, expected_count):
    hashes = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            sha = line.strip().upper()
            if not sha:
                continue
            if not SHA_PATTERN.fullmatch(sha):
                raise ValueError("Invalid SHA-256 at line {}".format(line_number))
            if sha in seen:
                raise ValueError("Duplicate SHA-256 at line {}: {}".format(line_number, sha))
            seen.add(sha)
            hashes.append(sha)
    if not hashes:
        raise ValueError("SHA-256 list is empty")
    if len(hashes) != expected_count:
        raise ValueError("Expected {} unique hashes, found {}".format(expected_count, len(hashes)))
    return hashes


def find_apks(directory, wanted):
    matches = {sha: [] for sha in wanted}
    ignored = 0
    for path in directory.iterdir():
        if path.suffix.lower() != ".apk":
            continue
        sha = path.stem.upper()
        if sha in matches:
            matches[sha].append(path)
        else:
            ignored += 1
    return matches, ignored


def stat_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def verify_one(sha, paths):
    result = {"expected_sha256": sha, "apk_names": sorted(path.name for path in paths)}
    if not paths:
        result["status"] = "missing"
        return result
    if len(paths) > 1:
        result["status"] = "duplicate_filenames"
        return result

    path = paths[0]
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("APK path is not a regular file (symlinks are not followed)")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(source.fileno())
        actual = digest.hexdigest().upper()
        result["actual_sha256"] = actual
        result["bytes"] = after.st_size
        if stat_identity(before) != stat_identity(after) or stat_identity(after) != stat_identity(path.stat()):
            result["status"] = "changed_during_check"
        else:
            result["status"] = "ok" if actual == sha else "hash_mismatch"
    except OSError as error:
        result["status"] = "read_error"
        result["error"] = str(error)
    return result


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent),
            prefix="." + path.name + ".", suffix=".tmp", delete=False,
        ) as destination:
            temporary = Path(destination.name)
            json.dump(report, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(str(temporary), str(path))
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha-list", type=Path,
                        default=SCRIPT_DIR / "google_play_apps_llm_sha256s_new_only_848.txt",
                        help="SHA-256 list; default is the 848-hash TXT next to this script")
    parser.add_argument("--apk-dir", type=Path, default=SCRIPT_DIR.parent / "apks",
                        help="Downloaded APK folder (default: ../apks)")
    parser.add_argument("--report", type=Path, default=SCRIPT_DIR / "apk_sha256_verification_report.json",
                        help="JSON report (default: code/apk_sha256_verification_report.json)")
    parser.add_argument("--expected-count", type=int, default=848,
                        help="Required number of unique hashes (default: 848)")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.expected_count < 1:
        raise ValueError("--expected-count must be positive")
    sha_list = args.sha_list.resolve()
    apk_dir = args.apk_dir.resolve()
    report_path = args.report.absolute()
    if not sha_list.is_file():
        raise ValueError("SHA-256 list not found: {}".format(sha_list))
    if not apk_dir.is_dir():
        raise ValueError("APK folder not found: {}".format(apk_dir))
    if report_path.suffix.lower() != ".json":
        raise ValueError("Report filename must end in .json")
    if report_path.is_symlink() or report_path.resolve() == sha_list:
        raise ValueError("Report cannot be a symlink or overwrite the SHA-256 list")

    hashes = load_hashes(sha_list, args.expected_count)
    matches, ignored = find_apks(apk_dir, set(hashes))
    print("List: {} unique hashes; APK folder: {}".format(len(hashes), apk_dir), flush=True)
    print("Other APKs outside this list are ignored: {}".format(ignored), flush=True)
    counts = Counter({status: 0 for status in STATUSES})
    records = []
    for index, sha in enumerate(hashes, 1):
        result = verify_one(sha, matches[sha])
        records.append(result)
        counts[result["status"]] += 1
        if result["status"] != "ok":
            print("[{} / {}] {}: {}".format(index, len(hashes), result["status"], sha), flush=True)
        if index % 25 == 0 or index == len(hashes):
            print("Checked {}/{}; verified={}; problems={}".format(
                index, len(hashes), counts["ok"], index - counts["ok"]
            ), flush=True)

    summary = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha_list": str(sha_list),
        "apk_dir": str(apk_dir),
        "expected_hashes": len(hashes),
        "status_counts": dict(counts),
        "other_apks_ignored": ignored,
        "all_verified": counts["ok"] == len(hashes),
    }
    write_report(report_path, {"summary": summary, "records": records})
    print("Summary: " + ", ".join("{}={}".format(status, counts[status]) for status in STATUSES), flush=True)
    print("Report: {}".format(report_path), flush=True)
    return 0 if summary["all_verified"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(2)
