#!/usr/bin/env python3
"""Test 100 missing Google Play packages against archived Wayback pages.

The pilot saves historical categories separately from current Play and
AppGoblin results. It uses only standard-library Python 3.10+ modules.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from fetch_google_play_categories import _JsonLdParser, _schema_nodes, load_mapping


ROOT = Path(__file__).resolve().parent.parent
CATEGORY_DIR = ROOT.parent / "google_play_dataset" / "data1" / "category"
DEFAULT_INPUT = CATEGORY_DIR / "appgoblin" / "remaining_gb_404.txt"
DEFAULT_OUTPUT = CATEGORY_DIR / "wayback_pilot"
MAPPING = ROOT / "code" / "google_play_category_mapping.json"
SEED = 20261004
SAMPLE_SIZE = 100
PACKAGE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+\Z")
WAYBACK_PATH_RE = re.compile(r"^/web/(\d{14})(?:[a-z_]+)?/(https?://.+)$")
MAX_HTML_BYTES = 4 * 1024 * 1024
FIELDS = [
    "package_name", "status", "store_category_id", "research_category_id",
    "research_category_zh", "mapping_status", "store_title_historical",
    "category_source", "snapshot_timestamp", "archive_url", "original_play_url",
    "category_observed_at_utc", "response_sha256", "parser_method",
    "availability_http_status", "snapshot_http_status", "error_code",
]
TERMINAL_STATUSES = {"found", "no_snapshot", "parse_error"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_population(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) < SAMPLE_SIZE or len(lines) != len(set(lines)):
        raise ValueError("Input must have at least 100 unique packages.")
    if any(not PACKAGE_RE.fullmatch(package) for package in lines):
        raise ValueError("Input contains an invalid package name.")
    return lines


def seeded_sample(population: list[str]) -> list[str]:
    # Keep the 12-package feasibility probe inside this larger random sample.
    rng = random.Random(SEED)
    first_twelve = rng.sample(population, 12)
    remaining = [p for p in population if p not in set(first_twelve)]
    return first_twelve + rng.sample(remaining, SAMPLE_SIZE - 12)


def prepare_sample(output: Path, population: list[str]) -> list[str]:
    sample = seeded_sample(population)
    path = output / "packages_100.txt"
    if path.exists():
        existing = path.read_text(encoding="utf-8-sig").splitlines()
        if existing != sample:
            raise ValueError("Existing sample differs from current source or fixed seed.")
    else:
        output.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sample) + "\n", encoding="utf-8")
    return sample


def unwrap_play_url(url: str) -> tuple[str, str | None]:
    """Return original Play URL and optional Wayback timestamp, or reject it."""
    parts = urllib.parse.urlsplit(url)
    stamp = None
    if parts.hostname == "web.archive.org":
        match = WAYBACK_PATH_RE.fullmatch(parts.path)
        if not match:
            raise ValueError("not_a_wayback_capture")
        stamp = match.group(1)
        url = match.group(2) + ("?" + parts.query if parts.query else "")
        parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"http", "https"} or parts.hostname != "play.google.com":
        raise ValueError("not_a_google_play_url")
    if parts.path != "/store/apps/details" and not parts.path.startswith("/store/apps/details/"):
        raise ValueError("not_an_app_details_url")
    return url, stamp


def url_matches_package(url: str, package: str) -> bool:
    try:
        original, _ = unwrap_play_url(url)
        return urllib.parse.parse_qs(urllib.parse.urlsplit(original).query).get("id") == [package]
    except ValueError:
        return False


def parse_archived_category(html: str, package: str) -> tuple[str, str]:
    parser = _JsonLdParser()
    parser.feed(html)
    matches = []
    for block in parser.blocks:
        try:
            data = json.loads(block)
        except (ValueError, RecursionError):
            continue
        for node in _schema_nodes(data):
            kind = node.get("@type")
            types = kind if isinstance(kind, list) else [kind]
            if (any(k in {"SoftwareApplication", "MobileApplication"} for k in types if isinstance(k, str))
                    and isinstance(node.get("url"), str)
                    and url_matches_package(node["url"], package)):
                matches.append(node)
    if not matches:
        raise ValueError("missing_identity_matched_app_schema")
    values = [node.get("applicationCategory") for node in matches]
    if any(not isinstance(value, str) for value in values):
        raise ValueError("missing_or_invalid_application_category")
    categories = set(values)
    if len(categories) != 1:
        raise ValueError("conflicting_application_categories")
    category = next(iter(categories))
    if not isinstance(category, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", category):
        raise ValueError("missing_or_invalid_application_category")
    title = matches[0].get("name")
    return category, title if isinstance(title, str) else ""


class Pacer:
    def __init__(self, interval: float):
        self.interval = interval
        self.next_at = 0.0

    def wait(self):
        delay = self.next_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self.next_at = time.monotonic() + self.interval


def get_bytes(url: str, pacer: Pacer, *, timeout: float, limit: int) -> tuple[bytes, int]:
    pacer.wait()
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; academic-category-research/1.0)",
        "Accept": "text/html,application/json",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read(limit + 1)
        if len(body) > limit:
            raise ValueError("response_too_large")
        return body, response.status


def lookup(package: str, mapping: dict[str, tuple[str, str]], pacer: Pacer) -> dict[str, str]:
    result = dict.fromkeys(FIELDS, "")
    result.update(package_name=package, category_source="wayback_google_play_page",
                  category_observed_at_utc=utc_now())
    target = f"https://play.google.com/store/apps/details?id={package}"
    availability = "https://archive.org/wayback/available?" + urllib.parse.urlencode({"url": target})
    try:
        body, code = get_bytes(availability, pacer, timeout=20, limit=256 * 1024)
        result["availability_http_status"] = str(code)
        payload = json.loads(body)
        snapshots = payload.get("archived_snapshots") if isinstance(payload, dict) else None
        if not isinstance(snapshots, dict):
            raise ValueError("invalid_availability_payload")
        capture = snapshots.get("closest") or {}
    except urllib.error.HTTPError as exc:
        retry_after = (exc.headers.get("Retry-After") or "") if exc.headers else ""
        retry_after = re.sub(r"[^A-Za-z0-9 :,;-]", "", retry_after)[:60]
        result.update(status="rate_limited" if exc.code == 429 else "availability_error",
                      availability_http_status=str(exc.code),
                      error_code=f"http_{exc.code}" + (f";retry_after={retry_after}" if retry_after else ""))
        return result
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, TypeError) as exc:
        result.update(status="availability_error", error_code=type(exc).__name__)
        return result
    if not capture:
        result["status"] = "no_snapshot"
        return result
    try:
        archive_url = str(capture["url"]).replace("http://web.archive.org/", "https://web.archive.org/", 1)
        original_url, stamp = unwrap_play_url(archive_url)
        if (capture.get("available") is not True or str(capture.get("status")) != "200"
                or stamp != capture.get("timestamp") or not url_matches_package(original_url, package)):
            raise ValueError("snapshot_identity_mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        result.update(status="snapshot_identity_error", error_code=str(exc))
        return result
    result.update(snapshot_timestamp=stamp, original_play_url=original_url)
    raw_url = f"https://web.archive.org/web/{stamp}id_/{original_url}"
    normal_url = f"https://web.archive.org/web/{stamp}/{original_url}"
    errors = []
    for method, url in (("raw", raw_url), ("rewritten", normal_url)):
        try:
            page_bytes, http_status = get_bytes(url, pacer, timeout=25, limit=MAX_HTML_BYTES)
            page = page_bytes.decode("utf-8", "replace")
            category, title = parse_archived_category(page, package)
            result.update(status="found", store_category_id=category,
                          store_title_historical=title, archive_url=url,
                          response_sha256=hashlib.sha256(page_bytes).hexdigest(),
                          snapshot_http_status=str(http_status), parser_method=f"wayback_{method}_jsonld_identity")
            if category in mapping:
                result["research_category_id"], result["research_category_zh"] = mapping[category]
                result["mapping_status"] = "mapped"
            else:
                result["mapping_status"] = "unmapped"
            return result
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = (exc.headers.get("Retry-After") or "") if exc.headers else ""
                retry_after = re.sub(r"[^A-Za-z0-9 :,;-]", "", retry_after)[:60]
                result.update(status="rate_limited",
                              error_code="http_429" + (f";retry_after={retry_after}" if retry_after else ""))
                return result
            errors.append(f"{method}_http_{exc.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            errors.append(f"{method}_{type(exc).__name__}")
        except ValueError as exc:
            errors.append(f"{method}_{str(exc)}")
    result.update(status="page_error" if any("URL" in e or "Timeout" in e or "http" in e for e in errors)
                  else "parse_error", error_code=";".join(errors)[:240], archive_url=normal_url)
    return result


def read_results(path: Path, sample: list[str]) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("Existing result schema differs from this script.")
        rows = list(reader)
    by_package = {row["package_name"]: row for row in rows}
    if len(rows) != len(by_package) or not set(by_package).issubset(sample):
        raise ValueError("Existing results contain duplicates or packages outside sample.")
    return by_package


def write_state(output: Path, sample: list[str], rows: dict[str, dict[str, str]],
                population_size: int, input_path: Path):
    csv_path = output / "results.csv"
    temporary = csv_path.with_name(csv_path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[p] for p in sample if p in rows)
    temporary.replace(csv_path)
    counts = Counter(row["status"] for row in rows.values())
    summary = {
        "input": str(input_path.resolve()), "population_size": population_size,
        "sample_seed": SEED, "sample_size": SAMPLE_SIZE,
        "sample_sha256": hashlib.sha256(("\n".join(sample) + "\n").encode()).hexdigest(),
        "looked_up": len(rows), "remaining": SAMPLE_SIZE - len(rows),
        "status_counts": dict(sorted(counts.items())),
        "valid_categories": counts["found"],
        "unmapped_categories": sum(r["status"] == "found" and r["mapping_status"] == "unmapped" for r in rows.values()),
        "updated_at_utc": utc_now(),
        "note": "Historical archived Google Play page; snapshot category may differ from current category.",
    }
    summary_path = output / "summary.json"
    temporary = summary_path.with_name(summary_path.name + ".tmp")
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--max-new", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--request-interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.max_new < 1 or args.request_interval < 0.5:
        parser.error("--max-new must be >=1 and --request-interval >=0.5")
    population = read_population(args.input)
    sample = prepare_sample(args.output_dir, population)
    rows = read_results(args.output_dir / "results.csv", sample)
    write_state(args.output_dir, sample, rows, len(population), args.input)
    if args.prepare_only:
        print(f"Prepared {len(sample)} packages from {len(population)} remaining GB 404 packages.")
        return 0
    mapping, _ = load_mapping(args.mapping)
    pacer = Pacer(args.request_interval)
    new_count = 0
    consecutive_errors = 0
    for index, package in enumerate(sample, 1):
        if package in rows and (rows[package]["status"] in TERMINAL_STATUSES or not args.retry_errors):
            continue
        row = lookup(package, mapping, pacer)
        rows[package] = row
        write_state(args.output_dir, sample, rows, len(population), args.input)
        new_count += 1
        print(f"{index}/{len(sample)} {package}: {row['status']} {row['store_category_id']}", flush=True)
        if row["status"] == "rate_limited":
            print("Wayback rate limit reached; stopped.", file=sys.stderr)
            return 3
        consecutive_errors = consecutive_errors + 1 if row["status"] in {"availability_error", "page_error"} else 0
        if consecutive_errors >= 5:
            print("Five consecutive network/page errors; stopped.", file=sys.stderr)
            return 3
        if new_count >= args.max_new:
            break
    print(f"Pilot progress: {len(rows)}/{len(sample)} checked; results in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
