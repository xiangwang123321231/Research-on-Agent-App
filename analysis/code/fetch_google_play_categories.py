#!/usr/bin/env python3
"""Fetch current Google Play categories by package, separately from APK features.

Python 3.10+; standard library only. See docs/google_play_category_collection.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    ROOT.parent
    / "google_play_dataset"
    / "data1"
    / "google_play_apps_llm_candidates_final.jsonl"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent / "category"
SCRIPT_VERSION = "google-play-category-collector-v1.1"
PARSER_VERSION = "google-play-identity-jsonld-v1"
SCHEMA_VERSION = "google-play-category-observation-v1"
PACKAGE_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+\Z")
CATEGORY_RE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
SYSTEMIC_ERRORS = {"network_error", "http_error", "parse_error"}
FIELDS = [
    "source_line", "package_name", "version_code", "sha256", "metadata_date",
    "status", "store_category_id", "research_category_id", "research_category_zh",
    "mapping_status", "category_source", "category_observed_at_utc", "country",
    "language", "store_title_current", "parser_method", "http_status", "attempts",
    "requested_url", "final_url", "response_sha256", "error_code", "error_message",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mapping(path: Path) -> tuple[dict[str, tuple[str, str]], str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload.get("version"), str) or not payload["version"]:
        raise ValueError("Mapping needs a non-empty version.")
    groups = payload.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Mapping needs a non-empty groups object.")
    mapping: dict[str, tuple[str, str]] = {}
    for group_id, group in groups.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", group_id):
            raise ValueError(f"Invalid research category ID: {group_id}")
        label = group.get("label_zh")
        ids = group.get("category_ids")
        if not isinstance(label, str) or not label or not isinstance(ids, list) or not ids:
            raise ValueError(f"Invalid mapping group: {group_id}")
        for category_id in ids:
            if not isinstance(category_id, str) or not CATEGORY_RE.fullmatch(category_id):
                raise ValueError(f"Invalid store category ID: {category_id}")
            if category_id in mapping:
                raise ValueError(f"Store category assigned more than once: {category_id}")
            mapping[category_id] = (group_id, label)
    return mapping, payload["version"]


def _identity(record: dict[str, Any], line: int) -> dict[str, Any]:
    app = (record.get("details") or {}).get("appDetails") or {}
    if not isinstance(app, dict):
        raise ValueError(f"Input line {line}: appDetails must be an object.")
    candidates = [record.get(k) for k in ("package_name", "packageName", "pkg_name", "docid")]
    candidates.append(app.get("packageName"))
    names = {str(x).strip() for x in candidates if x is not None and str(x).strip()}
    package = next(iter(names)) if len(names) == 1 else None
    error = ""
    if not names:
        error = "missing_package_name"
    elif len(names) != 1:
        error = "conflicting_package_names"
    elif not PACKAGE_RE.fullmatch(package):
        error = "invalid_package_name"
    versions = [record.get(k) for k in ("version_code", "versionCode", "vercode")]
    versions.append(app.get("versionCode"))
    version = next((x for x in versions if x is not None and x != ""), None)
    sha = record.get("sha256")
    sha = str(sha).strip().upper() if sha is not None else None
    if sha and not re.fullmatch(r"[A-F0-9]{64}", sha):
        raise ValueError(f"Input line {line}: invalid SHA-256.")
    return {
        "source_line": line, "package_name": package, "version_code": version,
        "sha256": sha or None,
        "metadata_date": record.get("metadata_date") or record.get("az_metadata_date"),
        "input_error": error,
    }


def load_input(path: Path, sha_path: Path | None = None) -> tuple[list[dict], str]:
    before = fingerprint(path)
    rows: list[dict] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            for line, record in enumerate(csv.DictReader(stream), 2):
                if None in record:
                    raise ValueError(f"CSV row {line}: more values than column names.")
                rows.append(_identity(record, line))
        else:
            for line, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                if path.suffix.lower() in {".txt", ".list"}:
                    if raw.lstrip().startswith("#"):
                        continue
                    record = {"package_name": raw.strip()}
                else:
                    try:
                        record = json.loads(raw)
                    except ValueError as exc:
                        raise ValueError(f"Input line {line}: invalid JSON.") from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"Input line {line}: expected an object.")
                rows.append(_identity(record, line))
    if fingerprint(path) != before:
        raise ValueError("Input changed while loading; retry with a stable input file.")
    if not rows or not any(not r["input_error"] for r in rows):
        raise ValueError("Input contains no valid package names.")
    if sha_path:
        payload = json.loads(sha_path.read_text(encoding="utf-8-sig"))
        sha_map: dict[tuple[str, str], set[str]] = {}
        for item in payload.get("details", []):
            name, version, sha = item.get("packageName"), item.get("versionCode"), item.get("sha256")
            if name is None or version is None or not sha:
                continue
            sha = str(sha).strip().upper()
            if not re.fullmatch(r"[A-F0-9]{64}", sha):
                raise ValueError("SHA-256 map contains an invalid hash.")
            sha_map.setdefault((str(name), str(version)), set()).add(sha)
        for row in rows:
            matches = sha_map.get((row["package_name"], str(row["version_code"])), set())
            if len(matches) > 1:
                raise ValueError(f"Ambiguous APK mapping for input line {row['source_line']}.")
            if matches:
                matched = next(iter(matches))
                if row["sha256"] and row["sha256"] != matched:
                    raise ValueError(f"Conflicting SHA-256 for input line {row['source_line']}.")
                row["sha256"] = matched
    return rows, before


class ParseError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _JsonLdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.blocks: list[str] = []
        self._parts: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "script":
            kind = dict(attrs).get("type") or ""
            if kind.lower().split(";", 1)[0].strip() == "application/ld+json":
                self._parts = []

    def handle_data(self, data):
        if self._parts is not None:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "script" and self._parts is not None:
            self.blocks.append("".join(self._parts))
            self._parts = None


def _schema_nodes(value):
    if isinstance(value, list):
        for item in value:
            yield from _schema_nodes(item)
    elif isinstance(value, dict):
        yield value
        # Do not recursively treat arbitrary nested recommendations as this app.
        if "@graph" in value:
            yield from _schema_nodes(value["@graph"])


def _url_is_app(url: str, package: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(url)
        return (parsed.scheme == "https" and parsed.hostname == "play.google.com"
                and (parsed.path == "/store/apps/details" or parsed.path.startswith("/store/apps/details/"))
                and urllib.parse.parse_qs(parsed.query).get("id") == [package])
    except ValueError:
        return False


def parse_category(page: str, package: str) -> dict:
    parser = _JsonLdParser()
    parser.feed(page)
    matches: list[dict] = []
    for block in parser.blocks:
        try:
            payload = json.loads(block)
        except (ValueError, RecursionError):
            continue
        for node in _schema_nodes(payload):
            kind = node.get("@type")
            types = kind if isinstance(kind, list) else [kind]
            if not any(isinstance(k, str) and k in {"SoftwareApplication", "MobileApplication"} for k in types):
                continue
            if isinstance(node.get("url"), str) and _url_is_app(node["url"], package):
                matches.append(node)
    if not matches:
        raise ParseError("missing_identity_matched_app_schema")
    ids = set()
    for node in matches:
        category = node.get("applicationCategory")
        if not isinstance(category, str) or not CATEGORY_RE.fullmatch(category):
            raise ParseError("missing_or_invalid_application_category")
        ids.add(category)
    if len(ids) != 1:
        raise ParseError("conflicting_application_categories")
    node = matches[0]
    return {
        "store_category_id": next(iter(ids)),
        "store_title_current": node.get("name") if isinstance(node.get("name"), str) else None,
        "parser_method": "identity_matched_jsonld_applicationCategory",
        "evidence": {k: node.get(k) for k in ("@type", "url", "name", "applicationCategory")},
    }


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return None


class PlayClient:
    def __init__(self, country="us", language="en", timeout=25.0,
                 delay=1.0, max_attempts=3, max_retry_wait=60.0,
                 opener: Callable | None = None, sleep: Callable = time.sleep,
                 monotonic: Callable = time.monotonic) -> None:
        self.country, self.language = country, language
        self.timeout, self.delay = timeout, delay
        self.max_attempts, self.max_retry_wait = max_attempts, max_retry_wait
        self._open = opener or urllib.request.urlopen
        self._sleep, self._clock = sleep, monotonic
        self._last_start: float | None = None

    def _throttle(self) -> None:
        if self._last_start is not None:
            remaining = self.delay - (self._clock() - self._last_start)
            if remaining > 0:
                self._sleep(remaining)
        self._last_start = self._clock()

    def fetch(self, package: str) -> dict:
        if not PACKAGE_RE.fullmatch(package):
            raise ValueError("Refusing to request an invalid package name.")
        url = "https://play.google.com/store/apps/details?" + urllib.parse.urlencode(
            {"id": package, "hl": self.language, "gl": self.country})
        base = {
            "schema_version": SCHEMA_VERSION, "parser_version": PARSER_VERSION,
            "script_version": SCRIPT_VERSION, "package_name": package,
            "country": self.country, "language": self.language,
            "category_source": "google_play_web", "requested_url": url,
        }
        for attempt in range(1, self.max_attempts + 1):
            self._throttle()
            result = {**base, "category_observed_at_utc": utc_now(), "attempts": attempt}
            wait = min(2.0 ** (attempt - 1), self.max_retry_wait)
            should_retry = False
            request = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; research-category-collector/1.0)",
                "Accept-Language": self.language, "Accept-Encoding": "identity",
            })
            try:
                with self._open(request, timeout=self.timeout) as response:
                    raw = response.read(MAX_RESPONSE_BYTES + 1)
                    result.update(http_status=response.status, final_url=response.url)
                    if not _url_is_app(response.url, package):
                        result.update(status="blocked", error_code="unexpected_redirect")
                        return result
                    if len(raw) > MAX_RESPONSE_BYTES:
                        result.update(status="parse_error", error_code="response_too_large")
                        return result
                    result["response_sha256"] = hashlib.sha256(raw).hexdigest()
                    page = raw.decode("utf-8-sig")
                    result.update(parse_category(page, package))
                    result["status"] = "success"
                    return result
            except urllib.error.HTTPError as exc:
                code = exc.code
                result.update(http_status=code, final_url=exc.geturl(), error_code=f"http_{code}")
                if code in {404, 410}:
                    result["status"] = "not_available_in_country"
                elif code in {401, 403}:
                    result["status"] = "blocked"
                elif code == 429:
                    result["status"] = "rate_limited"
                    requested_wait = retry_after_seconds(exc.headers.get("Retry-After"))
                    result["retry_after_seconds"] = requested_wait
                    if requested_wait is not None and requested_wait > self.max_retry_wait:
                        # Do not shorten the server's minimum wait and request too early.
                        exc.close()
                        return result
                    wait = max(wait, requested_wait or 0.0)
                    should_retry = True
                else:
                    result["status"] = "http_error"
                    should_retry = code in {408, 425} or 500 <= code <= 599
                exc.close()
            except ParseError as exc:
                result.update(status="parse_error", error_code=exc.code)
            except UnicodeError:
                result.update(status="parse_error", error_code="invalid_utf8_response")
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError, http.client.HTTPException) as exc:
                result.update(status="network_error", error_code=type(exc).__name__,
                              error_message=str(exc)[:400])
                should_retry = True
            if not should_retry or attempt == self.max_attempts:
                return result
            self._sleep(wait)
        raise AssertionError("No lookup result")


def _compatible(record: dict, country: str, language: str) -> bool:
    return (record.get("schema_version") == SCHEMA_VERSION
            and record.get("parser_version") == PARSER_VERSION
            and record.get("country") == country and record.get("language") == language)


def read_journal(path: Path, country: str, language: str) -> tuple[dict[str, dict], dict]:
    latest: dict[str, dict] = {}
    recovery: dict = {}
    if not path.exists():
        return latest, recovery
    end_offset = 0
    valid_without_newline = False
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            offset = end_offset
            end_offset += len(raw)
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
                if not isinstance(record, dict) or not isinstance(record.get("package_name"), str):
                    raise ValueError("Invalid observation object")
                if record.get("status") not in {"success", "not_available_in_country", "blocked",
                                                "rate_limited", "parse_error", "http_error", "network_error"}:
                    raise ValueError("Invalid observation status")
                if record["status"] == "success" and not CATEGORY_RE.fullmatch(str(record.get("store_category_id", ""))):
                    raise ValueError("Success record has no valid category")
            except (ValueError, UnicodeError) as exc:
                if raw.endswith(b"\n"):
                    raise ValueError(f"Journal line {line_number} is corrupted; no outputs were rewritten.") from exc
                # A crash may interrupt only the last append. Preserve it first.
                backup = path.with_name(path.name + f".partial-{uuid.uuid4().hex}.bin")
                with backup.open("xb") as out:
                    out.write(raw)
                    out.flush()
                    os.fsync(out.fileno())
                recovery = {"partial_tail_backup": str(backup), "truncated_at_byte": offset}
                break
            if _compatible(record, country, language):
                latest[record["package_name"]] = record
            valid_without_newline = not raw.endswith(b"\n")
    if recovery:
        with path.open("r+b") as stream:
            stream.truncate(recovery["truncated_at_byte"])
    elif valid_without_newline:
        with path.open("ab") as stream:
            stream.write(b"\n")
    return latest, recovery


@contextmanager
def output_lock(path: Path):
    # OS locks are released even if the process crashes. Keep the lock inode.
    stream = path.open("a+b")
    acquired = False
    try:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError as exc:
            raise ValueError("Another collector is using this output directory.") from exc
        yield
    finally:
        if acquired:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _atomic_write(path: Path, writer: Callable) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig" if path.suffix == ".csv" else "utf-8", newline="") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_snapshot(rows: list[dict], latest: dict, mapping: dict, directory: Path, context: dict) -> dict:
    records: list[dict] = []
    for row in rows:
        record = {k: row.get(k) for k in ("source_line", "package_name", "version_code", "sha256", "metadata_date")}
        observation = latest.get(row["package_name"])
        if row["input_error"]:
            record.update(status="invalid_input", error_code=row["input_error"])
        elif observation:
            record.update({k: observation.get(k) for k in FIELDS if k not in record})
            if observation["status"] == "success":
                mapped = mapping.get(observation["store_category_id"])
                if mapped:
                    record.update(research_category_id=mapped[0], research_category_zh=mapped[1], mapping_status="mapped")
                else:
                    record["mapping_status"] = "unmapped"
        else:
            record["status"] = "not_queried"
        records.append(record)

    def csv_writer(stream):
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    _atomic_write(directory / "categories.csv", csv_writer)
    unique = {r["package_name"] for r in rows if not r["input_error"]}
    package_counts = Counter(latest[p]["status"] if p in latest else "not_queried" for p in unique)
    row_counts = Counter(r["status"] for r in records)
    unmapped = Counter(r["store_category_id"] for r in records if r.get("mapping_status") == "unmapped")
    summary = {
        **context, "schema_version": "google-play-category-summary-v1", "script_version": SCRIPT_VERSION,
        "parser_version": PARSER_VERSION, "snapshot_created_at_utc": utc_now(),
        "input_rows": len(rows), "valid_unique_packages": len(unique),
        "row_status_counts": dict(row_counts), "unique_package_status_counts": dict(package_counts),
        "research_category_row_counts": dict(Counter(r["research_category_zh"] for r in records if r.get("mapping_status") == "mapped")),
        "unmapped_store_categories": dict(unmapped),
        "pass_complete": package_counts["not_queried"] == 0 and row_counts["invalid_input"] == 0,
        "all_categories_obtained": package_counts["success"] == len(unique) and row_counts["invalid_input"] == 0,
        "all_categories_mapped": package_counts["success"] == len(unique) and not unmapped and row_counts["invalid_input"] == 0,
        "interpretation": "Current store observations; not historical APK functionality or LLM confirmation. 404/410 indicates unavailability in the requested country, not confirmed global removal.",
    }
    summary["snapshot_csv_sha256"] = fingerprint(directory / "categories.csv")
    _atomic_write(directory / "summary.json", lambda stream: json.dump(summary, stream, ensure_ascii=False, indent=2))
    return summary


def run(args, client: PlayClient | None = None) -> dict:
    source, directory, map_path = args.input.resolve(), args.output_dir.resolve(), args.mapping.resolve()
    sha_path = args.sha256_map.resolve() if args.sha256_map else None
    protected = [source, map_path] + ([sha_path] if sha_path else [])
    targets = [directory / name for name in ("observations.jsonl", "categories.csv", "summary.json", "collector.lock")]
    for target in targets:
        if any(os.path.normcase(str(target.resolve())) == os.path.normcase(str(p))
               or (target.exists() and p.exists() and target.samefile(p)) for p in protected):
            raise ValueError("An output path aliases an input/mapping file; refusing to overwrite it.")
    mapping, mapping_version = load_mapping(map_path)
    rows, input_sha = load_input(source, sha_path)
    limit = 0 if args.export_only else args.limit
    directory.mkdir(parents=True, exist_ok=True)
    client = client or PlayClient(args.country, args.language, args.timeout, args.delay, args.attempts, args.max_retry_wait)
    with output_lock(directory / "collector.lock"):
        journal = directory / "observations.jsonl"
        latest, recovery = read_journal(journal, args.country, args.language)
        context = {
            "input_file": str(source), "input_sha256": input_sha,
            "sha256_map_file": str(sha_path) if sha_path else None,
            "sha256_map_sha256": fingerprint(sha_path) if sha_path else None,
            "mapping_file": str(map_path), "mapping_version": mapping_version,
            "mapping_sha256": fingerprint(map_path), "country": args.country, "language": args.language,
            "export_only": args.export_only, "requested_limit": limit,
            "run_started_at_utc": utc_now(), "requests_configuration": {
                "timeout_seconds": args.timeout, "minimum_request_interval_seconds": args.delay,
                "max_attempts": args.attempts, "max_retry_wait_seconds": args.max_retry_wait,
            }, "attempted_unique_packages_this_run": 0, "stopped_reason": None,
            "journal_recovery": recovery,
        }
        consecutive_errors = 0
        packages = list(dict.fromkeys(r["package_name"] for r in rows if not r["input_error"]))
        interrupted = False
        try:
            with journal.open("a", encoding="utf-8", newline="\n") as out:
                for package in packages:
                    existing = latest.get(package)
                    if existing and not args.refresh and (existing["status"] == "success" or not args.retry_failed):
                        continue
                    if limit is not None and context["attempted_unique_packages_this_run"] >= limit:
                        context["stopped_reason"] = "export_only" if args.export_only else "limit_reached"
                        break
                    result = client.fetch(package)
                    # Flush each event before updating the in-memory snapshot.
                    out.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    out.flush()
                    os.fsync(out.fileno())
                    latest[package] = result
                    context["attempted_unique_packages_this_run"] += 1
                    n = context["attempted_unique_packages_this_run"]
                    print(f"[{n}] {package}: {result['status']} {result.get('store_category_id', '')}", flush=True)
                    consecutive_errors = consecutive_errors + 1 if result["status"] in SYSTEMIC_ERRORS else 0
                    if result["status"] in {"blocked", "rate_limited"}:
                        context["stopped_reason"] = result["status"]
                        break
                    if consecutive_errors >= args.stop_after_errors:
                        context["stopped_reason"] = "consecutive_systemic_errors"
                        break
        except KeyboardInterrupt:
            interrupted = True
            context["stopped_reason"] = "interrupted"
        context["run_finished_at_utc"] = utc_now()
        summary = write_snapshot(rows, latest, mapping, directory, context)
        if interrupted:
            raise KeyboardInterrupt
        return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Candidate JSONL, CSV, or one-package-per-line TXT. "
            "Defaults to google_play_dataset/data1/"
            "google_play_apps_llm_candidates_final.jsonl."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mapping", type=Path, default=Path(__file__).with_name("google_play_category_mapping.json"))
    parser.add_argument("--sha256-map", type=Path, help="Optional final_sha256s.json for raw metadata/CSV input.")
    parser.add_argument("--country", default="us", help="Two-letter store country; each country has an independent cache.")
    parser.add_argument("--language", default="en", help="Two-letter language; defaults to English.")
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--delay", type=float, default=1.0, help="Minimum seconds between request starts.")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-retry-wait", type=float, default=60.0)
    parser.add_argument("--stop-after-errors", type=int, default=5, help="Stop after consecutive network/HTTP/parser errors.")
    parser.add_argument("--limit", type=int, help="Maximum new/retried unique packages this run; cache hits do not count.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry previously unsuccessful observations.")
    parser.add_argument("--refresh", action="store_true", help="Query previously observed packages again; retain history.")
    parser.add_argument("--export-only", action="store_true", help="Regenerate the CSV using existing observations; no network requests.")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    args.country, args.language = args.country.lower(), args.language.lower()
    if not re.fullmatch(r"[a-z]{2}", args.country) or not re.fullmatch(r"[a-z]{2}", args.language):
        parser.error("Country and language must be two ASCII letters.")
    if any(not math.isfinite(x) or x <= 0 for x in (args.timeout, args.max_retry_wait)):
        parser.error("Timeout and max-retry-wait must be positive finite numbers.")
    if not math.isfinite(args.delay) or args.delay < 0 or args.attempts < 1 or args.stop_after_errors < 1:
        parser.error("Delay must be nonnegative; attempts and stop-after-errors must be positive.")
    if args.limit is not None and args.limit < 0:
        parser.error("Limit must be nonnegative.")
    if args.export_only:
        args.limit = 0
    try:
        summary = run(args)
    except KeyboardInterrupt:
        print("Interrupted; completed observations retained. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "unique_package_status_counts": summary["unique_package_status_counts"],
                      "stopped_reason": summary["stopped_reason"],
                      "all_categories_obtained": summary["all_categories_obtained"]}, ensure_ascii=False, indent=2))
    return 1 if summary["stopped_reason"] in {"blocked", "rate_limited", "consecutive_systemic_errors"} else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
