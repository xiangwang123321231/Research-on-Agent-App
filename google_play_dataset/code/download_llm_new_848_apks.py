#!/usr/bin/env python3
"""Download the 848 new-only LLM-candidate APKs from AndroZoo.

On the server, defaults are ../data/google_play_apps_llm_sha256s_new_only_848.txt
and ../apks_new_848 relative to this script's code directory. The default run is a
read-only preflight; add --execute to download. Network requests connect directly
by default and ignore environment proxy settings; use --proxy for an explicit proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import requests


API_URL = "https://androzoo.uni.lu/api/download"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_HASH_FILE = PROJECT_DIR / "data" / "google_play_apps_llm_sha256s_new_only_848.txt"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "apks_new_848"
SHA_PATTERN = re.compile(r"[0-9A-Fa-f]{64}\Z")
HEADERS = {"User-Agent": "LLM-APK-research-downloader/1.0"}


def read_hashes(path: Path):
    hashes = []
    seen = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            sha = line.strip().upper()
            if not SHA_PATTERN.fullmatch(sha):
                raise ValueError("Invalid SHA-256 on line {}: {!r}".format(line_number, line.strip()))
            if sha in seen:
                raise ValueError("Duplicate SHA-256 on line {}: {}".format(line_number, sha))
            seen.add(sha)
            hashes.append(sha)
    if len(hashes) != 848:
        raise ValueError("Expected 848 unique hashes, found {}".format(len(hashes)))
    return hashes


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def find_existing(directory: Path, wanted):
    found = {}
    if not directory.exists():
        return found
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("Output path is not a regular directory: {}".format(directory))
    for path in directory.iterdir():
        if path.suffix.lower() != ".apk":
            continue
        sha = path.stem.upper()
        if sha not in wanted:
            continue
        if sha in found:
            raise ValueError("Two existing APK filenames match {}: {} and {}".format(sha, found[sha], path))
        if path.is_symlink() or not path.is_file():
            raise ValueError("Existing APK path is not a regular file: {}".format(path))
        found[sha] = path
    return found


def resolve_api_key(key_file: Path):
    key = os.environ.get("ANDROZOO_API_KEY", "").strip()
    if key:
        return key
    if key_file.is_file():
        for line in key_file.read_text(encoding="utf-8").splitlines():
            key = line.strip()
            if key:
                return key
    raise ValueError(
        "AndroZoo API key missing; set ANDROZOO_API_KEY or use --api-key-file"
    )


@contextmanager
def download_response(sha, api_key, timeout, proxy=None):
    # Each request owns its session, so concurrent workers do not share mutable state.
    with requests.Session() as session:
        session.trust_env = False
        if proxy:
            session.proxies = {"http": proxy, "https": proxy}
        # Keep custom CA bundles usable while ignoring all environment proxies.
        ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
        if ca_bundle:
            session.verify = ca_bundle
        with session.get(
            API_URL,
            params={"apikey": api_key, "sha256": sha},
            headers=HEADERS,
            stream=True,
            timeout=(30, timeout),
        ) as response:
            yield response


def download_one(sha, output_dir, api_key, retries, timeout, backoff, stop_event, proxy=None):
    destination = output_dir / (sha + ".apk")
    if destination.exists():
        if destination.is_file() and not destination.is_symlink() and file_sha256(destination) == sha:
            return "existing", "verified"
        return "failed", "existing_file_has_wrong_sha256"

    last_reason = "not_attempted"
    for attempt in range(1, retries + 1):
        if stop_event.is_set():
            return "aborted", "another_request_failed_authentication"
        temp_path = None
        try:
            with download_response(sha, api_key, timeout, proxy) as response:
                status = response.status_code
                if status in (401, 403):
                    stop_event.set()
                    return "failed", "HTTP_{};_check_API_key_or_access".format(status)
                if status == 404:
                    return "failed", "HTTP_404"
                if status != 200:
                    last_reason = "HTTP_{}".format(status)
                    if 400 <= status < 500 and status != 429:
                        return "failed", last_reason
                else:
                    digest = hashlib.sha256()
                    with tempfile.NamedTemporaryFile(
                        mode="wb", prefix="." + sha + ".", suffix=".part",
                        dir=str(output_dir), delete=False,
                    ) as temp:
                        temp_path = Path(temp.name)
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                temp.write(chunk)
                                digest.update(chunk)
                        temp.flush()
                        os.fsync(temp.fileno())
                    if digest.hexdigest().upper() != sha:
                        last_reason = "downloaded_content_sha256_mismatch"
                    else:
                        try:
                            # Exclusive publication: never replace a pre-existing APK.
                            os.link(temp_path, destination)
                        except FileExistsError:
                            if file_sha256(destination) == sha:
                                return "existing", "verified_after_concurrent_download"
                            return "failed", "destination_conflict"
                        return "downloaded", "sha256_verified"
        except requests.RequestException as error:
            # Do not print exception text: it can contain the URL and API key.
            last_reason = "network_{}".format(type(error).__name__)
        except OSError as error:
            last_reason = "filesystem_{}".format(type(error).__name__)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
        if attempt < retries and not stop_event.is_set():
            time.sleep(backoff * (2 ** (attempt - 1)))
    return "failed", last_reason


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH_FILE,
                        help="848-line SHA-256 TXT file (default: ../data/google_play_apps_llm_sha256s_new_only_848.txt)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="APK destination (default: ../apks_new_848)")
    parser.add_argument("--api-key-file", type=Path, default=SCRIPT_DIR / "apikey.txt",
                        help="API key file if ANDROZOO_API_KEY is not set (default: code/apikey.txt)")
    parser.add_argument("--proxy", default=None,
                        help="Explicit proxy URL; default is direct connection, ignoring environment proxies")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent downloads, 1 to 20 (default: 4)")
    parser.add_argument("--retries", type=int, default=3,
                        help="Maximum attempts per APK (default: 3)")
    parser.add_argument("--timeout", type=int, default=180,
                        help="Read timeout in seconds (default: 180)")
    parser.add_argument("--backoff", type=float, default=5.0,
                        help="Initial retry delay in seconds (default: 5)")
    parser.add_argument("--execute", action="store_true",
                        help="Download files; without this flag only preflight checks run")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.hash_file.is_file():
        raise FileNotFoundError("SHA-256 list not found: {}".format(args.hash_file))
    if not (1 <= args.workers <= 20) or args.retries < 1 or args.timeout < 1 or args.backoff < 0:
        raise ValueError("Invalid workers/retries/timeout/backoff setting")
    if args.output_dir.is_symlink():
        raise ValueError("Output directory cannot be a symlink")
    hashes = read_hashes(args.hash_file)
    output_dir = args.output_dir.resolve(strict=False)
    existing = find_existing(output_dir, set(hashes))
    for sha, path in existing.items():
        if file_sha256(path) != sha:
            raise ValueError("Existing APK has incorrect content hash; refusing to overwrite: {}".format(path))
    pending = [sha for sha in hashes if sha not in existing]
    print(
        "List: 848; already downloaded and verified: {}; to download: {}; output: {}".format(
            len(existing), len(pending), output_dir
        ),
        flush=True,
    )
    print("Connection: {} (environment proxies disabled)".format(
        "explicit proxy" if args.proxy else "direct"
    ), flush=True)
    if not args.execute:
        print("Dry run complete; no network requests or file changes. Add --execute to download.", flush=True)
        return 0
    if not pending:
        print("All 848 APKs are already present and verified.", flush=True)
        return 0
    api_key = resolve_api_key(args.api_key_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    counts = {"downloaded": 0, "existing": 0, "failed": 0, "aborted": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one, sha, output_dir, api_key, args.retries,
                args.timeout, args.backoff, stop_event, args.proxy,
            ): sha for sha in pending
        }
        for number, future in enumerate(as_completed(futures), 1):
            sha = futures[future]
            try:
                status, reason = future.result()
            except Exception as error:
                status, reason = "failed", "unexpected_{}".format(type(error).__name__)
            counts[status] += 1
            if status in ("failed", "aborted"):
                failures.append((sha, reason))
            print("[{}/{}] {} {} ({})".format(number, len(pending), status, sha, reason), flush=True)
    print(
        "Done: downloaded={}, already_verified={}, failed={}, aborted={}".format(
            counts["downloaded"], len(existing) + counts["existing"],
            counts["failed"], counts["aborted"]
        ),
        flush=True,
    )
    for sha, reason in failures:
        print("FAILED {} {}".format(sha, reason), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
