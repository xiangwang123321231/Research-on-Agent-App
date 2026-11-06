#!/usr/bin/env python3
"""Remove APKs whose SHA-256 is listed in a text file, searching several folders.

The APKs are named "<SHA256>.apk", so matching is done on the file name.

Default (no --execute): read-only scan, lists what would be removed.
--execute            : MOVE the matched APKs into a quarantine folder (reversible).
--execute --purge    : delete them permanently instead.

Setup for this dataset (212 bad APKs, split across three folders):

  python3 remove_listed_apks.py \\
      --sha-list /home/ubuntu/workspace/jwang/data1/google_play_apps_llm_sha256s_retry_212.txt \\
      --dirs /home/ubuntu/workspace/jwang/apks1 \\
             /home/ubuntu/workspace/jwang/apks2 \\
             /home/ubuntu/workspace/jwang/apks3

  ... add --execute to move them to .../jwang/removed_apks (nothing lost yet)
  ... add --execute --purge only after the quarantine looks right

Symlinks are rejected, nothing is overwritten, and a manifest CSV is written so a
quarantined APK can be put back where it came from.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path


SHA_PATTERN = re.compile(r"[0-9A-Fa-f]{64}\Z")

DEFAULT_SHA_LIST = "/home/ubuntu/workspace/jwang/data1/google_play_apps_llm_sha256s_retry_212.txt"

LINK_FALLBACK_ERRNOS = (
    errno.EXDEV,
    errno.EPERM,
    errno.EACCES,
    getattr(errno, "EOPNOTSUPP", errno.EPERM),
    getattr(errno, "ENOTSUP", errno.EPERM),
    errno.EINVAL,
    errno.EMLINK,
)


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def copy_exclusive_then_remove(source: Path, destination: Path):
    created = False
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            created = True
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if file_sha256(destination) != file_sha256(source):
            raise ValueError("Copied APK differs from source: {}".format(source))
        source.unlink()
    except Exception:
        if created and destination.exists():
            destination.unlink()
        raise


def move_without_overwrite(source: Path, destination: Path):
    if destination.exists():
        raise FileExistsError("Destination already exists: {}".format(destination))
    try:
        os.link(source, destination)
    except OSError as error:
        if error.errno not in LINK_FALLBACK_ERRNOS:
            raise
        copy_exclusive_then_remove(source, destination)
    else:
        source.unlink()


def load_sha_list(path: Path):
    """Return (ordered SHA list, invalid lines). Accepts '<SHA>' or '<SHA>.apk' or 'SHA  name'."""
    shas = {}
    invalid = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for lineno, line in enumerate(stream, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            token = text.split()[0]
            if token.lower().endswith(".apk"):
                token = token[:-4]
            if not SHA_PATTERN.fullmatch(token):
                invalid.append((lineno, text))
                continue
            shas[token.upper()] = shas.get(token.upper(), 0) + 1
    return list(shas), invalid


def scan_matches(folders, wanted):
    """Return (hits, duplicates within one folder) where hit = (sha, path, folder)."""
    hits = []
    duplicates = []
    for folder in folders:
        seen = set()
        for directory, _, names in os.walk(folder, followlinks=False):
            for name in sorted(names):
                if Path(name).suffix.lower() != ".apk":
                    continue
                stem = Path(name).stem.upper()
                if stem not in wanted:
                    continue
                path = Path(directory) / name
                if path.is_symlink():
                    raise ValueError("Symlink is not supported: {}".format(path))
                if stem in seen:
                    duplicates.append((folder, stem, path))
                    continue
                seen.add(stem)
                hits.append((stem, path, folder))
    return hits, duplicates


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sha-list", type=Path, default=Path(DEFAULT_SHA_LIST),
                        help="Text file with one SHA-256 per line (default: %(default)s).")
    parser.add_argument("--dirs", type=Path, nargs="+", required=True,
                        help="Folders to search for those APKs (searched recursively).")
    parser.add_argument("--quarantine", type=Path, default=None,
                        help="Where matched APKs are moved (default: <parent of the first "
                             "--dirs>/removed_apks).")
    parser.add_argument("--purge", action="store_true",
                        help="Delete permanently instead of quarantining.")
    parser.add_argument("--execute", action="store_true",
                        help="Act on the files; otherwise only report.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    sha_list = Path(os.path.abspath(str(args.sha_list)))
    if not sha_list.is_file():
        raise ValueError("SHA list does not exist: {}".format(sha_list))
    folders = []
    for raw in args.dirs:
        folder = Path(os.path.abspath(str(raw)))
        if folder.is_symlink():
            raise ValueError("Symlink folder is not supported: {}".format(folder))
        if not folder.is_dir():
            raise ValueError("Folder does not exist: {}".format(folder))
        folders.append(folder)
    if len(set(str(f).lower() for f in folders)) != len(folders):
        raise ValueError("--dirs must be distinct.")

    wanted_list, invalid = load_sha_list(sha_list)
    wanted = set(wanted_list)
    if not wanted:
        raise ValueError("No valid SHA-256 found in {}".format(sha_list))

    hits, duplicates = scan_matches(folders, wanted)
    found = set(hit[0] for hit in hits)
    missing = [sha for sha in wanted_list if sha not in found]
    per_folder = {}
    for _, _, folder in hits:
        per_folder[folder] = per_folder.get(folder, 0) + 1
    total_bytes = sum(hit[1].stat().st_size for hit in hits)

    print("SHA list:   {}  ({} unique SHA-256{})".format(
        sha_list, len(wanted),
        "; {} invalid lines ignored".format(len(invalid)) if invalid else ""))
    print("Folders:    {}".format(len(folders)))
    for folder in folders:
        print("  {:<40} matched {:>4}".format(str(folder), per_folder.get(folder, 0)))
    print("Matched:    {} APKs, {:.2f} GB".format(len(hits), total_bytes / (1024.0 ** 3)))
    print("Missing:    {} of {} listed SHA-256 not found in those folders".format(
        len(missing), len(wanted_list)))
    for lineno, text in invalid[:5]:
        print("  Invalid line {}: {}".format(lineno, text))
    for folder, sha, path in duplicates[:5]:
        print("  Duplicate copy: {} ({})".format(path, folder))
    for sha, path, _ in hits[:10]:
        print("  Would remove: {}".format(path))
    if len(hits) > 10:
        print("  ... and {} more".format(len(hits) - 10))
    if missing:
        for sha in missing[:10]:
            print("  Not found: {}".format(sha))
        if len(missing) > 10:
            print("  ... and {} more not found".format(len(missing) - 10))
        print("  Hint: the rest may still be in the un-split source folder; add it to --dirs.")

    if not hits:
        print("Nothing to remove.")
        return 0
    if not args.execute:
        print("Dry run only; no files changed. Add --execute to {} {} APKs.".format(
            "DELETE" if args.purge else "quarantine", len(hits)))
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest = None
    writer = None
    if args.purge:
        print("Purging {} APKs permanently...".format(len(hits)))
    else:
        quarantine = args.quarantine
        if quarantine is None:
            quarantine = folders[0].parent / "removed_apks"
        quarantine = Path(os.path.abspath(str(quarantine)))
        if any(quarantine == f or f in quarantine.parents or quarantine in f.parents for f in folders):
            raise ValueError("Quarantine folder must be outside the searched folders: "
                             "{}".format(quarantine))
        quarantine.mkdir(parents=True, exist_ok=True)
        manifest_path = quarantine / "removed_manifest_{}.csv".format(stamp)
        manifest = manifest_path.open("w", encoding="utf-8", newline="")
        writer = csv.writer(manifest)
        writer.writerow(["sha256", "original_path", "quarantine_path", "bytes"])
        print("Quarantining {} APKs into {}".format(len(hits), quarantine))

    done = 0
    try:
        for sha, path, folder in hits:
            try:
                if not path.is_file():
                    raise ValueError("File disappeared: {}".format(path))
                size = path.stat().st_size
                if args.purge:
                    path.unlink()
                    target = Path("")
                else:
                    relative = path.relative_to(folder)
                    target = quarantine / folder.name / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    move_without_overwrite(path, target)
            except (OSError, ValueError) as error:
                print("Stopped after {} files. Failed: {}".format(done, path), file=sys.stderr)
                raise error
            done += 1
            if writer is not None:
                writer.writerow([sha, str(path), str(target), size])
            if done % 50 == 0 or done == len(hits):
                print("Processed: {}/{}".format(done, len(hits)), flush=True)
                if manifest is not None:
                    manifest.flush()
    finally:
        if manifest is not None:
            manifest.close()

    if args.purge:
        print("Done: deleted {} APKs.".format(done))
    else:
        print("Done: quarantined {} APKs.".format(done))
        print("Manifest: {}".format(manifest_path))
        print("Undo: move the files back from {} (paths recorded in the manifest),".format(quarantine))
        print("      then re-run with --execute --purge when you are sure.")
    if missing:
        print("Still missing from these folders: {}".format(len(missing)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
