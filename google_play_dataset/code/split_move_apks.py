#!/usr/bin/env python3
"""Move a fixed number of APKs (default 2800) out of one directory into another.

Moves APKs out of /home/ubuntu/workspace/jwang/metadata/apks. Two runs total:

  # run 1: move 2800 into apks1
  python3 split_move_apks.py --dest /home/ubuntu/workspace/jwang/apks1
  python3 split_move_apks.py --dest /home/ubuntu/workspace/jwang/apks1 --execute

  # run 2: move the next 2800 into apks2
  python3 split_move_apks.py --dest /home/ubuntu/workspace/jwang/apks2 --execute

Without --execute nothing is touched: it only reports what would be moved.
APKs land directly in --dest (flat). Add --keep-tree to mirror source subfolders.

Safety: symlinks rejected, nothing overwritten, every run preflighted before any
file is touched, and the cross-filesystem fallback copies + verifies SHA-256
before deleting the source.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path


SHA_PATTERN = re.compile(r"[0-9A-Fa-f]{64}\Z")

DEFAULT_SOURCE = "/home/ubuntu/workspace/jwang/metadata/apks"

# os.link may be unsupported or forbidden on some filesystems / volumes.
LINK_FALLBACK_ERRNOS = (
    errno.EXDEV,
    errno.EPERM,
    errno.EACCES,
    getattr(errno, "EOPNOTSUPP", errno.EPERM),
    getattr(errno, "ENOTSUP", errno.EPERM),
    errno.EINVAL,
    errno.EMLINK,
)
FILE_LIMIT = 2800


def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def copy_exclusive_then_remove(source: Path, destination: Path):
    """Cross-filesystem fallback: copy to a new path, verify, then remove source."""
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
        # Only this call could have created destination; keep source on failure.
        if created and destination.exists():
            destination.unlink()
        raise


def move_without_overwrite(source: Path, destination: Path):
    if destination.exists():
        raise FileExistsError("Destination already exists: {}".format(destination))
    try:
        # A hard link is exclusive and does not overwrite an existing target.
        os.link(source, destination)
    except OSError as error:
        if error.errno not in LINK_FALLBACK_ERRNOS:
            raise
        copy_exclusive_then_remove(source, destination)
    else:
        source.unlink()


def checked_path(path: Path):
    """Reject symlink components before resolving a filesystem target."""
    path = Path(os.path.abspath(str(path)))
    for component in (path,) + tuple(path.parents):
        if component.is_symlink():
            raise ValueError("Symlink paths are not supported: {}".format(component))
    return path.resolve()


def is_within(path: Path, root: Path):
    return path == root or root in path.parents


def scan_apks(source: Path):
    def walk_error(error):
        raise error

    files = []
    for directory, subdirs, names in os.walk(source, followlinks=False, onerror=walk_error):
        subdirs.sort()
        for name in subdirs:
            child = Path(directory) / name
            if child.is_symlink():
                raise ValueError("Symlink directory is not supported: {}".format(child))
        for name in sorted(names):
            if Path(name).suffix.lower() != ".apk":
                continue
            path = Path(directory) / name
            resolved = checked_path(path)
            if not is_within(resolved, source) or not path.is_file():
                raise ValueError("Not a regular APK within source: {}".format(path))
            files.append(path)
    return files


def validate_destination(destination: Path, target: Path):
    resolved = checked_path(destination)
    if not is_within(resolved, target):
        raise ValueError("Destination escapes target: {}".format(destination))
    if os.path.lexists(destination):
        raise FileExistsError("Destination already exists: {}".format(destination))
    for parent in destination.parents:
        if parent.exists() and not parent.is_dir():
            raise ValueError("Destination parent is not a directory: {}".format(parent))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=Path(DEFAULT_SOURCE),
                        help="Directory the APKs are moved out of (default: %(default)s).")
    parser.add_argument("--dest", type=Path, required=True,
                        help="Directory the APKs are moved into, e.g. .../jwang/apks1.")
    parser.add_argument("--count", type=int, default=FILE_LIMIT,
                        help="How many APKs to move in this run (default: 2800).")
    parser.add_argument("--keep-tree", action="store_true",
                        help="Keep the source subdirectory layout instead of flattening.")
    parser.add_argument("--verify-sha256", action="store_true",
                        help="Additionally require every APK filename to be its correct SHA-256.")
    parser.add_argument("--execute", action="store_true",
                        help="Move files; otherwise only perform read-only preflight.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.count <= 0:
        raise ValueError("--count must be a positive integer.")

    source = checked_path(args.source_dir)
    target = checked_path(args.dest)
    if not source.is_dir():
        raise ValueError("Source directory does not exist: {}".format(source))
    if is_within(source, target) or is_within(target, source):
        raise ValueError("Source and destination must be different, non-nested directories.")
    if target.exists() and not target.is_dir():
        raise ValueError("Destination is not a directory: {}".format(target))

    files = scan_apks(source)
    if not files:
        raise ValueError("No APK left in source: {}".format(source))
    selected = files[:args.count]
    leftover = len(files) - len(selected)

    plans = []
    seen = set()
    for path in selected:
        relative = path.relative_to(source) if args.keep_tree else Path(path.name)
        destination = target / relative
        key = str(destination).lower()
        if key in seen:
            raise ValueError("Two APKs map to the same destination name: {}".format(destination))
        seen.add(key)
        plans.append((path, destination))

    conflicts = []
    for _, destination in plans:
        try:
            validate_destination(destination, target)
        except (OSError, ValueError) as error:
            conflicts.append(str(error))

    print("Source:      {}  ({} APKs)".format(source, len(files)))
    print("Destination: {}".format(target))
    print("To move:     {}{}".format(
        len(plans), "; left in source: {}".format(leftover) if leftover else ""))
    print("Layout:      {}".format("keep subdirectories" if args.keep_tree else "flat"))
    print("Conflicts:   {}".format(len(conflicts)))
    for conflict in conflicts[:20]:
        print("  Conflict: {}".format(conflict))
    if conflicts:
        raise ValueError("Preflight failed; no APKs were moved.")

    if args.verify_sha256:
        for path, _ in plans:
            if not SHA_PATTERN.fullmatch(path.stem):
                raise ValueError("SHA verification needs a SHA-256 filename: {}".format(path))
            if file_sha256(path) != path.stem.upper():
                raise ValueError("SHA-256 mismatch; no APKs were moved: {}".format(path))
        print("All APK filename SHA-256 checks passed.")

    if not args.execute:
        print("Dry run only; no files changed. Add --execute to move {} APKs.".format(len(plans)))
        return 0

    target.mkdir(parents=True, exist_ok=True)
    moved = 0
    for path, destination in plans:
        try:
            # Recheck paths immediately before mutation, including newly created directories.
            if checked_path(path) != path or not path.is_file():
                raise ValueError("Source changed since preflight: {}".format(path))
            validate_destination(destination, target)
            destination.parent.mkdir(parents=True, exist_ok=True)
            validate_destination(destination, target)
            move_without_overwrite(path, destination)
            moved += 1
            if moved % 200 == 0 or moved == len(plans):
                print("Moved: {}/{}".format(moved, len(plans)), flush=True)
        except (OSError, ValueError) as error:
            print("Stopped after {} successful moves. Failed file: {}".format(moved, path),
                  file=sys.stderr)
            raise error
    print("Done: moved {} APKs to {}".format(moved, target))
    print("APKs left in source: {}".format(len(files) - moved))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
