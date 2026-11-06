#!/usr/bin/env python3
"""Move all APKs, preserving relative subdirectories, without overwriting files.

Default: read-only preflight. Add --execute to move.
No SHA list, count restriction, or previously excluded hash is used.
Other files are left in place. Symlinks are rejected, never followed.
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
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=Path("/home/ubuntu/workspace/yzhou/apks"))
    parser.add_argument("--target-dir", type=Path,
                        default=Path("/home/ubuntu/workspace/jwang/metadata/apks"))
    parser.add_argument("--execute", action="store_true",
                        help="Move files; otherwise only perform read-only preflight.")
    parser.add_argument("--verify-sha256", action="store_true",
                        help="Additionally require every APK filename to be its correct SHA-256.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    source = checked_path(args.source_dir)
    target = checked_path(args.target_dir)
    if not source.is_dir():
        raise ValueError("Source directory does not exist: {}".format(source))
    if is_within(source, target) or is_within(target, source):
        raise ValueError("Source and target must be different, non-nested directories.")
    if target.exists() and not target.is_dir():
        raise ValueError("Target is not a directory: {}".format(target))

    files = scan_apks(source)
    plans = [(path, target / path.relative_to(source)) for path in files]
    conflicts = []
    for _, destination in plans:
        try:
            validate_destination(destination, target)
        except (OSError, ValueError) as error:
            conflicts.append(str(error))

    print("Source: {}".format(source))
    print("Target: {}".format(target))
    print("APKs found: {}; conflicts: {}".format(len(plans), len(conflicts)))
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
        print("Dry run only; no files changed. Add --execute to move all {} APKs.".format(len(plans)))
        return 0

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
            if moved % 500 == 0 or moved == len(plans):
                print("Moved: {}/{}".format(moved, len(plans)), flush=True)
        except (OSError, ValueError) as error:
            print("Stopped after {} successful moves. Failed file: {}".format(moved, path),
                  file=sys.stderr)
            raise error
    print("Done: moved {} APKs. Non-APK files were left unchanged.".format(moved))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print("ERROR: {}".format(error), file=sys.stderr)
        raise SystemExit(1)
