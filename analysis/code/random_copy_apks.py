"""Reproducibly copy a random sample of APK files; never move the originals.

Linux defaults:
    python random_copy_apks.py

Only the source directory's immediate files are sampled unless --recursive is
specified. This samples files, not unique binary contents, and does not silently
exclude training APKs. See random_copy_apks_README.md before using it for testing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SOURCE = Path("/home/ubuntu/workspace/jwang/metadata/apks")
DEFAULT_SEED = 20261016


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_sample(
    source: Path,
    output: Path | None = None,
    count: int = 200,
    seed: int = DEFAULT_SEED,
    recursive: bool = False,
) -> dict:
    source = Path(source).expanduser().resolve()
    if count < 1:
        raise ValueError("Sample count must be a positive integer.")
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")

    output = (
        Path(output).expanduser().resolve()
        if output is not None
        else source.with_name(f"{source.name}_random_{count}")
    )
    try:
        output.relative_to(source)
    except ValueError:
        pass
    else:
        raise ValueError("Output must be outside the source directory.")
    if output.exists():
        raise ValueError(f"Output already exists; nothing overwritten: {output}")

    entries = source.rglob("*") if recursive else source.iterdir()
    pool = sorted(
        (
            path for path in entries
            if path.suffix.lower() == ".apk"
            and not path.is_symlink()
            and path.is_file()
        ),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    # Reject any file reached through a symlinked parent directory as well.
    pool = [path for path in pool if path.resolve() == path]
    if len(pool) < count:
        raise ValueError(
            f"Found only {len(pool)} APK files; {count} requested. "
            "Nothing copied. Use --recursive if APKs are in subdirectories."
        )

    selected = random.Random(seed).sample(pool, count)
    manifest = {
        "status": "copying",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(source),
        "output_directory": str(output),
        "sampling_unit": "APK file (not deduplicated by content)",
        "sampling_method": "simple random sampling without replacement",
        "seed": seed,
        "recursive": recursive,
        "training_samples_automatically_excluded": False,
        "candidate_count": len(pool),
        "requested_count": count,
        "copied_count": 0,
        "candidate_relative_paths": [
            path.relative_to(source).as_posix() for path in pool
        ],
        "samples": [
            {"relative_path": path.relative_to(source).as_posix(), "status": "pending"}
            for path in selected
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "sampling_manifest.json"
    save_manifest(manifest_path, manifest)

    print(f"Source: {source}")
    print(f"Candidates: {len(pool)}; sample: {count}; seed: {seed}")
    print(f"Output: {output}", flush=True)
    try:
        for index, (source_file, record) in enumerate(
            zip(selected, manifest["samples"]), start=1
        ):
            destination = output / source_file.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation prevents overwriting even if another process
            # unexpectedly creates a file inside this new output directory.
            with destination.open("xb") as target:
                with source_file.open("rb") as original:
                    shutil.copyfileobj(original, target, length=1024 * 1024)
            shutil.copystat(source_file, destination)
            source_hash = sha256_file(source_file)
            if sha256_file(destination) != source_hash:
                raise RuntimeError(f"SHA-256 verification failed: {source_file}")
            record.update(
                status="copied",
                sha256=source_hash,
                size_bytes=destination.stat().st_size,
            )
            manifest["copied_count"] = index
            save_manifest(manifest_path, manifest)
            print(f"[{index}/{count}] {record['relative_path']}", flush=True)
    except (OSError, shutil.Error, RuntimeError, KeyboardInterrupt) as error:
        manifest["status"] = "incomplete"
        manifest["error"] = str(error) or type(error).__name__
        save_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["unique_sample_sha256_count"] = len(
        {record["sha256"] for record in manifest["samples"]}
    )
    save_manifest(manifest_path, manifest)
    print(f"Completed: {count} APK files copied. Originals were not moved.")
    if manifest["unique_sample_sha256_count"] < count:
        print("WARNING: Some selected files contain identical APK binaries.")
    print(f"Manifest: {manifest_path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories.")
    args = parser.parse_args()
    try:
        copy_sample(args.source, args.output, args.count, args.seed, args.recursive)
    except (OSError, shutil.Error, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. Partial copies and their manifest were preserved.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
