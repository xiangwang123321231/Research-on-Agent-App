"""Export package names and original Google Play store category IDs.

All original GAME_* categories are represented as one GAME category. Other
category IDs are preserved without translation or research-category mapping.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path


def export_categories(source: Path, destination: Path, *, overwrite: bool) -> tuple[int, int]:
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output paths must differ")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination} (pass --overwrite to replace)")

    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    total = 0
    with source.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"package_name", "resolved_category_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Input is missing columns: {sorted(required - set(reader.fieldnames or []))}")

        for line_number, row in enumerate(reader, start=2):
            total += 1
            category_id = (row["resolved_category_id"] or "").strip()
            if not category_id:
                continue

            package_name = (row["package_name"] or "").strip()
            if not package_name:
                raise ValueError(f"Missing package name at CSV line {line_number}")
            if package_name in seen:
                raise ValueError(f"Duplicate resolved package at CSV line {line_number}: {package_name}")
            seen.add(package_name)

            exported_category = "GAME" if category_id.startswith("GAME_") else category_id
            records.append((package_name, exported_category))

    if not records:
        raise ValueError("Input has no resolved categories")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8-sig", newline="", suffix=".tmp",
            prefix=destination.name + ".", dir=destination.parent, delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            writer = csv.writer(output_file)
            writer.writerow(("package_name", "store_category_id"))
            writer.writerows(records)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return total, len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Merged categories CSV")
    parser.add_argument("--output", type=Path, required=True, help="Two-column output CSV")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    args = parser.parse_args()
    total, exported = export_categories(args.input, args.output, overwrite=args.overwrite)
    print(f"Exported {exported} of {total} packages to {args.output}")


if __name__ == "__main__":
    main()
