#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rename plot files to match plots_manifest.csv.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("plots_manifest.csv"),
        help="Path to plots_manifest.csv",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repo root (contains config folders).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned renames without changing files.",
    )
    return parser.parse_args()


def rename_from_manifest(manifest_path: Path, root: Path, dry_run: bool) -> int:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    renamed = 0
    skipped = 0
    missing = 0

    with manifest_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            config = row.get("config", "").strip()
            default_name = row.get("default_name", "").strip()
            filename = row.get("filename", "").strip()

            if not config or not default_name or not filename:
                print(f"Skipping malformed row: {row}")
                continue

            plots_dir = root / config / "plots"
            src = plots_dir / default_name
            dst = plots_dir / filename

            if dst.exists():
                skipped += 1
                continue

            if not src.exists():
                missing += 1
                print(f"Missing source: {src}")
                continue

            if dry_run:
                print(f"[dry-run] {src} -> {dst}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
                print(f"Renamed: {src} -> {dst}")
            renamed += 1

    print(
        f"Done. Renamed: {renamed}, Skipped: {skipped}, Missing: {missing}",
    )
    return 0


def main() -> int:
    args = parse_args()
    manifest_path = args.root / args.manifest
    return rename_from_manifest(manifest_path, args.root, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
