from __future__ import annotations

import argparse
from pathlib import Path


def _should_keep(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    lowered_name = path.name.lower()
    return any("vggt" in part for part in lowered_parts) or "vggt" in lowered_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete JSON files under output/, preserving VGGT directories and VGGT-related JSON files."
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Root output directory to clean. Default: output",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be removed without deleting them.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"[clean_output_json] directory not found: {output_dir}")
        return

    removed: list[Path] = []
    kept: list[Path] = []
    for path in sorted(output_dir.rglob("*.json")):
        rel = path.relative_to(output_dir)
        if _should_keep(rel):
            kept.append(path)
            continue
        removed.append(path)
        if not args.dry_run:
            path.unlink()

    mode = "would remove" if args.dry_run else "removed"
    print(f"[clean_output_json] {mode} {len(removed)} JSON file(s)")
    for path in removed:
        print(f"  {path}")
    print(f"[clean_output_json] kept {len(kept)} VGGT-related JSON file(s)")


if __name__ == "__main__":
    main()
