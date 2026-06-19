"""Package per-node combined QGIS GeoJSON exports into a flat zip archive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = ROOT / "output" / "multiview"
DEFAULT_OUTPUT_ZIP = ROOT / "output" / "qgis_geojson_layers.zip"


def _node_file_name(node_dir: Path, *, prefix: str) -> str:
    name = node_dir.name
    if name.startswith("node"):
        return f"{prefix}{name[len('node'):]}.geojson"
    return f"{name}.geojson"


def _iter_combined_geojson(input_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for node_dir in sorted(input_root.glob("node*")):
        if not node_dir.is_dir():
            continue
        for geojson_path in sorted(node_dir.glob("*_qgis_combined.geojson")):
            pairs.append((node_dir, geojson_path))
    return pairs


def package_qgis_geojson(
    input_root: Path = DEFAULT_INPUT_ROOT,
    output_zip: Path = DEFAULT_OUTPUT_ZIP,
    *,
    file_prefix: str = "nodo",
    overwrite: bool = False,
) -> int:
    input_root = input_root.resolve()
    output_zip = output_zip.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if output_zip.exists() and not overwrite:
        raise FileExistsError(f"Output zip already exists: {output_zip}")

    files = _iter_combined_geojson(input_root)
    if not files:
        raise FileNotFoundError(f"No combined QGIS GeoJSON files found under: {input_root}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for node_dir, geojson_path in files:
            archive.write(geojson_path, _node_file_name(node_dir, prefix=file_prefix))

    return len(files)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a flat zip archive with one combined QGIS GeoJSON file per node."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root containing node*/ folders. Default: output/multiview.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ZIP,
        help="Destination zip path. Default: output/qgis_geojson_layers.zip.",
    )
    parser.add_argument(
        "--file-prefix",
        default="nodo",
        help="Prefix for GeoJSON files inside the zip. Default: nodo.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output zip if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    n_files = package_qgis_geojson(
        args.input_root,
        args.output,
        file_prefix=args.file_prefix,
        overwrite=args.overwrite,
    )
    print(f"Created {args.output} with {n_files} GeoJSON file(s).")


if __name__ == "__main__":
    main()
