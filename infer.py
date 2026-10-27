"""Esegue la ricostruzione da una cartella di immagini ed esporta per QGIS."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ricostruisce tutti i nodi presenti in una cartella di immagini e "
            "genera i GeoJSON pronti per l'importazione in QGIS."
        )
    )
    parser.add_argument(
        "images",
        type=Path,
        help="Cartella contenente file node<id>_<frame>.jpg.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Directory di destinazione. Default: ./output.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Checkpoint YOLO alternativo. Default: ./models/yolo_detector.pt.",
    )
    parser.add_argument(
        "--geometry-source",
        choices=["fake", "auto", "vggt"],
        default="fake",
        help="Geometria dell'export QGIS. Default: fake, con layout CAD.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    images_dir = args.images.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not images_dir.is_dir():
        raise SystemExit(f"Cartella immagini non trovata: {images_dir}")

    os.environ["IMAGES_DIR"] = str(images_dir)
    os.environ["WORK_DIR"] = str(output_dir)
    if args.model is not None:
        model_path = args.model.expanduser().resolve()
        if not model_path.is_file():
            raise SystemExit(f"Checkpoint YOLO non trovato: {model_path}")
        os.environ["YOLO_MODEL_PATH"] = str(model_path)

    from graph_reconstruction.pipeline.orchestrator import run_multiview_pipeline
    from graph_reconstruction.qgis import convert_fused_graph_to_qgis
    from graph_reconstruction.shared.detection import discover_nodes

    nodes = discover_nodes(images_dir)
    if not nodes:
        raise SystemExit(
            "Nessuna immagine valida trovata. Atteso: node<id>_<frame>.jpg"
        )

    generated: list[Path] = []
    for node_id, image_paths in nodes.items():
        result = run_multiview_pipeline(node_id, image_paths)
        if result.get("status") != "ok":
            raise SystemExit(f"Inferenza non completata per il nodo {node_id}: {result}")

        node_dir = output_dir / "multiview" / f"node{node_id}"
        fused_graph = node_dir / "fused_graph.json"
        paths = convert_fused_graph_to_qgis(
            fused_graph,
            node_dir,
            geometry_source=args.geometry_source,
        )
        generated.append(paths["combined"])

    print("\nOutput QGIS generati:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
