"""
Entry point standalone per il flusso:
  PDF -> diagram candidates -> best cluster -> multimodal API -> reference_graph.json

Uso:
  python run_pdf_to_reference_graph.py
  python run_pdf_to_reference_graph.py --pdf folder_with_pdf/node9.pdf
"""
from __future__ import annotations

from pathlib import Path
import sys
import json

from config import PDF_DIR
from project_prior import (
    extract_pdf_prior,
    extract_diagram_candidates,
    build_reference_graph_from_pdf_prior,
    save_reference_graph_overlay,
    save_reference_graph_schema_view,
)


def _resolve_arg(argv: list[str], name: str) -> str | None:
    if name in argv:
        idx = argv.index(name)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def main() -> None:
    pdf_path = _resolve_arg(sys.argv, "--pdf")
    if not pdf_path:
        pdf_path = PDF_DIR / "node9.pdf"
    parsed = extract_pdf_prior(pdf_path)
    diagrams = extract_diagram_candidates(parsed)
    result = build_reference_graph_from_pdf_prior(parsed, diagram_result=diagrams)

    print(f"[PDF->Graph] pdf={pdf_path}")
    print(f"[PDF->Graph] parse_status={parsed.status} backend={parsed.backend_used} pages={len(parsed.pages)}")
    total_candidates = sum(len(items) for items in diagrams.page_candidates.values())
    print(f"[PDF->Graph] diagram_candidates={total_candidates}")
    print(
        f"[PDF->Graph] status={result.status} model={result.model} "
        f"components={len(result.components)} connections={len(result.connections)}"
    )
    print(f"[PDF->Graph] debug_dir={result.debug_paths.get('root')}")
    for warning in result.warnings:
        print(f"[PDF->Graph][warn] {warning}")

    graph_arg = _resolve_arg(sys.argv, "--graph")
    image_arg = _resolve_arg(sys.argv, "--image")

    graph_path = Path(graph_arg) if graph_arg else Path(
        "output/pdf_prior/node9/llm_reference_graph/graph/reference_graph.json"
    )
    image_path = Path(image_arg) if image_arg else Path(
        "output/pdf_prior/node9/llm_reference_graph/inputs/page0_best_cluster.png"
    )

    if not graph_path.exists():
        raise FileNotFoundError(f"Reference graph JSON non trovato: {graph_path}")

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    out_dir = graph_path.parent.parent / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)

    schema_paths = save_reference_graph_schema_view(payload, out_dir)
    overlay_paths = {}
    if image_path.exists():
        overlay_paths = save_reference_graph_overlay(image_path, payload, out_dir)

    print(f"[ReferenceGraphViz] graph={graph_path}")
    print(f"[ReferenceGraphViz] visuals_dir={out_dir}")
    for key, value in {**schema_paths, **overlay_paths}.items():
        print(f"[ReferenceGraphViz] {key}={value}")


if __name__ == "__main__":
    main()