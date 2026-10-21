from __future__ import annotations

import argparse
import html
import shutil
from pathlib import Path

try:
    from .confronta_pipeline_con_standard import MODULE_DIR, load_geojson_graph, natural_key
    from .visualizza_best_match import draw_graph, edge_set
except ImportError:  # direct script execution
    from confronta_pipeline_con_standard import MODULE_DIR, load_geojson_graph, natural_key
    from visualizza_best_match import draw_graph, edge_set


DEFAULT_STANDARDS = MODULE_DIR / "grafi_campione_manuali_geojson"
DEFAULT_OUTPUT = MODULE_DIR.parents[1] / "output" / "standard_comparison" / "standards"


def render_svg(graph: dict, output_path: Path) -> None:
    width = 920
    height = 680
    highlighted_nodes = {node["id"] for node in graph["nodes"]}
    highlighted_edges = edge_set(graph)
    inventory = " | ".join(f"{key}x{value}" for key, value in sorted(graph["label_counts"].items()))

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#f6f8fb'/>",
        f"<text x='28' y='36' font-family='Segoe UI, Arial' font-size='24' font-weight='700' fill='#172033'>{html.escape(graph['id'].upper())}</text>",
        f"<text x='28' y='63' font-family='Consolas, monospace' font-size='13' fill='#526071'>{html.escape(inventory)}</text>",
    ]
    lines.extend(
        draw_graph(
            graph,
            28,
            86,
            864,
            550,
            f"Standard corretto: {graph['id'].upper()}",
            highlighted_nodes,
            highlighted_edges,
            "standard",
        )
    )
    lines.append("<text x='28' y='662' font-family='Segoe UI, Arial' font-size='12' fill='#526071'>Numero nel nodo = shurikode; etichetta esterna = id canonico.</text>")
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_index(output_dir: Path, rows: list[dict]) -> None:
    cards = "".join(
        f"<article><h2>{row['id'].upper()}</h2><p>{row['nodes']} nodi, {row['edges']} archi</p>"
        f"<a href='{row['file']}'><img src='{row['file']}' alt='{row['id']}' /></a></article>"
        for row in rows
    )
    page = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"/><title>Standard idraulici corretti</title>
<style>
body{{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f6f8fb;color:#172033}}
header{{padding:22px 28px;background:white;border-bottom:1px solid #d8dee9}}
h1{{margin:0 0 6px;font-size:24px}} header p,article p{{margin:0;color:#526071}}
main{{padding:20px 28px 36px;display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
article{{background:white;border:1px solid #d8dee9;border-radius:8px;overflow:hidden}}
article h2{{margin:0;padding:12px 14px 2px;font-size:17px}} article p{{padding:0 14px 10px;font-size:13px}}
img{{display:block;width:100%;border-top:1px solid #edf1f6}}
</style></head><body><header><h1>Standard idraulici corretti</h1>
<p>Galleria N01-N12 ricostruita dal PDF e dalla mappatura manuale.</p></header><main>{cards}</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--standards", type=Path, default=DEFAULT_STANDARDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    rows = []
    for path in sorted(args.standards.glob("n*.geojson"), key=natural_key):
        graph = load_geojson_graph(path)
        file_name = f"{graph['id']}.svg"
        render_svg(graph, args.output / file_name)
        rows.append({"id": graph["id"], "file": file_name, "nodes": len(graph["nodes"]), "edges": len(graph["edges"])})
        print(f"{graph['id'].upper()} -> {args.output / file_name}")

    write_index(args.output, rows)
    print(f"Galleria: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
