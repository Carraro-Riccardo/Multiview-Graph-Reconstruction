from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

try:
    from .confronta_pipeline_con_standard import (
        DEFAULT_OUTPUT, DEFAULT_PIPELINE, MODULE_DIR, compare_graphs,
        comparison_sort_key, load_geojson_graph, natural_key,
    )
except ImportError:  # direct script execution
    from confronta_pipeline_con_standard import (
        DEFAULT_OUTPUT, DEFAULT_PIPELINE, MODULE_DIR, compare_graphs,
        comparison_sort_key, load_geojson_graph, natural_key,
    )


DEFAULT_STANDARDS = MODULE_DIR / "grafi_campione_manuali_geojson"
DEFAULT_VIS_OUTPUT = DEFAULT_OUTPUT


def edge_set(graph: dict) -> set[tuple[str, str]]:
    return {tuple(sorted((edge["source"], edge["target"]))) for edge in graph["edges"]}


def graph_bounds(graph: dict):
    coords = [node.get("coordinates") for node in graph["nodes"] if node.get("coordinates")]
    xs = [float(coord[0]) for coord in coords]
    ys = [float(coord[1]) for coord in coords]
    if not xs or not ys:
        return -1.0, -1.0, 1.0, 1.0
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        min_y -= 1.0
        max_y += 1.0
    return min_x, min_y, max_x, max_y


def make_projector(graph: dict, x: float, y: float, width: float, height: float, pad: float = 36.0):
    min_x, min_y, max_x, max_y = graph_bounds(graph)
    span_x = max_x - min_x
    span_y = max_y - min_y
    scale = min((width - pad * 2) / span_x, (height - pad * 2) / span_y)

    graph_w = span_x * scale
    graph_h = span_y * scale
    offset_x = x + (width - graph_w) / 2
    offset_y = y + (height - graph_h) / 2

    def project(coords):
        px = offset_x + (float(coords[0]) - min_x) * scale
        py = offset_y + graph_h - (float(coords[1]) - min_y) * scale
        return px, py

    return project


def node_label(node: dict) -> str:
    return f"{node['id']} / {node['label']}"


def draw_graph(
    graph: dict,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    highlighted_nodes: set[str],
    highlighted_edges: set[tuple[str, str]],
    role: str,
) -> list[str]:
    project = make_projector(graph, x, y + 34, width, height - 34)
    nodes = {node["id"]: node for node in graph["nodes"]}
    lines = [
        f"<rect x='{x}' y='{y}' width='{width}' height='{height}' rx='8' fill='#ffffff' stroke='#d8dee9'/>",
        f"<text x='{x + 16}' y='{y + 23}' font-family='Segoe UI, Arial' font-size='15' font-weight='700' fill='#172033'>{html.escape(title)}</text>",
        f"<text x='{x + width - 132}' y='{y + 23}' font-family='Consolas, monospace' font-size='12' fill='#526071'>{len(graph['nodes'])}N / {len(graph['edges'])}A</text>",
    ]

    for edge in graph["edges"]:
        source = nodes.get(edge["source"])
        target = nodes.get(edge["target"])
        if not source or not target:
            continue
        x1, y1 = project(source["coordinates"])
        x2, y2 = project(target["coordinates"])
        key = tuple(sorted((edge["source"], edge["target"])))
        is_optional_53 = source["label"] == 53 or target["label"] == 53
        is_common = key in highlighted_edges
        color = "#4d82c4" if is_optional_53 else ("#18915f" if is_common else "#b9c2d0")
        stroke_width = 4 if is_common else 1.5
        opacity = 1.0 if is_common else 0.75
        lines.append(
            f"<line x1='{x1:.2f}' y1='{y1:.2f}' x2='{x2:.2f}' y2='{y2:.2f}' stroke='{color}' stroke-width='{stroke_width}' opacity='{opacity}' stroke-linecap='round'/>"
        )

    for node in graph["nodes"]:
        px, py = project(node["coordinates"])
        is_optional_53 = node["label"] == 53
        is_common = node["id"] in highlighted_nodes
        fill = "#8fb7e8" if is_optional_53 else ("#24b36b" if is_common else ("#f6a04d" if role == "pipeline" else "#e4e9f0"))
        stroke = "#356cae" if is_optional_53 else ("#0b6d42" if is_common else ("#b56718" if role == "pipeline" else "#8090a3"))
        radius = 10 if is_common else 8
        lines.append(
            f"<circle cx='{px:.2f}' cy='{py:.2f}' r='{radius}' fill='{fill}' stroke='{stroke}' stroke-width='2'/>"
        )
        label = html.escape(str(node["label"]))
        lines.append(
            f"<text x='{px:.2f}' y='{py + 4:.2f}' text-anchor='middle' font-family='Consolas, monospace' font-size='9' fill='#172033'>{label}</text>"
        )
        id_text = html.escape(str(node["id"]))
        lines.append(
            f"<text x='{px + 11:.2f}' y='{py - 10:.2f}' font-family='Consolas, monospace' font-size='8' fill='#526071'>{id_text}</text>"
        )

    return lines


def common_edges(query: dict, standard: dict, mapping: dict[str, str]):
    q_common = set()
    s_common = set()
    s_edges = edge_set(standard)
    for edge in query["edges"]:
        u = edge["source"]
        v = edge["target"]
        if u not in mapping or v not in mapping:
            continue
        mapped = tuple(sorted((mapping[u], mapping[v])))
        if mapped in s_edges:
            q_common.add(tuple(sorted((u, v))))
            s_common.add(mapped)
    return q_common, s_common


def render_match_svg(query: dict, standard: dict, comparison: dict) -> str:
    mapping = comparison["mapping"]
    q_nodes = set(mapping.keys())
    s_nodes = set(mapping.values())
    q_edges, s_edges = common_edges(query, standard, mapping)

    width = 1280
    height = 760
    panel_y = 108
    panel_h = 560
    panel_w = 590
    gap = 30

    coverage = comparison["query_coverage"]
    status = "MCS completo sul core pipeline" if coverage == 1.0 else ("MCS ampio" if coverage >= 0.75 else ("MCS parziale" if coverage >= 0.5 else "MCS limitato"))

    lines = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='1280' height='760' viewBox='0 0 1280 760'>",
        "<rect width='100%' height='100%' fill='#f6f8fb'/>",
        f"<text x='32' y='38' font-family='Segoe UI, Arial' font-size='24' font-weight='700' fill='#172033'>{html.escape(query['id'])} vs {html.escape(standard['id'])}</text>",
        f"<text x='32' y='65' font-family='Segoe UI, Arial' font-size='14' fill='#526071'>Best match: {html.escape(standard['id'])} | stato: {status}</text>",
        f"<text x='32' y='90' font-family='Consolas, monospace' font-size='13' fill='#172033'>MCS componenti={comparison['mcs_nodes']}/{comparison['query_nodes']} | MCS collegamenti={comparison['mcs_edges']}/{comparison['query_edges']} | GEDproxy={comparison['ged_proxy']} | 53 opzionali={comparison['query_53_optional']}/{comparison['standard_53_optional']}</text>",
        "<rect x='930' y='26' width='18' height='18' rx='4' fill='#24b36b' stroke='#0b6d42'/>",
        "<text x='956' y='40' font-family='Segoe UI, Arial' font-size='12' fill='#172033'>sottografo comune</text>",
        "<rect x='930' y='52' width='18' height='18' rx='4' fill='#f6a04d' stroke='#b56718'/>",
        "<text x='956' y='66' font-family='Segoe UI, Arial' font-size='12' fill='#172033'>extra pipeline</text>",
        "<rect x='930' y='78' width='18' height='18' rx='4' fill='#e4e9f0' stroke='#8090a3'/>",
        "<text x='956' y='92' font-family='Segoe UI, Arial' font-size='12' fill='#172033'>extra standard</text>",
        "<rect x='1110' y='78' width='18' height='18' rx='4' fill='#8fb7e8' stroke='#356cae'/>",
        "<text x='1136' y='92' font-family='Segoe UI, Arial' font-size='12' fill='#172033'>53 ignorato</text>",
    ]

    lines.extend(
        draw_graph(
            query,
            32,
            panel_y,
            panel_w,
            panel_h,
            f"Pipeline: {query['id']}",
            q_nodes,
            q_edges,
            "pipeline",
        )
    )
    lines.extend(
        draw_graph(
            standard,
            32 + panel_w + gap,
            panel_y,
            panel_w,
            panel_h,
            f"Standard aziendale: {standard['id']}",
            s_nodes,
            s_edges,
            "standard",
        )
    )

    lines.append(
        f"<text x='32' y='707' font-family='Segoe UI, Arial' font-size='12' fill='#526071'>Verde = MCS sul grafo core. Blu = condotta 53 opzionale, esclusa da MCS, GED e ranking.</text>"
    )
    lines.append("</svg>")
    return "\n".join(lines)


def make_index(output_dir: Path, rows: list[dict]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"<article><h2>{html.escape(row['query'])} -> {html.escape(row['standard'])}</h2>"
            f"<p>MCS {row['mcs_nodes']}/{row['query_nodes']} nodi, {row['mcs_edges']} archi; GEDproxy {row['ged_proxy']}; 53 opzionali {row['query_53_optional']}/{row['standard_53_optional']}</p>"
            f"<a href='{html.escape(row['file'])}'><img src='{html.escape(row['file'])}' alt='{html.escape(row['query'])}' /></a></article>"
        )
    html_doc = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <title>Visualizzazioni best match</title>
  <style>
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #f6f8fb; color: #172033; }}
    header {{ padding: 24px 28px 10px; background: white; border-bottom: 1px solid #d8dee9; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    p {{ margin: 0 0 12px; color: #526071; }}
    main {{ padding: 20px 28px 36px; display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    article {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; overflow: hidden; }}
    article h2 {{ margin: 0; padding: 12px 14px 0; font-size: 16px; }}
    article p {{ padding: 4px 14px 10px; font-size: 13px; }}
    img {{ width: 100%; display: block; border-top: 1px solid #edf1f6; }}
  </style>
</head>
<body>
  <header>
    <h1>Visualizzazioni best match</h1>
    <p>Pipeline a sinistra, standard aziendale piu simile a destra. Verde = sottografo comune.</p>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_doc, encoding="utf-8")


def make_standalone_report(output_dir: Path, rows: list[dict]) -> None:
    navigation = "".join(
        f"<a href='#{html.escape(row['query'])}'>{html.escape(row['query'])} -> {html.escape(row['standard'])}</a>"
        for row in rows
    )
    sections = []
    summary_rows = []
    for row in rows:
        svg = row["svg"]
        summary_rows.append(
            f"<tr><td>{html.escape(row['query'])}</td><td>{html.escape(row['standard'])}</td>"
            f"<td>{row['mcs_nodes']}/{row['query_nodes']}</td>"
            f"<td>{row['mcs_edges']}/{row['query_edges']}</td><td>{row['ged_proxy']}</td>"
            f"<td>{row['query_53_optional']}/{row['standard_53_optional']}</td></tr>"
        )
        sections.append(
            f"<section id='{html.escape(row['query'])}'>"
            f"<div class='section-head'><h2>{html.escape(row['query'])} -> {html.escape(row['standard'])}</h2>"
            f"<p>MCS {row['mcs_nodes']}/{row['query_nodes']} nodi, {row['mcs_edges']} archi | GEDproxy {row['ged_proxy']} | 53 opzionali {row['query_53_optional']}/{row['standard_53_optional']}</p></div>"
            f"<div class='svg-wrap'>{svg}</div></section>"
        )
    document = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Report completo best match</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: Segoe UI, Arial, sans-serif; background: #eef2f7; color: #172033; }}
    header {{ position: sticky; top: 0; z-index: 10; padding: 18px 24px 12px; background: rgba(255,255,255,.97); border-bottom: 1px solid #d8dee9; }}
    h1 {{ margin: 0 0 5px; font-size: 23px; }}
    header p {{ margin: 0 0 12px; color: #526071; }}
    nav {{ display: flex; gap: 7px; overflow-x: auto; padding-bottom: 4px; }}
    nav a {{ flex: 0 0 auto; padding: 6px 9px; border: 1px solid #cbd4e1; border-radius: 6px; color: #234a82; background: #fff; text-decoration: none; font-size: 12px; }}
    main {{ width: min(1320px, calc(100% - 32px)); margin: 20px auto 40px; display: grid; gap: 22px; }}
    .summary {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #edf1f6; text-align: left; white-space: nowrap; }}
    th {{ background: #f3f6fb; }}
    section {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; overflow: hidden; scroll-margin-top: 125px; }}
    .section-head {{ display: flex; align-items: baseline; justify-content: space-between; gap: 20px; padding: 12px 16px; border-bottom: 1px solid #edf1f6; }}
    h2 {{ margin: 0; font-size: 17px; }}
    .section-head p {{ margin: 0; color: #526071; font-family: Consolas, monospace; font-size: 12px; }}
    .svg-wrap svg {{ display: block; width: 100%; height: auto; }}
    @media print {{ header {{ position: static; }} nav {{ display: none; }} section {{ break-inside: avoid; page-break-after: always; }} main {{ width: 100%; margin: 0; }} }}
  </style>
</head>
<body>
  <header><h1>Pipeline vs standard manuali: best match</h1>
  <p>Unico report autosufficiente. Verde = massimo sottografo comune.</p><nav>{navigation}</nav></header>
  <main><div class="summary"><table><thead><tr><th>Nodo ricostruito</th><th>Schema standard piu simile</th><th>MCS componenti comuni / rilevati</th><th>MCS collegamenti comuni / rilevati</th><th>GED proxy</th><th>Condotte 53 rilevate / standard</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>{''.join(sections)}</main>
</body>
</html>"""
    (output_dir / "report_completo.html").write_text(document, encoding="utf-8")


def generate_best_match_report(
    *,
    standards_dir: Path = DEFAULT_STANDARDS,
    pipeline_dir: Path = DEFAULT_PIPELINE,
    output_dir: Path = DEFAULT_VIS_OUTPUT,
    node_ids: list[str] | None = None,
) -> Path | None:
    """Create one self-contained HTML comparing every graph with its best match."""
    standard_files = sorted(standards_dir.glob("*.geojson"), key=natural_key)
    pipeline_files = sorted(
        pipeline_dir.glob("node*/fused_graph_qgis_combined.geojson"),
        key=lambda p: natural_key(p.parent),
    )
    if node_ids is not None:
        selected = {value if value.startswith("node") else f"node{value}" for value in node_ids}
        pipeline_files = [path for path in pipeline_files if path.parent.name in selected]

    if not standard_files or not pipeline_files:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    standards = [load_geojson_graph(path) for path in standard_files]
    rows = []

    for pipeline_path in pipeline_files:
        query = load_geojson_graph(pipeline_path)
        query["id"] = pipeline_path.parent.name
        query["path"] = str(pipeline_path)
        comparisons = [compare_graphs(query, standard) for standard in standards]
        comparisons.sort(key=comparison_sort_key)
        if not comparisons:
            continue
        best = comparisons[0]
        standard = next(item for item in standards if item["id"] == best["standard_id"])
        svg = render_match_svg(query, standard, best)
        rows.append(
            {
                "query": query["id"],
                "standard": standard["id"],
                "svg": svg,
                "mcs_nodes": best["mcs_nodes"],
                "mcs_edges": best["mcs_edges"],
                "query_nodes": best["query_nodes"],
                "query_edges": best["query_edges"],
                "ged_proxy": best["ged_proxy"],
                "query_53_optional": best["query_53_optional"],
                "standard_53_optional": best["standard_53_optional"],
            }
        )
        print(f"{query['id']} -> {standard['id']}")

    if not rows:
        return None
    make_standalone_report(output_dir, rows)
    return output_dir / "report_completo.html"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standards", type=Path, default=DEFAULT_STANDARDS)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_VIS_OUTPUT)
    parser.add_argument("--only", help="Generate one pipeline node only, e.g. node5.")
    args = parser.parse_args()

    report_path = generate_best_match_report(
        standards_dir=args.standards,
        pipeline_dir=args.pipeline,
        output_dir=args.output,
        node_ids=[args.only] if args.only else None,
    )
    if report_path is None:
        raise SystemExit("No pipeline/standard graph available for the HTML report.")
    print(f"Report unico: {report_path}")


if __name__ == "__main__":
    main()
