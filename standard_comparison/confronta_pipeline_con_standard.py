from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parent
ROOT = MODULE_DIR.parent
DEFAULT_STANDARDS = MODULE_DIR / "grafi_campione_manuali_geojson"
DEFAULT_PIPELINE = ROOT / "output" / "multiview"
DEFAULT_OUTPUT = ROOT / "output" / "standard_comparison"


def as_int(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def coord_key(coords, ndigits=9):
    if not coords or len(coords) < 2:
        return None
    return (round(float(coords[0]), ndigits), round(float(coords[1]), ndigits))


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def load_geojson_graph(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", [])

    raw_points = {}
    point_id_by_coord = {}
    referenced_ids = set()

    for feature in features:
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        if geom.get("type") == "LineString":
            if props.get("da") is not None:
                referenced_ids.add(str(props.get("da")))
            if props.get("a") is not None:
                referenced_ids.add(str(props.get("a")))

    for feature in features:
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        node_id = props.get("id")
        shurikode = as_int(props.get("shurikode"))
        if node_id is None or shurikode is None:
            continue

        tipo = props.get("tipo_layer")
        is_master = str(tipo).lower() == "master"
        if is_master and str(node_id) not in referenced_ids:
            continue

        node = {
            "id": str(node_id),
            "label": shurikode,
            "name": props.get("nome") or props.get("etichetta") or str(shurikode),
            "tipo_layer": tipo,
            "coordinates": geom.get("coordinates"),
        }
        raw_points[node["id"]] = node
        key = coord_key(node["coordinates"])
        if key is not None:
            point_id_by_coord[key] = node["id"]

    warnings = []
    edges = []
    seen_edges = set()

    for feature in features:
        props = feature.get("properties") or {}
        geom = feature.get("geometry") or {}
        if geom.get("type") != "LineString":
            continue

        coords = geom.get("coordinates") or []
        da = str(props.get("da")) if props.get("da") is not None else None
        a = str(props.get("a")) if props.get("a") is not None else None

        if da not in raw_points and coords:
            fixed = point_id_by_coord.get(coord_key(coords[0]))
            if fixed:
                warnings.append({
                    "type": "edge_endpoint_fixed_from_geometry",
                    "edge_id": props.get("id"),
                    "field": "da",
                    "original": da,
                    "fixed": fixed,
                })
                da = fixed
        if a not in raw_points and len(coords) >= 2:
            fixed = point_id_by_coord.get(coord_key(coords[-1]))
            if fixed:
                warnings.append({
                    "type": "edge_endpoint_fixed_from_geometry",
                    "edge_id": props.get("id"),
                    "field": "a",
                    "original": a,
                    "fixed": fixed,
                })
                a = fixed

        if da not in raw_points or a not in raw_points or da == a:
            warnings.append({
                "type": "edge_dropped_bad_reference",
                "edge_id": props.get("id"),
                "da": da,
                "a": a,
            })
            continue

        u, v = sorted([da, a])
        edge_key = (u, v)
        if edge_key in seen_edges:
            warnings.append({"type": "duplicate_edge_dropped", "edge_id": props.get("id"), "da": da, "a": a})
            continue
        seen_edges.add(edge_key)
        edges.append({"id": str(props.get("id")), "source": u, "target": v})

    nodes = list(raw_points.values())
    adjacency = {node["id"]: set() for node in nodes}
    for edge in edges:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])

    components = connected_components(adjacency)
    label_counts = Counter(node["label"] for node in nodes)
    degree_sequence = sorted((len(adjacency[node["id"]]) for node in nodes), reverse=True)

    return {
        "id": path.stem,
        "path": str(path),
        "nodes": nodes,
        "edges": edges,
        "adjacency": {key: sorted(value) for key, value in adjacency.items()},
        "label_counts": dict(sorted(label_counts.items())),
        "degree_sequence": degree_sequence,
        "is_connected": len(components) <= 1,
        "components": components,
        "warnings": warnings,
    }


def connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    unseen = set(adjacency)
    components = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        comp = [start]
        while stack:
            node = stack.pop()
            for nxt in adjacency[node]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    stack.append(nxt)
                    comp.append(nxt)
        components.append(sorted(comp))
    return components


def induced_edge_set(graph: dict, subset: set[str] | None = None) -> set[tuple[str, str]]:
    if subset is None:
        subset = {node["id"] for node in graph["nodes"]}
    edges = set()
    for edge in graph["edges"]:
        if edge["source"] in subset and edge["target"] in subset:
            edges.add(tuple(sorted((edge["source"], edge["target"]))))
    return edges


def contract_optional_53(graph: dict) -> dict:
    """Remove shurikode 53 without breaking paths between real components."""
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    optional = {node_id for node_id, node in nodes_by_id.items() if node["label"] == 53}
    core = set(nodes_by_id) - optional
    original_adj = {node: set(neigh) for node, neigh in graph["adjacency"].items()}
    core_edges = {
        tuple(sorted((edge["source"], edge["target"])))
        for edge in graph["edges"]
        if edge["source"] in core and edge["target"] in core
    }

    unseen = set(optional)
    while unseen:
        start = unseen.pop()
        stack = [start]
        component = {start}
        boundary = set()
        while stack:
            current = stack.pop()
            for nxt in original_adj.get(current, set()):
                if nxt in optional and nxt not in component:
                    component.add(nxt)
                    unseen.discard(nxt)
                    stack.append(nxt)
                elif nxt in core:
                    boundary.add(nxt)
        boundary = sorted(boundary)
        for index, source in enumerate(boundary):
            for target in boundary[index + 1 :]:
                core_edges.add(tuple(sorted((source, target))))

    core_nodes = [nodes_by_id[node_id] for node_id in nodes_by_id if node_id in core]
    adjacency = {node["id"]: set() for node in core_nodes}
    edges = []
    for index, (source, target) in enumerate(sorted(core_edges), start=1):
        adjacency[source].add(target)
        adjacency[target].add(source)
        edges.append({"id": f"core_edge#{index}", "source": source, "target": target})

    result = dict(graph)
    result["nodes"] = core_nodes
    result["edges"] = edges
    result["adjacency"] = {node: sorted(neigh) for node, neigh in adjacency.items()}
    result["label_counts"] = dict(sorted(Counter(node["label"] for node in core_nodes).items()))
    result["optional_53_count"] = len(optional)
    return result


def maximum_common_induced_mapping(query: dict, standard: dict) -> dict:
    q_nodes = query["nodes"]
    s_nodes = standard["nodes"]
    q_by_id = {node["id"]: node for node in q_nodes}
    s_by_id = {node["id"]: node for node in s_nodes}
    q_adj = {node: set(neigh) for node, neigh in query["adjacency"].items()}
    s_adj = {node: set(neigh) for node, neigh in standard["adjacency"].items()}

    candidates = {
        q["id"]: [s["id"] for s in s_nodes if s["label"] == q["label"]]
        for q in q_nodes
    }

    ordered_q = sorted(
        [node["id"] for node in q_nodes if candidates[node["id"]]],
        key=lambda nid: (len(candidates[nid]), -len(q_adj[nid]), q_by_id[nid]["label"], nid),
    )

    best = {}
    best_edge_count = -1
    current = {}
    used_s = set()

    def compatible(q_id: str, s_id: str) -> bool:
        for mapped_q, mapped_s in current.items():
            q_has = mapped_q in q_adj[q_id]
            s_has = mapped_s in s_adj[s_id]
            if q_has != s_has:
                return False
        return True

    def backtrack(index: int) -> None:
        nonlocal best, best_edge_count
        if len(current) + (len(ordered_q) - index) < len(best):
            return
        if index == len(ordered_q):
            mapped_nodes = set(current)
            edge_count = len(induced_edge_set(query, mapped_nodes))
            if len(current) > len(best) or (len(current) == len(best) and edge_count > best_edge_count):
                best = dict(current)
                best_edge_count = edge_count
            return

        q_id = ordered_q[index]
        for s_id in candidates[q_id]:
            if s_id in used_s:
                continue
            if not compatible(q_id, s_id):
                continue
            current[q_id] = s_id
            used_s.add(s_id)
            backtrack(index + 1)
            used_s.remove(s_id)
            del current[q_id]

        backtrack(index + 1)

    backtrack(0)

    mapped_q = set(best)
    mapped_s = set(best.values())
    q_edges = induced_edge_set(query, mapped_q)
    s_edges = induced_edge_set(standard, mapped_s)
    edge_count = len(q_edges)

    return {
        "mapping": best,
        "mcs_nodes": len(best),
        "mcs_edges": edge_count,
        "query_coverage": len(best) / len(q_nodes) if q_nodes else 0.0,
        "standard_coverage": len(best) / len(s_nodes) if s_nodes else 0.0,
        "mapped_query_labels": Counter(q_by_id[qid]["label"] for qid in best),
        "mapped_standard_labels": Counter(s_by_id[sid]["label"] for sid in best.values()),
    }


def compare_graphs(query: dict, standard: dict) -> dict:
    query_core = contract_optional_53(query)
    standard_core = contract_optional_53(standard)
    mcs = maximum_common_induced_mapping(query_core, standard_core)

    q_edges = len(query_core["edges"])
    s_edges = len(standard_core["edges"])
    ged_proxy = (
        (len(query_core["nodes"]) - mcs["mcs_nodes"])
        + (len(standard_core["nodes"]) - mcs["mcs_nodes"])
        + (q_edges - mcs["mcs_edges"])
        + (s_edges - mcs["mcs_edges"])
    )

    return {
        "standard_id": standard["id"],
        "standard_path": standard["path"],
        "standard_nodes": len(standard_core["nodes"]),
        "standard_edges": len(standard_core["edges"]),
        "query_nodes": len(query_core["nodes"]),
        "query_edges": len(query_core["edges"]),
        "standard_nodes_total": len(standard["nodes"]),
        "query_nodes_total": len(query["nodes"]),
        "standard_53_optional": standard_core["optional_53_count"],
        "query_53_optional": query_core["optional_53_count"],
        "common_53_optional": min(query_core["optional_53_count"], standard_core["optional_53_count"]),
        "mcs_nodes": mcs["mcs_nodes"],
        "mcs_edges": mcs["mcs_edges"],
        "query_coverage": mcs["query_coverage"],
        "standard_coverage": mcs["standard_coverage"],
        "ged_proxy": ged_proxy,
        "mapping": mcs["mapping"],
    }


def comparison_sort_key(row: dict):
    return (-row["mcs_nodes"], -row["mcs_edges"], row["ged_proxy"], row["standard_id"])


def graph_summary(graph: dict) -> dict:
    return {
        "id": graph["id"],
        "path": graph["path"],
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "label_counts": graph["label_counts"],
        "degree_sequence": graph["degree_sequence"],
        "is_connected": graph["is_connected"],
        "warning_count": len(graph["warnings"]),
        "warnings": graph["warnings"],
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_company_csv(path: Path, rows: list[dict]) -> None:
    headers = [
        "Nodo ricostruito",
        "Schema standard piu simile",
        "Componenti comuni",
        "Componenti ricostruiti valutati",
        "Collegamenti comuni",
        "Collegamenti ricostruiti valutati",
        "Differenze residue (GED proxy)",
        "Condotte 53 nella ricostruzione (opzionali)",
        "Condotte 53 nello standard (opzionali)",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    row["query_id"],
                    row["best_standard"],
                    row["mcs_nodes"],
                    row["query_nodes"],
                    row["mcs_edges"],
                    row["query_edges"],
                    row["ged_proxy"],
                    row["query_53_optional"],
                    row["standard_53_optional"],
                ]
            )


def write_similarity_csv(path: Path, rows: list[dict]) -> None:
    """Persist every pipeline/standard comparison, not only the best match."""
    fields = [
        "query_id", "rank", "standard_id", "mcs_nodes", "mcs_edges",
        "query_nodes", "query_edges", "standard_nodes", "standard_edges",
        "query_coverage", "standard_coverage", "ged_proxy",
        "query_53_optional", "standard_53_optional",
    ]
    write_csv(path, [{key: row[key] for key in fields} for row in rows])


def evaluate_pipeline_against_standards(
    pipeline_dir: Path = DEFAULT_PIPELINE,
    standards_dir: Path = DEFAULT_STANDARDS,
    output_dir: Path = DEFAULT_OUTPUT,
    node_ids: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Compare reconstructed graphs with all standards and write evaluation artifacts."""
    standard_files = sorted(standards_dir.glob("*.geojson"), key=natural_key)
    pipeline_files = sorted(
        pipeline_dir.glob("node*/fused_graph_qgis_combined.geojson"),
        key=lambda path: natural_key(path.parent),
    )
    if node_ids is not None:
        selected = {value if value.startswith("node") else f"node{value}" for value in node_ids}
        pipeline_files = [path for path in pipeline_files if path.parent.name in selected]

    standards = [load_geojson_graph(path) for path in standard_files]
    queries = [
        load_geojson_graph(path) | {"id": path.parent.name, "path": str(path)}
        for path in pipeline_files
    ]
    if not standards:
        raise FileNotFoundError(f"Nessuno standard GeoJSON trovato in {standards_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    best_rows: list[dict] = []
    all_rows: list[dict] = []
    for query in queries:
        comparisons = [compare_graphs(query, standard) for standard in standards]
        comparisons.sort(key=comparison_sort_key)
        serializable = []
        for rank, comparison in enumerate(comparisons, start=1):
            row = {
                "query_id": query["id"],
                "rank": rank,
                **{key: value for key, value in comparison.items() if key != "mapping"},
            }
            row["query_coverage"] = round(row["query_coverage"], 6)
            row["standard_coverage"] = round(row["standard_coverage"], 6)
            all_rows.append(row)
            serializable.append({**row, "mapping": comparison["mapping"]})

        if comparisons:
            best = comparisons[0]
            best_rows.append({
                "query_id": query["id"], "best_standard": best["standard_id"],
                "query_nodes": best["query_nodes"], "query_edges": best["query_edges"],
                "standard_nodes": best["standard_nodes"], "standard_edges": best["standard_edges"],
                "mcs_nodes": best["mcs_nodes"], "mcs_edges": best["mcs_edges"],
                "query_coverage": round(best["query_coverage"], 6),
                "standard_coverage": round(best["standard_coverage"], 6),
                "ged_proxy": best["ged_proxy"],
                "query_53_optional": best["query_53_optional"],
                "standard_53_optional": best["standard_53_optional"],
                "common_53_optional": best["common_53_optional"],
            })

        # Keep node-scoped evaluation beside the graph that it describes.
        write_json(Path(query["path"]).parent / "standard_similarity.json", {
            "query_id": query["id"],
            "metric": "label-aware maximum common induced subgraph",
            "ranking": "mcs_nodes desc, mcs_edges desc, ged_proxy asc",
            "optional_shurikode": 53,
            "comparisons": serializable,
        })

    write_company_csv(output_dir / "best_matches.csv", best_rows)
    write_similarity_csv(output_dir / "similarity_table.csv", all_rows)
    write_json(output_dir / "similarity_table.json", all_rows)
    make_html(output_dir / "similarity_table.html", best_rows, all_rows)
    return best_rows, all_rows


def format_ascii_table(headers: list[str], rows: list[list[object]]) -> str:
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render(row):
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"

    lines = [separator, render(headers), separator]
    lines.extend(render(row) for row in text_rows)
    lines.append(separator)
    return "\n".join(lines)


def best_matches_table(rows: list[dict]) -> str:
    table_rows = [
        [
            row["query_id"],
            row["best_standard"],
            f"{row['mcs_nodes']}/{row['query_nodes']}",
            f"{row['mcs_edges']}/{row['query_edges']}",
            row["ged_proxy"],
            f"{row['query_53_optional']}/{row['standard_53_optional']}",
        ]
        for row in rows
    ]
    return format_ascii_table(
        [
            "Nodo ricostruito",
            "Schema standard piu simile",
            "Componenti comuni / rilevati",
            "Collegamenti comuni / rilevati",
            "Differenze residue",
            "Condotte 53 rilevate / standard",
        ],
        table_rows,
    )


def all_matches_table(rows: list[dict]) -> str:
    table_rows = [
        [
            row["query_id"],
            row["rank"],
            row["standard_id"],
            f"{row['mcs_nodes']}/{row['query_nodes']}",
            row["mcs_edges"],
            f"{row['query_coverage']:.1%}",
            f"{row['standard_coverage']:.1%}",
            row["ged_proxy"],
            f"{row['query_53_optional']}/{row['standard_53_optional']}",
        ]
        for row in rows
    ]
    return format_ascii_table(
        ["Pipeline", "Rank", "Standard", "MCS nodi", "MCS archi", "Cop. pipeline", "Cop. standard", "GED proxy", "53 opz. P/S"],
        table_rows,
    )


def make_html(path: Path, summary_rows: list[dict], comparison_rows: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_html = []
    for row in summary_rows:
        rows_html.append(
            "<tr>"
            f"<td>{row['query_id']}</td>"
            f"<td>{row['best_standard']}</td>"
            f"<td>{row['mcs_nodes']}/{row['query_nodes']}</td>"
            f"<td>{row['mcs_edges']}</td>"
            f"<td>{row['standard_coverage']:.2%}</td>"
            f"<td>{row['ged_proxy']}</td>"
            f"<td>{row['query_53_optional']}/{row['standard_53_optional']}</td>"
            "</tr>"
        )
    detail_rows = []
    for row in comparison_rows or []:
        detail_rows.append(
            "<tr>"
            f"<td>{row['query_id']}</td><td>{row['rank']}</td><td>{row['standard_id']}</td>"
            f"<td>{row['mcs_nodes']}/{row['query_nodes']}</td>"
            f"<td>{row['mcs_edges']}/{row['query_edges']}</td>"
            f"<td>{row['query_coverage']:.1%}</td><td>{row['standard_coverage']:.1%}</td>"
            f"<td>{row['ged_proxy']}</td></tr>"
        )
    html = f"""<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <title>Confronto pipeline vs standard aziendali</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 28px; color: #172033; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 7px 8px; text-align: left; }}
    th {{ background: #f3f6fb; }}
    h1 {{ font-size: 22px; margin-bottom: 6px; }}
    p {{ color: #526071; }}
  </style>
</head>
<body>
  <h1>Confronto pipeline vs standard aziendali</h1>
  <p>Best match ordinato per MCS massimo, poi archi MCS, poi GED proxy minima. Shurikode 53 escluso dalle metriche.</p>
  <table>
    <thead><tr><th>Nodo ricostruito</th><th>Schema standard piu simile</th><th>Componenti comuni / rilevati</th><th>Collegamenti comuni</th><th>Standard coperto</th><th>Differenze residue</th><th>Condotte 53 rilevate / standard</th></tr></thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  <h2>Tabella completa di similarità</h2>
  <table>
    <thead><tr><th>Nodo</th><th>Rank</th><th>Standard</th><th>MCS nodi</th><th>MCS archi</th><th>Copertura pipeline</th><th>Copertura standard</th><th>GED proxy</th></tr></thead>
    <tbody>{''.join(detail_rows)}</tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--standards", type=Path, default=DEFAULT_STANDARDS)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nodes", nargs="*", help="Node IDs to evaluate, e.g. 2 7 node9")
    args = parser.parse_args()
    best_rows, all_rows = evaluate_pipeline_against_standards(
        args.pipeline, args.standards, args.output, args.nodes
    )
    best_table = best_matches_table(best_rows)

    print(f"Confronti calcolati: {len(all_rows)}")
    print(f"Output pipeline caricati: {len(best_rows)}")
    print(f"Output scritto in: {args.output}")
    print("\nBest match:\n")
    print(best_table)


if __name__ == "__main__":
    main()
