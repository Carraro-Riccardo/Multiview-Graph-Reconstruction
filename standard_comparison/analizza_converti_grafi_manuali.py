from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "grafi_campione_manuali"
OUTPUT_DIR = ROOT / "grafi_campione_manuali_geojson"


CATALOG = {
    11: (8, "giunto gibault"),
    15: (1, "saracinesca aperta"),
    37: (21, "idrante soprassuolo"),
    53: (22, "tubo condotta"),
    75: (-1, "riduzione fl-fl"),
    95: (6, "Bout a bicchiere"),
    111: (19, "curva 45"),
    116: (2, "Tee fl-fl-fl"),
    175: (13, "curva 90 flangiata"),
    192: (3, "croce"),
    201: (4, "Tee bic-fl-bic"),
    207: (7, "Bout normale"),
    237: (15, "Manicotto o bigiunto"),
    255: (5, "Tee FL ridotto"),
}

EXPECTED_JUNCTION_DEGREE = {116: 3, 192: 4, 201: 3, 255: 3}

MISSING_LINK_FIXES = {
    "nodo4": [
        ("192#1", "15#3"), ("15#3", "95#4"), ("95#4", "53#1"),
        ("192#1", "15#2"), ("15#2", "95#2"), ("95#2", "53#2"),
        ("192#1", "15#1"), ("15#1", "95#1"), ("95#1", "53#4"),
        ("192#1", "15#4"), ("15#4", "95#3"), ("95#3", "53#3"),
    ],
    "nodo5": [
        ("116#1", "95#1"), ("95#1", "53#1"),
        ("116#1", "95#3"), ("95#3", "53#3"),
        ("116#1", "15#1"), ("15#1", "95#2"), ("95#2", "53#2"),
    ],
}

NODE_LABEL_FIXES = {
    "nodo4": {
        "53#5": ("95#2", "95"),
        "53#6": ("95#3", "95"),
        "53#7": ("95#4", "95"),
    }
}


def natural_number(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    return int(match.group()) if match else 0


def connected_components(node_ids: set[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    adjacency = {node_id: set() for node_id in node_ids}
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    unseen = set(node_ids)
    components = []
    while unseen:
        start = unseen.pop()
        queue = deque([start])
        component = [start]
        while queue:
            current = queue.popleft()
            for nxt in adjacency[current]:
                if nxt in unseen:
                    unseen.remove(nxt)
                    queue.append(nxt)
                    component.append(nxt)
        components.append(sorted(component))
    return components


def analyze(path: Path) -> tuple[dict, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = data.get("nodes")
    links = data.get("links")
    errors = []
    warnings = []

    if not isinstance(nodes, list) or not isinstance(links, list):
        return data, {"file": path.name, "errors": ["Il JSON deve contenere array nodes e links."], "warnings": []}

    if not links and path.stem in MISSING_LINK_FIXES:
        links = [{"from": source, "to": target} for source, target in MISSING_LINK_FIXES[path.stem]]
        data["links"] = links
        warnings.append(
            f"Il sorgente aveva links vuoto: ricostruiti {len(links)} collegamenti dalle coordinate e dalla geometria dello schema."
        )

    fixes = NODE_LABEL_FIXES.get(path.stem, {})
    for old_id, (new_id, new_label) in fixes.items():
        node = next((item for item in nodes if str(item.get("id")) == old_id), None)
        if node is None:
            continue
        node["id"] = new_id
        node["label"] = new_label
        for link in links:
            if str(link.get("from")) == old_id:
                link["from"] = new_id
            if str(link.get("to")) == old_id:
                link["to"] = new_id
        warnings.append(f"{old_id} corretto in {new_id}: il componente intermedio e un Bout 95, non un tubo 53.")

    ids = [str(node.get("id")) for node in nodes]
    id_counts = Counter(ids)
    duplicate_ids = sorted(node_id for node_id, count in id_counts.items() if count > 1)
    if duplicate_ids:
        errors.append(f"ID nodo duplicati: {duplicate_ids}")
    node_by_id = {str(node.get("id")): node for node in nodes}

    for node in nodes:
        node_id = str(node.get("id"))
        label = str(node.get("label"))
        prefix = node_id.split("#", 1)[0]
        if prefix != label:
            errors.append(f"{node_id}: label {label} diversa dal prefisso ID {prefix}")
        try:
            shurikode = int(label)
        except ValueError:
            errors.append(f"{node_id}: label non numerica")
            continue
        if shurikode not in CATALOG:
            errors.append(f"{node_id}: shurikode {shurikode} non presente nella mappatura")
        for field in ("x", "y"):
            value = node.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                errors.append(f"{node_id}: coordinata {field} non valida")

    raw_edges = []
    bad_references = []
    self_loops = []
    for index, link in enumerate(links, start=1):
        source = str(link.get("from"))
        target = str(link.get("to"))
        if source not in node_by_id or target not in node_by_id:
            bad_references.append(f"link#{index} {source}->{target}")
            continue
        if source == target:
            self_loops.append(f"link#{index} {source}")
        raw_edges.append(tuple(sorted((source, target))))
    if bad_references:
        errors.append(f"Link con riferimenti inesistenti: {bad_references}")
    if self_loops:
        errors.append(f"Self-loop: {self_loops}")

    edge_counts = Counter(raw_edges)
    duplicate_edges = [edge for edge, count in edge_counts.items() if count > 1]
    if duplicate_edges:
        errors.append(f"Link duplicati: {duplicate_edges}")
    edges = set(raw_edges)
    components = connected_components(set(ids), edges)
    if len(components) != 1:
        errors.append(f"Grafo disconnesso: {components}")
    is_tree = len(components) == 1 and len(edges) == len(nodes) - 1
    if not is_tree:
        warnings.append(f"Il grafo non e un albero: {len(nodes)} nodi, {len(edges)} archi")

    degree = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1
    for node_id, node in node_by_id.items():
        try:
            shurikode = int(node["label"])
        except (ValueError, TypeError):
            continue
        node_degree = degree[node_id]
        expected = EXPECTED_JUNCTION_DEGREE.get(shurikode)
        if expected is not None and node_degree != expected:
            warnings.append(f"{node_id}: grado {node_degree}, atteso {expected} per shurikode {shurikode}")
        if expected is None and node_degree > 2:
            warnings.append(f"{node_id}: componente lineare shurikode {shurikode} con grado anomalo {node_degree}")
        if node_degree == 0:
            errors.append(f"{node_id}: nodo isolato")

    inventory = Counter(int(node["label"]) for node in nodes if str(node.get("label", "")).isdigit())
    report = {
        "file": path.name,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "is_connected": len(components) == 1,
        "is_tree": is_tree,
        "inventory": dict(sorted(inventory.items())),
        "degree_sequence": sorted(degree.values(), reverse=True),
        "errors": errors,
        "warnings": warnings,
    }
    return data, report


def to_geojson(source_path: Path, data: dict, report: dict) -> dict:
    features = []
    coordinates = {}
    for node in data["nodes"]:
        node_id = str(node["id"])
        shurikode = int(node["label"])
        component_id, name = CATALOG[shurikode]
        # Le annotazioni usano coordinate schermo con Y verso il basso.
        coords = [float(node["x"]), -float(node["y"])]
        coordinates[node_id] = coords
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": node_id,
                    "tipo_layer": "componente",
                    "nome": name,
                    "component_id": component_id,
                    "shurikode": shurikode,
                    "etichetta": f"{shurikode} - {name}",
                    "annotation_x": node["x"],
                    "annotation_y": node["y"],
                },
                "geometry": {"type": "Point", "coordinates": coords},
            }
        )
    for index, link in enumerate(data["links"], start=1):
        source = str(link["from"])
        target = str(link["to"])
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": f"edge#{index}",
                    "tipo_layer": "collegamento",
                    "da": source,
                    "a": target,
                    "etichetta": f"{source} -> {target}",
                },
                "geometry": {"type": "LineString", "coordinates": [coordinates[source], coordinates[target]]},
            }
        )
    number = natural_number(source_path)
    return {
        "type": "FeatureCollection",
        "name": f"standard_manuale_n{number:02d}",
        "metadata": {
            "standard_id": f"N{number:02d}",
            "authority": "annotazione manuale validata",
            "source_annotation": str(source_path),
            "coordinate_mode": "schematic",
            "coordinate_transform": "GeoJSON y = -annotation_y",
            "source_summary": {
                "n_nodes": report["node_count"],
                "n_edges": report["edge_count"],
                "is_connected": report["is_connected"],
                "is_tree": report["is_tree"],
            },
        },
        "features": features,
    }


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    reports = []
    converted = 0
    for path in sorted(SOURCE_DIR.glob("*.json"), key=natural_number):
        data, report = analyze(path)
        reports.append(report)
        if report["errors"]:
            print(f"{path.name}: NON convertito, {len(report['errors'])} errori")
            continue
        output = to_geojson(path, data, report)
        output_path = OUTPUT_DIR / f"n{natural_number(path):02d}.geojson"
        output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        converted += 1
        print(f"{path.name}: OK -> {output_path.name}")

    summary = {
        "source_directory": str(SOURCE_DIR),
        "converted_directory": str(OUTPUT_DIR),
        "files_found": len(reports),
        "files_converted": converted,
        "errors_total": sum(len(report["errors"]) for report in reports),
        "warnings_total": sum(len(report["warnings"]) for report in reports),
        "graphs": reports,
    }
    (OUTPUT_DIR / "_report_validazione.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {OUTPUT_DIR / '_report_validazione.json'}")
    print(f"Convertiti {converted}/{len(reports)}, errori {summary['errors_total']}, warning {summary['warnings_total']}")


if __name__ == "__main__":
    main()
