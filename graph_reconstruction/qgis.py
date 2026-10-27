"""Convert fused graph JSON files to QGIS-friendly GeoJSON layers.

The fused graph produced by the pipeline is topological, not georeferenced: it
stores components and connections but no real-world coordinates. This adapter
creates temporary coordinates around an origin so the graph can be inspected in
QGIS with a compact, readable attribute schema. It exports one combined file
whose geometry classes can be selected separately by QGIS.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from graph_reconstruction.visualization.shurikode_symbols import (
    DEFAULT_EDGE_LENGTH_FACTOR,
    DEFAULT_ICON_SIZE,
    DEFAULT_PORT_CATALOG_PATH,
    cad_port_layout,
    load_port_catalog,
)


DEFAULT_ORIGIN_LON = 0.0
DEFAULT_ORIGIN_LAT = 0.0
DEFAULT_RADIUS_DEGREES = 1.0
DEFAULT_CRS_EPSG = 7795
from graph_reconstruction.config import WORK_DIR

DEFAULT_BATCH_ROOT = WORK_DIR / "multiview"
DEFAULT_LAYOUT = "cad"
DEFAULT_EDGE_ROUTING = "orthogonal"
DEFAULT_MASTER_ORIENTATION_DEG = None
DEFAULT_MASTER_POLYGON_SIZE_FACTOR = 0.03


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    if "nodes" not in payload or "edges" not in payload:
        raise ValueError(f"{path} must contain 'nodes' and 'edges' arrays.")
    return payload


def _as_qgis_property(value: Any) -> Any:
    """Keep scalar values native and serialize nested values as compact JSON."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _feature_properties(item: dict[str, Any], *, exclude: set[str] | None = None) -> dict[str, Any]:
    exclude = exclude or set()
    return {
        key: _as_qgis_property(value)
        for key, value in item.items()
        if key not in exclude
    }


def _sorted_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(nodes, key=lambda node: (int(node.get("code", 0)), str(node.get("id", ""))))


def _default_node_name(input_path: Path) -> str:
    if input_path.stem == "fused_graph" and input_path.parent.name:
        return input_path.parent.name
    return input_path.stem


def _fake_positions(
    nodes: list[dict[str, Any]],
    *,
    origin_x: float,
    origin_y: float,
    radius: float,
    layout: str,
) -> dict[str, tuple[float, float]]:
    if not nodes:
        return {}

    if layout == "zero":
        return {
            str(node["id"]): (origin_x, origin_y)
            for node in nodes
        }

    ordered_nodes = _sorted_nodes(nodes)
    n_nodes = len(ordered_nodes)
    positions: dict[str, tuple[float, float]] = {}

    for index, node in enumerate(ordered_nodes):
        angle = (2.0 * math.pi * index) / max(n_nodes, 1)
        x = origin_x + radius * math.cos(angle)
        y = origin_y + radius * math.sin(angle)
        positions[str(node["id"])] = (x, y)

    return positions


def _graph_from_payload(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> nx.Graph:
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(str(node["id"]), **node)
    for edge in edges:
        graph.add_edge(str(edge["node_a"]), str(edge["node_b"]), **edge)
    return graph


def _median_edge_length(
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
) -> float:
    lengths = [
        math.hypot(positions[u][0] - positions[v][0], positions[u][1] - positions[v][1])
        for u, v in graph.edges()
        if u in positions and v in positions
    ]
    if not lengths:
        return 1.0
    lengths = sorted(lengths)
    mid = len(lengths) // 2
    if len(lengths) % 2:
        return lengths[mid]
    return (lengths[mid - 1] + lengths[mid]) / 2.0


def _cad_positions(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    origin_x: float,
    origin_y: float,
    edge_length: float,
    port_catalog_path: Path = DEFAULT_PORT_CATALOG_PATH,
) -> dict[str, tuple[float, float]]:
    graph = _graph_from_payload(nodes, edges)
    if graph.number_of_nodes() == 0:
        return {}

    port_catalog = load_port_catalog(port_catalog_path)
    screen_positions, _, _, _ = cad_port_layout(
        graph,
        icon_size=DEFAULT_ICON_SIZE,
        edge_length_factor=DEFAULT_EDGE_LENGTH_FACTOR,
        min_width=820,
        min_height=560,
        port_catalog=port_catalog,
    )
    if not screen_positions:
        return {}

    xs = [xy[0] for xy in screen_positions.values()]
    ys = [xy[1] for xy in screen_positions.values()]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    median_len = _median_edge_length(graph, screen_positions)
    scale = edge_length / max(median_len, 1e-9)

    return {
        node_id: (
            origin_x + (x - center_x) * scale,
            origin_y - (y - center_y) * scale,
        )
        for node_id, (x, y) in screen_positions.items()
    }


def _load_vggt_scene_geometry(input_path: Path, debug_path: Path | None = None) -> dict[str, dict[str, Any]]:
    if debug_path is None:
        debug_path = input_path.parent / "vggt_geometry.json"
        legacy_path = input_path.parent / "vggt_edge_debug.json"
        if not debug_path.exists() and legacy_path.exists():
            debug_path = legacy_path
    if not debug_path.exists():
        return {}
    payload = _load_json_object(debug_path)
    scene_geometry = payload.get("scene_geometry", {})
    return scene_geometry if isinstance(scene_geometry, dict) else {}


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _project_centroids_to_plane(centroids: np.ndarray) -> np.ndarray:
    if len(centroids) == 1:
        return np.zeros((1, 2), dtype=float)
    mean = centroids.mean(axis=0)
    centered = centroids - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _vggt_positions(
    nodes: list[dict[str, Any]],
    scene_geometry: dict[str, dict[str, Any]],
) -> tuple[dict[str, tuple[float, float]], dict[str, dict[str, Any]]]:
    node_ids = [str(node["id"]) for node in _sorted_nodes(nodes)]
    usable_ids = [
        node_id
        for node_id in node_ids
        if node_id in scene_geometry and "centroid" in scene_geometry[node_id]
    ]
    if not usable_ids:
        return {}, {}

    centroids = np.array(
        [scene_geometry[node_id]["centroid"][:3] for node_id in usable_ids],
        dtype=float,
    )
    coords2d = _project_centroids_to_plane(centroids)

    positions = {
        node_id: (float(coords2d[index, 0]), float(coords2d[index, 1]))
        for index, node_id in enumerate(usable_ids)
    }
    properties = {}
    for node_id in usable_ids:
        geom = scene_geometry[node_id]
        centroid = geom.get("centroid", [None, None, None])
        axis = geom.get("axis") or [None, None, None]
        properties[node_id] = {
            "vggt_centroid_x": centroid[0],
            "vggt_centroid_y": centroid[1],
            "vggt_centroid_z": centroid[2],
            "vggt_axis_x": axis[0],
            "vggt_axis_y": axis[1],
            "vggt_axis_z": axis[2],
            "vggt_n_observations": geom.get("n_observations"),
        }
    return positions, properties


def _node_features(
    nodes: list[dict[str, Any]],
    positions: dict[str, tuple[float, float]],
    geometry_properties: dict[str, dict[str, Any]],
    *,
    include_technical_fields: bool,
) -> list[dict[str, Any]]:
    features = []
    for node in _sorted_nodes(nodes):
        node_id = str(node["id"])
        x, y = positions[node_id]
        props = {
            "id": node_id,
            "tipo_layer": "componente",
            "nome": node.get("name"),
            "shurikode": node.get("code"),
            "etichetta": f"{node.get('code')} - {node.get('name')}",
            "diametro": node.get("diameter_mm"),
            "angolo_rotazione": None,
        }
        if include_technical_fields:
            props.update({
                f"src_{key}": value
                for key, value in _feature_properties(node).items()
                if key not in props
            })
            props["qgis_x"] = x
            props["qgis_y"] = y
            props.update(geometry_properties.get(node_id, {}))
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [x, y],
            },
            "properties": props,
        })
    return features


def _edge_features(
    edges: list[dict[str, Any]],
    positions: dict[str, tuple[float, float]],
    *,
    include_technical_fields: bool,
    edge_routing: str,
) -> list[dict[str, Any]]:
    features = []
    for edge_index, edge in enumerate(edges, start=1):
        node_a = str(edge["node_a"])
        node_b = str(edge["node_b"])
        if node_a not in positions or node_b not in positions:
            raise ValueError(f"Edge {node_a!r}-{node_b!r} references a missing node.")

        props = {
            "id": f"edge#{edge_index}",
            "tipo_layer": "collegamento",
            "da": node_a,
            "a": node_b,
            "etichetta": f"{node_a} -> {node_b}",
            "diametro": edge.get("diameter_mm"),
            "angolo_rotazione": None,
        }
        if include_technical_fields:
            props.update({
                f"src_{key}": value
                for key, value in _feature_properties(edge).items()
                if key not in {"node_a", "node_b"}
            })
        ax, ay = positions[node_a]
        bx, by = positions[node_b]
        coordinates = [[ax, ay], [bx, by]]
        if edge_routing == "orthogonal" and abs(ax - bx) > 1e-9 and abs(ay - by) > 1e-9:
            coordinates = [[ax, ay], [bx, ay], [bx, by]]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": props,
        })
    return features


def _master_geometry(origin_x: float, origin_y: float, *, size: float) -> dict[str, Any]:
    half_size = max(size, 1e-9) / 2.0
    ring = [
        [origin_x - half_size, origin_y - half_size],
        [origin_x + half_size, origin_y - half_size],
        [origin_x + half_size, origin_y + half_size],
        [origin_x - half_size, origin_y + half_size],
        [origin_x - half_size, origin_y - half_size],
    ]
    return {
        "type": "Polygon",
        "coordinates": [ring],
    }


def _master_feature(
    origin_x: float,
    origin_y: float,
    *,
    node_name: str,
    master_id: str,
    master_date: str | None,
    master_orientation_deg: float | None,
    polygon_size: float,
) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": _master_geometry(origin_x, origin_y, size=polygon_size),
        "properties": {
            "id": master_id,
            "tipo_layer": "master",
            "nome": "Nodo master",
            "node_id": master_id,
            "etichetta": master_id,
            "data": master_date,
            "orientamento": master_orientation_deg,
            "coordinata_x": origin_x,
            "coordinata_y": origin_y,
            "dimensione_poligono": polygon_size,
        },
    }


def _feature_collection(
    features: list[dict[str, Any]],
    *,
    name: str,
    metadata: dict[str, Any],
    crs_epsg: int | None,
) -> dict[str, Any]:
    payload = {
        "type": "FeatureCollection",
        "name": name,
        "metadata": metadata,
        "features": features,
    }
    if crs_epsg is not None:
        payload["crs"] = {
            "type": "name",
            "properties": {
                "name": f"urn:ogc:def:crs:EPSG::{crs_epsg}",
            },
        }
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def convert_fused_graph_to_qgis(
    input_path: Path,
    output_dir: Path | None = None,
    *,
    origin_lon: float = DEFAULT_ORIGIN_LON,
    origin_lat: float = DEFAULT_ORIGIN_LAT,
    radius_degrees: float = DEFAULT_RADIUS_DEGREES,
    crs_epsg: int | None = DEFAULT_CRS_EPSG,
    layout: str = DEFAULT_LAYOUT,
    geometry_source: str = "fake",
    vggt_debug_path: Path | None = None,
    include_technical_fields: bool = False,
    edge_routing: str = DEFAULT_EDGE_ROUTING,
    master_id: str | None = None,
    master_date: str | None = None,
    master_orientation_deg: float | None = DEFAULT_MASTER_ORIENTATION_DEG,
    master_polygon_size_factor: float = DEFAULT_MASTER_POLYGON_SIZE_FACTOR,
) -> dict[str, Path]:
    graph = _load_json(input_path)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("'nodes' and 'edges' must be arrays.")

    output_dir = output_dir or input_path.parent
    stem = input_path.stem
    node_name = _default_node_name(input_path)
    geometry_properties: dict[str, dict[str, Any]] = {}
    scene_geometry: dict[str, dict[str, Any]] = {}
    if geometry_source in {"auto", "vggt"}:
        scene_geometry = _load_vggt_scene_geometry(input_path, vggt_debug_path)
        positions, geometry_properties = _vggt_positions(nodes, scene_geometry)
        if geometry_source == "vggt" and not positions:
            raise ValueError(
                "VGGT geometry requested, but no scene_geometry was found. "
                "Expected vggt_geometry.json next to fused_graph.json."
            )
    else:
        positions = {}

    effective_geometry_source = "vggt_pca" if positions else "fake"
    if not positions:
        if layout == "cad":
            positions = _cad_positions(
                nodes,
                edges,
                origin_x=origin_lon,
                origin_y=origin_lat,
                edge_length=radius_degrees,
            )
        else:
            positions = _fake_positions(
                nodes,
                origin_x=origin_lon,
                origin_y=origin_lat,
                radius=radius_degrees,
                layout=layout,
            )

    metadata = {
        "source": str(input_path),
        "crs_hint": f"EPSG:{crs_epsg}" if crs_epsg is not None else "not declared",
        "coordinate_mode": effective_geometry_source,
        "layout": layout,
        "origin_x": origin_lon,
        "origin_y": origin_lat,
        "radius": radius_degrees,
        "crs_epsg": crs_epsg,
        "note": (
            "Coordinates are for QGIS inspection. In fake mode the graph is laid "
            "out around a global reference point. Replace the synthetic offsets "
            "with a measured top-view layout when available. The cad layout "
            "uses the same topological placement as the SVG symbol renderer."
        ),
        "source_summary": {
            "total_images": graph.get("total_images"),
            "n_nodes": graph.get("n_nodes", len(nodes)),
            "n_edges": graph.get("n_edges", len(edges)),
            "is_connected": graph.get("is_connected"),
            "n_vggt_scene_geometries": len(scene_geometry),
            "geometry_source_requested": geometry_source,
            "geometry_source_used": effective_geometry_source,
            "vggt_debug_path": str(vggt_debug_path or input_path.parent / "vggt_geometry.json"),
        },
    }

    node_payload = _feature_collection(
        _node_features(
            nodes,
            positions,
            geometry_properties,
            include_technical_fields=include_technical_fields,
        ),
        name=f"{stem}_nodes",
        metadata=metadata,
        crs_epsg=crs_epsg,
    )
    edge_payload = _feature_collection(
        _edge_features(
            edges,
            positions,
            include_technical_fields=include_technical_fields,
            edge_routing=edge_routing,
        ),
        name=f"{stem}_edges",
        metadata=metadata,
        crs_epsg=crs_epsg,
    )
    master_payload = _feature_collection(
        [_master_feature(
            origin_lon,
            origin_lat,
            node_name=node_name,
            master_id=master_id or f"{node_name}_master",
            master_date=master_date,
            master_orientation_deg=master_orientation_deg,
            polygon_size=radius_degrees * master_polygon_size_factor,
        )],
        name=f"{stem}_master",
        metadata=metadata,
        crs_epsg=crs_epsg,
    )
    paths = {
        "combined": output_dir / f"{stem}_qgis_combined.geojson",
    }
    combined_payload = _feature_collection(
        node_payload["features"] + edge_payload["features"] + master_payload["features"],
        name=f"{stem}_combined",
        metadata=metadata,
        crs_epsg=crs_epsg,
    )
    _write_json(paths["combined"], combined_payload)
    return paths


def find_fused_graphs(root: Path = DEFAULT_BATCH_ROOT) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("*/fused_graph.json"))


def convert_all_fused_graphs_to_qgis(
    root: Path = DEFAULT_BATCH_ROOT,
    *,
    origin_lon: float = DEFAULT_ORIGIN_LON,
    origin_lat: float = DEFAULT_ORIGIN_LAT,
    radius_degrees: float = DEFAULT_RADIUS_DEGREES,
    crs_epsg: int | None = DEFAULT_CRS_EPSG,
    layout: str = DEFAULT_LAYOUT,
    geometry_source: str = "fake",
    include_technical_fields: bool = False,
    edge_routing: str = DEFAULT_EDGE_ROUTING,
    master_date: str | None = None,
    master_orientation_deg: float | None = DEFAULT_MASTER_ORIENTATION_DEG,
) -> dict[Path, dict[str, Path]]:
    results: dict[Path, dict[str, Path]] = {}
    for input_path in find_fused_graphs(root):
        results[input_path] = convert_fused_graph_to_qgis(
            input_path,
            input_path.parent,
            origin_lon=origin_lon,
            origin_lat=origin_lat,
            radius_degrees=radius_degrees,
            crs_epsg=crs_epsg,
            layout=layout,
            geometry_source=geometry_source,
            vggt_debug_path=None,
            include_technical_fields=include_technical_fields,
            edge_routing=edge_routing,
            master_date=master_date,
            master_orientation_deg=master_orientation_deg,
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pipeline fused_graph.json files to QGIS-importable GeoJSON."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="Path to a fused_graph.json file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated GeoJSON files. Defaults to the input file directory.",
    )
    origin_x = parser.add_mutually_exclusive_group()
    origin_x.add_argument(
        "--origin-x",
        type=float,
        default=DEFAULT_ORIGIN_LON,
        help="Global reference X coordinate for the layout center. Default: 0.0.",
    )
    origin_x.add_argument(
        "--origin-lon",
        type=float,
        dest="origin_x",
        help="Deprecated alias for --origin-x.",
    )
    origin_y = parser.add_mutually_exclusive_group()
    origin_y.add_argument(
        "--origin-y",
        type=float,
        default=DEFAULT_ORIGIN_LAT,
        help="Global reference Y coordinate for the layout center. Default: 0.0.",
    )
    origin_y.add_argument(
        "--origin-lat",
        type=float,
        dest="origin_y",
        help="Deprecated alias for --origin-y.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        dest="radius_degrees",
        default=DEFAULT_RADIUS_DEGREES,
        help="Synthetic layout radius around the reference point. Default: 1.0.",
    )
    parser.add_argument(
        "--radius-degrees",
        type=float,
        dest="radius_degrees",
        help="Deprecated alias for --radius.",
    )
    parser.add_argument(
        "--crs-epsg",
        type=int,
        default=DEFAULT_CRS_EPSG,
        help="EPSG code declared in exported GeoJSON files. Default: 7795.",
    )
    parser.add_argument(
        "--layout",
        choices=["cad", "circle", "zero"],
        default=DEFAULT_LAYOUT,
        help=(
            "Use 'cad' for the top-view SVG-like layout, 'circle' for a simple "
            "test layout, or 'zero' to stack all nodes at the origin. Default: cad."
        ),
    )
    parser.add_argument(
        "--geometry-source",
        choices=["auto", "vggt", "fake"],
        default="fake",
        help=(
            "Use synthetic coordinates ('fake'), use VGGT scene geometry when available "
            "('auto'), or require VGGT geometry ('vggt'). Default: fake."
        ),
    )
    parser.add_argument(
        "--vggt-debug",
        type=Path,
        default=None,
        help="Optional path to vggt_geometry.json. Only used in single-file mode.",
    )
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=DEFAULT_BATCH_ROOT,
        help="Root scanned in batch mode. Default: output/multiview.",
    )
    parser.add_argument(
        "--include-technical-fields",
        action="store_true",
        help="Also export internal pipeline attributes with src_ prefixes for debugging.",
    )
    parser.add_argument(
        "--edge-routing",
        choices=["orthogonal", "straight"],
        default=DEFAULT_EDGE_ROUTING,
        help="Route edges as right-angle polylines or direct straight segments. Default: orthogonal.",
    )
    parser.add_argument(
        "--master-id",
        default=None,
        help=(
            "Optional ID for the master node in single-file mode. Defaults to "
            "'<node_folder>_master' for fused_graph.json inputs."
        ),
    )
    parser.add_argument(
        "--master-date",
        default=None,
        help="Optional date attribute for the master node, for example 2026-06-10.",
    )
    parser.add_argument(
        "--master-orientation",
        type=float,
        default=DEFAULT_MASTER_ORIENTATION_DEG,
        dest="master_orientation_deg",
        help="Optional master-node orientation angle in degrees.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.input is None:
        results = convert_all_fused_graphs_to_qgis(
            args.batch_root,
            origin_lon=args.origin_x,
            origin_lat=args.origin_y,
            radius_degrees=args.radius_degrees,
            crs_epsg=args.crs_epsg,
            layout=args.layout,
            geometry_source=args.geometry_source,
            include_technical_fields=args.include_technical_fields,
            edge_routing=args.edge_routing,
            master_date=args.master_date,
            master_orientation_deg=args.master_orientation_deg,
        )
        if not results:
            raise SystemExit(f"No fused_graph.json files found under {args.batch_root}")

        print(f"Generated QGIS GeoJSON files for {len(results)} fused graph(s):")
        for input_path, paths in results.items():
            print(f"  {input_path.parent}:")
            for layer_name, path in paths.items():
                print(f"    {layer_name}: {path}")
        return

    paths = convert_fused_graph_to_qgis(
        args.input,
        args.output_dir,
        origin_lon=args.origin_x,
        origin_lat=args.origin_y,
        radius_degrees=args.radius_degrees,
        crs_epsg=args.crs_epsg,
        layout=args.layout,
        geometry_source=args.geometry_source,
        vggt_debug_path=args.vggt_debug,
        include_technical_fields=args.include_technical_fields,
        edge_routing=args.edge_routing,
        master_id=args.master_id,
        master_date=args.master_date,
        master_orientation_deg=args.master_orientation_deg,
    )
    print("Generated QGIS GeoJSON files:")
    for layer_name, path in paths.items():
        print(f"  {layer_name}: {path}")


if __name__ == "__main__":
    main()
