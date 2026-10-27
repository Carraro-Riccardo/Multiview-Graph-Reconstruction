"""Render fused graphs with Shurikode SVG component symbols.

This script is intentionally separate from ``visualize.py``. It keeps the
existing color-based plot intact and adds a symbol-based SVG view:

    python visualize_shurikode_symbols.py 10
    python visualize_shurikode_symbols.py 2 7 9
    python visualize_shurikode_symbols.py --all

For every graph node, the component ``code`` is matched against
``graph_reconstruction/assets/shurikode_svgs/shurikode_mapping.csv`` and
``<code>.svg``.
If the SVG is missing, the node is rendered as a red dot.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import networkx as nx


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_reconstruction.config import ASSETS_DIR, WORK_DIR

DEFAULT_WORK_DIR = WORK_DIR / "multiview"
DEFAULT_SYMBOL_DIR = ASSETS_DIR / "shurikode_svgs"
DEFAULT_PORT_CATALOG_PATH = ASSETS_DIR / "shurikode_component_ports.json"
DEFAULT_OUTPUT_NAME = "fused_graph_symbols.svg"
DEFAULT_ICON_SIZE = 58
DEFAULT_EDGE_LENGTH_FACTOR = 0.96
_PATH_TOKEN_RE = re.compile(r"[AaCcHhLlMmQqSsTtVvZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")

# Screen-space direction vectors. Y grows downward in SVG coordinates.
DIR_W = (-1.0, 0.0)
DIR_E = (1.0, 0.0)
DIR_N = (0.0, -1.0)
DIR_S = (0.0, 1.0)
DIR_NW = (-math.sqrt(0.5), -math.sqrt(0.5))
DIR_NE = (math.sqrt(0.5), -math.sqrt(0.5))
DIR_SW = (-math.sqrt(0.5), math.sqrt(0.5))
DIR_SE = (math.sqrt(0.5), math.sqrt(0.5))
CANONICAL_DIRS = (DIR_E, DIR_W, DIR_N, DIR_S, DIR_NE, DIR_NW, DIR_SE, DIR_SW)

# Fallback local ports in the orientation used by the SVG asset. The primary
# source is graph_reconstruction/assets/shurikode_component_ports.json; this dict keeps old renders
# working if the catalog is missing or incomplete.
PORT_TEMPLATES: dict[int, tuple[tuple[float, float], ...]] = {
    37: (DIR_W,),
    159: (DIR_W,),
    250: (DIR_W,),
    252: (DIR_W,),
    116: (DIR_W, DIR_E, DIR_N),
    201: (DIR_W, DIR_E, DIR_N),
    255: (DIR_W, DIR_E, DIR_N),
    192: (DIR_W, DIR_E, DIR_N, DIR_S),
    111: (DIR_E, DIR_NW),
    175: (DIR_W, DIR_S),
    190: (DIR_E, DIR_NW),
    195: (DIR_E, DIR_NW),
    235: (DIR_W, DIR_S),
    245: (DIR_W, DIR_NE),
    54: (DIR_W, DIR_NE),
}


@dataclass(frozen=True)
class SymbolInfo:
    code: int
    name: str
    block: str
    note: str
    svg_path: Path | None

    @property
    def has_svg(self) -> bool:
        return self.svg_path is not None and self.svg_path.exists()


def _parse_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_symbol_index(symbol_dir: Path = DEFAULT_SYMBOL_DIR) -> dict[int, SymbolInfo]:
    """Load code -> SVG metadata from the mapping CSV."""
    mapping_path = symbol_dir / "shurikode_mapping.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_path}")

    symbols: dict[int, SymbolInfo] = {}
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = _parse_int(row.get("shurikode"), default=-1)
            if code < 0:
                continue
            svg_path = symbol_dir / f"{code}.svg"
            symbols[code] = SymbolInfo(
                code=code,
                name=(row.get("nome_componente") or f"code_{code}").strip(),
                block=(row.get("blocco_dxf") or "").strip(),
                note=(row.get("note") or "").strip(),
                svg_path=svg_path if svg_path.exists() else None,
            )
    return symbols


def load_port_catalog(catalog_path: Path = DEFAULT_PORT_CATALOG_PATH) -> dict:
    """Load component port/orientation annotations."""
    if not catalog_path.exists():
        return {"components": {}}
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def _node_folder_name(node_id: str) -> str:
    node_id = str(node_id).strip()
    return node_id if node_id.startswith("node") else f"node{node_id}"


def graph_json_path(node_id: str, work_dir: Path = DEFAULT_WORK_DIR) -> Path:
    return work_dir / _node_folder_name(node_id) / "fused_graph.json"


def load_fused_graph(node_id: str, work_dir: Path = DEFAULT_WORK_DIR) -> tuple[nx.Graph, dict]:
    """Load a ``fused_graph.json`` file into a NetworkX graph."""
    path = graph_json_path(node_id, work_dir)
    if not path.exists():
        raise FileNotFoundError(f"fused_graph.json not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    graph = nx.Graph()
    for node in data.get("nodes", []):
        node_id_value = str(node["id"])
        graph.add_node(node_id_value, **node)
    for edge in data.get("edges", []):
        graph.add_edge(str(edge["node_a"]), str(edge["node_b"]), **edge)
    return graph, data


def _layout_distance(edge_attrs: dict) -> float:
    support = _parse_float(edge_attrs.get("support_count"), 0.0)
    forced = bool(edge_attrs.get("forced", False))
    reason = edge_attrs.get("forced_reason") or ""
    if forced and "structural_fallback" in reason:
        return 8.0
    if forced:
        return 4.0
    if support <= 0:
        return 3.0
    return 1.0 / (support + 0.5)


def weighted_layout(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    """Layout similar to ``visualize.py``, with low-confidence edges longer."""
    if graph.number_of_nodes() == 0:
        return {}

    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(graph.nodes())
    for u, v, attrs in graph.edges(data=True):
        layout_graph.add_edge(u, v, weight=_layout_distance(attrs))

    isolated = [n for n in graph.nodes() if graph.degree(n) == 0]
    connected = [n for n in graph.nodes() if graph.degree(n) > 0]

    pos: dict[str, tuple[float, float]] = {}
    if connected:
        sub = layout_graph.subgraph(connected)
        scale = max(3.0, 0.8 * len(connected))
        try:
            sub_pos = nx.kamada_kawai_layout(sub, weight="weight", scale=scale)
        except Exception:
            sub_pos = nx.spring_layout(
                sub,
                weight="weight",
                seed=42,
                iterations=200,
                scale=scale,
            )
        sub_pos = nx.spring_layout(
            sub,
            pos=sub_pos,
            seed=0,
            iterations=30,
            scale=scale,
        )
        pos.update({str(node): (float(xy[0]), float(xy[1])) for node, xy in sub_pos.items()})

    if isolated:
        y_bottom = min((xy[1] for xy in pos.values()), default=0.0) - 1.6
        x_start = -0.75 * (len(isolated) - 1)
        for i, node in enumerate(isolated):
            pos[str(node)] = (x_start + i * 1.5, y_bottom)

    return pos


def _normalize_positions(
    pos: dict[str, tuple[float, float]],
    width: int,
    height: int,
    margin: int,
) -> dict[str, tuple[float, float]]:
    if not pos:
        return {}

    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    out: dict[str, tuple[float, float]] = {}
    for node, (x, y) in pos.items():
        px = margin + ((x - min_x) / span_x) * (width - 2 * margin)
        py = margin + (1.0 - ((y - min_y) / span_y)) * (height - 2 * margin)
        out[node] = (px, py)
    return out


def _median(values: list[float]) -> float:
    if not values:
        return 1.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _fit_collapsed_positions(
    graph: nx.Graph,
    raw_pos: dict[str, tuple[float, float]],
    icon_size: int,
    edge_length_factor: float,
    min_width: int,
    min_height: int,
) -> tuple[dict[str, tuple[float, float]], int, int]:
    """Scale the layout so adjacent component symbols nearly touch."""
    if not raw_pos:
        return {}, min_width, min_height

    edge_lengths = [
        math.hypot(raw_pos[u][0] - raw_pos[v][0], raw_pos[u][1] - raw_pos[v][1])
        for u, v in graph.edges()
        if u in raw_pos and v in raw_pos
    ]
    target_edge_length = icon_size * edge_length_factor
    if edge_lengths:
        scale = target_edge_length / max(_median(edge_lengths), 1e-9)
    else:
        scale = icon_size * 1.1

    scaled = {
        node: (xy[0] * scale, -xy[1] * scale)
        for node, xy in raw_pos.items()
    }
    xs = [xy[0] for xy in scaled.values()]
    ys = [xy[1] for xy in scaled.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    side_margin = 72
    top_margin = 118
    bottom_margin = 170
    half_icon = icon_size / 2
    graph_width = max_x - min_x + icon_size
    graph_height = max_y - min_y + icon_size
    width = max(min_width, int(math.ceil(graph_width + side_margin * 2)))
    height = max(min_height, int(math.ceil(graph_height + top_margin + bottom_margin)))

    tx = (width - graph_width) / 2 + half_icon - min_x
    ty = top_margin + half_icon - min_y
    return {node: (x + tx, y + ty) for node, (x, y) in scaled.items()}, width, height


def _vector_key(vec: tuple[float, float]) -> tuple[int, int]:
    return (int(round(vec[0] * 1000)), int(round(vec[1] * 1000)))


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _unit(vec: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(vec[0], vec[1])
    if length <= 1e-9:
        return DIR_E
    return vec[0] / length, vec[1] / length


def _opposite(vec: tuple[float, float]) -> tuple[float, float]:
    return -vec[0], -vec[1]


def _angle_deg(vec: tuple[float, float]) -> float:
    return math.degrees(math.atan2(vec[1], vec[0]))


def _rotate_vec(vec: tuple[float, float], degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    return vec[0] * cos_a - vec[1] * sin_a, vec[0] * sin_a + vec[1] * cos_a


def _normalize_degrees(degrees: float) -> float:
    normalized = ((degrees + 180.0) % 360.0) - 180.0
    return 0.0 if abs(normalized) < 1e-6 else normalized


def _snap_dir(vec: tuple[float, float]) -> tuple[float, float]:
    vec = _unit(vec)
    return max(CANONICAL_DIRS, key=lambda candidate: _dot(vec, candidate))


def _angle_delta(a: float, b: float) -> float:
    return abs(((a - b + 180.0) % 360.0) - 180.0)


def _component_annotation(port_catalog: dict | None, code: int) -> dict:
    if not port_catalog:
        return {}
    components = port_catalog.get("components") or {}
    return components.get(str(code)) or {}


def _catalog_allowed_rotations(port_catalog: dict | None, code: int) -> list[float] | None:
    annotation = _component_annotation(port_catalog, code)
    rotations = annotation.get("allowed_rotations")
    if rotations is None:
        return None
    out: list[float] = []
    for value in rotations:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out or None


def _catalog_ports(port_catalog: dict | None, code: int) -> tuple[tuple[float, float], ...]:
    return tuple(port for port, _ in _catalog_port_entries(port_catalog, code))


def _catalog_port_entries(
    port_catalog: dict | None,
    code: int,
) -> tuple[tuple[tuple[float, float], float], ...]:
    annotation = _component_annotation(port_catalog, code)
    ports: list[tuple[tuple[float, float], float]] = []
    for port in annotation.get("ports") or []:
        if port.get("connectable", True) is False:
            continue
        raw_dir = port.get("dir")
        if not isinstance(raw_dir, list) or len(raw_dir) != 2:
            continue
        role = str(port.get("connection_role") or "").strip().lower()
        weight = 0.0 if role == "restricted" else 0.15
        try:
            ports.append((_unit((float(raw_dir[0]), float(raw_dir[1]))), weight))
        except (TypeError, ValueError):
            continue
    return tuple(ports)


def _node_code(graph: nx.Graph, node: str) -> int:
    return _parse_int(graph.nodes[node].get("code"), 0)


def _node_port_template(
    graph: nx.Graph,
    node: str,
    port_catalog: dict | None = None,
) -> tuple[tuple[float, float], ...]:
    attrs = graph.nodes[node]
    code = _parse_int(attrs.get("code"), 0)
    catalog_ports = _catalog_ports(port_catalog, code)
    if catalog_ports:
        return catalog_ports
    if code in PORT_TEMPLATES:
        return PORT_TEMPLATES[code]

    degree_hint = _parse_int(attrs.get("max_degree"), graph.degree(node))
    if degree_hint <= 1 or graph.degree(node) <= 1:
        return (DIR_W,)
    if degree_hint == 3:
        return (DIR_W, DIR_E, DIR_N)
    if degree_hint >= 4:
        return (DIR_W, DIR_E, DIR_N, DIR_S)
    return (DIR_W, DIR_E)


def _global_ports(
    graph: nx.Graph,
    node: str,
    rotation: float,
    port_catalog: dict | None = None,
) -> list[tuple[float, float]]:
    ports: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for port in _node_port_template(graph, node, port_catalog):
        snapped = _snap_dir(_rotate_vec(port, rotation))
        key = _vector_key(snapped)
        if key not in seen:
            ports.append(snapped)
            seen.add(key)
    return ports


def _raw_hint_vector(
    raw_pos: dict[str, tuple[float, float]],
    u: str,
    v: str,
) -> tuple[float, float]:
    if u not in raw_pos or v not in raw_pos:
        return DIR_E
    ux, uy = raw_pos[u]
    vx, vy = raw_pos[v]
    return _unit((vx - ux, -(vy - uy)))


def _choose_anchor(graph: nx.Graph, port_catalog: dict | None = None) -> str:
    orienting_codes = {116, 201, 255, 192}
    return max(
        graph.nodes(),
        key=lambda node: (
            bool(
                (_component_annotation(port_catalog, _node_code(graph, node)).get("connection_policy") or {}).get(
                    "global_orientation_anchor",
                    False,
                )
            )
            or _parse_int(graph.nodes[node].get("code"), 0) in orienting_codes,
            graph.degree(node),
            _parse_float(graph.nodes[node].get("support_count"), 0.0),
            -len(str(node)),
            str(node),
        ),
    )


def _orientation_for_parent(
    graph: nx.Graph,
    node: str,
    target_parent_dir: tuple[float, float],
    port_catalog: dict | None = None,
) -> float:
    code = _node_code(graph, node)
    port_entries = _catalog_port_entries(port_catalog, code)
    if port_entries:
        ports = tuple(port for port, _ in port_entries)
    else:
        ports = _node_port_template(graph, node, port_catalog)
        port_entries = tuple((port, 0.0) for port in ports)
    allowed = _catalog_allowed_rotations(port_catalog, code)
    if allowed:
        return _normalize_degrees(
            max(
                allowed,
                key=lambda rotation: max(
                    _dot(_unit(_rotate_vec(port, rotation)), target_parent_dir) + weight
                    for port, weight in port_entries
                ),
            )
        )

    target_angle = _angle_deg(target_parent_dir)
    best_port = max(ports, key=lambda port: _dot(_snap_dir(port), target_parent_dir))
    return _normalize_degrees(target_angle - _angle_deg(best_port))


def _port_alignment_score(
    graph: nx.Graph,
    node: str,
    target_parent_dir: tuple[float, float],
    port_catalog: dict | None = None,
) -> float:
    code = _node_code(graph, node)
    port_entries = _catalog_port_entries(port_catalog, code)
    if port_entries:
        ports = tuple(port for port, _ in port_entries)
    else:
        ports = _node_port_template(graph, node, port_catalog)
        port_entries = tuple((port, 0.0) for port in ports)
    allowed = _catalog_allowed_rotations(port_catalog, code)
    if allowed:
        return max(
            _dot(_unit(_rotate_vec(port, rotation)), target_parent_dir) + weight
            for rotation in allowed
            for port, weight in port_entries
        )
    return max(_dot(_snap_dir(port), target_parent_dir) for port in ports)


def _choose_child_direction(
    graph: nx.Graph,
    raw_pos: dict[str, tuple[float, float]],
    positions: dict[str, tuple[float, float]],
    occupied: dict[tuple[int, int], str],
    used_ports: dict[str, set[tuple[int, int]]],
    rotations: dict[str, float],
    pitch: float,
    port_catalog: dict | None,
    parent: str,
    child: str,
) -> tuple[float, float]:
    used = used_ports.setdefault(parent, set())
    hint = _raw_hint_vector(raw_pos, parent, child)
    candidates = [
        direction
        for direction in _global_ports(graph, parent, rotations[parent], port_catalog)
        if _vector_key(direction) not in used
    ]
    if not candidates:
        candidates = [direction for direction in CANONICAL_DIRS if _vector_key(direction) not in used]
    if not candidates:
        candidates = list(CANONICAL_DIRS)

    px, py = positions[parent]

    def score(direction: tuple[float, float]) -> tuple[float, float]:
        target = (
            int(round((px + direction[0] * pitch) / pitch)),
            int(round((py + direction[1] * pitch) / pitch)),
        )
        collision = 1.0 if target in occupied else 0.0
        child_alignment = _port_alignment_score(graph, child, _opposite(direction), port_catalog)
        return collision, -child_alignment, -_dot(direction, hint)

    return min(candidates, key=score)


def _fit_screen_positions(
    raw_positions: dict[str, tuple[float, float]],
    icon_size: int,
    min_width: int,
    min_height: int,
) -> tuple[dict[str, tuple[float, float]], int, int]:
    if not raw_positions:
        return {}, min_width, min_height

    xs = [xy[0] for xy in raw_positions.values()]
    ys = [xy[1] for xy in raw_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    side_margin = 72
    top_margin = 118
    bottom_margin = 170
    graph_width = max_x - min_x + icon_size
    graph_height = max_y - min_y + icon_size
    width = max(min_width, int(math.ceil(graph_width + side_margin * 2)))
    height = max(min_height, int(math.ceil(graph_height + top_margin + bottom_margin)))
    tx = (width - graph_width) / 2 + icon_size / 2 - min_x
    ty = top_margin + icon_size / 2 - min_y
    return {node: (x + tx, y + ty) for node, (x, y) in raw_positions.items()}, width, height


def cad_port_layout(
    graph: nx.Graph,
    icon_size: int,
    edge_length_factor: float,
    min_width: int,
    min_height: int,
    port_catalog: dict | None = None,
) -> tuple[dict[str, tuple[float, float]], dict[str, float], int, int]:
    """Place connected components by snapping edges to component ports."""
    if graph.number_of_nodes() == 0:
        return {}, {}, min_width, min_height

    pitch = icon_size * edge_length_factor
    raw_hint = weighted_layout(graph)
    anchor = _choose_anchor(graph, port_catalog)
    raw_positions: dict[str, tuple[float, float]] = {anchor: (0.0, 0.0)}
    rotations: dict[str, float] = {anchor: 0.0}
    used_ports: dict[str, set[tuple[int, int]]] = {anchor: set()}
    occupied: dict[tuple[int, int], str] = {(0, 0): anchor}

    queue = [anchor]
    while queue:
        parent = queue.pop(0)
        neighbors = [n for n in graph.neighbors(parent) if n not in raw_positions]
        neighbors.sort(key=lambda child: -_dot(_raw_hint_vector(raw_hint, parent, child), DIR_E))

        for child in neighbors:
            direction = _choose_child_direction(
                graph,
                raw_hint,
                raw_positions,
                occupied,
                used_ports,
                rotations,
                pitch,
                port_catalog,
                parent,
                child,
            )
            px, py = raw_positions[parent]
            cx = px + direction[0] * pitch
            cy = py + direction[1] * pitch
            target_key = (int(round(cx / pitch)), int(round(cy / pitch)))
            if target_key in occupied:
                cx = px + direction[0] * pitch * 1.04
                cy = py + direction[1] * pitch * 1.04
                target_key = (int(round(cx / pitch)), int(round(cy / pitch)))

            raw_positions[child] = (cx, cy)
            occupied[target_key] = child
            rotations[child] = _orientation_for_parent(
                graph,
                child,
                _opposite(direction),
                port_catalog,
            )
            used_ports.setdefault(parent, set()).add(_vector_key(direction))
            used_ports.setdefault(child, set()).add(_vector_key(_opposite(direction)))
            queue.append(child)

    unplaced = [node for node in graph.nodes() if node not in raw_positions]
    if unplaced:
        max_y = max(y for _, y in raw_positions.values())
        x_start = -0.5 * (len(unplaced) - 1) * pitch
        for index, node in enumerate(unplaced):
            raw_positions[node] = (x_start + index * pitch, max_y + pitch * 1.6)
            rotations[node] = 0.0

    positions, width, height = _fit_screen_positions(raw_positions, icon_size, min_width, min_height)
    return positions, rotations, width, height


def _trim_line(
    a: tuple[float, float],
    b: tuple[float, float],
    trim: float,
) -> tuple[float, float, float, float]:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length <= 1e-9 or length <= 2 * trim:
        return ax, ay, bx, by
    ux, uy = dx / length, dy / length
    return ax + ux * trim, ay + uy * trim, bx - ux * trim, by - uy * trim


def _edge_style(attrs: dict) -> tuple[str, float, str, float, str]:
    support = _parse_float(attrs.get("support_count"), 0.0)
    forced = bool(attrs.get("forced", False))
    reason = attrs.get("forced_reason") or ""
    evidence_sources = set(attrs.get("evidence_sources") or [])

    if forced and "structural_fallback" in reason:
        return "#64748b", 1.8, "8 8", 0.45, "fallback"
    if forced:
        return "#d97706", 2.0, "7 6", 0.62, "forced"
    if "vggt_3d" in evidence_sources:
        return "#0284c7", 2.2 + min(support, 4.0) * 0.25, "", 0.82, "VGGT"
    if support > 0:
        return "#334155", 2.0 + min(support, 4.0) * 0.28, "", 0.78, "observed"
    return "#94a3b8", 1.6, "4 6", 0.55, "weak"


def _is_zero(value: str | None) -> bool:
    if value is None:
        return True
    try:
        return float(value) == 0.0
    except ValueError:
        return value.strip() in {"0", "0.0", "0px"}


def _svg_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _is_command(token: str) -> bool:
    return len(token) == 1 and token.isalpha()


def _path_points(path_d: str) -> list[tuple[float, float]]:
    tokens = _PATH_TOKEN_RE.findall(path_d.replace(",", " "))
    points: list[tuple[float, float]] = []
    i = 0
    cmd = ""
    current = (0.0, 0.0)
    start = (0.0, 0.0)
    param_counts = {
        "L": 2, "H": 1, "V": 1, "C": 6, "S": 4,
        "Q": 4, "T": 2, "A": 7,
    }

    while i < len(tokens):
        if _is_command(tokens[i]):
            cmd = tokens[i]
            i += 1
        if not cmd:
            break

        upper = cmd.upper()
        relative = cmd.islower()

        if upper == "Z":
            current = start
            points.append(current)
            cmd = ""
            continue

        if upper == "M":
            needed = 2
        else:
            needed = param_counts.get(upper)
        if needed is None:
            break

        while i < len(tokens) and not _is_command(tokens[i]):
            if i + needed > len(tokens):
                break
            try:
                values = [float(tokens[i + j]) for j in range(needed)]
            except ValueError:
                break
            i += needed

            cx, cy = current
            if upper == "M":
                x, y = values
                current = (cx + x, cy + y) if relative else (x, y)
                start = current
                points.append(current)
                cmd = "l" if relative else "L"
                upper = "L"
                needed = 2
            elif upper == "L":
                x, y = values
                current = (cx + x, cy + y) if relative else (x, y)
                points.append(current)
            elif upper == "H":
                x = values[0] + cx if relative else values[0]
                current = (x, cy)
                points.append(current)
            elif upper == "V":
                y = values[0] + cy if relative else values[0]
                current = (cx, y)
                points.append(current)
            elif upper == "C":
                coords = list(zip(values[0::2], values[1::2]))
                abs_coords = [(cx + x, cy + y) for x, y in coords] if relative else coords
                points.extend(abs_coords)
                current = abs_coords[-1]
            elif upper == "S":
                coords = list(zip(values[0::2], values[1::2]))
                abs_coords = [(cx + x, cy + y) for x, y in coords] if relative else coords
                points.extend(abs_coords)
                current = abs_coords[-1]
            elif upper == "Q":
                coords = list(zip(values[0::2], values[1::2]))
                abs_coords = [(cx + x, cy + y) for x, y in coords] if relative else coords
                points.extend(abs_coords)
                current = abs_coords[-1]
            elif upper == "T":
                x, y = values
                current = (cx + x, cy + y) if relative else (x, y)
                points.append(current)
            elif upper == "A":
                x, y = values[-2], values[-1]
                current = (cx + x, cy + y) if relative else (x, y)
                points.append(current)

    return points


def _svg_stroke_padding(root: ET.Element) -> float:
    max_width = 0.0
    for element in root.iter():
        style = element.attrib.get("style") or ""
        style_match = re.search(r"stroke-width\s*:\s*([0-9.]+)", style)
        if style_match:
            max_width = max(max_width, float(style_match.group(1)))
        attr_width = element.attrib.get("stroke-width")
        if attr_width:
            try:
                max_width = max(max_width, float(attr_width))
            except ValueError:
                pass
        if element.text and _svg_local_name(element.tag) == "style":
            for match in re.finditer(r"stroke-width\s*:\s*([0-9.]+)", element.text):
                max_width = max(max_width, float(match.group(1)))
    return max_width / 2.0


def _visible_svg_bbox(root: ET.Element) -> tuple[float, float, float, float] | None:
    points: list[tuple[float, float]] = []
    for element in root.iter():
        tag = _svg_local_name(element.tag)
        if tag == "path" and element.attrib.get("d"):
            points.extend(_path_points(element.attrib["d"]))
        elif tag == "line":
            points.extend([
                (_parse_float(element.attrib.get("x1")), _parse_float(element.attrib.get("y1"))),
                (_parse_float(element.attrib.get("x2")), _parse_float(element.attrib.get("y2"))),
            ])
        elif tag == "circle":
            cx = _parse_float(element.attrib.get("cx"))
            cy = _parse_float(element.attrib.get("cy"))
            r = _parse_float(element.attrib.get("r"))
            points.extend([(cx - r, cy - r), (cx + r, cy + r)])
        elif tag == "ellipse":
            cx = _parse_float(element.attrib.get("cx"))
            cy = _parse_float(element.attrib.get("cy"))
            rx = _parse_float(element.attrib.get("rx"))
            ry = _parse_float(element.attrib.get("ry"))
            points.extend([(cx - rx, cy - ry), (cx + rx, cy + ry)])

    if not points:
        return None

    pad = _svg_stroke_padding(root) * 1.25
    min_x = min(x for x, _ in points) - pad
    min_y = min(y for _, y in points) - pad
    max_x = max(x for x, _ in points) + pad
    max_y = max(y for _, y in points) + pad
    return min_x, min_y, max_x, max_y


def _transparent_svg_bytes(svg_path: Path) -> bytes:
    """Return SVG bytes with the background removed and viewBox cropped."""
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(svg_path)
    root = tree.getroot()

    for child in list(root):
        if _svg_local_name(child.tag) != "rect":
            continue
        if not _is_zero(child.attrib.get("x")) or not _is_zero(child.attrib.get("y")):
            continue
        fill = (child.attrib.get("fill") or "").strip().lower()
        style = (child.attrib.get("style") or "").replace(" ", "").lower()
        if fill not in {"", "none", "transparent"} or "fill:" in style:
            root.remove(child)
            break

    bbox = _visible_svg_bbox(root)
    if bbox is not None:
        min_x, min_y, max_x, max_y = bbox
        width = max(max_x - min_x, 1.0)
        height = max(max_y - min_y, 1.0)
        root.attrib["viewBox"] = f"{min_x:.3f} {min_y:.3f} {width:.3f} {height:.3f}"
        root.attrib["width"] = f"{width:.3f}"
        root.attrib["height"] = f"{height:.3f}"

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def _data_uri(svg_path: Path) -> str:
    encoded = base64.b64encode(_transparent_svg_bytes(svg_path)).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _node_display_name(node_attrs: dict, symbol_info: SymbolInfo | None) -> str:
    if symbol_info is not None and symbol_info.name:
        return symbol_info.name
    return str(node_attrs.get("name") or f"code_{node_attrs.get('code', '?')}")


def _render_symbol_node(
    node_id: str,
    attrs: dict,
    x: float,
    y: float,
    symbol_info: SymbolInfo | None,
    icon_size: int,
    show_labels: bool,
    rotation_deg: float = 0.0,
) -> list[str]:
    code = _parse_int(attrs.get("code"), 0)
    support = _parse_float(attrs.get("support_count"), 0.0)
    confidence = _parse_float(attrs.get("confidence"), 0.0)
    name = _node_display_name(attrs, symbol_info)
    title = escape(
        f"{node_id} | code {code} | {name} | support {support:g} | conf {confidence:.2f}"
    )

    lines = [
        f'<g class="node" data-node="{escape(node_id)}" data-code="{code}" '
        f'data-rotation="{rotation_deg:.2f}">'
    ]
    lines.append(f"<title>{title}</title>")

    if symbol_info is not None and symbol_info.has_svg:
        href = _data_uri(symbol_info.svg_path)  # type: ignore[arg-type]
        left = x - icon_size / 2
        top = y - icon_size / 2
        transform = ""
        if abs(rotation_deg) > 1e-6:
            transform = f' transform="rotate({rotation_deg:.2f} {x:.2f} {y:.2f})"'
        lines.append(
            f'<image x="{left:.2f}" y="{top:.2f}" width="{icon_size}" height="{icon_size}" '
            f'href="{href}" xlink:href="{href}" preserveAspectRatio="xMidYMid meet"{transform} />'
        )
    else:
        radius = icon_size * 0.28
        lines.append(
            f'<circle class="missing-symbol" cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" />'
        )
        lines.append(
            f'<text class="missing-code" x="{x:.2f}" y="{y + 4:.2f}">{code}</text>'
        )

    if show_labels:
        label_y = y + icon_size * 0.68
        lines.append(
            f'<text class="node-label" x="{x:.2f}" y="{label_y:.2f}">'
            f'<tspan x="{x:.2f}">{escape(node_id)} - {code}</tspan>'
            f'<tspan x="{x:.2f}" dy="13">{escape(name)}</tspan>'
            f"</text>"
        )
    lines.append("</g>")
    return lines


def _render_legend(
    x: int,
    y: int,
    missing_codes: Iterable[int],
) -> list[str]:
    missing = sorted(set(missing_codes))
    missing_text = ", ".join(str(code) for code in missing) if missing else "none"
    if len(missing_text) > 54:
        missing_text = missing_text[:51] + "..."

    return [
        f'<g class="legend" transform="translate({x},{y})">',
        '<rect x="0" y="0" width="330" height="116" rx="8" />',
        '<text x="14" y="24" class="legend-title">Edges</text>',
        '<line x1="16" y1="42" x2="70" y2="42" class="edge-sample observed" />',
        '<text x="82" y="46">observed / VGGT-supported</text>',
        '<line x1="16" y1="64" x2="70" y2="64" class="edge-sample forced" />',
        '<text x="82" y="68">forced / fallback</text>',
        '<circle cx="43" cy="89" r="9" class="missing-symbol" />',
        '<text x="82" y="94">missing SVG: ' + escape(missing_text) + "</text>",
        "</g>",
    ]


def render_graph_svg(
    graph: nx.Graph,
    graph_data: dict,
    node_id: str,
    symbols: dict[int, SymbolInfo],
    port_catalog: dict | None,
    output_path: Path,
    width: int = 820,
    height: int = 560,
    icon_size: int = DEFAULT_ICON_SIZE,
    edge_length_factor: float = DEFAULT_EDGE_LENGTH_FACTOR,
    show_labels: bool = False,
    show_edge_labels: bool = False,
    layout_mode: str = "cad",
) -> Path:
    """Render one graph to a self-contained SVG file."""
    if layout_mode == "collapsed":
        pos, width, height = _fit_collapsed_positions(
            graph,
            weighted_layout(graph),
            icon_size,
            edge_length_factor,
            width,
            height,
        )
        rotations = {node: 0.0 for node in graph.nodes()}
    else:
        pos, rotations, width, height = cad_port_layout(
            graph,
            icon_size,
            edge_length_factor,
            width,
            height,
            port_catalog,
        )
    n_views = graph_data.get("total_images", "?")
    connected = graph_data.get("is_connected")
    if connected is None:
        connected = nx.is_connected(graph) if graph.number_of_nodes() else True

    missing_codes: set[int] = set()
    for _, attrs in graph.nodes(data=True):
        code = _parse_int(attrs.get("code"), 0)
        info = symbols.get(code)
        if info is None or not info.has_svg:
            missing_codes.add(code)

    svg: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<style>",
        "svg { background: #f8fafc; font-family: Arial, Helvetica, sans-serif; }",
        ".title { font-size: 24px; font-weight: 700; fill: #111827; }",
        ".subtitle { font-size: 13px; fill: #475569; }",
        ".edge { fill: none; stroke-linecap: round; }",
        ".edge-label { font-size: 9px; fill: #334155; text-anchor: middle; paint-order: stroke; stroke: #f8fafc; stroke-width: 4px; }",
        ".missing-symbol { fill: #dc2626; stroke: #ffffff; stroke-width: 4px; filter: drop-shadow(0 3px 7px rgba(15,23,42,0.25)); }",
        ".missing-code { font-size: 11px; font-weight: 700; fill: #ffffff; text-anchor: middle; }",
        ".node-label { font-size: 10px; fill: #0f172a; text-anchor: middle; paint-order: stroke; stroke: #f8fafc; stroke-width: 5px; }",
        ".legend rect { fill: rgba(255,255,255,0.92); stroke: #cbd5e1; }",
        ".legend text { font-size: 12px; fill: #334155; }",
        ".legend-title { font-weight: 700; fill: #111827; }",
        ".edge-sample { stroke-width: 4px; stroke-linecap: round; }",
        ".edge-sample.observed { stroke: #0284c7; }",
        ".edge-sample.forced { stroke: #d97706; stroke-dasharray: 7 6; }",
        "</style>",
        f'<text class="title" x="36" y="46">Node {escape(_node_folder_name(node_id))} - Shurikode symbols ({escape(layout_mode)})</text>',
        (
            f'<text class="subtitle" x="36" y="70">'
            f'{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, '
            f'{n_views} views, {"connected" if connected else "disconnected"}'
            f"</text>"
        ),
        '<g class="edges">',
    ]

    for u, v, attrs in graph.edges(data=True):
        if u not in pos or v not in pos:
            continue
        color, width_px, dash, alpha, label = _edge_style(attrs)
        x1, y1, x2, y2 = _trim_line(pos[u], pos[v], icon_size * 0.46)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        svg.append(
            f'<line class="edge" x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" '
            f'stroke-width="{width_px:.2f}" opacity="{alpha:.2f}"{dash_attr} />'
        )
        support = _parse_float(attrs.get("support_count"), 0.0)
        if show_edge_labels and support > 0 and not attrs.get("forced", False):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            svg.append(
                f'<text class="edge-label" x="{mx:.2f}" y="{my - 6:.2f}">'
                f'{escape(label)}</text>'
            )

    svg.append("</g>")
    svg.append('<g class="nodes">')
    for current_node in graph.nodes():
        if current_node not in pos:
            continue
        attrs = graph.nodes[current_node]
        code = _parse_int(attrs.get("code"), 0)
        svg.extend(
            _render_symbol_node(
                current_node,
                attrs,
                pos[current_node][0],
                pos[current_node][1],
                symbols.get(code),
                icon_size,
                show_labels,
                rotations.get(current_node, 0.0),
            )
        )
    svg.append("</g>")
    svg.extend(_render_legend(width - 370, height - 150, missing_codes))
    svg.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def _discover_node_ids(work_dir: Path) -> list[str]:
    ids: list[str] = []
    for path in sorted(work_dir.glob("node*/fused_graph.json")):
        ids.append(path.parent.name.removeprefix("node"))
    return ids


def render_node(
    node_id: str,
    work_dir: Path,
    symbol_dir: Path,
    port_catalog_path: Path,
    output_name: str,
    icon_size: int = DEFAULT_ICON_SIZE,
    edge_length_factor: float = DEFAULT_EDGE_LENGTH_FACTOR,
    show_labels: bool = False,
    show_edge_labels: bool = False,
    layout_mode: str = "cad",
) -> Path:
    symbols = load_symbol_index(symbol_dir)
    port_catalog = load_port_catalog(port_catalog_path)
    graph, data = load_fused_graph(node_id, work_dir)
    out = work_dir / _node_folder_name(node_id) / output_name
    return render_graph_svg(
        graph,
        data,
        node_id,
        symbols,
        port_catalog,
        out,
        icon_size=icon_size,
        edge_length_factor=edge_length_factor,
        show_labels=show_labels,
        show_edge_labels=show_edge_labels,
        layout_mode=layout_mode,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render fused graph visualizations using Shurikode SVG symbols.",
    )
    parser.add_argument("nodes", nargs="*", help="Node IDs, e.g. 10 or node10.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render every node*/fused_graph.json found in the work directory.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help=f"Directory containing node*/fused_graph.json files. Default: {DEFAULT_WORK_DIR}",
    )
    parser.add_argument(
        "--symbol-dir",
        type=Path,
        default=DEFAULT_SYMBOL_DIR,
        help=f"Directory containing shurikode_mapping.csv and SVGs. Default: {DEFAULT_SYMBOL_DIR}",
    )
    parser.add_argument(
        "--port-catalog",
        type=Path,
        default=DEFAULT_PORT_CATALOG_PATH,
        help=f"Component port/orientation catalog JSON. Default: {DEFAULT_PORT_CATALOG_PATH}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output SVG file name inside each node folder. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--icon-size",
        type=int,
        default=DEFAULT_ICON_SIZE,
        help=f"Rendered symbol size in pixels. Default: {DEFAULT_ICON_SIZE}",
    )
    parser.add_argument(
        "--edge-length-factor",
        type=float,
        default=DEFAULT_EDGE_LENGTH_FACTOR,
        help=(
            "Target connected-node center distance as a fraction of icon size. "
            f"Default: {DEFAULT_EDGE_LENGTH_FACTOR}"
        ),
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show node labels under symbols. Labels are hidden by default in the collapsed view.",
    )
    parser.add_argument(
        "--show-edge-labels",
        action="store_true",
        help="Show edge support labels. Hidden by default in the collapsed view.",
    )
    parser.add_argument(
        "--layout",
        choices=("cad", "collapsed"),
        default="cad",
        help="Layout strategy. 'cad' snaps edges to component ports; 'collapsed' keeps the previous compact graph layout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.all:
        node_ids = _discover_node_ids(args.work_dir)
    else:
        node_ids = args.nodes or ["2"]

    if not node_ids:
        print(f"No fused_graph.json files found under {args.work_dir}", file=sys.stderr)
        return 1

    try:
        symbols = load_symbol_index(args.symbol_dir)
        port_catalog = load_port_catalog(args.port_catalog)
    except Exception as exc:
        print(f"[symbols] {exc}", file=sys.stderr)
        return 1

    exit_code = 0
    for node_id in node_ids:
        try:
            graph, data = load_fused_graph(node_id, args.work_dir)
            out = args.work_dir / _node_folder_name(node_id) / args.output_name
            render_graph_svg(
                graph,
                data,
                node_id,
                symbols,
                port_catalog,
                out,
                icon_size=args.icon_size,
                edge_length_factor=args.edge_length_factor,
                show_labels=args.show_labels,
                show_edge_labels=args.show_edge_labels,
                layout_mode=args.layout,
            )
            print(f"[{_node_folder_name(node_id)}] saved: {out.resolve()}")
        except Exception as exc:
            exit_code = 1
            print(f"[{_node_folder_name(node_id)}] {exc}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
