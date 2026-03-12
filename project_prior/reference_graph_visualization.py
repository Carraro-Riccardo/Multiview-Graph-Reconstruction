"""
Visualizzazione dedicata del reference graph estratto via LLM.

Questo modulo evita di legare la lettura del grafo esclusivamente
all'overlay sull'immagine del cluster, che puo' risultare corretto
semanticamente ma non preciso geometricamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover
    plt = None

try:
    import networkx as nx  # type: ignore
except ImportError:  # pragma: no cover
    nx = None


TYPE_COLORS = {
    "valve": "#2e86de",
    "tee": "#27ae60",
    "cross": "#e67e22",
    "reducer": "#8e44ad",
    "pipe": "#16a085",
    "flange": "#c0392b",
    "elbow": "#f1c40f",
    "unknown": "#7f8c8d",
}


def _graph_positions_from_payload(payload: dict[str, Any]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for idx, component in enumerate(payload.get("components", [])):
        comp_id = component.get("id", f"component_{idx:03d}")
        center = component.get("center")
        if isinstance(center, list) and len(center) == 2:
            try:
                positions[comp_id] = (float(center[0]), -float(center[1]))
                continue
            except Exception:
                pass
        positions[comp_id] = (float(idx), 0.0)
    return positions


def save_reference_graph_overlay(
    image_path: Path,
    payload: dict[str, Any],
    out_dir: Path,
) -> dict[str, str]:
    if cv2 is None:
        return {}
    image = cv2.imread(str(image_path))
    if image is None:
        return {}

    overlay = image.copy()
    numbered = image.copy()
    labels = image.copy()
    by_id = {item["id"]: item for item in payload.get("components", []) if isinstance(item, dict) and "id" in item}

    for connection in payload.get("connections", []):
        if not isinstance(connection, dict):
            continue
        src = by_id.get(connection.get("from"))
        dst = by_id.get(connection.get("to"))
        if not src or not dst or not src.get("center") or not dst.get("center"):
            continue
        a = tuple(int(round(v)) for v in src["center"])
        b = tuple(int(round(v)) for v in dst["center"])
        cv2.line(overlay, a, b, (0, 255, 0), 2)
        cv2.line(numbered, a, b, (0, 255, 0), 2)
        cv2.line(labels, a, b, (0, 255, 0), 2)

    for idx, component in enumerate(payload.get("components", []), start=1):
        if not isinstance(component, dict):
            continue
        bbox = component.get("bbox")
        center = component.get("center")
        label = component.get("label_text") or component.get("type") or "unknown"
        text_main = f"{component.get('id', '?')} | {label}"[:80]
        text_num = f"{idx}:{component.get('id', '?')}"

        if bbox:
            x0, y0, x1, y1 = [int(round(v)) for v in bbox]
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 165, 255), 2)
            cv2.rectangle(numbered, (x0, y0), (x1, y1), (0, 165, 255), 2)
            cv2.rectangle(labels, (x0, y0), (x1, y1), (0, 165, 255), 2)
            cv2.putText(overlay, text_main, (x0, max(18, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(labels, text_main, (x0, max(18, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        if center:
            cx, cy = [int(round(v)) for v in center]
            cv2.circle(overlay, (cx, cy), 6, (255, 0, 255), -1)
            cv2.circle(numbered, (cx, cy), 6, (255, 0, 255), -1)
            cv2.circle(labels, (cx, cy), 6, (255, 0, 255), -1)
            cv2.putText(numbered, text_num, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    overlay_path = out_dir / "reference_graph_overlay.png"
    numbered_path = out_dir / "reference_graph_numbered.png"
    labels_path = out_dir / "reference_graph_labels.png"
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(numbered_path), numbered)
    cv2.imwrite(str(labels_path), labels)
    return {
        "reference_graph_overlay": str(overlay_path),
        "reference_graph_numbered": str(numbered_path),
        "reference_graph_labels": str(labels_path),
    }


def save_reference_graph_schema_view(
    payload: dict[str, Any],
    out_dir: Path,
) -> dict[str, str]:
    if plt is None or nx is None:
        return {}

    graph = nx.Graph()
    positions = _graph_positions_from_payload(payload)

    for component in payload.get("components", []):
        if not isinstance(component, dict):
            continue
        comp_id = component.get("id")
        if not comp_id:
            continue
        graph.add_node(
            comp_id,
            type=component.get("type", "unknown"),
            label_text=component.get("label_text"),
            confidence=component.get("confidence"),
        )

    for connection in payload.get("connections", []):
        if not isinstance(connection, dict):
            continue
        src = connection.get("from")
        dst = connection.get("to")
        if src in graph.nodes and dst in graph.nodes:
            graph.add_edge(src, dst, confidence=connection.get("confidence"))

    if graph.number_of_nodes() == 0:
        return {}

    missing = [node for node in graph.nodes if node not in positions]
    if missing:
        fallback = nx.spring_layout(graph, seed=13)
        for node in missing:
            positions[node] = fallback[node]

    node_colors = [
        TYPE_COLORS.get(graph.nodes[node].get("type", "unknown"), TYPE_COLORS["unknown"])
        for node in graph.nodes
    ]
    labels = {}
    for node in graph.nodes:
        node_type = graph.nodes[node].get("type", "unknown")
        label_text = graph.nodes[node].get("label_text")
        labels[node] = f"{node}\n{node_type}" + (f"\n{label_text}" if label_text else "")

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(graph, positions, ax=ax, width=2.2, edge_color="#4cd137")
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=node_colors, node_size=1800, edgecolors="#111111", linewidths=1.2)
    nx.draw_networkx_labels(graph, positions, labels=labels, ax=ax, font_size=8, font_color="#111111")
    ax.set_title("Reference Graph Schema View")
    ax.axis("off")
    plt.tight_layout()
    schema_path = out_dir / "reference_graph_schema.png"
    fig.savefig(schema_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    numbered_labels = {node: f"{idx + 1}: {node}" for idx, node in enumerate(graph.nodes)}
    nx.draw_networkx_edges(graph, positions, ax=ax, width=2.2, edge_color="#4cd137")
    nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=node_colors, node_size=1600, edgecolors="#111111", linewidths=1.2)
    nx.draw_networkx_labels(graph, positions, labels=numbered_labels, ax=ax, font_size=8, font_color="#111111")
    ax.set_title("Reference Graph Numbered Schema View")
    ax.axis("off")
    plt.tight_layout()
    numbered_path = out_dir / "reference_graph_schema_numbered.png"
    fig.savefig(numbered_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "reference_graph_schema": str(schema_path),
        "reference_graph_schema_numbered": str(numbered_path),
    }

