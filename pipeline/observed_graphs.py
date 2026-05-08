"""
Step 1 - per-view detection loading.

KNN adjacency is no longer built here — graph edges come exclusively from
VGGT 3D geometry in the new 3D-only pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import (
    COMPONENT_INFO,
    CONF_THRESHOLD,
    DIAMETER_CODES,
)
from core.detection import detect_and_decode_cached


@dataclass
class ObservedNode:
    observed_node_id: str
    image_id: str
    code: int
    name: str
    confidence: float
    bbox: list
    center: tuple
    local_degree_estimate: int
    nearby_codes: list


@dataclass
class ObservedLocalGraph:
    image_id: str
    nodes: dict
    edges: list = field(default_factory=list)
    local_patterns: list = field(default_factory=list)


def build_observed_graph(image_path: Path) -> Optional[ObservedLocalGraph]:
    image_id = image_path.stem
    dets_raw, _ = detect_and_decode_cached(image_path)
    if not dets_raw:
        return None

    dets = [
        det for det in dets_raw
        if det.get("confidence", det.get("conf", 0.0)) >= CONF_THRESHOLD
        and not det.get("is_diameter", det["code"] in DIAMETER_CODES)
    ]
    if not dets:
        return None

    nodes: dict = {}
    for idx, det in enumerate(dets):
        observed_node_id = f"{image_id}_{idx}"
        confidence = det.get("confidence", det.get("conf", 0.0))
        _, name = COMPONENT_INFO.get(det["code"], (2, f"code_{det['code']}"))
        nodes[observed_node_id] = ObservedNode(
            observed_node_id=observed_node_id,
            image_id=image_id,
            code=det["code"],
            name=name,
            confidence=confidence,
            bbox=det["bbox_norm"],
            center=(det["cx"], det["cy"]),
            local_degree_estimate=0,
            nearby_codes=[],
        )

    return ObservedLocalGraph(image_id=image_id, nodes=nodes)


def build_all_observed_graphs(image_paths: list) -> list:
    graphs: list = []
    for image_path in image_paths:
        path = Path(image_path)
        print(f"  [ObsGraph] {path.name}", end="")
        observed_graph = build_observed_graph(path)
        if observed_graph is None:
            print(" -> no valid detections, skipped")
            continue
        print(f" -> {len(observed_graph.nodes)} nodes")
        graphs.append(observed_graph)
    return graphs


def serialize_observed_graphs(graphs: list) -> dict:
    serialized_graphs = []
    for graph in graphs:
        serialized_graphs.append({
            "image_id": graph.image_id,
            "n_nodes": len(graph.nodes),
            "nodes": [
                {
                    "observed_node_id": node.observed_node_id,
                    "image_id": node.image_id,
                    "code": node.code,
                    "name": node.name,
                    "confidence": node.confidence,
                    "bbox": node.bbox,
                    "center": list(node.center),
                }
                for node in graph.nodes.values()
            ],
        })

    return {
        "n_views": len(graphs),
        "observed_graphs": serialized_graphs,
    }
