"""
Step 1 - per-view local graph hypothesis extraction.

Per-view edges are observational hypotheses, not guaranteed final scene edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .branch_fragments import extract_branch_fragments
from config import (
    COOC_COUNTS,
    COMPONENT_INFO,
    CONF_THRESHOLD,
    DIAMETER_CODES,
    DIST_MULTIPLIER,
    FORBIDDEN_PAIRS,
    K_NEIGHBORS,
    TRIPLET_COUNTS,
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
class ObservedEdge:
    source_id: str
    target_id: str
    image_id: str
    confidence: float
    distance: float
    cooc_support: float
    forbidden: bool
    edge_status: str
    direct_visible: bool
    shortcut_candidate: bool
    support_reason: str


@dataclass
class ObservedLocalGraph:
    image_id: str
    nodes: dict
    edges: list
    local_patterns: list
    branch_fragments: list = field(default_factory=list)


def _norm_pair(a: int, b: int) -> tuple:
    return (min(a, b), max(a, b))


def _cooc_support(code_a: int, code_b: int) -> float:
    pair = _norm_pair(code_a, code_b)
    count = COOC_COUNTS.get(pair, 0)
    if count == 0:
        return 0.0
    max_count = max(COOC_COUNTS.values()) if COOC_COUNTS else 1
    return count / max_count


def _compute_knn(dets: list, k: int = K_NEIGHBORS, dist_multiplier: float = DIST_MULTIPLIER):
    n = len(dets)
    if n < 2:
        return [[] for _ in dets], float("inf")

    centers = np.array([[d["cx"], d["cy"]] for d in dets])
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)

    nn_dists = np.array([
        np.min(distances[i, [j for j in range(n) if j != i]])
        for i in range(n)
    ])
    cutoff = float(np.median(nn_dists)) * dist_multiplier

    neighbour_indices = []
    for idx in range(n):
        candidates = sorted(
            (distances[idx, j], j)
            for j in range(n)
            if j != idx and distances[idx, j] <= cutoff
        )
        neighbour_indices.append([j for _, j in candidates[:k]])

    return neighbour_indices, cutoff


def _classify_observed_edge(distance: float, cutoff: float, forbidden: bool) -> tuple[str, bool, bool, str]:
    if cutoff <= 0:
        cutoff = 1.0
    relative_distance = distance / cutoff
    if forbidden or relative_distance >= 0.8:
        return "shortcut_candidate", False, True, "distant_or_suspicious_local_adjacency"
    if relative_distance <= 0.45:
        return "direct_visible", True, False, "tight_local_proximity"
    return "plausible_local_adjacency", False, False, "partial_view_local_adjacency"


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

    knn_neighbours, cutoff = _compute_knn(dets)

    nodes: dict = {}
    for idx, det in enumerate(dets):
        observed_node_id = f"{image_id}_{idx}"
        confidence = det.get("confidence", det.get("conf", 0.0))
        _, name = COMPONENT_INFO.get(det["code"], (2, f"code_{det['code']}"))
        nearby_codes = [dets[j]["code"] for j in knn_neighbours[idx]]
        nodes[observed_node_id] = ObservedNode(
            observed_node_id=observed_node_id,
            image_id=image_id,
            code=det["code"],
            name=name,
            confidence=confidence,
            bbox=det["bbox_norm"],
            center=(det["cx"], det["cy"]),
            local_degree_estimate=len(nearby_codes),
            nearby_codes=nearby_codes,
        )

    observed_ids = list(nodes.keys())
    edges: list = []
    seen_pairs: set = set()
    for idx, observed_id_a in enumerate(observed_ids):
        for neighbour_idx in knn_neighbours[idx]:
            observed_id_b = observed_ids[neighbour_idx]
            pair = (min(observed_id_a, observed_id_b), max(observed_id_a, observed_id_b))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            code_a = dets[idx]["code"]
            code_b = dets[neighbour_idx]["code"]
            conf_a = dets[idx].get("confidence", dets[idx].get("conf", 0.0))
            conf_b = dets[neighbour_idx].get("confidence", dets[neighbour_idx].get("conf", 0.0))
            distance = float(np.sqrt(
                (dets[idx]["cx"] - dets[neighbour_idx]["cx"]) ** 2 +
                (dets[idx]["cy"] - dets[neighbour_idx]["cy"]) ** 2
            ))
            cooc = _cooc_support(code_a, code_b)
            forbidden = _norm_pair(code_a, code_b) in FORBIDDEN_PAIRS
            edge_status, direct_visible, shortcut_candidate, support_reason = _classify_observed_edge(
                distance,
                cutoff,
                forbidden,
            )

            edges.append(ObservedEdge(
                source_id=observed_id_a,
                target_id=observed_id_b,
                image_id=image_id,
                confidence=(conf_a + conf_b) / 2.0 * (0.5 + 0.5 * cooc),
                distance=distance,
                cooc_support=cooc,
                forbidden=forbidden,
                edge_status=edge_status,
                direct_visible=direct_visible,
                shortcut_candidate=shortcut_candidate,
                support_reason=support_reason,
            ))

    local_patterns = _extract_local_patterns(dets, knn_neighbours)
    observed_graph = ObservedLocalGraph(
        image_id=image_id,
        nodes=nodes,
        edges=edges,
        local_patterns=local_patterns,
    )
    observed_graph.branch_fragments = extract_branch_fragments(observed_graph)
    return observed_graph


def _extract_local_patterns(dets: list, knn_neighbours: list) -> list:
    patterns: list = []
    seen_pairs: set = set()

    for idx, neighbours in enumerate(knn_neighbours):
        for neighbour_idx in neighbours:
            pair = _norm_pair(dets[idx]["code"], dets[neighbour_idx]["code"])
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                patterns.append({
                    "type": "pair",
                    "codes": list(pair),
                    "cooc_count": COOC_COUNTS.get(pair, 0),
                    "forbidden": pair in FORBIDDEN_PAIRS,
                })

    for center_idx, neighbours in enumerate(knn_neighbours):
        if len(neighbours) < 2:
            continue
        for idx_a in range(len(neighbours)):
            for idx_b in range(idx_a + 1, len(neighbours)):
                center_code = dets[center_idx]["code"]
                code_a = dets[neighbours[idx_a]]["code"]
                code_b = dets[neighbours[idx_b]]["code"]
                key = (min(code_a, code_b), center_code, max(code_a, code_b))
                patterns.append({
                    "type": "triplet",
                    "center_code": center_code,
                    "neighbor_codes": [code_a, code_b],
                    "triplet_key": list(key),
                    "triplet_count": TRIPLET_COUNTS.get(key, 0),
                })

    return patterns


def build_all_observed_graphs(image_paths: list) -> list:
    graphs: list = []
    for image_path in image_paths:
        path = Path(image_path)
        print(f"  [ObsGraph] {path.name}", end="")
        observed_graph = build_observed_graph(path)
        if observed_graph is None:
            print(" -> no valid detections, skipped")
            continue
        print(f" -> {len(observed_graph.nodes)} nodes, {len(observed_graph.edges)} edges")
        graphs.append(observed_graph)
    return graphs


def serialize_observed_graphs(graphs: list) -> dict:
    from .branch_fragments import serialize_branch_fragments

    serialized_graphs = []
    for graph in graphs:
        serialized_graphs.append({
            "image_id": graph.image_id,
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "n_branch_fragments": len(graph.branch_fragments),
            "nodes": [
                {
                    "observed_node_id": node.observed_node_id,
                    "image_id": node.image_id,
                    "code": node.code,
                    "name": node.name,
                    "confidence": node.confidence,
                    "bbox": node.bbox,
                    "center": list(node.center),
                    "local_degree_estimate": node.local_degree_estimate,
                    "nearby_codes": node.nearby_codes,
                }
                for node in graph.nodes.values()
            ],
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "image_id": edge.image_id,
                    "confidence": edge.confidence,
                    "distance": edge.distance,
                    "cooc_support": edge.cooc_support,
                    "forbidden": edge.forbidden,
                    "edge_status": edge.edge_status,
                    "direct_visible": edge.direct_visible,
                    "shortcut_candidate": edge.shortcut_candidate,
                    "support_reason": edge.support_reason,
                }
                for edge in graph.edges
            ],
            "local_patterns": graph.local_patterns,
            "branch_fragments": serialize_branch_fragments(graph.branch_fragments),
        })

    return {
        "n_views": len(graphs),
        "observed_graphs": serialized_graphs,
    }
