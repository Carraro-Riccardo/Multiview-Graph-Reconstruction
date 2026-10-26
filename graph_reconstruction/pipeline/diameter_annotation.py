"""Associazione e propagazione delle etichette di diametro sul grafo fuso."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import networkx as nx

from graph_reconstruction.config import (
    CONF_THRESHOLD,
    DIAMETER_BREAK_CODES,
    DIAMETER_CODES,
    DIAMETER_VALUE_BY_CODE,
)
from graph_reconstruction.shared.detection import detect_and_decode_cached
from .node_reid import build_observed_to_scene_map


def _edge_key(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))


def _point_segment_distance(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    px, py = point
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5, 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    clamped = max(0.0, min(1.0, t))
    cx = ax + clamped * dx
    cy = ay + clamped * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5, t


def _load_view_detections(
    image_path: Path,
    obs_to_scene: dict[str, str],
) -> tuple[dict[str, tuple[float, float]], list[dict[str, Any]]]:
    image_id = image_path.stem
    dets_raw, _ = detect_and_decode_cached(image_path)

    accepted = [
        det for det in dets_raw
        if det.get("confidence", det.get("conf", 0.0)) >= CONF_THRESHOLD
    ]
    topo = [det for det in accepted if det.get("code") not in DIAMETER_CODES]
    diameters = [det for det in accepted if det.get("code") in DIAMETER_VALUE_BY_CODE]

    scene_centers: dict[str, tuple[float, float]] = {}
    for idx, det in enumerate(topo):
        obs_id = f"{image_id}_{idx}"
        scene_id = obs_to_scene.get(obs_id)
        if scene_id is None:
            continue
        scene_centers[scene_id] = (float(det["cx"]), float(det["cy"]))

    diameter_observations = []
    for idx, det in enumerate(diameters):
        code = int(det["code"])
        diameter_observations.append({
            "observation_id": f"{image_id}_diameter_{idx}",
            "image_id": image_id,
            "code": code,
            "diameter": DIAMETER_VALUE_BY_CODE[code],
            "confidence": float(det.get("confidence", det.get("conf", 0.0))),
            "center": [float(det["cx"]), float(det["cy"])],
            "bbox": det.get("bbox_norm"),
        })

    return scene_centers, diameter_observations


def _collect_edge_seeds(
    graph: nx.Graph,
    image_paths: list[Path],
    obs_to_scene: dict[str, str],
    *,
    max_segment_distance: float,
) -> tuple[dict[tuple[str, str], list[dict[str, Any]]], list[dict[str, Any]]]:
    """Associa ogni etichetta al segmento visibile più vicino nella stessa vista."""
    seeds: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    all_observations: list[dict[str, Any]] = []

    graph_edges = [_edge_key(a, b) for a, b in graph.edges()]
    for raw_path in image_paths:
        image_path = Path(raw_path)
        centers, diameter_observations = _load_view_detections(image_path, obs_to_scene)
        for obs in diameter_observations:
            best: tuple[float, float, tuple[str, str]] | None = None
            point = tuple(obs["center"])
            for edge in graph_edges:
                a, b = edge
                if a not in centers or b not in centers:
                    continue
                dist, t = _point_segment_distance(point, centers[a], centers[b])
                if t < -0.35 or t > 1.35:
                    continue
                candidate = (dist, abs(t - 0.5), edge)
                if best is None or candidate < best:
                    best = candidate

            if best is None:
                obs["association"] = "unmatched"
                all_observations.append(dict(obs))
                continue

            dist, _, edge = best
            obs["nearest_edge"] = list(edge)
            obs["distance_to_edge"] = dist
            if dist > max_segment_distance:
                obs["association"] = "too_far_from_edge"
                all_observations.append(dict(obs))
                continue
            obs["association"] = "edge_seed"
            seeds[edge].append(dict(obs))
            all_observations.append(dict(obs))

    return dict(seeds), all_observations


def _edge_components(graph: nx.Graph) -> list[list[tuple[str, str]]]:
    """Raggruppa gli archi lungo linee continue, fermandosi ai cambi diametro."""
    edges = [_edge_key(a, b) for a, b in graph.edges()]
    if not edges:
        return []

    parent = {edge: edge for edge in edges}

    def find(edge: tuple[str, str]) -> tuple[str, str]:
        while parent[edge] != edge:
            parent[edge] = parent[parent[edge]]
            edge = parent[edge]
        return edge

    def union(a: tuple[str, str], b: tuple[str, str]) -> None:
        parent[find(a)] = find(b)

    incident: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        a, b = edge
        incident[a].append(edge)
        incident[b].append(edge)

    for node_id, node_edges in incident.items():
        code = int(graph.nodes[node_id].get("code", 0))
        if code in DIAMETER_BREAK_CODES:
            continue
        for edge in node_edges[1:]:
            union(node_edges[0], edge)

    groups: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        groups[find(edge)].append(edge)
    return list(groups.values())


def _choose_diameter(observations: list[dict[str, Any]]) -> tuple[int | None, dict[str, Any]]:
    """Sceglie il diametro pesando confidenza e distanza dall'arco."""
    if not observations:
        return None, {"confidence": 0.0, "observations": 0, "candidates": {}}

    weights: dict[int, float] = defaultdict(float)
    counts: Counter[int] = Counter()
    for obs in observations:
        diameter = int(obs["diameter"])
        distance = float(obs.get("distance_to_edge", 0.0))
        confidence = float(obs.get("confidence", 1.0))
        weights[diameter] += confidence / max(distance, 0.015)
        counts[diameter] += 1

    best = max(weights, key=lambda dia: (weights[dia], counts[dia]))
    total_weight = sum(weights.values())
    debug = {
        "confidence": weights[best] / total_weight if total_weight else 0.0,
        "observations": len(observations),
        "candidates": {str(dia): counts[dia] for dia in sorted(counts)},
        "weighted_candidates": {
            str(dia): round(weights[dia], 4) for dia in sorted(weights)
        },
        "conflict": len(counts) > 1,
    }
    return best, debug


def annotate_graph_diameters(
    graph: nx.Graph,
    scene_nodes: list,
    image_paths: list[Path],
    *,
    max_segment_distance: float = 0.14,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Associa ``diameter_mm`` ad archi e nodi usando le etichette rilevate.

    Le etichette di diametro non sono nodi topologici: inizializzano l'arco
    visibile più vicino e il valore scelto viene propagato lungo la stessa linea.
    Riduzioni e adattatori elencati in ``DIAMETER_BREAK_CODES`` interrompono la
    propagazione.
    """

    obs_to_scene = build_observed_to_scene_map(scene_nodes)
    seeds, observations = _collect_edge_seeds(
        graph,
        [Path(path) for path in image_paths],
        obs_to_scene,
        max_segment_distance=max_segment_distance,
    )

    # La scelta avviene per intera linea, non per singolo arco: viste diverse
    # possono quindi contribuire allo stesso diametro propagato.
    edge_debug: dict[str, dict[str, Any]] = {}
    for group_index, component_edges in enumerate(_edge_components(graph), start=1):
        component_observations: list[dict[str, Any]] = []
        for edge in component_edges:
            component_observations.extend(seeds.get(edge, []))

        diameter, choice_debug = _choose_diameter(component_observations)
        for edge in component_edges:
            a, b = edge
            if diameter is not None:
                graph[a][b]["diameter_mm"] = diameter
                graph[a][b]["diameter_source"] = "detected_propagated"
                graph[a][b]["diameter_confidence"] = choice_debug["confidence"]
                graph[a][b]["diameter_observation_count"] = choice_debug["observations"]
                graph[a][b]["diameter_conflict"] = choice_debug["conflict"]
            else:
                graph[a][b]["diameter_mm"] = None
                graph[a][b]["diameter_source"] = None
                graph[a][b]["diameter_confidence"] = 0.0
                graph[a][b]["diameter_observation_count"] = 0
                graph[a][b]["diameter_conflict"] = False

            edge_debug[f"{a}--{b}"] = {
                "line_group": group_index,
                "diameter_mm": diameter,
                "seed_observations": len(seeds.get(edge, [])),
                **choice_debug,
            }

    # Un nodo espone il diametro solo se tutti gli archi incidenti concordano;
    # in caso contrario conserva i candidati e segnala il conflitto.
    for node_id in graph.nodes():
        incident_diameters = sorted({
            graph[a][b].get("diameter_mm")
            for a, b in graph.edges(node_id)
            if graph[a][b].get("diameter_mm") is not None
        })
        graph.nodes[node_id]["diameter_mm"] = (
            incident_diameters[0] if len(incident_diameters) == 1 else None
        )
        graph.nodes[node_id]["diameter_candidates_mm"] = incident_diameters
        graph.nodes[node_id]["diameter_conflict"] = len(incident_diameters) > 1

    debug = {
        "diameter_code_map": DIAMETER_VALUE_BY_CODE,
        "break_codes": sorted(DIAMETER_BREAK_CODES),
        "max_segment_distance": max_segment_distance,
        "n_diameter_observations": len(observations),
        "n_seeded_edges": len(seeds),
        "observations": observations,
        "edge_debug": edge_debug,
    }
    return graph, debug
