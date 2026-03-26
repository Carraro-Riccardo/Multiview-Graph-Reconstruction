"""pipeline/direct_graph_builder.py

Direct per-image adjacency graph builder.
Replaces Steps 3-6 of the pipeline (multiview_fusion, candidate_scene_graph,
structural_reconciliation, graph_refinement).

Algorithm
---------
1. For every view, map each observed detection -> scene node ID (from ReID).
2. Aggregate direct 2D adjacency votes from the observed KNN graph.
3. Optionally fuse extra 3D votes from VGGT world points when available.
4. Apply shortcut suppression for weak 2D-only long jumps.
5. Build the final graph greedily, hub-first, respecting forbidden pairs,
   cycle avoidance, and physical degree constraints.
6. Return also a list of BridgeEdge objects so the existing spanning-tree pass
   can still close any remaining gaps.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from config import (
    COMPONENT_INFO,
    DEGREE_SATURATION,
    DIAMETER_CODES,
    FORBIDDEN_PAIRS,
    VGGT_3D_VOTE_WEIGHT,
)

# A pair is "adjacent" in a view if distance <= median-KNN-dist * this factor.
# Tighter than DIST_MULTIPLIER (2.5) to reduce A-C shortcuts.
ADJACENCY_FACTOR: float = 1.8


def _max_allowed_degree(code: int) -> int:
    physical_valence = COMPONENT_INFO.get(code, (2, ""))[0]
    if code in DIAMETER_CODES:
        return 0
    if DEGREE_SATURATION.get(code) == 0.0:
        return min(1, physical_valence)
    return physical_valence


def _norm(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))


def _norm_code(ca: int, cb: int) -> tuple[int, int]:
    return (min(ca, cb), max(ca, cb))


@dataclass
class BridgeEdge:
    """Lightweight drop-in replacement for CandidateEdge used by force_spanning_tree."""

    scene_node_a: str
    scene_node_b: str
    vote_count: float
    supporting_images: list = field(default_factory=list)
    forbidden_pair_flag: bool = False
    support_count: float = 0
    branch_support_count: int = 0
    knn_support_count: int = 0
    triplet_support: int = 0
    cooc_count: int = 0
    evidence_sources: list = field(default_factory=lambda: ["direct_adjacency"])
    weakened_by_reconciliation: bool = False
    direct_visible_support_count: int = 0
    vggt_support_count: int = 0
    vggt_score: float = 0.0
    vggt_min_distance: float | None = None
    vggt_port_distance: float | None = None
    vggt_axis_alignment: float = 0.0
    edge_id: str = ""


def build_direct_graph(
    scene_nodes: list,
    observed_graphs: list,
    n_total_views: int,
    vggt_edge_hints: dict | None = None,
) -> tuple[nx.Graph, list, dict]:
    """Build the fused graph directly from 2D adjacency votes plus optional 3D hints."""

    obs_to_scene: dict[str, str] = {}
    sn_code: dict[str, int] = {}

    for sn in scene_nodes:
        sn_code[sn.scene_node_id] = sn.fused_code
        for obs_id in sn.supporting_observed_ids:
            obs_to_scene[obs_id] = sn.scene_node_id

    max_degree = {
        sn.scene_node_id: _max_allowed_degree(sn.fused_code)
        for sn in scene_nodes
    }
    anchor_ids = {sid for sid, md in max_degree.items() if md >= 3}

    direct_vote_count: dict[tuple[str, str], int] = defaultdict(int)
    direct_vote_views: dict[tuple[str, str], list[str]] = defaultdict(list)

    for og in observed_graphs:
        for edge in og.edges:
            sn_a = obs_to_scene.get(edge.source_id)
            sn_b = obs_to_scene.get(edge.target_id)
            if not sn_a or not sn_b or sn_a == sn_b:
                continue
            ca, cb = sn_code.get(sn_a, 0), sn_code.get(sn_b, 0)
            if _norm_code(ca, cb) in FORBIDDEN_PAIRS:
                continue
            pair = _norm(sn_a, sn_b)
            direct_vote_count[pair] += 1
            direct_vote_views[pair].append(og.image_id)

    vggt_edge_hints = vggt_edge_hints or {}
    combined_pairs = set(direct_vote_count.keys()) | set(vggt_edge_hints.keys())

    def _combined_score(pair: tuple[str, str]) -> float:
        direct_votes = direct_vote_count.get(pair, 0)
        vggt_votes = float(vggt_edge_hints.get(pair, {}).get("vote_count", 0))
        return float(direct_votes) + vggt_votes * VGGT_3D_VOTE_WEIGHT

    shortcut_pairs: set[tuple[str, str]] = set()
    min_path_votes = 1

    for pair_ac in list(direct_vote_count.keys()):
        if vggt_edge_hints.get(pair_ac, {}).get("vote_count", 0) > 0:
            continue
        a, c = pair_ac
        for b in sn_code:
            if b in (a, c):
                continue
            if max_degree.get(b, 2) != 2:
                continue
            v_ab = direct_vote_count.get(_norm(a, b), 0)
            v_bc = direct_vote_count.get(_norm(b, c), 0)
            if v_ab >= min_path_votes and v_bc >= min_path_votes:
                shortcut_pairs.add(pair_ac)
                break

    def _priority(pair: tuple[str, str]) -> tuple:
        a, b = pair
        hub_bonus = max(max(0, max_degree.get(a, 2) - 2), max(0, max_degree.get(b, 2) - 2))
        shortcut_penalty = 1 if pair in shortcut_pairs else 0
        vggt_votes = vggt_edge_hints.get(pair, {}).get("vote_count", 0)
        return (vggt_votes, hub_bonus, -shortcut_penalty, _combined_score(pair))

    sorted_pairs = sorted(combined_pairs, key=_priority, reverse=True)

    graph = nx.Graph()
    for sn in scene_nodes:
        graph.add_node(
            sn.scene_node_id,
            code=sn.fused_code,
            name=sn.fused_name,
            confidence=sn.fused_confidence,
            support_count=sn.support_count,
            support_images=list(getattr(sn, "support_images", [])),
            max_degree=max_degree[sn.scene_node_id],
            anchor_hint_score=getattr(sn, "anchor_hint_score", 0.0),
            is_anchor=sn.scene_node_id in anchor_ids,
        )

    all_nodes = [sn.scene_node_id for sn in scene_nodes]
    uf_parent = {n: n for n in all_nodes}

    def _uf_find(x: str) -> str:
        while uf_parent[x] != x:
            uf_parent[x] = uf_parent[uf_parent[x]]
            x = uf_parent[x]
        return x

    def _uf_union(x: str, y: str) -> None:
        uf_parent[_uf_find(x)] = _uf_find(y)

    def _same_component(x: str, y: str) -> bool:
        return _uf_find(x) == _uf_find(y)

    degree_used: dict[str, int] = defaultdict(int)
    accepted_log: list[dict] = []
    rejected_log: list[dict] = []

    for pair in sorted_pairs:
        a, b = pair
        direct_votes = direct_vote_count.get(pair, 0)
        vggt_hint = vggt_edge_hints.get(pair, {})
        vggt_votes = int(vggt_hint.get("vote_count", 0))
        combined_votes = _combined_score(pair)

        if max_degree.get(a, 0) == 0 or max_degree.get(b, 0) == 0:
            rejected_log.append({"pair": list(pair), "reason": "diameter_node", "votes": combined_votes})
            continue
        if degree_used[a] >= max_degree.get(a, 2):
            rejected_log.append({"pair": list(pair), "reason": f"deg_sat_{a}", "votes": combined_votes})
            continue
        if degree_used[b] >= max_degree.get(b, 2):
            rejected_log.append({"pair": list(pair), "reason": f"deg_sat_{b}", "votes": combined_votes})
            continue
        if _same_component(a, b):
            rejected_log.append({"pair": list(pair), "reason": "would_create_cycle", "votes": combined_votes})
            continue

        is_shortcut = pair in shortcut_pairs
        degree_used[a] += 1
        degree_used[b] += 1
        _uf_union(a, b)

        supporting_images = sorted(
            set(direct_vote_views.get(pair, [])) | set(vggt_hint.get("supporting_images", []))
        )
        evidence_sources = []
        if direct_votes > 0:
            evidence_sources.append("direct_adjacency")
        if vggt_votes > 0:
            evidence_sources.append("vggt_3d")
        if is_shortcut:
            evidence_sources.append("shortcut_candidate")

        graph.add_edge(
            a,
            b,
            vote_count=combined_votes,
            support_count=combined_votes,
            supporting_images=supporting_images,
            branch_support_count=0,
            knn_support_count=direct_votes,
            vggt_support_count=vggt_votes,
            vggt_score=vggt_hint.get("score", 0.0),
            vggt_min_distance=vggt_hint.get("min_dist"),
            vggt_port_distance=vggt_hint.get("scene_port_dist", vggt_hint.get("min_port_dist")),
            vggt_axis_alignment=vggt_hint.get("scene_axis_alignment", vggt_hint.get("axis_alignment", 0.0)),
            cooc_count=0,
            triplet_support=0,
            forbidden_flag=False,
            evidence_sources=evidence_sources,
            shortcut_suppressed=is_shortcut,
            forced=False,
            forced_reason=None,
        )
        accepted_log.append(
            {
                "pair": list(pair),
                "votes": combined_votes,
                "direct_votes": direct_votes,
                "vggt_votes": vggt_votes,
                "shortcut": is_shortcut,
            }
        )

    bridge_edges: list[BridgeEdge] = []
    for a, b in sorted(combined_pairs, key=lambda pair: _combined_score(pair), reverse=True):
        pair = _norm(a, b)
        direct_votes = direct_vote_count.get(pair, 0)
        vggt_hint = vggt_edge_hints.get(pair, {})
        vggt_votes = int(vggt_hint.get("vote_count", 0))
        evidence_sources = []
        if direct_votes > 0:
            evidence_sources.append("direct_adjacency")
        if vggt_votes > 0:
            evidence_sources.append("vggt_3d")
        bridge_edges.append(
            BridgeEdge(
                scene_node_a=a,
                scene_node_b=b,
                vote_count=_combined_score(pair),
                supporting_images=sorted(
                    set(direct_vote_views.get(pair, [])) | set(vggt_hint.get("supporting_images", []))
                ),
                forbidden_pair_flag=False,
                support_count=_combined_score(pair),
                branch_support_count=0,
                knn_support_count=direct_votes,
                triplet_support=0,
                cooc_count=0,
                evidence_sources=evidence_sources,
                weakened_by_reconciliation=False,
                direct_visible_support_count=direct_votes,
                vggt_support_count=vggt_votes,
                vggt_score=float(vggt_hint.get("score", 0.0)),
                vggt_min_distance=vggt_hint.get("min_dist"),
                vggt_port_distance=vggt_hint.get("scene_port_dist", vggt_hint.get("min_port_dist")),
                vggt_axis_alignment=vggt_hint.get("scene_axis_alignment", vggt_hint.get("axis_alignment", 0.0)),
                edge_id=f"DA_{a}_{b}",
            )
        )

    n_isolated = sum(1 for n in graph.nodes() if graph.degree(n) == 0)
    is_conn = nx.is_connected(graph) if graph.number_of_nodes() > 0 else True

    debug = {
        "n_pairs_voted": len(combined_pairs),
        "n_pairs_with_vggt": sum(
            1 for hint in vggt_edge_hints.values() if hint.get("vote_count", 0) > 0
        ),
        "n_shortcut_suppressed": len(shortcut_pairs),
        "n_edges_accepted": graph.number_of_edges(),
        "n_isolated": n_isolated,
        "is_connected": is_conn,
        "adjacency_factor": ADJACENCY_FACTOR,
        "vggt_vote_weight": VGGT_3D_VOTE_WEIGHT,
        "vote_distribution": sorted(
            [
                {
                    "pair": list(pair),
                    "votes": _combined_score(pair),
                    "direct_votes": direct_vote_count.get(pair, 0),
                    "vggt_votes": vggt_edge_hints.get(pair, {}).get("vote_count", 0),
                    "shortcut": pair in shortcut_pairs,
                }
                for pair in combined_pairs
            ],
            key=lambda item: (-item["votes"], -item["vggt_votes"], -item["direct_votes"]),
        )[:40],
        "accepted_edges": accepted_log[:30],
        "rejected_edges": rejected_log[:30],
    }

    print(
        f"  [DirectGraph] voted_pairs={len(combined_pairs)}"
        f"  vggt_pairs={debug['n_pairs_with_vggt']}"
        f"  shortcuts={len(shortcut_pairs)}"
        f"  edges={graph.number_of_edges()}"
        f"  isolated={n_isolated}"
        f"  {'CONNECTED' if is_conn else 'DISCONNECTED'}"
    )
    return graph, bridge_edges, debug
