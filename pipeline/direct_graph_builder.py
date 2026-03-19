"""pipeline/direct_graph_builder.py

Direct per-image adjacency graph builder.
Replaces Steps 3–6 of the pipeline (multiview_fusion, candidate_scene_graph,
structural_reconciliation, graph_refinement).

Algorithm
---------
1. For every view, map each observed detection → scene node ID (from ReID).
2. For every pair of scene nodes visible in the same view, measure the pixel
   distance between their detection centres.  If the distance is below a
   per-view threshold (median KNN distance × ADJACENCY_FACTOR), cast one
   adjacency vote for that pair.
3. After collecting votes across all views, apply shortcut suppression:
   if A–C votes exist but A–B and B–C also exist (through a degree-2 chain
   node B), deprioritise A–C so the path A–B–C is preferred.
4. Build the final graph greedily, hub-first (max_degree ≥ 3 nodes first),
   respecting forbidden pairs and physical degree constraints.
5. Return also a list of BridgeEdge objects (all voted pairs) so the existing
   force_spanning_tree pass can still close any remaining gaps.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from config import (
    COMPONENT_INFO,
    DEGREE_SATURATION,
    DIAMETER_CODES,
    FORBIDDEN_PAIRS,
)

# A pair is "adjacent" in a view if distance ≤ median-KNN-dist × this factor.
# Tighter than DIST_MULTIPLIER (2.5) to reduce A–C shortcuts.
ADJACENCY_FACTOR: float = 1.8


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── BridgeEdge ────────────────────────────────────────────────────────────────

@dataclass
class BridgeEdge:
    """Lightweight drop-in replacement for CandidateEdge used by force_spanning_tree."""
    scene_node_a: str
    scene_node_b: str
    vote_count: int
    supporting_images: list = field(default_factory=list)
    forbidden_pair_flag: bool = False
    support_count: int = 0
    branch_support_count: int = 0
    knn_support_count: int = 0
    triplet_support: int = 0
    cooc_count: int = 0
    evidence_sources: list = field(default_factory=lambda: ["direct_adjacency"])
    weakened_by_reconciliation: bool = False
    direct_visible_support_count: int = 0
    edge_id: str = ""


# ── Main entry point ──────────────────────────────────────────────────────────

def build_direct_graph(
    scene_nodes: list,
    observed_graphs: list,
    n_total_views: int,
) -> tuple[nx.Graph, list, dict]:
    """
    Build the fused graph directly from per-image scene node adjacency votes.

    Parameters
    ----------
    scene_nodes      : list of SceneNode objects (from ReID, Step 2)
    observed_graphs  : list of ObservedGraph objects (from Step 1)
    n_total_views    : total number of views

    Returns
    -------
    graph            : nx.Graph with scene nodes and directly-voted edges
    bridge_edges     : list[BridgeEdge] — ALL voted pairs (for spanning-tree)
    debug            : dict with diagnostics
    """

    # ── Build lookup tables ───────────────────────────────────────────────
    obs_to_scene: dict[str, str] = {}
    sn_code:    dict[str, int]   = {}
    sn_support: dict[str, int]   = {}
    sn_images:  dict[str, list]  = {}

    for sn in scene_nodes:
        sn_code[sn.scene_node_id]    = sn.fused_code
        sn_support[sn.scene_node_id] = sn.support_count
        sn_images[sn.scene_node_id]  = list(getattr(sn, "support_images", []))
        for obs_id in sn.supporting_observed_ids:
            obs_to_scene[obs_id] = sn.scene_node_id

    max_degree = {
        sn.scene_node_id: _max_allowed_degree(sn.fused_code)
        for sn in scene_nodes
    }
    anchor_ids = {sid for sid, md in max_degree.items() if md >= 3}

    # ── Step 1 — per-view adjacency votes ────────────────────────────────
    # Use the KNN edges already built in each ObservedGraph (K=3, per-view).
    # Each edge = two detections that are KNN neighbours in that view.
    # We map detection ids → scene node ids and aggregate votes per pair.
    vote_count: dict[tuple, int]  = defaultdict(int)
    vote_views: dict[tuple, list] = defaultdict(list)

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
            vote_count[pair] += 1
            vote_views[pair].append(og.image_id)

    # ── Step 2 — shortcut suppression ────────────────────────────────────
    # If both A–B and B–C have votes (B a degree-2 chain node), the direct
    # edge A–C is a likely shortcut.  Mark it — it will be considered last.
    shortcut_pairs: set[tuple] = set()
    MIN_PATH_VOTES = 1

    for pair_ac in list(vote_count.keys()):
        a, c = pair_ac
        for b in sn_code:
            if b in (a, c):
                continue
            if max_degree.get(b, 2) != 2:       # only suppress via chain nodes
                continue
            v_ab = vote_count.get(_norm(a, b), 0)
            v_bc = vote_count.get(_norm(b, c), 0)
            if v_ab >= MIN_PATH_VOTES and v_bc >= MIN_PATH_VOTES:
                shortcut_pairs.add(pair_ac)
                break

    # ── Step 3 — hub-first greedy graph building ──────────────────────────
    def _priority(pair: tuple) -> tuple:
        a, b = pair
        hub_bonus  = max(max(0, max_degree.get(a, 2) - 2),
                         max(0, max_degree.get(b, 2) - 2))
        is_shortcut = 1 if pair in shortcut_pairs else 0
        return (hub_bonus, -is_shortcut, vote_count[pair])

    sorted_pairs = sorted(vote_count.keys(), key=_priority, reverse=True)

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

    # Union-Find for cycle detection
    all_nodes = [sn.scene_node_id for sn in scene_nodes]
    _uf_parent = {n: n for n in all_nodes}

    def _uf_find(x: str) -> str:
        while _uf_parent[x] != x:
            _uf_parent[x] = _uf_parent[_uf_parent[x]]
            x = _uf_parent[x]
        return x

    def _uf_union(x: str, y: str) -> None:
        _uf_parent[_uf_find(x)] = _uf_find(y)

    def _same_component(x: str, y: str) -> bool:
        return _uf_find(x) == _uf_find(y)

    degree_used: dict[str, int] = defaultdict(int)
    accepted_log: list[dict]    = []
    rejected_log: list[dict]    = []

    for pair in sorted_pairs:
        a, b = pair
        vc   = vote_count[pair]

        # Diameter nodes never connect
        if max_degree.get(a, 0) == 0 or max_degree.get(b, 0) == 0:
            rejected_log.append({"pair": list(pair), "reason": "diameter_node", "votes": vc})
            continue

        # Degree saturation
        if degree_used[a] >= max_degree.get(a, 2):
            rejected_log.append({"pair": list(pair), "reason": f"deg_sat_{a}", "votes": vc})
            continue
        if degree_used[b] >= max_degree.get(b, 2):
            rejected_log.append({"pair": list(pair), "reason": f"deg_sat_{b}", "votes": vc})
            continue

        # Cycle check: skip if both endpoints already connected
        if _same_component(a, b):
            rejected_log.append({"pair": list(pair), "reason": "would_create_cycle", "votes": vc})
            continue

        is_shortcut = pair in shortcut_pairs
        degree_used[a] += 1
        degree_used[b] += 1
        _uf_union(a, b)
        graph.add_edge(
            a, b,
            vote_count=vc,
            support_count=vc,
            supporting_images=vote_views[pair],
            branch_support_count=0,
            knn_support_count=vc,
            cooc_count=0,
            triplet_support=0,
            forbidden_flag=False,
            evidence_sources=["direct_adjacency"] + (["shortcut_candidate"] if is_shortcut else []),
            shortcut_suppressed=is_shortcut,
            forced=False,
            forced_reason=None,
        )
        accepted_log.append({"pair": list(pair), "votes": vc, "shortcut": is_shortcut})

    # ── Build BridgeEdge list for spanning-tree fallback ─────────────────
    bridge_edges: list[BridgeEdge] = [
        BridgeEdge(
            scene_node_a=a,
            scene_node_b=b,
            vote_count=vc,
            supporting_images=vote_views[(min(a,b), max(a,b))],
            forbidden_pair_flag=False,
            support_count=vc,
            branch_support_count=0,
            knn_support_count=vc,
            triplet_support=0,
            cooc_count=0,
            evidence_sources=["direct_adjacency"],
            weakened_by_reconciliation=False,
            direct_visible_support_count=vc,
            edge_id=f"DA_{a}_{b}",
        )
        for (a, b), vc in sorted(
            vote_count.items(), key=lambda x: x[1], reverse=True
        )
    ]

    # ── Debug summary ─────────────────────────────────────────────────────
    n_isolated  = sum(1 for n in graph.nodes() if graph.degree(n) == 0)
    is_conn     = nx.is_connected(graph) if graph.number_of_nodes() > 0 else True

    debug = {
        "n_pairs_voted":         len(vote_count),
        "n_shortcut_suppressed": len(shortcut_pairs),
        "n_edges_accepted":      graph.number_of_edges(),
        "n_isolated":            n_isolated,
        "is_connected":          is_conn,
        "adjacency_factor":      ADJACENCY_FACTOR,
        "vote_distribution": sorted(
            [
                {
                    "pair":     list(p),
                    "votes":    v,
                    "shortcut": p in shortcut_pairs,
                }
                for p, v in vote_count.items()
            ],
            key=lambda x: -x["votes"],
        )[:40],
        "accepted_edges": accepted_log[:30],
        "rejected_edges": rejected_log[:30],
    }

    print(
        f"  [DirectGraph] voted_pairs={len(vote_count)}"
        f"  shortcuts={len(shortcut_pairs)}"
        f"  edges={graph.number_of_edges()}"
        f"  isolated={n_isolated}"
        f"  {'CONNECTED' if is_conn else 'DISCONNECTED'}"
    )
    return graph, bridge_edges, debug
