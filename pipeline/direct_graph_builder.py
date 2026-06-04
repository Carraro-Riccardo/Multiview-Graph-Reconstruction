"""pipeline/direct_graph_builder.py

3D-only graph builder: edges come exclusively from VGGT edge hints.
Per-image 2D adjacency votes are no longer used.

Algorithm
---------
1. Sort VGGT hint pairs by priority (vote_count, cooc-weighted score).
2. Build the final graph greedily, respecting forbidden pairs,
   cycle avoidance, and physical degree constraints.
3. Return BridgeEdge objects for the spanning-tree pass.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from config import (
    COOC_COUNTS,
    COMPONENT_INFO,
    DIAMETER_CODES,
    FORBIDDEN_PAIRS,
)


def _max_allowed_degree(code: int) -> int:
    physical_valence = COMPONENT_INFO.get(code, (2, ""))[0]
    if code in DIAMETER_CODES:
        return 0
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
    triplet_support: int = 0
    cooc_count: int = 0
    evidence_sources: list = field(default_factory=lambda: ["vggt_3d"])
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
    """Build the fused graph from VGGT 3D edge hints only."""

    sn_code: dict[str, int] = {}
    for sn in scene_nodes:
        sn_code[sn.scene_node_id] = sn.fused_code

    max_degree = {
        sn.scene_node_id: _max_allowed_degree(sn.fused_code)
        for sn in scene_nodes
    }
    anchor_ids = {sid for sid, md in max_degree.items() if md >= 3}

    vggt_edge_hints = vggt_edge_hints or {}
    vggt_pairs = set(vggt_edge_hints.keys())

    _max_cooc = max(COOC_COUNTS.values()) if COOC_COUNTS else 1

    def _vggt_score(pair: tuple[str, str]) -> float:
        hint = vggt_edge_hints.get(pair, {})
        return float(hint.get("vote_count", 0)) * float(hint.get("score", 0.0))

    def _priority(pair: tuple[str, str]) -> tuple:
        a, b = pair
        hint = vggt_edge_hints.get(pair, {})
        ca, cb = sn_code.get(a, 0), sn_code.get(b, 0)
        cooc_norm = COOC_COUNTS.get(_norm_code(ca, cb), 0) / _max_cooc
        vggt_votes = hint.get("vote_count", 0)
        vggt_score = hint.get("score", 0.0)
        return (vggt_votes, vggt_score * (1 + cooc_norm))

    sorted_pairs = sorted(vggt_pairs, key=_priority, reverse=True)

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
    rewired_log: list[dict] = []

    def _edge_attrs(pair: tuple[str, str], combined_votes: float, vggt_hint: dict, vggt_votes: int) -> dict:
        supporting_images = sorted(set(vggt_hint.get("supporting_images", [])))
        evidence_sources = ["vggt_3d"] if vggt_votes > 0 else []
        return {
            "vote_count": combined_votes,
            "support_count": combined_votes,
            "supporting_images": supporting_images,
            "vggt_support_count": vggt_votes,
            "vggt_score": vggt_hint.get("score", 0.0),
            "vggt_min_distance": vggt_hint.get("min_dist"),
            "vggt_port_distance": vggt_hint.get("scene_port_dist", vggt_hint.get("min_port_dist")),
            "vggt_axis_alignment": vggt_hint.get("scene_axis_alignment", vggt_hint.get("axis_alignment", 0.0)),
            "cooc_count": 0,
            "triplet_support": 0,
            "forbidden_flag": False,
            "evidence_sources": evidence_sources,
            "forced": False,
            "forced_reason": None,
        }

    for pair in sorted_pairs:
        a, b = pair
        vggt_hint = vggt_edge_hints.get(pair, {})
        vggt_votes = int(vggt_hint.get("vote_count", 0))
        combined_votes = _vggt_score(pair)

        ca, cb = sn_code.get(a, 0), sn_code.get(b, 0)
        if _norm_code(ca, cb) in FORBIDDEN_PAIRS:
            rejected_log.append({"pair": list(pair), "reason": "forbidden_pair", "votes": combined_votes})
            continue
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

        degree_used[a] += 1
        degree_used[b] += 1
        _uf_union(a, b)

        graph.add_edge(
            a,
            b,
            **_edge_attrs(pair, combined_votes, vggt_hint, vggt_votes),
        )
        accepted_log.append(
            {
                "pair": list(pair),
                "votes": combined_votes,
                "vggt_votes": vggt_votes,
            }
        )

    # Local structural repair for single/sparse-view scenes: the greedy pass can
    # saturate a Tee with a weak valve edge before accepting a plausible Tee-Tee
    # bridge. If replacing that weak edge preserves connectivity and improves
    # evidence, prefer the anchor-anchor connection.
    for pair in sorted_pairs:
        a, b = pair
        if graph.has_edge(a, b):
            continue
        if a not in anchor_ids or b not in anchor_ids:
            continue
        if not nx.has_path(graph, a, b):
            continue

        vggt_hint = vggt_edge_hints.get(pair, {})
        vggt_votes = int(vggt_hint.get("vote_count", 0))
        candidate_score = _vggt_score(pair)
        if candidate_score <= 0:
            continue

        path = nx.shortest_path(graph, a, b)
        removable_edges: list[tuple[float, str, str]] = []
        for u, v in zip(path, path[1:]):
            if u not in {a, b} and v not in {a, b}:
                continue
            other = v if u in {a, b} else u
            if other in anchor_ids:
                continue
            if graph.degree(other) <= 1:
                continue
            edge_score = float(graph[u][v].get("support_count", 0.0))
            if candidate_score <= edge_score * 1.2:
                continue
            removable_edges.append((edge_score, u, v))

        if not removable_edges:
            continue

        removed_score, rem_u, rem_v = min(removable_edges, key=lambda item: item[0])
        removed_attrs = dict(graph[rem_u][rem_v])
        graph.remove_edge(rem_u, rem_v)
        degree_used[rem_u] -= 1
        degree_used[rem_v] -= 1

        if degree_used[a] >= max_degree.get(a, 2) or degree_used[b] >= max_degree.get(b, 2):
            graph.add_edge(rem_u, rem_v, **removed_attrs)
            degree_used[rem_u] += 1
            degree_used[rem_v] += 1
            continue

        graph.add_edge(a, b, **_edge_attrs(pair, candidate_score, vggt_hint, vggt_votes))
        degree_used[a] += 1
        degree_used[b] += 1
        if not nx.is_connected(graph):
            graph.remove_edge(a, b)
            degree_used[a] -= 1
            degree_used[b] -= 1
            graph.add_edge(rem_u, rem_v, **removed_attrs)
            degree_used[rem_u] += 1
            degree_used[rem_v] += 1
            continue

        rewired_log.append({
            "added": list(pair),
            "removed": [rem_u, rem_v],
            "added_score": candidate_score,
            "removed_score": removed_score,
            "reason": "anchor_bridge_rewire",
        })

    bridge_edges: list[BridgeEdge] = []
    for a, b in sorted(vggt_pairs, key=lambda pair: _vggt_score(pair), reverse=True):
        pair = _norm(a, b)
        vggt_hint = vggt_edge_hints.get(pair, {})
        vggt_votes = int(vggt_hint.get("vote_count", 0))
        bridge_edges.append(
            BridgeEdge(
                scene_node_a=a,
                scene_node_b=b,
                vote_count=_vggt_score(pair),
                supporting_images=sorted(set(vggt_hint.get("supporting_images", []))),
                forbidden_pair_flag=False,
                support_count=_vggt_score(pair),
                triplet_support=0,
                cooc_count=0,
                evidence_sources=["vggt_3d"] if vggt_votes > 0 else [],
                vggt_support_count=vggt_votes,
                vggt_score=float(vggt_hint.get("score", 0.0)),
                vggt_min_distance=vggt_hint.get("min_dist"),
                vggt_port_distance=vggt_hint.get("scene_port_dist", vggt_hint.get("min_port_dist")),
                vggt_axis_alignment=vggt_hint.get("scene_axis_alignment", vggt_hint.get("axis_alignment", 0.0)),
                edge_id=f"VGGT_{a}_{b}",
            )
        )

    n_isolated = sum(1 for n in graph.nodes() if graph.degree(n) == 0)
    is_conn = nx.is_connected(graph) if graph.number_of_nodes() > 0 else True

    debug = {
        "mode": "vggt_3d_only",
        "n_vggt_pairs": len(vggt_pairs),
        "n_pairs_with_vggt": sum(
            1 for hint in vggt_edge_hints.values() if hint.get("vote_count", 0) > 0
        ),
        "n_edges_accepted": graph.number_of_edges(),
        "n_isolated": n_isolated,
        "is_connected": is_conn,
        "vote_distribution": sorted(
            [
                {
                    "pair": list(pair),
                    "votes": _vggt_score(pair),
                    "vggt_votes": vggt_edge_hints.get(pair, {}).get("vote_count", 0),
                }
                for pair in vggt_pairs
            ],
            key=lambda item: (-item["votes"], -item["vggt_votes"]),
        )[:40],
        "accepted_edges": accepted_log[:30],
        "rejected_edges": rejected_log[:30],
        "rewired_edges": rewired_log,
    }

    print(
        f"  [DirectGraph] vggt_pairs={len(vggt_pairs)}"
        f"  edges={graph.number_of_edges()}"
        f"  isolated={n_isolated}"
        f"  {'CONNECTED' if is_conn else 'DISCONNECTED'}"
    )
    return graph, bridge_edges, debug
