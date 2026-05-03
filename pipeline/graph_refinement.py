"""Spanning-tree forcing and graph serialization."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from config import COOC_COUNTS, COMPONENT_INFO, DIAMETER_CODES, FORBIDDEN_PAIRS, TRIPLET_COUNTS


def _max_allowed_degree(code: int) -> int:
    physical_valence = COMPONENT_INFO.get(code, (2, ""))[0]
    if code in DIAMETER_CODES:
        return 0
    return physical_valence


def force_spanning_tree(
    graph: nx.Graph,
    candidate_edges: list,
) -> tuple[nx.Graph, dict]:
    """
    Guarantee the graph is connected (spanning tree, no new cycles).

    Strategy (in order of preference):
    1. Candidate-bridge pass: among all candidate edges that cross two distinct
       connected components, pick the best ones (by evidence score) using
       Kruskal's algorithm.
    2. Structural-fallback pass: for components still unconnected after step 1,
       find the highest-COOC_COUNTS compatible pair of nodes (one per component)
       and add a synthetic forced edge.

    Degree constraints are relaxed for isolated nodes (they have no edges and
    must be connected).  Diameter-code nodes (max_degree == 0) are skipped.
    All added edges are marked forced=True.
    """
    if graph.number_of_nodes() == 0 or nx.is_connected(graph):
        return graph, {"forced_edges": [], "already_connected": True, "n_forced": 0}

    degree_used = {n: graph.degree(n) for n in graph.nodes()}
    max_deg     = {n: graph.nodes[n]["max_degree"] for n in graph.nodes()}

    # ── Union-Find over node indices ─────────────────────────────────────────
    nodes   = list(graph.nodes())
    idx_map = {n: i for i, n in enumerate(nodes)}
    parent  = list(range(len(nodes)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> bool:
        rx, ry = _find(x), _find(y)
        if rx == ry:
            return False
        parent[rx] = ry
        return True

    for a, b in graph.edges():
        _union(idx_map[a], idx_map[b])

    def _different_components(a: str, b: str) -> bool:
        return _find(idx_map[a]) != _find(idx_map[b])

    def _can_connect(a: str, b: str, override_degree: bool = False) -> bool:
        if max_deg[a] == 0 and degree_used[a] > 0:
            return False
        if max_deg[b] == 0 and degree_used[b] > 0:
            return False
        if override_degree:
            return True
        if max_deg[a] > 0 and degree_used[a] >= max_deg[a]:
            return False
        if max_deg[b] > 0 and degree_used[b] >= max_deg[b]:
            return False
        return True

    def _add_forced(a: str, b: str, attrs: dict) -> None:
        _union(idx_map[a], idx_map[b])
        degree_used[a] += 1
        degree_used[b] += 1
        graph.add_edge(a, b, forced=True, **attrs)

    forced_log: list[dict] = []

    # ── Pass 1: candidate bridges ─────────────────────────────────────────────
    def _cand_priority(edge) -> tuple:
        return (
            edge.vggt_support_count,
            edge.branch_support_count * 2 + edge.knn_support_count,
            edge.support_count,
            edge.triplet_support + edge.cooc_count,
            0 if edge.forbidden_pair_flag else 1,
        )

    for edge in sorted(candidate_edges, key=_cand_priority, reverse=True):
        a, b = edge.scene_node_a, edge.scene_node_b
        if a not in idx_map or b not in idx_map:
            continue
        if not _different_components(a, b):
            continue
        if not _can_connect(a, b):
            continue
        if edge.forbidden_pair_flag:
            continue

        _add_forced(a, b, {
            "support_count":              edge.support_count,
            "supporting_images":          edge.supporting_images,
            "branch_support_count":       edge.branch_support_count,
            "knn_support_count":          edge.knn_support_count,
            "vggt_support_count":         edge.vggt_support_count,
            "vggt_score":                 edge.vggt_score,
            "vggt_min_distance":          edge.vggt_min_distance,
            "vggt_port_distance":         edge.vggt_port_distance,
            "vggt_axis_alignment":        edge.vggt_axis_alignment,
            "cooc_count":                 edge.cooc_count,
            "triplet_support":            edge.triplet_support,
            "forbidden_flag":             edge.forbidden_pair_flag,
            "evidence_sources":           edge.evidence_sources,
            "forced_reason":              "spanning_tree_candidate_bridge",
        })
        forced_log.append({"a": a, "b": b, "via": "candidate_bridge",
                            "priority": str(_cand_priority(edge))})

        if nx.is_connected(graph):
            break

    # ── Pass 2: structural fallback (degree-respecting) ───────────────────────
    if not nx.is_connected(graph):
        _max_cooc   = max(COOC_COUNTS.values())   if COOC_COUNTS   else 1
        _max_triplet = max(TRIPLET_COUNTS.values()) if TRIPLET_COUNTS else 1

        def _norm(x: int, y: int) -> tuple:
            return (min(x, y), max(x, y))

        def _triplet_completion(a: str, b: str) -> float:
            ca = graph.nodes[a]["code"]
            cb = graph.nodes[b]["code"]
            score = 0.0
            for nb in graph.neighbors(b):
                cn = graph.nodes[nb]["code"]
                key = (min(ca, cn), cb, max(ca, cn))
                score += TRIPLET_COUNTS.get(key, 0) / _max_triplet
            for na in graph.neighbors(a):
                cn = graph.nodes[na]["code"]
                key = (min(cb, cn), ca, max(cb, cn))
                score += TRIPLET_COUNTS.get(key, 0) / _max_triplet
            return score

        def _structural_score(a: str, b: str) -> float:
            ca = graph.nodes[a]["code"]
            cb = graph.nodes[b]["code"]
            if _norm(ca, cb) in FORBIDDEN_PAIRS:
                return -1.0
            cooc    = COOC_COUNTS.get(_norm(ca, cb), 0) / _max_cooc
            support = (graph.nodes[a]["support_count"] + graph.nodes[b]["support_count"]) / 20.0
            triplet = _triplet_completion(a, b) * 0.5
            return cooc + support + triplet

        def _run_fallback_pass(override: bool) -> None:
            for _ in range(len(nodes)):
                if nx.is_connected(graph):
                    break
                comps = sorted(nx.connected_components(graph), key=len, reverse=True)
                main  = comps[0]

                best_score, best_a, best_b = -999.0, None, None
                for other in comps[1:]:
                    for a in main:
                        for b in other:
                            if not _can_connect(a, b, override_degree=override):
                                continue
                            if override:
                                ca = graph.nodes[a]["code"]
                                cb = graph.nodes[b]["code"]
                                if COOC_COUNTS.get(_norm(ca, cb), 0) == 0:
                                    continue
                            s = _structural_score(a, b)
                            if s > best_score:
                                best_score, best_a, best_b = s, a, b

                if best_a is None or best_score < 0:
                    break

                reason = "spanning_tree_override" if override else "spanning_tree_structural_fallback"
                _add_forced(best_a, best_b, {
                    "support_count":        0,
                    "supporting_images":    [],
                    "branch_support_count": 0,
                    "knn_support_count":    0,
                    "cooc_count":           0,
                    "triplet_support":      0,
                    "forbidden_flag":       False,
                    "evidence_sources":     ["structural_fallback"],
                    "forced_reason":        reason,
                    "structural_score":     best_score,
                    "override_degree":      override,
                })
                forced_log.append({
                    "a": best_a, "b": best_b,
                    "via": "structural_fallback" if not override else "override",
                    "structural_score": best_score,
                })

        _run_fallback_pass(override=False)

        if not nx.is_connected(graph):
            _run_fallback_pass(override=True)

    debug = {
        "already_connected":   False,
        "n_forced":            len(forced_log),
        "is_connected_after":  nx.is_connected(graph),
        "forced_edges":        forced_log,
    }
    print(
        f"  [SpanTree] forced {len(forced_log)} edge(s)  "
        f"connected={'Y' if debug['is_connected_after'] else 'N'}"
    )
    return graph, debug


def serialize_fused_graph(graph: nx.Graph, n_total_views: int) -> dict:
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append({
            "id": node_id,
            "code": attrs["code"],
            "name": attrs["name"],
            "confidence": attrs["confidence"],
            "support_count": attrs["support_count"],
            "support_images": attrs["support_images"],
            "degree": graph.degree(node_id),
            "max_degree": attrs["max_degree"],
            "is_anchor": attrs["is_anchor"],
        })

    edges = []
    for node_a, node_b, attrs in graph.edges(data=True):
        edges.append({
            "node_a": node_a,
            "node_b": node_b,
            "support_count": attrs.get("support_count", 0),
            "supporting_images": attrs.get("supporting_images", []),
            "branch_support_count": attrs.get("branch_support_count", 0),
            "knn_support_count": attrs.get("knn_support_count", 0),
            "vggt_support_count": attrs.get("vggt_support_count", 0),
            "vggt_score": attrs.get("vggt_score", 0.0),
            "vggt_min_distance": attrs.get("vggt_min_distance", None),
            "vggt_port_distance": attrs.get("vggt_port_distance", None),
            "vggt_axis_alignment": attrs.get("vggt_axis_alignment", 0.0),
            "cooc_count": attrs.get("cooc_count", 0),
            "triplet_support": attrs.get("triplet_support", 0),
            "forbidden_flag": attrs.get("forbidden_flag", False),
            "evidence_sources": attrs.get("evidence_sources", []),
            "expanded_by_paths": attrs.get("expanded_by_paths", []),
            "forced": attrs.get("forced", False),
            "forced_reason": attrs.get("forced_reason", None),
            "consistency_score": attrs.get("consistency_score", None),
            "consistency_n_visible": attrs.get("consistency_n_visible", None),
        })

    return {
        "total_images": n_total_views,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "is_connected": nx.is_connected(graph) if graph.number_of_nodes() > 0 else True,
        "nodes": nodes,
        "edges": edges,
    }
