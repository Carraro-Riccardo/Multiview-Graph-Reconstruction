"""Step 6/7 - global refinement and spanning-tree forcing."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from config import COOC_COUNTS, COMPONENT_INFO, DEGREE_SATURATION, DIAMETER_CODES, FORBIDDEN_PAIRS, TRIPLET_COUNTS


def _max_allowed_degree(code: int) -> int:
    physical_valence = COMPONENT_INFO.get(code, (2, ""))[0]
    if code in DIAMETER_CODES:
        return 0
    if DEGREE_SATURATION.get(code) == 0.0:
        return min(1, physical_valence)
    return physical_valence


def _edge_priority(edge, n_total_views: int) -> tuple:
    direct_support = edge.branch_support_count * 2 + edge.knn_support_count
    prior_support = edge.triplet_support + edge.cooc_count
    shortcut_penalty = 1 if edge.weakened_by_reconciliation else 0
    forbidden_penalty = 1 if edge.forbidden_pair_flag else 0
    normalized_views = edge.support_count / max(n_total_views, 1)
    return (
        direct_support,
        normalized_views,
        prior_support,
        -shortcut_penalty,
        -forbidden_penalty,
    )


def refine_candidate_graph(
    scene_nodes: list,
    anchor_nodes: list,
    scene_branches: list,
    candidate_edges: list,
    n_total_views: int,
) -> tuple[nx.Graph, dict]:
    anchor_ids = {anchor.scene_node_id for anchor in anchor_nodes}
    degree_used: dict = defaultdict(int)
    max_degree = {
        scene_node.scene_node_id: _max_allowed_degree(scene_node.fused_code)
        for scene_node in scene_nodes
    }

    graph = nx.Graph()
    for scene_node in scene_nodes:
        graph.add_node(
            scene_node.scene_node_id,
            code=scene_node.fused_code,
            name=scene_node.fused_name,
            confidence=scene_node.fused_confidence,
            support_count=scene_node.support_count,
            support_images=scene_node.support_images,
            max_degree=max_degree[scene_node.scene_node_id],
            anchor_hint_score=scene_node.anchor_hint_score,
            is_anchor=scene_node.scene_node_id in anchor_ids,
        )

    debug = {
        "accepted_edges": [],
        "rejected_edges": [],
        "branch_acceptance": [],
        "branch_pruning": [],
        "triplet_validation": [],
    }

    candidate_by_pair = {
        (min(edge.scene_node_a, edge.scene_node_b), max(edge.scene_node_a, edge.scene_node_b)): edge
        for edge in candidate_edges
    }
    accepted_pairs: set[tuple[str, str]] = set()

    sorted_branches = sorted(
        scene_branches,
        key=lambda branch: (
            -branch.support_count,
            -len(branch.attachment_anchors),
            -len(branch.fused_ordered_scene_node_ids),
            -branch.confidence,
        ),
    )

    for scene_branch in sorted_branches:
        ordered_nodes = scene_branch.fused_ordered_scene_node_ids
        if len(ordered_nodes) < 2:
            debug["branch_pruning"].append({
                "scene_branch_id": scene_branch.scene_branch_id,
                "reason": "branch_has_fewer_than_two_scene_nodes",
            })
            continue

        branch_edges = []
        branch_reject_reason = None
        for idx in range(len(ordered_nodes) - 1):
            pair = (min(ordered_nodes[idx], ordered_nodes[idx + 1]),
                    max(ordered_nodes[idx], ordered_nodes[idx + 1]))
            edge = candidate_by_pair.get(pair)
            if edge is None:
                branch_reject_reason = f"missing_candidate_edge_for_{pair}"
                break
            if edge.weakened_by_reconciliation:
                branch_reject_reason = f"shortcut_edge_replaced_by_hidden_path_{pair}"
                break
            if edge.forbidden_pair_flag and edge.branch_support_count == 0:
                branch_reject_reason = f"forbidden_without_branch_support_{pair}"
                break

            a, b = pair
            if degree_used[a] >= max_degree[a]:
                branch_reject_reason = f"degree_saturation_at_{a}"
                break
            if degree_used[b] >= max_degree[b]:
                branch_reject_reason = f"degree_saturation_at_{b}"
                break
            branch_edges.append((pair, edge))

        if branch_reject_reason is not None:
            debug["branch_pruning"].append({
                "scene_branch_id": scene_branch.scene_branch_id,
                "reason": branch_reject_reason,
                "branch_support_count": scene_branch.support_count,
            })
            continue

        for pair, edge in branch_edges:
            if pair in accepted_pairs:
                continue
            a, b = pair
            degree_used[a] += 1
            degree_used[b] += 1
            graph.add_edge(
                a,
                b,
                support_count=edge.support_count,
                supporting_images=edge.supporting_images,
                branch_support_count=edge.branch_support_count,
                knn_support_count=edge.knn_support_count,
                cooc_count=edge.cooc_count,
                triplet_support=edge.triplet_support,
                forbidden_flag=edge.forbidden_pair_flag,
                evidence_sources=edge.evidence_sources,
                weakened_by_reconciliation=edge.weakened_by_reconciliation,
                expanded_by_paths=edge.expanded_by_paths,
            )
            accepted_pairs.add(pair)
            debug["accepted_edges"].append({
                "edge_id": edge.edge_id,
                "nodes": [a, b],
                "accepted_via": "scene_branch",
                "evidence_sources": edge.evidence_sources,
            })

        debug["branch_acceptance"].append({
            "scene_branch_id": scene_branch.scene_branch_id,
            "accepted_pairs": [list(pair) for pair, _ in branch_edges],
            "support_count": scene_branch.support_count,
            "attachment_anchors": scene_branch.attachment_anchors,
            "reasons": scene_branch.acceptance_reasons,
        })

    residual_edges = sorted(
        [
            edge for edge in candidate_edges
            if (min(edge.scene_node_a, edge.scene_node_b), max(edge.scene_node_a, edge.scene_node_b))
            not in accepted_pairs
        ],
        key=lambda edge: _edge_priority(edge, n_total_views),
        reverse=True,
    )

    for edge in residual_edges:
        pair = (min(edge.scene_node_a, edge.scene_node_b), max(edge.scene_node_a, edge.scene_node_b))
        a, b = pair
        reject_reason = None
        if edge.weakened_by_reconciliation:
            reject_reason = "shortcut_edge_weakened_by_reconciliation"
        elif max_degree[a] == 0 or max_degree[b] == 0:
            reject_reason = "diameter_label_node"
        elif degree_used[a] >= max_degree[a]:
            reject_reason = f"degree_saturation_at_{a}"
        elif degree_used[b] >= max_degree[b]:
            reject_reason = f"degree_saturation_at_{b}"
        elif edge.forbidden_pair_flag and edge.branch_support_count == 0 and edge.triplet_support == 0:
            reject_reason = "forbidden_pair_without_branch_or_triplet_support"

        if reject_reason is not None:
            debug["rejected_edges"].append({
                "edge_id": edge.edge_id,
                "nodes": [a, b],
                "reason": reject_reason,
                "evidence_sources": edge.evidence_sources,
                "reconciliation_notes": edge.reconciliation_notes,
            })
            continue

        degree_used[a] += 1
        degree_used[b] += 1
        graph.add_edge(
            a,
            b,
            support_count=edge.support_count,
            supporting_images=edge.supporting_images,
            branch_support_count=edge.branch_support_count,
            knn_support_count=edge.knn_support_count,
            cooc_count=edge.cooc_count,
            triplet_support=edge.triplet_support,
            forbidden_flag=edge.forbidden_pair_flag,
            evidence_sources=edge.evidence_sources,
            weakened_by_reconciliation=edge.weakened_by_reconciliation,
            expanded_by_paths=edge.expanded_by_paths,
        )
        debug["accepted_edges"].append({
            "edge_id": edge.edge_id,
            "nodes": [a, b],
            "accepted_via": "residual_candidate",
            "evidence_sources": edge.evidence_sources,
        })

    for node_id in graph.nodes():
        neighbours = list(graph.neighbors(node_id))
        if len(neighbours) < 2:
            continue
        center_code = graph.nodes[node_id]["code"]
        for idx_a in range(len(neighbours)):
            for idx_b in range(idx_a + 1, len(neighbours)):
                code_a = graph.nodes[neighbours[idx_a]]["code"]
                code_b = graph.nodes[neighbours[idx_b]]["code"]
                triplet_key = (min(code_a, code_b), center_code, max(code_a, code_b))
                debug["triplet_validation"].append({
                    "center": node_id,
                    "triplet_key": list(triplet_key),
                    "triplet_count": TRIPLET_COUNTS.get(triplet_key, 0),
                    "validated": TRIPLET_COUNTS.get(triplet_key, 0) > 0,
                })

    debug["is_connected"] = nx.is_connected(graph) if graph.number_of_nodes() > 0 else True
    debug["n_isolated"] = sum(1 for node_id in graph.nodes() if graph.degree(node_id) == 0)

    print(
        f"  [Refinement] accepted={graph.number_of_edges()} "
        f"pruned_branches={len(debug['branch_pruning'])} "
        f"rejected_edges={len(debug['rejected_edges'])}"
    )
    return graph, debug


def force_spanning_tree(
    graph: nx.Graph,
    candidate_edges: list,
) -> tuple[nx.Graph, dict]:
    """
    Step 7 — guarantee the graph is a spanning tree (connected, no new cycles).

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

    # Initialise union-find from current edges
    for a, b in graph.edges():
        _union(idx_map[a], idx_map[b])

    def _different_components(a: str, b: str) -> bool:
        return _find(idx_map[a]) != _find(idx_map[b])

    def _can_connect(a: str, b: str) -> bool:
        """True if neither endpoint is a saturated diameter node."""
        if max_deg[a] == 0 and degree_used[a] > 0:   # diameter node already used
            return False
        if max_deg[b] == 0 and degree_used[b] > 0:
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

        _add_forced(a, b, {
            "support_count":              edge.support_count,
            "supporting_images":          edge.supporting_images,
            "branch_support_count":       edge.branch_support_count,
            "knn_support_count":          edge.knn_support_count,
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

    # ── Pass 2: structural fallback ───────────────────────────────────────────
    if not nx.is_connected(graph):
        _max_cooc = max(COOC_COUNTS.values()) if COOC_COUNTS else 1

        def _norm(x: int, y: int) -> tuple:
            return (min(x, y), max(x, y))

        def _structural_score(a: str, b: str) -> float:
            ca = graph.nodes[a]["code"]
            cb = graph.nodes[b]["code"]
            if _norm(ca, cb) in FORBIDDEN_PAIRS:
                return -1.0
            cooc    = COOC_COUNTS.get(_norm(ca, cb), 0) / _max_cooc
            support = (graph.nodes[a]["support_count"] + graph.nodes[b]["support_count"]) / 20.0
            return cooc + support

        for _ in range(len(nodes)):          # safety cap
            if nx.is_connected(graph):
                break
            comps = sorted(nx.connected_components(graph), key=len, reverse=True)
            main  = comps[0]

            best_score, best_a, best_b = -999.0, None, None
            for other in comps[1:]:
                for a in main:
                    for b in other:
                        if not _can_connect(a, b):
                            continue
                        s = _structural_score(a, b)
                        if s > best_score:
                            best_score, best_a, best_b = s, a, b

            if best_a is None:
                break   # no valid pair found at all

            _add_forced(best_a, best_b, {
                "support_count":        0,
                "supporting_images":    [],
                "branch_support_count": 0,
                "knn_support_count":    0,
                "cooc_count":           0,
                "triplet_support":      0,
                "forbidden_flag":       False,
                "evidence_sources":     ["structural_fallback"],
                "forced_reason":        "spanning_tree_structural_fallback",
                "structural_score":     best_score,
            })
            forced_log.append({"a": best_a, "b": best_b, "via": "structural_fallback",
                                "structural_score": best_score})

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
            "cooc_count": attrs.get("cooc_count", 0),
            "triplet_support": attrs.get("triplet_support", 0),
            "forbidden_flag": attrs.get("forbidden_flag", False),
            "evidence_sources": attrs.get("evidence_sources", []),
            "weakened_by_reconciliation": attrs.get("weakened_by_reconciliation", False),
            "expanded_by_paths": attrs.get("expanded_by_paths", []),
            "forced": attrs.get("forced", False),
            "forced_reason": attrs.get("forced_reason", None),
        })

    return {
        "total_images": n_total_views,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "is_connected": nx.is_connected(graph) if graph.number_of_nodes() > 0 else True,
        "nodes": nodes,
        "edges": edges,
    }
