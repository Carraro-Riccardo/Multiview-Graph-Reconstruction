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


def _edge_priority(edge, n_total_views: int, max_degree: dict | None = None) -> tuple:
    direct_support = edge.branch_support_count * 2 + edge.knn_support_count
    prior_support = edge.triplet_support + edge.cooc_count
    shortcut_penalty = 1 if edge.weakened_by_reconciliation else 0
    forbidden_penalty = 1 if edge.forbidden_pair_flag else 0
    normalized_views = edge.support_count / max(n_total_views, 1)
    # Hub bonus: edges incident to high-valence nodes (Tee, Croce, …) go first.
    # Bonus = max extra valence beyond 2, so Tee(3)→1, Croce(4)→2, others→0.
    hub_bonus = 0
    if max_degree:
        da = max_degree.get(edge.scene_node_a, 2)
        db = max_degree.get(edge.scene_node_b, 2)
        hub_bonus = max(max(0, da - 2), max(0, db - 2))
    return (
        hub_bonus,
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

    # ── Hub pre-claim pass ────────────────────────────────────────────────────
    # Hub nodes (max_degree >= 3: Tee FL, Croce, …) are the structural backbone.
    # Before branch/residual processing consumes their neighbours, pre-claim their
    # best-evidenced connections (knn_support >= 2) so branch processing cannot
    # saturate hub-adjacent nodes before the hub fills its slots.
    hub_preclaim_log = []
    hub_cands = sorted(
        [
            e for e in candidate_edges
            if not e.forbidden_pair_flag
            and not e.weakened_by_reconciliation
            and e.knn_support_count >= 2
            and (
                max_degree.get(e.scene_node_a, 2) >= 3
                or max_degree.get(e.scene_node_b, 2) >= 3
            )
        ],
        key=lambda e: (
            max(max_degree.get(e.scene_node_a, 2), max_degree.get(e.scene_node_b, 2)),
            e.branch_support_count * 2 + e.knn_support_count,
            e.support_count,
        ),
        reverse=True,
    )
    for edge in hub_cands:
        pair = (min(edge.scene_node_a, edge.scene_node_b),
                max(edge.scene_node_a, edge.scene_node_b))
        a, b = pair
        if pair in accepted_pairs:
            continue
        if degree_used[a] >= max_degree[a] or degree_used[b] >= max_degree[b]:
            continue
        degree_used[a] += 1
        degree_used[b] += 1
        graph.add_edge(
            a, b,
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
        hub_preclaim_log.append({"edge_id": edge.edge_id, "nodes": [a, b],
                                  "accepted_via": "hub_preclaim"})
        debug["accepted_edges"].append({"edge_id": edge.edge_id, "nodes": [a, b],
                                         "accepted_via": "hub_preclaim",
                                         "evidence_sources": edge.evidence_sources})
    debug["hub_preclaim"] = hub_preclaim_log

    def _branch_hub_score(branch) -> int:
        """Sum of extra valence beyond 2 for each hub node in the branch."""
        return sum(
            max(0, max_degree.get(nid, 2) - 2)
            for nid in branch.fused_ordered_scene_node_ids
        )

    sorted_branches = sorted(
        scene_branches,
        key=lambda branch: (
            -_branch_hub_score(branch),   # hub-rich branches go first
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
            if edge.forbidden_pair_flag:
                branch_reject_reason = f"forbidden_pair_{pair}"
                break

            a, b = pair
            # If this pair was already pre-claimed (hub pass), it is already in
            # the graph and degree_used — don't check degree again, just include
            # it so the branch can pass through without being rejected.
            if pair not in accepted_pairs:
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
        key=lambda edge: _edge_priority(edge, n_total_views, max_degree),
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
        elif edge.forbidden_pair_flag:
            reject_reason = "forbidden_pair"

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


def fix_boutbic_antipattern(
    graph: nx.Graph,
) -> tuple[nx.Graph, list]:
    """Remove structurally impossible (15, 95, 15) patterns from the graph.

    In the full dataset, Bout bic. (code 95) NEVER connects two Saracinesca
    (code 15) — the pattern has zero GT occurrences.  When the greedy builder
    produces such a pattern (usually because a marginal vote tied 95 to two
    different 15 nodes), the weaker of the two 15--95 edges is removed so the
    freed Saracinesca can reconnect via the spanning tree.

    "Weaker" is defined as lower (knn_support_count + vggt_support_count).
    Ties are broken by removing the edge with the lower vote_count / support_count.
    """
    _BOUTBIC  = 95
    _SARAC    = 15
    removed: list[dict] = []

    for nid in list(graph.nodes()):
        if graph.nodes[nid].get("code") != _BOUTBIC:
            continue
        if graph.degree(nid) != 2:
            continue
        nb0, nb1 = list(graph.neighbors(nid))
        if graph.nodes[nb0].get("code") != _SARAC or graph.nodes[nb1].get("code") != _SARAC:
            continue

        # Both neighbours are Saracinesca — structurally impossible
        e0 = graph.edges[nid, nb0]
        e1 = graph.edges[nid, nb1]

        def _strength(e: dict) -> tuple:
            return (
                e.get("knn_support_count", 0) + e.get("vggt_support_count", 0),
                e.get("vote_count", e.get("support_count", 0)),
            )

        if _strength(e0) <= _strength(e1):
            weaker_nb = nb0
        else:
            weaker_nb = nb1

        graph.remove_edge(nid, weaker_nb)
        # Seal the boutbic: it now holds its one valid connection.
        # Lowering max_degree to 1 prevents force_spanning_tree from routing
        # the freed Saracinesca back through this same boutbic node.
        graph.nodes[nid]["max_degree"] = 1

        removed.append({
            "boutbic_node":   nid,
            "removed_edge":   [nid, weaker_nb],
            "kept_edge":      [nid, nb1 if weaker_nb == nb0 else nb0],
            "reason":         "boutbic_15_95_15_antipattern",
        })
        print(f"  [BoutBicFix] removed impossible edge {nid}--{weaker_nb} (15-95-15 antipattern)")

    return graph, removed


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

    def _can_connect(a: str, b: str, override_degree: bool = False) -> bool:
        """True if both endpoints can accept one more edge.

        override_degree=True is used as absolute last resort: allows exceeding
        max_degree but still blocks diameter-code nodes that are already used.
        """
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
        da = max_deg.get(edge.scene_node_a, 2)
        db = max_deg.get(edge.scene_node_b, 2)
        hub_bonus = max(max(0, da - 2), max(0, db - 2))
        return (
            edge.vggt_support_count,
            hub_bonus,
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
            """Sum of known-triplet counts completed by adding edge A--B.

            For each existing neighbor N of B (B is the hub): check the
            triplet (ca, cb, cn) where cb is the center.
            For each existing neighbor N of A (A is the hub): check the
            triplet (cb, ca, cn) where ca is the center.
            """
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
            # Co-occurrence: how often this code pair appears adjacent in the dataset
            cooc    = COOC_COUNTS.get(_norm(ca, cb), 0) / _max_cooc
            support = (graph.nodes[a]["support_count"] + graph.nodes[b]["support_count"]) / 20.0
            # Hub bonus: reduced weight so COOC/triplet priors can compete
            da = max_deg.get(a, 2)
            db = max_deg.get(b, 2)
            hub_bonus = max(max(0, da - 2), max(0, db - 2)) * 0.3
            # Triplet completion: reward edges that complete known 3-component patterns
            triplet = _triplet_completion(a, b) * 0.5
            return cooc + support + hub_bonus + triplet

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
                            # In override mode, only allow code pairs with at least
                            # one observed co-occurrence — never create absurd edges.
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

        # Degree-respecting pass
        _run_fallback_pass(override=False)

        # Override pass: last resort for still-isolated nodes — allow exceeding
        # max_degree only when the code pair has at least one co-occurrence in the
        # dataset (COOC_COUNTS > 0), so we never produce physically absurd edges.
        if not nx.is_connected(graph):
            _run_fallback_pass(override=True)

    # Note: if the graph is still disconnected here, the remaining isolated nodes
    # could not be connected without violating physical degree constraints or
    # adding forbidden-pair edges.  We intentionally accept disconnection rather
    # than produce a physically impossible graph.  These nodes are likely
    # phantom detections from re-ID over-segmentation.

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
            "weakened_by_reconciliation": attrs.get("weakened_by_reconciliation", False),
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
