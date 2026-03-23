"""Step 7.5 – spatial consistency pruning.

After the spanning-tree pass, every edge in the fused graph is projected back
onto each source image.  Edges that are spatially violated in the majority of
views where both endpoints are visible are pruned (or swapped for a better
candidate bridge if removing would disconnect the graph).

An edge is a prune candidate when ALL of the following hold:
    consistency  < MIN_CONSISTENCY        (violated in >60% of visible views)
    support_count <= LOW_SUPPORT_THRESHOLD (weak original evidence)

Consistency metric per edge:
    n_views_where_both_visible_AND_dist <= cutoff*FACTOR
    -------------------------------------------------------
    n_views_where_both_visible
"""

from __future__ import annotations

import math
from collections import defaultdict

import networkx as nx
import numpy as np

from config import DIST_MULTIPLIER

# ── Config ─────────────────────────────────────────────────────────────────────
# Same spatial factor used in instance_spatial_edges.py
CONSISTENCY_CUTOFF_FACTOR: float = 1.8

# Edge is a prune candidate if consistent in fewer than this fraction of visible views
MIN_CONSISTENCY: float = 0.40

# Only prune edges with at most this many supporting views.
# High-support edges (e.g. support_count=8) earned their place even if
# occasionally violated by an occluded/partial view.
LOW_SUPPORT_THRESHOLD: int = 2
# ───────────────────────────────────────────────────────────────────────────────


def _build_position_maps(
    scene_nodes: list,
    observed_graphs: list,
) -> tuple[dict, dict, dict]:
    """
    Returns:
      sn_pos_per_view : (scene_node_id, image_id) -> (cx, cy)
      view_cutoffs    : image_id -> KNN-median-distance cutoff  (same as pipeline)
      sn_to_views     : scene_node_id -> set of image_ids
    """
    obs_to_scene: dict[str, str] = {}
    for sn in scene_nodes:
        for obs_id in sn.supporting_observed_ids:
            obs_to_scene[obs_id] = sn.scene_node_id

    sn_pos_per_view: dict[tuple, tuple] = {}
    for og in observed_graphs:
        for obs_id, obs_node in og.nodes.items():
            sn_id = obs_to_scene.get(obs_id)
            if sn_id is not None:
                key = (sn_id, og.image_id)
                if key not in sn_pos_per_view:
                    sn_pos_per_view[key] = obs_node.center

    view_cutoffs: dict[str, float] = {}
    for og in observed_graphs:
        centers = np.array([n.center for n in og.nodes.values()])
        if len(centers) < 2:
            view_cutoffs[og.image_id] = float("inf")
        else:
            dists = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            view_cutoffs[og.image_id] = float(np.median(dists.min(axis=1))) * DIST_MULTIPLIER

    sn_to_views: dict[str, set] = defaultdict(set)
    for (sn_id, img_id) in sn_pos_per_view:
        sn_to_views[sn_id].add(img_id)

    return sn_pos_per_view, view_cutoffs, sn_to_views


def _edge_consistency(
    a: str,
    b: str,
    sn_pos_per_view: dict,
    view_cutoffs: dict,
    sn_to_views: dict,
) -> dict:
    """Compute consistency stats for a single edge (a, b)."""
    shared = sn_to_views.get(a, set()) & sn_to_views.get(b, set())
    n_visible = n_supported = n_violated = 0
    for img_id in shared:
        pa = sn_pos_per_view.get((a, img_id))
        pb = sn_pos_per_view.get((b, img_id))
        if pa is None or pb is None:
            continue
        dist = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        cutoff = view_cutoffs.get(img_id, float("inf"))
        n_visible += 1
        if dist <= cutoff * CONSISTENCY_CUTOFF_FACTOR:
            n_supported += 1
        else:
            n_violated += 1
    consistency = n_supported / n_visible if n_visible > 0 else None
    return {
        "consistency": consistency,
        "n_visible": n_visible,
        "n_supported": n_supported,
        "n_violated": n_violated,
    }


def run_consistency_pass(
    scene_nodes: list,
    observed_graphs: list,
    fused_graph: nx.Graph,
    candidate_edges: list,
) -> tuple[nx.Graph, dict]:
    """
    Main entry point.  Modifies fused_graph in-place (edges may be removed or
    replaced) and annotates every remaining edge with 'consistency_score' and
    'consistency_n_visible' attributes.

    Returns (fused_graph, debug_dict).
    """
    sn_pos_per_view, view_cutoffs, sn_to_views = _build_position_maps(
        scene_nodes, observed_graphs
    )

    # ── Compute consistency for every current edge ────────────────────────────
    edge_cons: dict[tuple, dict] = {}
    for a, b in list(fused_graph.edges()):
        stats = _edge_consistency(a, b, sn_pos_per_view, view_cutoffs, sn_to_views)
        edge_cons[(a, b)] = stats
        fused_graph[a][b]["consistency_score"]    = stats["consistency"]
        fused_graph[a][b]["consistency_n_visible"] = stats["n_visible"]

    # ── Identify prune candidates ─────────────────────────────────────────────
    prune_candidates = []
    for (a, b), stats in edge_cons.items():
        cons    = stats["consistency"]
        support = fused_graph[a][b].get("support_count", 0)
        if cons is None or cons >= MIN_CONSISTENCY:
            continue
        if support > LOW_SUPPORT_THRESHOLD:
            continue
        prune_candidates.append((a, b, cons, support, stats))

    # Worst consistency first
    prune_candidates.sort(key=lambda x: x[2])

    # ── Candidate edge index (sorted best-first for swap search) ──────────────
    def _cand_score(e) -> tuple:
        return (
            e.branch_support_count * 2 + e.knn_support_count,
            e.support_count,
            e.direct_visible_support_count,
        )

    sorted_cands = sorted(candidate_edges, key=_cand_score, reverse=True)

    debug = {
        "n_edges_checked":   len(edge_cons),
        "n_prune_candidates": len(prune_candidates),
        "pruned":             [],
        "swapped":            [],
        "kept_for_connectivity": [],
        "edge_consistency":  [
            {
                "edge": [a, b],
                "consistency": stats["consistency"],
                "n_visible":   stats["n_visible"],
                "n_violated":  stats["n_violated"],
                "support_count": fused_graph[a][b].get("support_count", 0)
                    if fused_graph.has_edge(a, b) else "removed",
            }
            for (a, b), stats in sorted(
                edge_cons.items(),
                key=lambda x: (x[1]["consistency"] if x[1]["consistency"] is not None else 2.0),
            )
        ],
    }

    # ── Prune / swap loop ─────────────────────────────────────────────────────
    for a, b, cons, support, stats in prune_candidates:
        if not fused_graph.has_edge(a, b):
            continue  # already removed by a previous iteration

        saved_attrs = dict(fused_graph[a][b])
        fused_graph.remove_edge(a, b)

        if fused_graph.number_of_nodes() == 0 or nx.is_connected(fused_graph):
            debug["pruned"].append({
                "edge": [a, b],
                "consistency": cons,
                "support":     support,
                "n_visible":   stats["n_visible"],
                "n_violated":  stats["n_violated"],
            })
            print(
                f"    [Consistency] pruned {a}-{b}  "
                f"cons={cons:.0%} sup={support} vis={stats['n_visible']} vio={stats['n_violated']}"
            )
            continue

        # ── Disconnected: look for a better bridge ────────────────────────────
        comp_of: dict[str, int] = {}
        for i, comp in enumerate(nx.connected_components(fused_graph)):
            for n in comp:
                comp_of[n] = i
        ca = comp_of.get(a, -1)
        cb = comp_of.get(b, -2)

        max_deg  = {n: fused_graph.nodes[n]["max_degree"] for n in fused_graph.nodes()}
        deg_used = dict(fused_graph.degree())

        best_swap = None
        for cand in sorted_cands:
            na, nb = cand.scene_node_a, cand.scene_node_b
            if na not in comp_of or nb not in comp_of:
                continue
            # Must bridge the exact two disconnected components
            if {comp_of[na], comp_of[nb]} != {ca, cb}:
                continue
            # Not the same edge we just removed
            if {na, nb} == {a, b}:
                continue
            if cand.forbidden_pair_flag:
                continue
            # Diameter nodes blocked
            if max_deg.get(na, 0) == 0 or max_deg.get(nb, 0) == 0:
                continue
            # Respect degree saturation — don't add edges to already-full nodes
            if max_deg.get(na, 0) > 0 and deg_used.get(na, 0) >= max_deg.get(na, 0):
                continue
            if max_deg.get(nb, 0) > 0 and deg_used.get(nb, 0) >= max_deg.get(nb, 0):
                continue
            # Require at least some direct spatial evidence
            if cand.direct_visible_support_count == 0 and cand.knn_support_count == 0:
                continue
            best_swap = cand
            break

        if best_swap is not None:
            na, nb    = best_swap.scene_node_a, best_swap.scene_node_b
            swap_cons = _edge_consistency(na, nb, sn_pos_per_view, view_cutoffs, sn_to_views)
            fused_graph.add_edge(na, nb, **{
                "support_count":           best_swap.support_count,
                "supporting_images":       best_swap.supporting_images,
                "branch_support_count":    best_swap.branch_support_count,
                "knn_support_count":       best_swap.knn_support_count,
                "vggt_support_count":      best_swap.vggt_support_count,
                "vggt_score":              best_swap.vggt_score,
                "vggt_min_distance":       best_swap.vggt_min_distance,
                "cooc_count":              best_swap.cooc_count,
                "triplet_support":         best_swap.triplet_support,
                "forbidden_flag":          best_swap.forbidden_pair_flag,
                "evidence_sources":        best_swap.evidence_sources,
                "forced":                  True,
                "forced_reason":           "consistency_bridge_swap",
                "consistency_score":       swap_cons["consistency"],
                "consistency_n_visible":   swap_cons["n_visible"],
            })
            new_cs = f"{swap_cons['consistency']:.0%}" if swap_cons["consistency"] is not None else "n/a"
            debug["swapped"].append({
                "removed":         [a, b],
                "added":           [na, nb],
                "old_consistency": cons,
                "new_consistency": swap_cons["consistency"],
                "old_support":     support,
            })
            print(
                f"    [Consistency] swapped {a}-{b} (cons={cons:.0%}) "
                f"-> {na}-{nb} (cons={new_cs})"
            )
        else:
            # No better bridge — restore original edge and warn
            fused_graph.add_edge(a, b, **saved_attrs)
            debug["kept_for_connectivity"].append({
                "edge":        [a, b],
                "consistency": cons,
                "support":     support,
                "reason":      "no_better_bridge_found",
            })
            print(
                f"    [Consistency] kept {a}-{b}  cons={cons:.0%} sup={support} "
                f"[no better bridge — connectivity preserved]"
            )

    print(
        f"  [Consistency] pruned={len(debug['pruned'])}  "
        f"swapped={len(debug['swapped'])}  "
        f"kept_for_connectivity={len(debug['kept_for_connectivity'])}"
    )
    return fused_graph, debug
