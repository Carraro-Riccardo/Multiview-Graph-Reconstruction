"""Local structural reconciliation for shortcut-like multi-view edges."""

from __future__ import annotations

from config import COOC_COUNTS, DEGREE_SATURATION, FORBIDDEN_PAIRS, TRIPLET_COUNTS


def _pair(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))


def _norm_pair(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def _triplet_count(code_a: int, code_b: int, code_c: int) -> int:
    return TRIPLET_COUNTS.get((min(code_a, code_c), code_b, max(code_a, code_c)), 0)


def _degree_note(scene_node) -> str:
    saturation = DEGREE_SATURATION.get(scene_node.fused_code)
    if saturation == 0.0:
        return "typically_terminal"
    if scene_node.max_degree == 2:
        return "degree2_chain_component"
    if scene_node.max_degree >= 3:
        return "anchor_or_hub_component"
    return "generic_component"


def reconcile_shortcut_edges(scene_nodes: list, candidate_edges: list) -> tuple[list, dict]:
    scene_by_id = {scene_node.scene_node_id: scene_node for scene_node in scene_nodes}
    edge_by_pair = {
        _pair(edge.scene_node_a, edge.scene_node_b): edge
        for edge in candidate_edges
    }

    expanded_cases = []
    weakened_cases = []
    kept_cases = []
    ambiguous_cases = []

    for direct_edge in candidate_edges:
        direct_pair = _pair(direct_edge.scene_node_a, direct_edge.scene_node_b)
        node_a = scene_by_id[direct_edge.scene_node_a]
        node_b = scene_by_id[direct_edge.scene_node_b]

        expansion_options = []
        for scene_c in scene_nodes:
            if scene_c.scene_node_id in direct_pair:
                continue

            # Skip expansions where either path leg is a forbidden pair —
            # a path through a physically impossible connection cannot explain away a valid edge.
            if _norm_pair(node_a.fused_code, scene_c.fused_code) in FORBIDDEN_PAIRS:
                continue
            if _norm_pair(scene_c.fused_code, node_b.fused_code) in FORBIDDEN_PAIRS:
                continue

            edge_ac = edge_by_pair.get(_pair(direct_edge.scene_node_a, scene_c.scene_node_id))
            edge_cb = edge_by_pair.get(_pair(scene_c.scene_node_id, direct_edge.scene_node_b))
            if edge_ac is None or edge_cb is None:
                continue

            path_views = set(edge_ac.supporting_images) | set(edge_cb.supporting_images)
            novel_views = sorted(path_views - set(direct_edge.supporting_images))
            triplet_support = _triplet_count(node_a.fused_code, scene_c.fused_code, node_b.fused_code)
            intermediate_degree_note = _degree_note(scene_c)
            degree_consistent = scene_c.max_degree == 2 and DEGREE_SATURATION.get(scene_c.fused_code, 1.0) != 0.0
            direct_is_weak = (
                direct_edge.direct_visible_support_count == 0
                or direct_edge.shortcut_candidate_support_count > 0
            )
            path_is_stronger = (
                (edge_ac.branch_support_count + edge_cb.branch_support_count) > direct_edge.branch_support_count
                or len(novel_views) > 0
            )
            prior_support = triplet_support > 0 or (
                COOC_COUNTS.get(_norm_pair(node_a.fused_code, scene_c.fused_code), 0) > 0
                and COOC_COUNTS.get(_norm_pair(scene_c.fused_code, node_b.fused_code), 0) > 0
            )

            positive_checks = sum([
                1 if direct_is_weak else 0,
                1 if path_is_stronger else 0,
                1 if prior_support else 0,
                1 if degree_consistent else 0,
            ])
            if positive_checks < 3:
                continue

            expansion_options.append({
                "intermediate_scene_node_id": scene_c.scene_node_id,
                "path": [direct_edge.scene_node_a, scene_c.scene_node_id, direct_edge.scene_node_b],
                "supporting_views": sorted(path_views),
                "novel_views": novel_views,
                "supporting_triplets": triplet_support,
                "degree_consistency_notes": {
                    "intermediate": intermediate_degree_note,
                    "direct_edge_weak": direct_is_weak,
                    "path_is_stronger": path_is_stronger,
                },
                "edge_ids": [edge_ac.edge_id, edge_cb.edge_id],
                "score_tuple": (
                    positive_checks,
                    triplet_support,
                    edge_ac.branch_support_count + edge_cb.branch_support_count,
                    len(novel_views),
                ),
            })

        if not expansion_options:
            kept_cases.append({
                "edge_id": direct_edge.edge_id,
                "involved_scene_nodes": [direct_edge.scene_node_a, direct_edge.scene_node_b],
                "decision": "direct_edge_kept",
                "reason": "no_strong_hidden_intermediate_evidence",
            })
            continue

        expansion_options.sort(key=lambda item: item["score_tuple"], reverse=True)
        best = expansion_options[0]
        if len(expansion_options) > 1 and expansion_options[0]["score_tuple"] == expansion_options[1]["score_tuple"]:
            ambiguous_cases.append({
                "edge_id": direct_edge.edge_id,
                "involved_scene_nodes": [direct_edge.scene_node_a, direct_edge.scene_node_b],
                "decision": "ambiguous_case_not_resolved",
                "candidate_expansions": [
                    {
                        "path": option["path"],
                        "supporting_views": option["supporting_views"],
                        "supporting_triplets": option["supporting_triplets"],
                    }
                    for option in expansion_options[:3]
                ],
            })
            continue

        direct_edge.weakened_by_reconciliation = True
        direct_edge.expanded_by_paths.append(best["path"])
        direct_edge.reconciliation_notes.extend([
            f"weakened_direct_edge_due_to_path={best['path']}",
            f"novel_views={best['novel_views']}",
            f"triplet_support={best['supporting_triplets']}",
        ])
        weakened_cases.append({
            "edge_id": direct_edge.edge_id,
            "involved_scene_nodes": [direct_edge.scene_node_a, direct_edge.scene_node_b],
            "decision": "direct_edge_weakened",
            "supporting_views": direct_edge.supporting_images,
            "explanation": best,
        })
        expanded_cases.append({
            "edge_id": direct_edge.edge_id,
            "involved_scene_nodes": [direct_edge.scene_node_a, direct_edge.scene_node_b],
            "decision": "direct_edge_expanded_into_2_step_path",
            "expanded_path": best["path"],
            "supporting_views": best["supporting_views"],
            "supporting_triplets": best["supporting_triplets"],
            "degree_consistency_notes": best["degree_consistency_notes"],
        })

    debug = {
        "direct_edge_kept": kept_cases,
        "direct_edge_weakened": weakened_cases,
        "direct_edge_expanded": expanded_cases,
        "ambiguous_cases_not_resolved": ambiguous_cases,
    }
    print(
        f"  [Reconciliation] expanded={len(expanded_cases)} "
        f"weakened={len(weakened_cases)} ambiguous={len(ambiguous_cases)}"
    )
    return candidate_edges, debug
