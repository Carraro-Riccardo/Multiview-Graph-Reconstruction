"""Branch-centric multiview fusion helpers."""

from __future__ import annotations

from .branch_fragments import (
    identify_anchor_nodes,
    match_branch_fragments,
    serialize_anchor_nodes,
    serialize_branch_fragments,
    serialize_scene_branches,
)
from .node_reid import build_observed_to_scene_map


def fuse_multiview_scene(scene_nodes: list, observed_graphs: list) -> tuple[dict, dict]:
    obs_to_scene = build_observed_to_scene_map(scene_nodes)
    anchor_nodes = identify_anchor_nodes(scene_nodes)
    anchor_scene_node_ids = {anchor.scene_node_id for anchor in anchor_nodes}

    branch_fragments = []
    per_view_fragment_counts = {}
    for observed_graph in observed_graphs:
        per_view_fragment_counts[observed_graph.image_id] = len(observed_graph.branch_fragments)
        branch_fragments.extend(observed_graph.branch_fragments)

    scene_branches, branch_debug = match_branch_fragments(
        branch_fragments,
        obs_to_scene,
        anchor_scene_node_ids=anchor_scene_node_ids,
    )

    attachment_hypotheses = []
    for scene_branch in scene_branches:
        for attachment in scene_branch.attachment_anchors:
            attachment_hypotheses.append({
                "scene_branch_id": scene_branch.scene_branch_id,
                **attachment,
            })

    fusion_state = {
        "scene_nodes": scene_nodes,
        "anchor_nodes": anchor_nodes,
        "branch_fragments": branch_fragments,
        "scene_branches": scene_branches,
        "obs_to_scene": obs_to_scene,
    }
    fusion_debug = {
        "per_view_fragment_counts": per_view_fragment_counts,
        "attachment_hypotheses": attachment_hypotheses,
        "branch_reid": branch_debug,
        "serialized": {
            "anchor_nodes": serialize_anchor_nodes(anchor_nodes),
            "branch_fragments": serialize_branch_fragments(branch_fragments),
            "scene_branches": serialize_scene_branches(scene_branches),
        },
    }
    return fusion_state, fusion_debug
