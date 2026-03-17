"""Step 4 - candidate scene graph from multi-view structural hypotheses."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from config import COOC_COUNTS, FORBIDDEN_PAIRS, TRIPLET_COUNTS


@dataclass
class CandidateEdge:
    edge_id: str
    scene_node_a: str
    scene_node_b: str
    code_a: int
    code_b: int
    evidence_sources: list[str]
    supporting_images: list[str]
    branch_ids: list[str]
    support_count: int
    branch_support_count: int
    knn_support_count: int
    residual_support_count: int
    direct_visible_support_count: int
    shortcut_candidate_support_count: int
    observational_status_votes: dict
    support_reasons: dict
    cooc_count: int
    triplet_support: int
    forbidden_pair_flag: bool
    a_remaining_degree: int
    b_remaining_degree: int
    acceptance_reasons: list[str]
    weakened_by_reconciliation: bool = False
    expanded_by_paths: list[list[str]] = field(default_factory=list)
    reconciliation_notes: list[str] = field(default_factory=list)


def _sorted_pair(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))


def _norm_pair(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def _triplet_support(code_a: int, code_b: int) -> int:
    return sum(count for key, count in TRIPLET_COUNTS.items() if code_a in key and code_b in key)


def _build_residual_knn_support(scene_nodes: list, observed_graphs: list) -> dict:
    obs_to_scene = {}
    for scene_node in scene_nodes:
        for observed_node_id in scene_node.supporting_observed_ids:
            obs_to_scene[observed_node_id] = scene_node.scene_node_id

    pair_support: dict = defaultdict(lambda: {
        "images": set(),
        "direct_visible_images": set(),
        "shortcut_candidate_images": set(),
        "status_votes": Counter(),
        "support_reasons": Counter(),
    })
    for observed_graph in observed_graphs:
        for observed_edge in observed_graph.edges:
            scene_a = obs_to_scene.get(observed_edge.source_id)
            scene_b = obs_to_scene.get(observed_edge.target_id)
            if scene_a is None or scene_b is None or scene_a == scene_b:
                continue
            pair = _sorted_pair(scene_a, scene_b)
            pair_support[pair]["images"].add(observed_graph.image_id)
            if observed_edge.direct_visible:
                pair_support[pair]["direct_visible_images"].add(observed_graph.image_id)
            if observed_edge.shortcut_candidate:
                pair_support[pair]["shortcut_candidate_images"].add(observed_graph.image_id)
            pair_support[pair]["status_votes"][observed_edge.edge_status] += 1
            pair_support[pair]["support_reasons"][observed_edge.support_reason] += 1
    return pair_support


def build_candidate_graph(
    scene_nodes: list,
    observed_graphs: list,
    anchor_nodes: list,
    scene_branches: list,
) -> tuple[list[CandidateEdge], dict]:
    scene_by_id = {scene_node.scene_node_id: scene_node for scene_node in scene_nodes}
    anchor_ids = {anchor.scene_node_id for anchor in anchor_nodes}
    residual_knn = _build_residual_knn_support(scene_nodes, observed_graphs)

    edge_evidence: dict = {}
    branch_debug = []

    for scene_branch in scene_branches:
        ordered_scene_nodes = scene_branch.fused_ordered_scene_node_ids
        if len(ordered_scene_nodes) < 2:
            continue

        for idx in range(len(ordered_scene_nodes) - 1):
            scene_a = ordered_scene_nodes[idx]
            scene_b = ordered_scene_nodes[idx + 1]
            if scene_a == scene_b:
                continue

            pair = _sorted_pair(scene_a, scene_b)
            evidence = edge_evidence.setdefault(pair, {
                "evidence_sources": set(),
                "supporting_images": set(),
                "branch_ids": set(),
                "acceptance_reasons": [],
                "branch_support_count": 0,
                "knn_support_count": 0,
                "residual_support_count": 0,
                "direct_visible_support_count": 0,
                "shortcut_candidate_support_count": 0,
                "observational_status_votes": Counter(),
                "support_reasons": Counter(),
            })
            evidence["evidence_sources"].add("scene_branch")
            evidence["supporting_images"].update(scene_branch.support_images)
            evidence["branch_ids"].add(scene_branch.scene_branch_id)
            evidence["branch_support_count"] += 1

            if scene_a in anchor_ids or scene_b in anchor_ids:
                reason = "anchor_attached_branch_step"
            else:
                reason = "ordered_branch_chain_step"
            evidence["acceptance_reasons"].append(reason)
            branch_debug.append({
                "scene_branch_id": scene_branch.scene_branch_id,
                "pair": list(pair),
                "reason": reason,
            })

    for pair, info in residual_knn.items():
        evidence = edge_evidence.setdefault(pair, {
            "evidence_sources": set(),
            "supporting_images": set(),
            "branch_ids": set(),
            "acceptance_reasons": [],
            "branch_support_count": 0,
            "knn_support_count": 0,
            "residual_support_count": 0,
            "direct_visible_support_count": 0,
            "shortcut_candidate_support_count": 0,
            "observational_status_votes": Counter(),
            "support_reasons": Counter(),
        })
        evidence["evidence_sources"].add("knn_residual")
        evidence["supporting_images"].update(info["images"])
        evidence["knn_support_count"] = len(info["images"])
        evidence["residual_support_count"] = len(info["images"])
        evidence["direct_visible_support_count"] = len(info["direct_visible_images"])
        evidence["shortcut_candidate_support_count"] = len(info["shortcut_candidate_images"])
        evidence["observational_status_votes"].update(info["status_votes"])
        evidence["support_reasons"].update(info["support_reasons"])
        evidence["acceptance_reasons"].append("residual_local_hypothesis_support")

    candidate_edges: list[CandidateEdge] = []
    for edge_idx, pair in enumerate(sorted(edge_evidence.keys())):
        scene_a, scene_b = pair
        node_a = scene_by_id[scene_a]
        node_b = scene_by_id[scene_b]
        prior_key = _norm_pair(node_a.fused_code, node_b.fused_code)
        evidence = edge_evidence[pair]

        if COOC_COUNTS.get(prior_key, 0) > 0:
            evidence["evidence_sources"].add("cooc_prior")
            evidence["acceptance_reasons"].append("cooc_prior_support")
        if _triplet_support(node_a.fused_code, node_b.fused_code) > 0:
            evidence["evidence_sources"].add("triplet_prior")
            evidence["acceptance_reasons"].append("triplet_pattern_support")

        candidate_edges.append(CandidateEdge(
            edge_id=f"CE_{edge_idx}",
            scene_node_a=scene_a,
            scene_node_b=scene_b,
            code_a=node_a.fused_code,
            code_b=node_b.fused_code,
            evidence_sources=sorted(evidence["evidence_sources"]),
            supporting_images=sorted(evidence["supporting_images"]),
            branch_ids=sorted(evidence["branch_ids"]),
            support_count=max(evidence["branch_support_count"], evidence["knn_support_count"]),
            branch_support_count=evidence["branch_support_count"],
            knn_support_count=evidence["knn_support_count"],
            residual_support_count=evidence["residual_support_count"],
            direct_visible_support_count=evidence["direct_visible_support_count"],
            shortcut_candidate_support_count=evidence["shortcut_candidate_support_count"],
            observational_status_votes=dict(evidence["observational_status_votes"]),
            support_reasons=dict(evidence["support_reasons"]),
            cooc_count=COOC_COUNTS.get(prior_key, 0),
            triplet_support=_triplet_support(node_a.fused_code, node_b.fused_code),
            forbidden_pair_flag=prior_key in FORBIDDEN_PAIRS,
            a_remaining_degree=node_a.max_degree,
            b_remaining_degree=node_b.max_degree,
            acceptance_reasons=evidence["acceptance_reasons"],
        ))

    debug = {
        "anchor_node_ids": sorted(anchor_ids),
        "branch_edge_steps": branch_debug,
        "residual_pairs": [
            {
                "pair": list(pair),
                "supporting_images": sorted(info["images"]),
                "status_votes": dict(info["status_votes"]),
                "support_reasons": dict(info["support_reasons"]),
            }
            for pair, info in sorted(residual_knn.items())
        ],
    }

    print(
        f"  [CandidateGraph] {len(candidate_edges)} candidates from "
        f"{len(scene_branches)} scene branches and {len(residual_knn)} residual pairs"
    )
    return candidate_edges, debug


def serialize_candidate_edges(candidate_edges: list[CandidateEdge]) -> list[dict]:
    return [
        {
            "edge_id": edge.edge_id,
            "scene_node_a": edge.scene_node_a,
            "scene_node_b": edge.scene_node_b,
            "code_a": edge.code_a,
            "code_b": edge.code_b,
            "evidence_sources": edge.evidence_sources,
            "supporting_images": edge.supporting_images,
            "branch_ids": edge.branch_ids,
            "support_count": edge.support_count,
            "branch_support_count": edge.branch_support_count,
            "knn_support_count": edge.knn_support_count,
            "residual_support_count": edge.residual_support_count,
            "direct_visible_support_count": edge.direct_visible_support_count,
            "shortcut_candidate_support_count": edge.shortcut_candidate_support_count,
            "observational_status_votes": edge.observational_status_votes,
            "support_reasons": edge.support_reasons,
            "cooc_count": edge.cooc_count,
            "triplet_support": edge.triplet_support,
            "forbidden_pair_flag": edge.forbidden_pair_flag,
            "a_remaining_degree": edge.a_remaining_degree,
            "b_remaining_degree": edge.b_remaining_degree,
            "acceptance_reasons": edge.acceptance_reasons,
            "weakened_by_reconciliation": edge.weakened_by_reconciliation,
            "expanded_by_paths": edge.expanded_by_paths,
            "reconciliation_notes": edge.reconciliation_notes,
        }
        for edge in candidate_edges
    ]
