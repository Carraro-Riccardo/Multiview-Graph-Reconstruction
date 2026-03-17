"""
Branch-centric data structures and fusion helpers for the multi-view pipeline.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from config import COMPONENT_INFO, DEGREE_SATURATION, FORBIDDEN_PAIRS, TRIPLET_COUNTS


ANCHOR_CODES: frozenset[int] = frozenset(
    code for code, (valence, _) in COMPONENT_INFO.items() if valence >= 3
)
TERMINAL_CODES: frozenset[int] = frozenset(
    code for code, (valence, _) in COMPONENT_INFO.items() if valence == 1
)
RARE_REDUCER_CODES: frozenset[int] = frozenset({75, 255})

FRAGMENT_MATCH_THRESHOLD = 0.45
AMBIGUOUS_MATCH_MARGIN = 0.12


def is_anchor(code: int) -> bool:
    return code in ANCHOR_CODES or code in RARE_REDUCER_CODES


def is_terminal(code: int) -> bool:
    if code in TERMINAL_CODES:
        return True
    sat = DEGREE_SATURATION.get(code)
    return sat is not None and sat == 0.0


@dataclass
class AnchorNode:
    anchor_id: str
    scene_node_id: str
    code: int
    name: str
    valence: int
    anchor_type: str
    is_hub: bool
    is_strong_endpoint: bool
    support_count: int
    support_images: list[str]
    anchor_rank: float
    evidence: dict = field(default_factory=dict)


@dataclass
class BranchFragment:
    branch_fragment_id: str
    image_id: str
    ordered_observed_node_ids: list[str]
    ordered_codes: list[int]
    endpoint_candidates: list[dict]
    anchor_attachment_hints: list[dict]
    confidence: float
    local_evidence_metadata: dict


@dataclass
class SceneBranch:
    scene_branch_id: str
    supporting_fragments: list[str]
    fused_ordered_scene_node_ids: list[str]
    fused_ordered_codes: list[int]
    support_count: int
    support_images: list[str]
    attachment_anchors: list[dict]
    confidence: float
    merged_fragment_ids: list[str]
    ambiguous_matches: list[dict]
    acceptance_reasons: list[str]
    pruning_reasons: list[str]
    fragment_merge_debug: list[dict]


def _node_name(code: int) -> str:
    return COMPONENT_INFO.get(code, (2, f"code_{code}"))[1]


def _node_role(code: int, degree: int) -> str:
    if is_anchor(code) or degree >= 3:
        return "anchor"
    if is_terminal(code) or degree <= 1:
        return "endpoint"
    return "chain"


def extract_branch_fragments(obs_graph) -> list[BranchFragment]:
    if len(obs_graph.nodes) < 2:
        return []

    G = nx.Graph()
    for nid in obs_graph.nodes:
        G.add_node(nid)
    for edge in obs_graph.edges:
        G.add_edge(
            edge.source_id,
            edge.target_id,
            distance=edge.distance,
            confidence=edge.confidence,
            forbidden=edge.forbidden,
            cooc_support=edge.cooc_support,
        )

    if G.number_of_edges() == 0:
        return []

    notable_ids: set[str] = set()
    anchor_ids: set[str] = set()
    terminal_ids: set[str] = set()
    for nid, node in obs_graph.nodes.items():
        degree = G.degree(nid)
        if is_anchor(node.code) or degree >= 3:
            anchor_ids.add(nid)
            notable_ids.add(nid)
        if is_terminal(node.code) or degree <= 1:
            terminal_ids.add(nid)
            notable_ids.add(nid)

    if not notable_ids:
        max_deg = max((G.degree(n) for n in G.nodes()), default=0)
        notable_ids = {n for n in G.nodes() if G.degree(n) >= max(2, max_deg)}
    if not notable_ids:
        notable_ids = set(G.nodes())

    fragments: list[BranchFragment] = []
    visited_edges: set[tuple[str, str]] = set()
    frag_counter = 0

    for start_id in sorted(notable_ids):
        for nbr in sorted(G.neighbors(start_id)):
            edge_key = (min(start_id, nbr), max(start_id, nbr))
            if edge_key in visited_edges:
                continue

            path = [start_id, nbr]
            prev, current = start_id, nbr
            while current not in notable_ids:
                candidates = [n for n in G.neighbors(current) if n != prev]
                if len(candidates) != 1:
                    break
                nxt = candidates[0]
                path.append(nxt)
                prev, current = current, nxt

            for idx in range(len(path) - 1):
                a, b = path[idx], path[idx + 1]
                visited_edges.add((min(a, b), max(a, b)))

            if len(path) < 2:
                continue

            ordered_codes = [obs_graph.nodes[nid].code for nid in path]
            confidences = [obs_graph.nodes[nid].confidence for nid in path]
            distances = []
            forbidden_pairs = []
            for idx in range(len(path) - 1):
                edge = G.get_edge_data(path[idx], path[idx + 1], default={})
                distances.append(edge.get("distance"))
                forbidden_pairs.append(
                    tuple(sorted((ordered_codes[idx], ordered_codes[idx + 1]))) in FORBIDDEN_PAIRS
                )

            endpoint_candidates = []
            attachment_hints = []
            for endpoint_idx, obs_id in ((0, path[0]), (-1, path[-1])):
                node = obs_graph.nodes[obs_id]
                degree = G.degree(obs_id)
                role = _node_role(node.code, degree)
                endpoint_candidates.append({
                    "observed_node_id": obs_id,
                    "code": node.code,
                    "name": node.name,
                    "local_degree": degree,
                    "role": role,
                    "is_anchor_candidate": role == "anchor",
                    "is_terminal_candidate": role == "endpoint",
                })
                if role == "anchor":
                    attachment_hints.append({
                        "side": "start" if endpoint_idx == 0 else "end",
                        "observed_node_id": obs_id,
                        "code": node.code,
                        "name": node.name,
                        "hint_type": "anchor_endpoint",
                    })

            triplets = []
            for idx in range(1, len(ordered_codes) - 1):
                left = ordered_codes[idx - 1]
                center = ordered_codes[idx]
                right = ordered_codes[idx + 1]
                triplet_key = (min(left, right), center, max(left, right))
                triplets.append({
                    "triplet_key": list(triplet_key),
                    "triplet_count": TRIPLET_COUNTS.get(triplet_key, 0),
                })

            fragments.append(BranchFragment(
                branch_fragment_id=f"{obs_graph.image_id}_frag_{frag_counter}",
                image_id=obs_graph.image_id,
                ordered_observed_node_ids=list(path),
                ordered_codes=ordered_codes,
                endpoint_candidates=endpoint_candidates,
                anchor_attachment_hints=attachment_hints,
                confidence=sum(confidences) / len(confidences),
                local_evidence_metadata={
                    "length": len(path),
                    "mean_detection_confidence": sum(confidences) / len(confidences),
                    "triplets": triplets,
                    "distances": distances,
                    "forbidden_adjacent_pairs": forbidden_pairs,
                },
            ))
            frag_counter += 1

    return fragments


def identify_anchor_nodes(scene_nodes: list) -> list[AnchorNode]:
    anchors: list[AnchorNode] = []
    for scene_node in scene_nodes:
        code = scene_node.fused_code
        hub = COMPONENT_INFO.get(code, (2, ""))[0] >= 3 or is_anchor(code)
        strong_endpoint = is_terminal(code)
        if not hub and not strong_endpoint:
            continue

        if hub and code in {192}:
            anchor_type = "cross"
            base_rank = 0.92
        elif hub and code in {116, 201, 255}:
            anchor_type = "tee"
            base_rank = 0.82
        elif code in RARE_REDUCER_CODES:
            anchor_type = "rare_reducer"
            base_rank = 0.74
        elif strong_endpoint:
            anchor_type = "strong_endpoint"
            base_rank = 0.62
        else:
            anchor_type = "distinctive_anchor"
            base_rank = 0.7

        support_boost = min(scene_node.support_count / 4.0, 1.0) * 0.18
        anchors.append(AnchorNode(
            anchor_id=f"AN_{scene_node.scene_node_id}",
            scene_node_id=scene_node.scene_node_id,
            code=code,
            name=scene_node.fused_name,
            valence=scene_node.max_degree,
            anchor_type=anchor_type,
            is_hub=hub,
            is_strong_endpoint=strong_endpoint,
            support_count=scene_node.support_count,
            support_images=scene_node.support_images,
            anchor_rank=min(1.0, base_rank + support_boost),
            evidence={
                "supporting_observed_ids": scene_node.supporting_observed_ids,
                "local_structural_signature": scene_node.local_structural_signature,
            },
        ))
    return sorted(anchors, key=lambda anchor: (-anchor.anchor_rank, anchor.scene_node_id))


def _counter_jaccard(codes_a: list[int], codes_b: list[int]) -> float:
    ca = Counter(codes_a)
    cb = Counter(codes_b)
    intersection = sum(min(ca[k], cb[k]) for k in (ca.keys() | cb.keys()))
    union = sum(max(ca[k], cb[k]) for k in (ca.keys() | cb.keys()))
    return intersection / union if union else 0.0


def _triplet_compatibility(codes_a: list[int], codes_b: list[int]) -> float:
    def triplets(codes: list[int]) -> list[tuple[int, int, int]]:
        out = []
        for idx in range(1, len(codes) - 1):
            left = codes[idx - 1]
            center = codes[idx]
            right = codes[idx + 1]
            out.append((min(left, right), center, max(left, right)))
        return out

    ta = triplets(codes_a)
    tb = triplets(codes_b)
    if not ta or not tb:
        return 0.0

    set_a = set(ta)
    set_b = set(tb)
    overlap = set_a & set_b
    if overlap:
        return 0.25 + 0.1 * min(
            sum(TRIPLET_COUNTS.get(key, 0) for key in overlap) / max(len(overlap), 1),
            1.0,
        )
    return -0.05


def _anchor_hint_compatibility(fragment: BranchFragment, cluster_codes: list[int]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if not fragment.anchor_attachment_hints:
        return 0.0, reasons

    first_code = cluster_codes[0] if cluster_codes else None
    last_code = cluster_codes[-1] if cluster_codes else None
    score = 0.0
    for hint in fragment.anchor_attachment_hints:
        if hint["code"] == first_code or hint["code"] == last_code:
            score += 0.18
            reasons.append(f"anchor_hint_matches_{hint['code']}")
        else:
            score -= 0.06
            reasons.append(f"anchor_hint_mismatch_{hint['code']}")
    return score, reasons


def _sequence_compatibility(fragment: BranchFragment, cluster_codes: list[int]) -> tuple[float, dict]:
    if not fragment.ordered_codes or not cluster_codes:
        return 0.0, {"reason": "empty_sequence"}

    forward = list(fragment.ordered_codes)
    reverse = list(reversed(fragment.ordered_codes))
    candidate_orders = [("forward", forward), ("reversed", reverse)]

    best_score = -1.0
    best_debug = {}
    for direction, codes in candidate_orders:
        score = _counter_jaccard(codes, cluster_codes)
        reasons = [f"multiset_jaccard={score:.2f}", f"direction={direction}"]

        if codes[0] == cluster_codes[0]:
            score += 0.14
            reasons.append("start_code_matches")
        if codes[-1] == cluster_codes[-1]:
            score += 0.14
            reasons.append("end_code_matches")

        triplet_bonus = _triplet_compatibility(codes, cluster_codes)
        score += triplet_bonus
        reasons.append(f"triplet_compatibility={triplet_bonus:.2f}")

        anchor_bonus, anchor_reasons = _anchor_hint_compatibility(fragment, cluster_codes)
        score += anchor_bonus
        reasons.extend(anchor_reasons)

        endpoint_roles = [candidate["role"] for candidate in fragment.endpoint_candidates]
        if "anchor" in endpoint_roles and (codes[0] in ANCHOR_CODES or codes[-1] in ANCHOR_CODES):
            score += 0.08
            reasons.append("anchor_endpoint_supported")

        if best_score < score:
            best_score = score
            best_debug = {
                "direction": direction,
                "fragment_codes": codes,
                "cluster_codes": cluster_codes,
                "reasons": reasons,
            }

    return best_score, best_debug


def _representative_fragment(fragments: list[BranchFragment]) -> BranchFragment:
    return max(
        fragments,
        key=lambda frag: (
            len(frag.ordered_observed_node_ids),
            frag.confidence,
            len(frag.anchor_attachment_hints),
        ),
    )


def _majority_codes(fragments: list[BranchFragment]) -> list[int]:
    representative = _representative_fragment(fragments)
    return list(representative.ordered_codes)


def _ordered_scene_nodes(fragment: BranchFragment, obs_to_scene: dict[str, str]) -> list[str]:
    ordered: list[str] = []
    for observed_id in fragment.ordered_observed_node_ids:
        scene_id = obs_to_scene.get(observed_id)
        if scene_id is None:
            continue
        if not ordered or ordered[-1] != scene_id:
            ordered.append(scene_id)
    return ordered


def match_branch_fragments(
    all_fragments: list[BranchFragment],
    obs_to_scene: dict[str, str],
    anchor_scene_node_ids: set[str] | None = None,
) -> tuple[list[SceneBranch], dict]:
    if anchor_scene_node_ids is None:
        anchor_scene_node_ids = set()
    if not all_fragments:
        return [], {
            "total_fragments": 0,
            "total_scene_branches": 0,
            "merged_groups": [],
            "ambiguous_fragment_matches": [],
        }

    clusters: list[dict] = []
    ambiguous_matches: list[dict] = []

    ordered_fragments = sorted(
        all_fragments,
        key=lambda frag: (
            -len(frag.ordered_observed_node_ids),
            -len(frag.anchor_attachment_hints),
            -frag.confidence,
        ),
    )

    for fragment in ordered_fragments:
        scored_candidates: list[tuple[float, int, dict]] = []
        for idx, cluster in enumerate(clusters):
            if fragment.image_id in cluster["image_ids"]:
                continue
            score, compat_debug = _sequence_compatibility(fragment, cluster["canonical_codes"])
            if score >= FRAGMENT_MATCH_THRESHOLD:
                scored_candidates.append((score, idx, compat_debug))

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        if not scored_candidates:
            clusters.append({
                "fragments": [fragment],
                "image_ids": {fragment.image_id},
                "canonical_codes": list(fragment.ordered_codes),
                "merge_debug": [{
                    "fragment_id": fragment.branch_fragment_id,
                    "decision": "seed_cluster",
                    "reasons": ["first_fragment_for_cluster"],
                }],
            })
            continue

        best_score, best_idx, best_debug = scored_candidates[0]
        if len(scored_candidates) > 1 and best_score - scored_candidates[1][0] <= AMBIGUOUS_MATCH_MARGIN:
            ambiguous_matches.append({
                "fragment_id": fragment.branch_fragment_id,
                "candidate_clusters": [
                    {
                        "cluster_index": cluster_idx,
                        "score": score,
                        "reasons": compat_debug["reasons"],
                    }
                    for score, cluster_idx, compat_debug in scored_candidates[:3]
                ],
            })

        clusters[best_idx]["fragments"].append(fragment)
        clusters[best_idx]["image_ids"].add(fragment.image_id)
        clusters[best_idx]["canonical_codes"] = _majority_codes(clusters[best_idx]["fragments"])
        clusters[best_idx]["merge_debug"].append({
            "fragment_id": fragment.branch_fragment_id,
            "decision": "merged",
            "score": best_score,
            "reasons": best_debug["reasons"],
        })

    scene_branches: list[SceneBranch] = []
    merged_groups = []
    for idx, cluster in enumerate(clusters):
        fragments = cluster["fragments"]
        representative = _representative_fragment(fragments)
        fused_scene_ids = _ordered_scene_nodes(representative, obs_to_scene)
        support_images = sorted(cluster["image_ids"])

        attachment_anchors = []
        for pos, scene_node_id in (("start", fused_scene_ids[0] if fused_scene_ids else None),
                                   ("end", fused_scene_ids[-1] if fused_scene_ids else None)):
            if scene_node_id is None:
                continue
            attachment_anchors.append({
                "side": pos,
                "scene_node_id": scene_node_id,
                "is_anchor": scene_node_id in anchor_scene_node_ids,
                "hypothesis": "direct_reid_from_fragment_endpoint",
            })

        pruning_reasons = []
        if len(fused_scene_ids) < 2:
            pruning_reasons.append("insufficient_scene_node_chain")
        if representative.local_evidence_metadata.get("forbidden_adjacent_pairs") and all(
            representative.local_evidence_metadata["forbidden_adjacent_pairs"]
        ):
            pruning_reasons.append("all_local_adjacencies_forbidden_prior")

        acceptance_reasons = [
            f"support_count={len(fragments)}",
            f"images={','.join(support_images)}",
            f"canonical_codes={cluster['canonical_codes']}",
        ]
        if attachment_anchors:
            acceptance_reasons.append("anchor_attachments_available")

        scene_branch = SceneBranch(
            scene_branch_id=f"SB_{idx}",
            supporting_fragments=[frag.branch_fragment_id for frag in fragments],
            fused_ordered_scene_node_ids=fused_scene_ids,
            fused_ordered_codes=list(cluster["canonical_codes"]),
            support_count=len(fragments),
            support_images=support_images,
            attachment_anchors=attachment_anchors,
            confidence=sum(frag.confidence for frag in fragments) / len(fragments),
            merged_fragment_ids=[frag.branch_fragment_id for frag in fragments],
            ambiguous_matches=[
                entry for entry in ambiguous_matches
                if entry["fragment_id"] in {frag.branch_fragment_id for frag in fragments}
            ],
            acceptance_reasons=acceptance_reasons,
            pruning_reasons=pruning_reasons,
            fragment_merge_debug=list(cluster["merge_debug"]),
        )
        scene_branches.append(scene_branch)
        merged_groups.append({
            "scene_branch_id": scene_branch.scene_branch_id,
            "merged_fragment_ids": scene_branch.merged_fragment_ids,
            "canonical_codes": scene_branch.fused_ordered_codes,
            "support_images": support_images,
            "attachment_anchors": attachment_anchors,
        })

    return scene_branches, {
        "total_fragments": len(all_fragments),
        "total_scene_branches": len(scene_branches),
        "match_threshold": FRAGMENT_MATCH_THRESHOLD,
        "ambiguous_margin": AMBIGUOUS_MATCH_MARGIN,
        "merged_groups": merged_groups,
        "ambiguous_fragment_matches": ambiguous_matches,
    }


def serialize_anchor_nodes(anchor_nodes: list[AnchorNode]) -> list[dict]:
    return [
        {
            "anchor_id": anchor.anchor_id,
            "scene_node_id": anchor.scene_node_id,
            "code": anchor.code,
            "name": anchor.name,
            "valence": anchor.valence,
            "anchor_type": anchor.anchor_type,
            "is_hub": anchor.is_hub,
            "is_strong_endpoint": anchor.is_strong_endpoint,
            "support_count": anchor.support_count,
            "support_images": anchor.support_images,
            "anchor_rank": anchor.anchor_rank,
            "evidence": anchor.evidence,
        }
        for anchor in anchor_nodes
    ]


def serialize_branch_fragments(fragments: list[BranchFragment]) -> list[dict]:
    return [
        {
            "branch_fragment_id": fragment.branch_fragment_id,
            "image_id": fragment.image_id,
            "ordered_observed_node_ids": fragment.ordered_observed_node_ids,
            "ordered_codes": fragment.ordered_codes,
            "endpoint_candidates": fragment.endpoint_candidates,
            "anchor_attachment_hints": fragment.anchor_attachment_hints,
            "confidence": fragment.confidence,
            "local_evidence_metadata": fragment.local_evidence_metadata,
        }
        for fragment in fragments
    ]


def serialize_scene_branches(scene_branches: list[SceneBranch]) -> list[dict]:
    return [
        {
            "scene_branch_id": branch.scene_branch_id,
            "supporting_fragments": branch.supporting_fragments,
            "fused_ordered_scene_node_ids": branch.fused_ordered_scene_node_ids,
            "fused_ordered_codes": branch.fused_ordered_codes,
            "support_count": branch.support_count,
            "support_images": branch.support_images,
            "attachment_anchors": branch.attachment_anchors,
            "confidence": branch.confidence,
            "merged_fragment_ids": branch.merged_fragment_ids,
            "ambiguous_matches": branch.ambiguous_matches,
            "acceptance_reasons": branch.acceptance_reasons,
            "pruning_reasons": branch.pruning_reasons,
            "fragment_merge_debug": branch.fragment_merge_debug,
        }
        for branch in scene_branches
    ]
