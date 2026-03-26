"""Step 2 - node re-identification: multi-seed co-association + soft post-merge.

Each ObservedNode (one detection in one image) is assigned to a SceneNode
(the unique physical component it represents) by:
  1. Running N single-seed assignments (one per image) using Hungarian matching
     on cosine similarity of ResNet-50 patch embeddings.
  2. Building a co-association matrix: how often each pair of detected instances
     was placed in the same group across all seed runs.
  3. Greedy clustering per component code using the co-association matrix.
  4. Optional soft post-merge: merge groups with high prototype similarity
     AND high cross-group co-association score.

Output: list[SceneNode] compatible with the rest of the multiview pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from .branch_fragments import is_anchor, is_terminal
from core.linear_assignment import linear_sum_assignment
from config import COMPONENT_INFO


# ── RE-ID hyper-parameters ──────────────────────────────────────────────────
REID_MATCH_THRESHOLD      = 0.55   # min cosine sim for seed assignment acceptance
REID_ALLOW_UNMATCHED      = True   # skip weak matches instead of forcing them
REID_VISUAL_WEIGHT        = 0.85   # appearance term in seed matching
REID_CONTEXT_WEIGHT       = 0.15   # local-neighbourhood term in seed matching

_COASSOC_MIN_RUNS_BASE    = 2      # immutable baseline for adaptive formula
COASSOC_MIN_RUNS          = 2      # effective value; overridden per-call in identify_scene_nodes
                                       # (adaptive formula: max(1, min(BASE, n_images // 3)))
COASSOC_MIN_RATIO         = 0.60   # min same_group / co_observed ratio
FINAL_GROUP_MIN_AFFINITY  = 0.60   # min mean affinity to join a greedy cluster

SOFT_MERGE_ENABLED            = True
SOFT_MERGE_MIN_PROTO_SIM      = 0.72
SOFT_MERGE_MIN_CROSS_COASSOC  = 0.20
SOFT_MERGE_MIN_COMBINED_SCORE = 0.58
SOFT_MERGE_ALPHA              = 0.55   # weight of prototype cosine similarity
SOFT_MERGE_BETA               = 0.45   # weight of cross-group co-association
SOFT_MERGE_ALLOW_IMAGE_SUBSET_REPAIR = True
SOFT_MERGE_MAX_SUBSET_GROUP_SUPPORT  = 3
SOFT_MERGE_MAX_REPAIRABLE_OVERLAP_IMAGES = 1
SOFT_MERGE_MIN_UNIQUE_IMAGE_GAIN = 2
SOFT_MERGE_MIN_REPAIR_KEEP_SCORE = 0.55
SOFT_MERGE_MIN_REPAIR_MARGIN = 0.05
SOFT_MERGE_REPAIRED_PROTO_EPS = 0.01
SOFT_MERGE_REPAIRED_MIN_CROSS_COASSOC = 0.50
SOFT_MERGE_REPAIRED_MIN_COMBINED_SCORE = 0.62

SAME_IMAGE_REPAIR_ENABLED        = True
SAME_IMAGE_REPAIR_ROUNDS         = 3
SAME_IMAGE_REPAIR_MIN_GAIN       = 0.08
SAME_IMAGE_REPAIR_MIN_SCORE      = 0.35
SAME_IMAGE_REPAIR_ALPHA          = 0.60   # prototype fit
SAME_IMAGE_REPAIR_BETA           = 0.40   # pairwise fit to support outside current image

PATCH_EXPAND_RATIO = 0.35
MIN_PATCH_SIZE     = 48

# ── Algorithm selection ───────────────────────────────────────────────────────
# True  → skip seed runs entirely; use direct pairwise cosine similarity
#         as affinity signal (cluster count emerges from data, not seed image)
# False → original multi-seed co-association matrix approach
REID_USE_DIRECT_PAIRWISE = False

# ── Instance-level neighbourhood enhancement ─────────────────────────────────
# After computing the base pair affinity (co-association or direct pairwise),
# run NBRHD_ROUNDS iterations of neighbourhood-augmented refinement.
#
# Each round blends two signals:
#   visual  — base co-association / pairwise cosine ratio  (weight NBRHD_ALPHA)
#   cls_nbr — dynamic: cosine sim between cluster-ID histograms of KNN
#             neighbours (which instances surround this detection?).
#             Recomputed every round as clustering improves.
#
# Combined affinity:  α·visual + (1−α)·cls_nbr
#
# Set NBRHD_ROUNDS = 0 to disable.
NBRHD_ROUNDS = 2
NBRHD_ALPHA  = 0.65   # weight for visual (co-assoc) signal vs neighbourhood

# ── VGGT 3D re-id constraints ─────────────────────────────────────────────────
# When VGGT point maps are available (search_roots passed to identify_scene_nodes),
# per-detection 3D centroids are extracted and used to hard-constrain the visual
# clustering:
#
#   dist ≤ CONFIDENT_DIST  → forced same instance (pair added/overridden in pair_stats)
#   dist > SPLIT_DIST      → forced different (pair removed from pair_stats)
#   in between             → visual co-association decides as before
#
# Set VGGT_REID_ENABLED = False to revert to pure visual re-id.
VGGT_REID_ENABLED        = True
VGGT_REID_CONFIDENT_DIST = 0.08   # m — hard merge (generous for per-detection centroid noise)
VGGT_REID_SPLIT_DIST     = None   # disabled: splits are handled by visual co-association ratio


# ── SceneNode (downstream-compatible) ───────────────────────────────────────
@dataclass
class SceneNode:
    scene_node_id: str
    fused_code: int
    fused_name: str
    fused_confidence: float
    support_count: int
    support_images: list[str]
    supporting_observed_ids: list[str]
    local_structural_signature: dict
    max_degree: int
    anchor_hint_score: float
    representative_branch_roles: list[str]
    representative_triplet_hints: list[list[int]]


# ── Embedding model (lazy singleton) ────────────────────────────────────────
_emb_model     = None
_emb_transform = None


def _get_embedding_model():
    global _emb_model, _emb_transform
    if _emb_model is not None:
        return _emb_model, _emb_transform
    import torch
    import torchvision.models as M
    import torchvision.transforms as T
    model = M.resnet50(weights=M.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _emb_model, _emb_transform = model, transform
    return model, transform


def _extract_embedding(patch_bgr: np.ndarray) -> np.ndarray | None:
    if patch_bgr.size == 0:
        return None
    try:
        import torch
        from PIL import Image
        model, transform = _get_embedding_model()
        pil = Image.fromarray(cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB))
        tensor = transform(pil).unsqueeze(0)
        with torch.no_grad():
            emb = model(tensor).squeeze(0).numpy()
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)
    except Exception:
        return None


def _expand_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple:
    pad_x = max(int((x2 - x1) * PATCH_EXPAND_RATIO), 0)
    pad_y = max(int((y2 - y1) * PATCH_EXPAND_RATIO), 0)
    ex1, ey1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    ex2, ey2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if ex2 - ex1 < MIN_PATCH_SIZE:
        cx = (ex1 + ex2) // 2
        ex1, ex2 = max(0, cx - MIN_PATCH_SIZE // 2), min(w, cx + MIN_PATCH_SIZE // 2)
    if ey2 - ey1 < MIN_PATCH_SIZE:
        cy = (ey1 + ey2) // 2
        ey1, ey2 = max(0, cy - MIN_PATCH_SIZE // 2), min(h, cy + MIN_PATCH_SIZE // 2)
    return ex1, ey1, ex2, ey2


# ── Items: ObservedNode + visual embedding ───────────────────────────────────
def _build_items(
    observed_graphs: list,
    image_paths: list[Path],
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Build one item per ObservedNode carrying the visual embedding needed for RE-ID.
    Returns (all_items, items_by_image).
    """
    path_by_stem = {p.stem: p for p in image_paths}
    all_items: list[dict] = []
    items_by_image: dict[str, list[dict]] = defaultdict(list)

    for og in observed_graphs:
        image_id = og.image_id
        img_path = path_by_stem.get(image_id)
        img_bgr  = cv2.imread(str(img_path)) if img_path else None
        h, w     = (img_bgr.shape[:2] if img_bgr is not None else (1, 1))

        for obs in og.nodes.values():
            emb = None
            if img_bgr is not None:
                x1n, y1n, x2n, y2n = obs.bbox
                x1, y1 = int(x1n * w), int(y1n * h)
                x2, y2 = int(x2n * w), int(y2n * h)
                ex1, ey1, ex2, ey2 = _expand_box(x1, y1, x2, y2, w, h)
                patch = img_bgr[ey1:ey2, ex1:ex2]
                emb   = _extract_embedding(patch)

            item = {
                "observed_node_id": obs.observed_node_id,
                "image_id":         image_id,
                "code":             obs.code,
                "confidence":       obs.confidence,
                "center":           obs.center,
                "embedding":        emb,
                "nearby_codes":     list(obs.nearby_codes),
            }
            all_items.append(item)
            items_by_image[image_id].append(item)

    return all_items, dict(items_by_image)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _pair_key(a: str, b: str) -> tuple:
    return (a, b) if a < b else (b, a)


def _cosine_sim(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))   # both are pre-normalised


def _multiset_jaccard(values_a: list[int], values_b: list[int]) -> float:
    if not values_a and not values_b:
        return 0.0
    counter_a = Counter(values_a)
    counter_b = Counter(values_b)
    keys = set(counter_a) | set(counter_b)
    intersection = sum(min(counter_a.get(key, 0), counter_b.get(key, 0)) for key in keys)
    union = sum(max(counter_a.get(key, 0), counter_b.get(key, 0)) for key in keys)
    if union == 0:
        return 0.0
    return float(intersection / union)


def _group_context_sim(group_member_ids: list[str], query_id: str, item_by_id: dict) -> float:
    query_codes = item_by_id[query_id]["nearby_codes"]
    if not query_codes:
        return 0.0
    vals = [
        _multiset_jaccard(item_by_id[member_id]["nearby_codes"], query_codes)
        for member_id in group_member_ids
        if item_by_id[member_id]["nearby_codes"]
    ]
    return float(np.mean(vals)) if vals else 0.0


def _blend_seed_similarity(visual_sim: float, context_sim: float, has_context: bool) -> float:
    if not has_context:
        return visual_sim
    total = REID_VISUAL_WEIGHT + REID_CONTEXT_WEIGHT
    if total <= 0:
        return visual_sim
    alpha = REID_VISUAL_WEIGHT / total
    beta = REID_CONTEXT_WEIGHT / total
    return alpha * visual_sim + beta * context_sim


def _mean_proto(ids: list[str], item_by_id: dict) -> np.ndarray | None:
    embs = [item_by_id[i]["embedding"] for i in ids if item_by_id[i]["embedding"] is not None]
    if not embs:
        return None
    proto = np.mean(np.stack(embs), axis=0)
    return proto / (np.linalg.norm(proto) + 1e-8)


def _valid_ids_for_image(image_id: str, items_by_image: dict, item_by_id: dict) -> list[str]:
    return [
        item["observed_node_id"]
        for item in items_by_image.get(image_id, [])
        if item["embedding"] is not None
    ]


# ── Single-seed assignment ────────────────────────────────────────────────────
def _run_single_seed(
    seed_img_id: str,
    image_ids: list[str],
    items_by_image: dict,
    item_by_id: dict,
) -> dict:
    assignment: dict[str, str] = {}
    groups:     dict[str, dict] = {}
    by_code:    dict[int, list[str]] = defaultdict(list)

    seed_ids = _valid_ids_for_image(seed_img_id, items_by_image, item_by_id)
    seed_by_code: dict[int, list[str]] = defaultdict(list)
    for iid in seed_ids:
        seed_by_code[item_by_id[iid]["code"]].append(iid)

    for code, ids in seed_by_code.items():
        ids = sorted(ids, key=lambda i: (-item_by_id[i]["confidence"], i))
        for k, iid in enumerate(ids, start=1):
            gid = f"{code}#{k}"
            assignment[iid] = gid
            groups[gid] = {
                "code":      code,
                "members":   [iid],
                "prototype": item_by_id[iid]["embedding"].copy(),
            }
            by_code[code].append(gid)

    for img_id in image_ids:
        if img_id == seed_img_id:
            continue
        q_by_code: dict[int, list[str]] = defaultdict(list)
        for item in items_by_image.get(img_id, []):
            if item["embedding"] is not None:
                q_by_code[item["code"]].append(item["observed_node_id"])

        for code, qids in q_by_code.items():
            cand_gids = by_code.get(code, [])
            qids = sorted(qids, key=lambda i: (-item_by_id[i]["confidence"], i))

            if cand_gids:
                S = np.zeros((len(cand_gids), len(qids)), dtype=np.float32)
                for r, gid in enumerate(cand_gids):
                    for c, qid in enumerate(qids):
                        visual_sim = _cosine_sim(groups[gid]["prototype"], item_by_id[qid]["embedding"])
                        context_sim = _group_context_sim(groups[gid]["members"], qid, item_by_id)
                        has_context = bool(item_by_id[qid]["nearby_codes"]) and any(
                            item_by_id[member_id]["nearby_codes"] for member_id in groups[gid]["members"]
                        )
                        S[r, c] = _blend_seed_similarity(visual_sim, context_sim, has_context)

                row_ind, col_ind = linear_sum_assignment(-S)
                for r, c in zip(row_ind, col_ind):
                    sim = float(S[r, c])
                    if REID_ALLOW_UNMATCHED and sim < REID_MATCH_THRESHOLD:
                        continue
                    gid = cand_gids[r]
                    qid = qids[c]
                    assignment[qid] = gid
                    groups[gid]["members"].append(qid)
                    new_proto = _mean_proto(groups[gid]["members"], item_by_id)
                    if new_proto is not None:
                        groups[gid]["prototype"] = new_proto

    return {"seed_image": seed_img_id, "assignment": assignment}


# ── Direct pairwise affinity (replaces co-association) ───────────────────────
def _build_direct_pairwise_affinity(all_items: list, item_by_id: dict) -> dict:
    """
    Compute pair_stats from raw pairwise cosine similarity instead of seed runs.

    For every pair of detections (same code, different images, both with valid
    embeddings) the 'ratio' is set to the direct cosine similarity.  The greedy
    clustering step then uses this as affinity, so the cluster count is driven
    by the data rather than by which image was chosen as seed.
    """
    by_code: dict[int, list[str]] = defaultdict(list)
    for item in all_items:
        if item["embedding"] is not None:
            by_code[item["code"]].append(item["observed_node_id"])

    pair_stats: dict[tuple, dict] = {}
    for ids in by_code.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if item_by_id[a]["image_id"] == item_by_id[b]["image_id"]:
                    continue
                sim = _cosine_sim(item_by_id[a]["embedding"], item_by_id[b]["embedding"])
                key = _pair_key(a, b)
                # co_observed is set to COASSOC_MIN_RUNS so the min-runs
                # filter in _affinity() is always satisfied.
                pair_stats[key] = {
                    "same_group": sim,
                    "co_observed": COASSOC_MIN_RUNS,
                    "ratio": sim,
                }

    return pair_stats


# ── Instance-level neighbourhood enhancement ─────────────────────────────────

def _counter_cosine_sim(c1: dict, c2: dict) -> float:
    """Cosine similarity between two frequency dicts (cluster-ID → count)."""
    if not c1 or not c2:
        return 0.0
    keys = set(c1) | set(c2)
    v1 = np.array([c1.get(k, 0) for k in keys], dtype=float)
    v2 = np.array([c2.get(k, 0) for k in keys], dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _get_cluster_assignment(
    all_items: list,
    item_by_id: dict,
    pair_stats: dict,
) -> dict[str, str]:
    """Run greedy clustering and return {detection_id: cluster_id} dict."""
    items_by_code: dict[int, list[str]] = defaultdict(list)
    for item in all_items:
        items_by_code[item["code"]].append(item["observed_node_id"])
    assignment: dict[str, str] = {}
    for code, code_ids in sorted(items_by_code.items()):
        clusters = _greedy_cluster_code(code, code_ids, item_by_id, pair_stats)
        for k, cluster in enumerate(clusters, start=1):
            cid = f"{code}#{k}"
            for det_id in cluster:
                assignment[det_id] = cid
    return assignment


# ── VGGT 3D geometry loader ───────────────────────────────────────────────────
def _load_obs_geometry(observed_graphs: list, search_roots: list) -> dict[str, dict]:
    """Extract per-detection 3D centroid from VGGT point maps.

    Imports the private helpers from vggt_3d_edges so we don't duplicate the
    point-map loading logic.  Returns {} if no point maps can be found.
    """
    try:
        from .vggt_3d_edges import (  # noqa: PLC0415
            _candidate_files_for_image,
            _estimate_observation_geometry,
            _load_pointmap,
        )
    except ImportError:
        return {}

    obs_geometry: dict[str, dict] = {}
    n_loaded = 0
    for og in observed_graphs:
        candidates = _candidate_files_for_image(og.image_id, search_roots)
        pointmap = None
        for path in candidates:
            pointmap = _load_pointmap(path)
            if pointmap is not None:
                break
        if pointmap is None:
            continue
        n_loaded += 1
        for obs_id, obs_node in og.nodes.items():
            geom = _estimate_observation_geometry(pointmap, list(obs_node.bbox))
            if geom is not None:
                obs_geometry[obs_id] = geom

    if n_loaded:
        print(
            f"  [VGGT-ReID] loaded {n_loaded}/{len(observed_graphs)} point maps "
            f"→ {len(obs_geometry)} detection geometries"
        )
    return obs_geometry


def _apply_vggt_3d_constraints(
    pair_stats: dict,
    obs_geometry: dict,
    all_items: list,
    item_by_id: dict,
) -> dict:
    """Hard-constrain pair affinities using VGGT 3D centroid distances.

    For every same-code, different-image pair (a, b) where both have 3D geometry:
      dist ≤ CONFIDENT_DIST  → ratio=1.0, force merge (add pair if missing)
      dist > SPLIT_DIST      → remove pair from pair_stats (affinity = 0)
      else                   → leave unchanged (visual signal decides)
    """
    by_code: dict[int, list[str]] = defaultdict(list)
    for item in all_items:
        by_code[item["code"]].append(item["observed_node_id"])

    n_forced_merge = n_forced_split = 0

    for det_ids in by_code.values():
        for i, a in enumerate(det_ids):
            geom_a = obs_geometry.get(a)
            if geom_a is None:
                continue
            for b in det_ids[i + 1:]:
                if item_by_id[a]["image_id"] == item_by_id[b]["image_id"]:
                    continue  # same image → always distinct
                geom_b = obs_geometry.get(b)
                if geom_b is None:
                    continue
                dist = float(np.linalg.norm(geom_a["centroid"] - geom_b["centroid"]))
                key = _pair_key(a, b)
                if dist <= VGGT_REID_CONFIDENT_DIST:
                    pair_stats[key] = {
                        "same_group":    1.0,
                        "co_observed":   _COASSOC_MIN_RUNS_BASE,
                        "ratio":         1.0,
                        "vggt_dist":     dist,
                        "vggt_override": "merge",
                    }
                    n_forced_merge += 1
                elif VGGT_REID_SPLIT_DIST is not None and dist > VGGT_REID_SPLIT_DIST:
                    if pair_stats.pop(key, None) is not None:
                        n_forced_split += 1

    print(
        f"  [VGGT-ReID] constraints: forced_merge={n_forced_merge}  "
        f"forced_split={n_forced_split}"
    )
    return pair_stats


def _enhance_with_instance_neighborhoods(
    base_pair_stats: dict,
    all_items: list,
    item_by_id: dict,
    observed_graphs: list,
) -> dict:
    """
    Iteratively augment pair affinity with instance-level neighbourhood signal.

    Each round blends:
        visual  — base co-association / pairwise cosine ratio  (weight NBRHD_ALPHA)
        cls_nbr — cosine sim between cluster-ID histograms of KNN neighbours.
                  Recomputed every round as clustering improves.

    Combined affinity:  α·visual + (1−α)·cls_nbr

    Only pairs already present in base_pair_stats are updated.
    """
    if NBRHD_ROUNDS == 0:
        return base_pair_stats

    # Build detection → set of KNN-neighbour detection IDs from observed graphs.
    neighbors_of: dict[str, set[str]] = defaultdict(set)
    for og in observed_graphs:
        for edge in og.edges:
            neighbors_of[edge.source_id].add(edge.target_id)
            neighbors_of[edge.target_id].add(edge.source_id)

    current_stats = base_pair_stats

    for round_idx in range(NBRHD_ROUNDS):
        # Step 1 — provisional clustering
        assignment = _get_cluster_assignment(all_items, item_by_id, current_stats)

        # Step 2 — cluster-ID neighbourhood profile per detection
        cls_nbr_profile: dict[str, dict[str, int]] = {}
        for det_id in item_by_id:
            profile: dict[str, int] = {}
            for nbr_id in neighbors_of.get(det_id, set()):
                cid = assignment.get(nbr_id)
                if cid:
                    profile[cid] = profile.get(cid, 0) + 1
            cls_nbr_profile[det_id] = profile

        # Step 3 — blend visual + cls_nbr
        new_stats: dict[tuple, dict] = {}
        n_signal = 0
        for key, st in base_pair_stats.items():
            a, b   = key
            visual = st["ratio"]
            cls_nbr = _counter_cosine_sim(
                cls_nbr_profile.get(a, {}),
                cls_nbr_profile.get(b, {}),
            )
            combined = NBRHD_ALPHA * visual + (1.0 - NBRHD_ALPHA) * cls_nbr
            if cls_nbr > 0:
                n_signal += 1
            new_stats[key] = {
                "same_group":  combined,
                "co_observed": st["co_observed"],
                "ratio":       combined,
                "visual_ratio": visual,
                "cls_nbr_sim":  cls_nbr,
            }

        current_stats = new_stats
        print(
            f"  [ReID-Nbr] round {round_idx + 1}/{NBRHD_ROUNDS}: "
            f"signal={n_signal}/{len(new_stats)}"
        )

    return current_stats


# ── Co-association matrix ─────────────────────────────────────────────────────
def _build_coassoc(single_seed_runs: list[dict], item_by_id: dict) -> dict:
    pair_stats: dict[tuple, dict] = {}

    for run in single_seed_runs:
        assignment = run["assignment"]
        assigned_by_code: dict[int, list[str]] = defaultdict(list)
        local_groups: dict[tuple, list[str]]   = defaultdict(list)

        for iid, gid in assignment.items():
            code = item_by_id[iid]["code"]
            assigned_by_code[code].append(iid)
            local_groups[(code, gid)].append(iid)

        for code, ids in assigned_by_code.items():
            for a, b in combinations(ids, 2):
                if item_by_id[a]["image_id"] == item_by_id[b]["image_id"]:
                    continue
                key = _pair_key(a, b)
                st  = pair_stats.setdefault(key, {"same_group": 0, "co_observed": 0, "ratio": 0.0})
                st["co_observed"] += 1

        for (code, gid), ids in local_groups.items():
            for a, b in combinations(ids, 2):
                if item_by_id[a]["image_id"] == item_by_id[b]["image_id"]:
                    continue
                key = _pair_key(a, b)
                st  = pair_stats.setdefault(key, {"same_group": 0, "co_observed": 0, "ratio": 0.0})
                st["same_group"] += 1

    for st in pair_stats.values():
        st["ratio"] = st["same_group"] / st["co_observed"] if st["co_observed"] > 0 else 0.0

    return pair_stats


# ── Greedy clustering per component code ─────────────────────────────────────
def _affinity(a: str, b: str, pair_stats: dict) -> float:
    st = pair_stats.get(_pair_key(a, b))
    if st is None or st["co_observed"] < COASSOC_MIN_RUNS or st["ratio"] < COASSOC_MIN_RATIO:
        return 0.0
    return st["ratio"]


def _greedy_cluster_code(
    code: int,
    code_ids: list[str],
    item_by_id: dict,
    pair_stats: dict,
) -> list[list[str]]:
    node_strength = {
        iid: sum(
            _affinity(iid, jid, pair_stats)
            for jid in code_ids
            if iid != jid and item_by_id[iid]["image_id"] != item_by_id[jid]["image_id"]
        )
        for iid in code_ids
    }
    ordered = sorted(
        code_ids,
        key=lambda i: (-node_strength[i], item_by_id[i]["image_id"], i),
    )

    unassigned = set(code_ids)
    clusters: list[list[str]] = []

    while unassigned:
        seed = next((i for i in ordered if i in unassigned), next(iter(unassigned)))
        group = [seed]
        unassigned.remove(seed)
        used_images = {item_by_id[seed]["image_id"]}

        while True:
            best_cand, best_score = None, -1.0
            for cand in list(unassigned):
                if item_by_id[cand]["image_id"] in used_images:
                    continue
                nonzero = [_affinity(cand, m, pair_stats) for m in group]
                nonzero = [x for x in nonzero if x > 0]
                if not nonzero:
                    continue
                avg = float(np.mean(nonzero))
                if avg >= FINAL_GROUP_MIN_AFFINITY and avg > best_score:
                    best_score, best_cand = avg, cand
            if best_cand is None:
                break
            group.append(best_cand)
            unassigned.remove(best_cand)
            used_images.add(item_by_id[best_cand]["image_id"])

        clusters.append(group)

    def _sort_key(cluster):
        centers = [item_by_id[i]["center"] for i in cluster]
        return (-len(cluster), float(np.mean([c[1] for c in centers])), float(np.mean([c[0] for c in centers])))

    return sorted(clusters, key=_sort_key)


# ── Soft post-merge ───────────────────────────────────────────────────────────
def _score_obs_to_support(
    obs_id: str,
    support_ids: list[str],
    item_by_id: dict,
    pair_stats: dict,
) -> dict:
    proto = _mean_proto(support_ids, item_by_id)
    proto_fit = _cosine_sim(item_by_id[obs_id]["embedding"], proto)
    pair_vals = [
        float(pair_stats[_pair_key(obs_id, other_id)]["ratio"])
        for other_id in support_ids
        if item_by_id[other_id]["image_id"] != item_by_id[obs_id]["image_id"]
        and _pair_key(obs_id, other_id) in pair_stats
    ]
    pair_fit = float(np.mean(pair_vals)) if pair_vals else 0.0
    score = SAME_IMAGE_REPAIR_ALPHA * proto_fit + SAME_IMAGE_REPAIR_BETA * pair_fit
    return {
        "score": score,
        "proto_fit": proto_fit,
        "pair_fit": pair_fit,
        "support_count": len(support_ids),
    }


def _repair_soft_merge_overlap(
    ids1: list[str],
    ids2: list[str],
    item_by_id: dict,
    pair_stats: dict,
) -> tuple[list[str], list[str], dict] | None:
    imgs1 = {item_by_id[i]["image_id"] for i in ids1}
    imgs2 = {item_by_id[i]["image_id"] for i in ids2}
    overlap_images = sorted(imgs1 & imgs2)
    if not overlap_images:
        return list(ids1), list(ids2), {
            "overlap_images": [],
            "repaired": False,
            "chosen": [],
            "dropped": [],
        }

    unique_gain = len(imgs1 | imgs2) - max(len(imgs1), len(imgs2))
    if (
        not SOFT_MERGE_ALLOW_IMAGE_SUBSET_REPAIR
        or len(overlap_images) > SOFT_MERGE_MAX_REPAIRABLE_OVERLAP_IMAGES
        or min(len(ids1), len(ids2)) > SOFT_MERGE_MAX_SUBSET_GROUP_SUPPORT
        or unique_gain < SOFT_MERGE_MIN_UNIQUE_IMAGE_GAIN
    ):
        return None

    remaining_support = [
        obs_id
        for obs_id in ids1 + ids2
        if item_by_id[obs_id]["image_id"] not in overlap_images
    ]
    if len(remaining_support) < 2:
        return None

    kept_by_image: dict[str, str] = {}
    chosen_meta: list[dict] = []
    dropped_meta: list[dict] = []
    support_context = list(remaining_support)

    for image_id in overlap_images:
        candidates = [
            obs_id
            for obs_id in ids1 + ids2
            if item_by_id[obs_id]["image_id"] == image_id
        ]
        scored = []
        for obs_id in candidates:
            meta = _score_obs_to_support(obs_id, support_context, item_by_id, pair_stats)
            scored.append((meta["score"], obs_id, meta))
        scored.sort(reverse=True)
        if not scored:
            return None
        best_score, best_obs_id, best_meta = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        if (
            best_score < SOFT_MERGE_MIN_REPAIR_KEEP_SCORE
            or best_score - second_score < SOFT_MERGE_MIN_REPAIR_MARGIN
        ):
            return None
        kept_by_image[image_id] = best_obs_id
        chosen_meta.append({
            "image_id": image_id,
            "obs_id": best_obs_id,
            "score": best_score,
            "proto_fit": best_meta["proto_fit"],
            "pair_fit": best_meta["pair_fit"],
        })
        for _, dropped_obs_id, dropped_meta_entry in scored[1:]:
            dropped_meta.append({
                "image_id": image_id,
                "obs_id": dropped_obs_id,
                "score": dropped_meta_entry["score"],
                "proto_fit": dropped_meta_entry["proto_fit"],
                "pair_fit": dropped_meta_entry["pair_fit"],
            })
        support_context.append(best_obs_id)

    repaired_ids1 = [
        obs_id for obs_id in ids1
        if item_by_id[obs_id]["image_id"] not in overlap_images or kept_by_image[item_by_id[obs_id]["image_id"]] == obs_id
    ]
    repaired_ids2 = [
        obs_id for obs_id in ids2
        if item_by_id[obs_id]["image_id"] not in overlap_images or kept_by_image[item_by_id[obs_id]["image_id"]] == obs_id
    ]
    if not repaired_ids1 or not repaired_ids2:
        return None

    return repaired_ids1, repaired_ids2, {
        "overlap_images": overlap_images,
        "unique_image_gain": unique_gain,
        "repaired": True,
        "chosen": chosen_meta,
        "dropped": dropped_meta,
    }


def _soft_merge_thresholds(overlap_repair: bool) -> tuple[float, float, float]:
    proto_threshold = SOFT_MERGE_MIN_PROTO_SIM
    cross_threshold = SOFT_MERGE_MIN_CROSS_COASSOC
    score_threshold = SOFT_MERGE_MIN_COMBINED_SCORE
    if overlap_repair:
        proto_threshold -= SOFT_MERGE_REPAIRED_PROTO_EPS
        cross_threshold = max(cross_threshold, SOFT_MERGE_REPAIRED_MIN_CROSS_COASSOC)
        score_threshold = max(score_threshold, SOFT_MERGE_REPAIRED_MIN_COMBINED_SCORE)
    return proto_threshold, cross_threshold, score_threshold


def _run_soft_post_merge(
    groups: dict,
    groups_by_code: dict,
    item_by_id: dict,
    pair_stats: dict,
) -> tuple[dict, dict]:
    if not SOFT_MERGE_ENABLED:
        return groups, groups_by_code

    def _proto(ids):
        return _mean_proto(ids, item_by_id)

    def _cross_coassoc(ids1, ids2):
        vals = [
            pair_stats[_pair_key(a, b)]["ratio"]
            for a in ids1 for b in ids2
            if a != b
            and item_by_id[a]["image_id"] != item_by_id[b]["image_id"]
            and _pair_key(a, b) in pair_stats
        ]
        return float(np.mean(vals)) if vals else 0.0

    def _can_merge(g1, g2):
        if g1["code"] != g2["code"]:
            return False, None
        ids1, ids2 = g1["member_ids"], g2["member_ids"]
        repaired = _repair_soft_merge_overlap(ids1, ids2, item_by_id, pair_stats)
        if repaired is None:
            return False, None
        eval_ids1, eval_ids2, repair_meta = repaired
        overlap_images = repair_meta["overlap_images"]
        overlap_repair = repair_meta["repaired"]
        proto_threshold, cross_threshold, score_threshold = _soft_merge_thresholds(overlap_repair)
        p1, p2 = _proto(eval_ids1), _proto(eval_ids2)
        proto_sim   = _cosine_sim(p1, p2)
        cross_coass = _cross_coassoc(eval_ids1, eval_ids2)
        score = SOFT_MERGE_ALPHA * proto_sim + SOFT_MERGE_BETA * cross_coass
        if proto_sim   < proto_threshold:               return False, None
        if cross_coass < cross_threshold:               return False, None
        if score       < score_threshold:               return False, None
        return True, {
            "proto_sim": proto_sim,
            "cross_coassoc": cross_coass,
            "score": score,
            "overlap_images": overlap_images,
            "overlap_repair": overlap_repair,
            "thresholds": {
                "proto": proto_threshold,
                "cross": cross_threshold,
                "score": score_threshold,
            },
            "repair_meta": repair_meta,
            "merged_member_ids": list(dict.fromkeys(eval_ids1 + eval_ids2)),
        }

    # Collect and sort candidate merges by score
    merge_candidates = []
    for code, gids in groups_by_code.items():
        for i in range(len(gids)):
            for j in range(i + 1, len(gids)):
                ok, ms = _can_merge(groups[gids[i]], groups[gids[j]])
                if ok:
                    merge_candidates.append((ms["score"], gids[i], gids[j]))
    merge_candidates.sort(reverse=True)

    active   = dict(groups)
    consumed: set[str] = set()
    merge_count_by_code: dict[int, int] = defaultdict(int)

    for _, gid1, gid2 in merge_candidates:
        if gid1 in consumed or gid2 in consumed:
            continue
        if gid1 not in active or gid2 not in active:
            continue
        ok, ms = _can_merge(active[gid1], active[gid2])
        if not ok:
            continue

        code = active[gid1]["code"]
        merge_count_by_code[code] += 1
        gid_new  = f"{code}#SM{merge_count_by_code[code]}"
        ids_union = list(ms.get("merged_member_ids") or dict.fromkeys(active[gid1]["member_ids"] + active[gid2]["member_ids"]))

        # Resolve image conflicts: keep the detection closest to the merged proto
        proto_union = _proto(ids_union)
        by_image: dict[str, list[str]] = defaultdict(list)
        for iid in ids_union:
            by_image[item_by_id[iid]["image_id"]].append(iid)
        kept = []
        for img_ids in by_image.values():
            if len(img_ids) == 1:
                kept.append(img_ids[0])
            else:
                best = max(img_ids, key=lambda i: _cosine_sim(item_by_id[i]["embedding"], proto_union))
                kept.append(best)

        active[gid_new] = {"code": code, "member_ids": sorted(kept, key=lambda i: (item_by_id[i]["image_id"], i))}
        consumed |= {gid1, gid2}
        del active[gid1], active[gid2]

    # Re-number groups stably per code
    def _group_sort_key(g):
        centers = [item_by_id[i]["center"] for i in g["member_ids"]]
        return (-len(g["member_ids"]), float(np.mean([c[1] for c in centers])), float(np.mean([c[0] for c in centers])))

    final_groups:   dict[str, dict] = {}
    final_by_code:  dict[int, list[str]] = defaultdict(list)
    by_code_groups: dict[int, list[dict]] = defaultdict(list)
    for g in active.values():
        by_code_groups[g["code"]].append(g)

    for code, group_list in by_code_groups.items():
        for k, g in enumerate(sorted(group_list, key=_group_sort_key), start=1):
            gid = f"{code}#{k}"
            g["group_id"] = gid
            final_groups[gid]   = g
            final_by_code[code].append(gid)

    return final_groups, dict(final_by_code)


# ── SceneNode assembly ────────────────────────────────────────────────────────
def _build_scene_nodes(groups: dict, item_by_id: dict, observed_graphs: list) -> list[SceneNode]:
    obs_node_by_id = {nid: node for og in observed_graphs for nid, node in og.nodes.items()}
    scene_nodes: list[SceneNode] = []

    for g in groups.values():
        gid        = g["group_id"]
        code       = g["code"]
        member_ids = g["member_ids"]
        valence, name = COMPONENT_INFO.get(code, (2, f"code_{code}"))

        support_images    = sorted({item_by_id[i]["image_id"] for i in member_ids})
        fused_confidence  = float(np.mean([item_by_id[i]["confidence"] for i in member_ids]))

        # Structural hints used by graph reconstruction downstream
        from collections import Counter
        all_nearby: list[int] = []
        anchor_count = terminal_count = 0
        for iid in member_ids:
            obs = obs_node_by_id.get(iid)
            if obs is None:
                continue
            all_nearby.extend(getattr(obs, "nearby_codes", []))
            if is_anchor(code) or getattr(obs, "local_degree_estimate", 0) >= 3:
                anchor_count += 1
            if is_terminal(code) or getattr(obs, "local_degree_estimate", 2) <= 1:
                terminal_count += 1

        anchor_hint_score = min(
            1.0,
            anchor_count / max(len(member_ids), 1) + (0.3 if is_anchor(code) else 0.0)
        )
        local_structural_signature = {
            "self_code":               code,
            "nearby_code_histogram":   dict(Counter(all_nearby)),
            "anchor_hint_score":       anchor_hint_score,
            "terminal_hint_count":     terminal_count,
        }

        scene_nodes.append(SceneNode(
            scene_node_id=gid,
            fused_code=code,
            fused_name=name,
            fused_confidence=fused_confidence,
            support_count=len(member_ids),
            support_images=support_images,
            supporting_observed_ids=list(member_ids),
            local_structural_signature=local_structural_signature,
            max_degree=valence,
            anchor_hint_score=anchor_hint_score,
            representative_branch_roles=[],
            representative_triplet_hints=[],
        ))

    return scene_nodes


# ── Main entry point ─────────────────────────────────────────────────────────
def _observation_image_counts(obs_ids: list[str], item_by_id: dict) -> dict[str, list[str]]:
    by_image: dict[str, list[str]] = defaultdict(list)
    for obs_id in obs_ids:
        by_image[item_by_id[obs_id]["image_id"]].append(obs_id)
    return {img: sorted(ids) for img, ids in sorted(by_image.items())}


def _observation_budget_violations(obs_ids: list[str], item_by_id: dict) -> list[dict]:
    violations = []
    by_image = _observation_image_counts(obs_ids, item_by_id)
    for image_id, image_obs in by_image.items():
        if len(image_obs) <= 1:
            continue
        violations.append({
            "image_id": image_id,
            "count": len(image_obs),
            "observed_ids": image_obs,
        })
    return violations


def _group_proto_similarity(ids1: list[str], ids2: list[str], item_by_id: dict) -> float:
    return _cosine_sim(_mean_proto(ids1, item_by_id), _mean_proto(ids2, item_by_id))


def _group_cross_coassoc(ids1: list[str], ids2: list[str], item_by_id: dict, pair_stats: dict) -> float:
    vals = [
        pair_stats[_pair_key(a, b)]["ratio"]
        for a in ids1 for b in ids2
        if a != b
        and item_by_id[a]["image_id"] != item_by_id[b]["image_id"]
        and _pair_key(a, b) in pair_stats
    ]
    return float(np.mean(vals)) if vals else 0.0


def _summarize_cross_pairs(ids1: list[str], ids2: list[str], item_by_id: dict, pair_stats: dict) -> dict:
    samples = []
    total_pairs = 0
    n_with_stats = 0
    n_effective = 0
    n_below_ratio = 0
    n_below_runs = 0

    for a in ids1:
        for b in ids2:
            if a == b or item_by_id[a]["image_id"] == item_by_id[b]["image_id"]:
                continue
            total_pairs += 1
            st = pair_stats.get(_pair_key(a, b))
            if st is None:
                continue
            n_with_stats += 1
            ratio = float(st.get("ratio", 0.0))
            co_observed = int(st.get("co_observed", 0))
            effective = ratio >= COASSOC_MIN_RATIO and co_observed >= COASSOC_MIN_RUNS
            if effective:
                n_effective += 1
            elif co_observed < COASSOC_MIN_RUNS:
                n_below_runs += 1
            else:
                n_below_ratio += 1
            samples.append({
                "a": a,
                "b": b,
                "ratio": ratio,
                "co_observed": co_observed,
                "same_group": float(st.get("same_group", 0.0)),
                "effective_affinity": ratio if effective else 0.0,
                "visual_ratio": float(st.get("visual_ratio", ratio)),
                "cls_nbr_sim": float(st.get("cls_nbr_sim", 0.0)),
                "vggt_dist": st.get("vggt_dist"),
                "vggt_override": st.get("vggt_override"),
            })

    samples_sorted = sorted(
        samples,
        key=lambda entry: (
            -entry["effective_affinity"],
            -entry["ratio"],
            -entry["co_observed"],
            entry["a"],
            entry["b"],
        ),
    )
    ratios = [entry["ratio"] for entry in samples]
    return {
        "n_cross_pairs": total_pairs,
        "n_with_pair_stats": n_with_stats,
        "n_missing_pair_stats": total_pairs - n_with_stats,
        "n_effective_affinity": n_effective,
        "n_below_ratio": n_below_ratio,
        "n_below_runs": n_below_runs,
        "mean_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "max_ratio": float(np.max(ratios)) if ratios else 0.0,
        "top_pairs": samples_sorted[:5],
        "bottom_pairs": list(reversed(samples_sorted[-5:])),
    }


def _soft_merge_decision(g1: dict, g2: dict, item_by_id: dict, pair_stats: dict) -> dict:
    ids1 = list(g1["member_ids"])
    ids2 = list(g2["member_ids"])
    repaired = _repair_soft_merge_overlap(ids1, ids2, item_by_id, pair_stats)
    if repaired is None:
        overlap_images = sorted(
            {item_by_id[i]["image_id"] for i in ids1}
            & {item_by_id[i]["image_id"] for i in ids2}
        )
        return {
            "blocked_by_image_overlap": bool(overlap_images),
            "overlap_images": overlap_images,
            "overlap_repair": False,
            "proto_sim": _group_proto_similarity(ids1, ids2, item_by_id),
            "cross_coassoc": _group_cross_coassoc(ids1, ids2, item_by_id, pair_stats),
            "score": SOFT_MERGE_ALPHA * _group_proto_similarity(ids1, ids2, item_by_id)
            + SOFT_MERGE_BETA * _group_cross_coassoc(ids1, ids2, item_by_id, pair_stats),
            "pass_proto": _group_proto_similarity(ids1, ids2, item_by_id) >= SOFT_MERGE_MIN_PROTO_SIM,
            "pass_cross_coassoc": _group_cross_coassoc(ids1, ids2, item_by_id, pair_stats) >= SOFT_MERGE_MIN_CROSS_COASSOC,
            "pass_score": (
                SOFT_MERGE_ALPHA * _group_proto_similarity(ids1, ids2, item_by_id)
                + SOFT_MERGE_BETA * _group_cross_coassoc(ids1, ids2, item_by_id, pair_stats)
            ) >= SOFT_MERGE_MIN_COMBINED_SCORE,
            "can_merge": False,
        }
    eval_ids1, eval_ids2, repair_meta = repaired
    overlap_images = repair_meta["overlap_images"]
    overlap_repair = repair_meta["repaired"]
    proto_threshold, cross_threshold, score_threshold = _soft_merge_thresholds(overlap_repair)
    proto_sim = _group_proto_similarity(eval_ids1, eval_ids2, item_by_id)
    cross_coassoc = _group_cross_coassoc(eval_ids1, eval_ids2, item_by_id, pair_stats)
    score = SOFT_MERGE_ALPHA * proto_sim + SOFT_MERGE_BETA * cross_coassoc
    return {
        "blocked_by_image_overlap": bool(overlap_images) and not overlap_repair,
        "overlap_images": overlap_images,
        "overlap_repair": overlap_repair,
        "repair_meta": repair_meta,
        "proto_sim": proto_sim,
        "cross_coassoc": cross_coassoc,
        "score": score,
        "pass_proto": proto_sim >= proto_threshold,
        "pass_cross_coassoc": cross_coassoc >= cross_threshold,
        "pass_score": score >= score_threshold,
        "thresholds": {
            "proto": proto_threshold,
            "cross": cross_threshold,
            "score": score_threshold,
        },
        "can_merge": (
            (not overlap_images or overlap_repair)
            and proto_sim >= proto_threshold
            and cross_coassoc >= cross_threshold
            and score >= score_threshold
        ),
    }


def _score_obs_to_group(
    obs_id: str,
    group_id: str,
    assignment: dict[str, str],
    item_by_id: dict,
    pair_stats: dict,
) -> dict:
    image_id = item_by_id[obs_id]["image_id"]
    support_ids = [
        other_id
        for other_id, other_group_id in assignment.items()
        if other_group_id == group_id and item_by_id[other_id]["image_id"] != image_id
    ]
    raw_vals = [
        float(pair_stats[_pair_key(obs_id, other_id)]["ratio"])
        for other_id in support_ids
        if _pair_key(obs_id, other_id) in pair_stats
    ]
    pair_fit = float(np.mean(raw_vals)) if raw_vals else 0.0
    proto = _mean_proto(support_ids, item_by_id)
    proto_fit = _cosine_sim(item_by_id[obs_id]["embedding"], proto)
    score = SAME_IMAGE_REPAIR_ALPHA * proto_fit + SAME_IMAGE_REPAIR_BETA * pair_fit
    return {
        "score": score,
        "proto_fit": proto_fit,
        "pair_fit": pair_fit,
        "support_count": len(support_ids),
    }


def _build_groups_from_assignment(
    assignment: dict[str, str],
    code: int,
    item_by_id: dict,
) -> tuple[dict[str, dict], list[str]]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for obs_id, group_id in assignment.items():
        by_group[group_id].append(obs_id)

    groups: dict[str, dict] = {}
    active_group_ids: list[str] = []
    for group_id in sorted(by_group):
        member_ids = sorted(by_group[group_id], key=lambda obs_id: (item_by_id[obs_id]["image_id"], obs_id))
        if not member_ids:
            continue
        groups[group_id] = {"code": code, "member_ids": member_ids, "group_id": group_id}
        active_group_ids.append(group_id)
    return groups, active_group_ids


def _repair_same_image_conflicts(
    groups: dict,
    groups_by_code: dict,
    item_by_id: dict,
    pair_stats: dict,
) -> tuple[dict, dict, list[dict]]:
    if not SAME_IMAGE_REPAIR_ENABLED:
        return groups, groups_by_code, []

    updated_groups = {gid: {"code": group["code"], "member_ids": list(group["member_ids"]), "group_id": gid} for gid, group in groups.items()}
    updated_groups_by_code = {code: list(group_ids) for code, group_ids in groups_by_code.items()}
    repair_log: list[dict] = []

    for code, group_ids in sorted(updated_groups_by_code.items()):
        if len(group_ids) <= 1:
            continue
        code_group_ids = [gid for gid in group_ids if gid in updated_groups]
        assignment = {
            obs_id: gid
            for gid in code_group_ids
            for obs_id in updated_groups[gid]["member_ids"]
        }
        if not assignment:
            continue

        code_obs_ids = sorted(assignment.keys(), key=lambda obs_id: (item_by_id[obs_id]["image_id"], obs_id))
        images = sorted({item_by_id[obs_id]["image_id"] for obs_id in code_obs_ids})

        for round_idx in range(SAME_IMAGE_REPAIR_ROUNDS):
            round_changed = False
            for image_id in images:
                image_obs_ids = [obs_id for obs_id in code_obs_ids if item_by_id[obs_id]["image_id"] == image_id]
                if len(image_obs_ids) <= 1:
                    continue

                active_group_ids = sorted({
                    group_id
                    for group_id in code_group_ids
                    if any(assigned_group_id == group_id for assigned_group_id in assignment.values())
                })
                if len(active_group_ids) <= 1:
                    continue

                row_by_obs_id = {obs_id: row for row, obs_id in enumerate(image_obs_ids)}
                col_by_group_id = {group_id: col for col, group_id in enumerate(active_group_ids)}
                score_matrix = np.zeros((len(image_obs_ids), len(active_group_ids)), dtype=float)
                for row, obs_id in enumerate(image_obs_ids):
                    for col, group_id in enumerate(active_group_ids):
                        meta = _score_obs_to_group(obs_id, group_id, assignment, item_by_id, pair_stats)
                        score_matrix[row, col] = meta["score"]

                current_pairs = []
                current_total = 0.0
                for row, obs_id in enumerate(image_obs_ids):
                    current_group_id = assignment[obs_id]
                    if current_group_id not in active_group_ids:
                        continue
                    col = active_group_ids.index(current_group_id)
                    current_pairs.append((row, col))
                    current_total += score_matrix[row, col]

                if not current_pairs:
                    continue

                row_ind, col_ind = linear_sum_assignment(-score_matrix)
                optimal_pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
                optimal_total = float(sum(score_matrix[row, col] for row, col in optimal_pairs))
                gain = optimal_total - current_total
                changed_assignments = [
                    (image_obs_ids[row], active_group_ids[col])
                    for row, col in optimal_pairs
                    if assignment[image_obs_ids[row]] != active_group_ids[col]
                ]
                if not changed_assignments:
                    continue

                min_optimal_score = min(score_matrix[row, col] for row, col in optimal_pairs)
                if gain < SAME_IMAGE_REPAIR_MIN_GAIN or min_optimal_score < SAME_IMAGE_REPAIR_MIN_SCORE:
                    continue

                before_assignment = {obs_id: assignment[obs_id] for obs_id in image_obs_ids}
                for obs_id, new_group_id in changed_assignments:
                    assignment[obs_id] = new_group_id

                round_changed = True
                repair_log.append({
                    "code": code,
                    "image_id": image_id,
                    "round": round_idx + 1,
                    "gain": gain,
                    "current_total": current_total,
                    "optimal_total": optimal_total,
                    "before": before_assignment,
                    "after": {obs_id: assignment[obs_id] for obs_id in image_obs_ids},
                    "changes": [
                        {
                            "obs_id": obs_id,
                            "from_group": before_assignment[obs_id],
                            "to_group": new_group_id,
                            "new_score": score_matrix[row_by_obs_id[obs_id], col_by_group_id[new_group_id]],
                        }
                        for obs_id, new_group_id in changed_assignments
                    ],
                })

            if not round_changed:
                break

        rebuilt_groups, rebuilt_group_ids = _build_groups_from_assignment(assignment, code, item_by_id)
        for gid in list(code_group_ids):
            updated_groups.pop(gid, None)
        updated_groups.update(rebuilt_groups)
        updated_groups_by_code[code] = rebuilt_group_ids

    if repair_log:
        print(
            f"  [ReID-Repair] same-image reassignments: {len(repair_log)}"
        )

    return updated_groups, updated_groups_by_code, repair_log


def _serialize_reid_debug(
    all_items: list[dict],
    item_by_id: dict,
    pair_stats: dict,
    groups: dict,
    groups_by_code: dict,
    scene_nodes: list[SceneNode],
    n_seed_runs: int,
    repair_log: list[dict],
) -> dict:
    item_payload = {
        item["observed_node_id"]: {
            "observed_node_id": item["observed_node_id"],
            "image_id": item["image_id"],
            "code": int(item["code"]),
            "confidence": float(item["confidence"]),
            "center": [float(item["center"][0]), float(item["center"][1])],
            "nearby_codes": [int(code) for code in item["nearby_codes"]],
            "has_embedding": item["embedding"] is not None,
        }
        for item in sorted(all_items, key=lambda entry: (entry["code"], entry["image_id"], entry["observed_node_id"]))
    }

    pair_payload = []
    for (a, b), st in sorted(pair_stats.items(), key=lambda entry: (item_by_id[entry[0][0]]["code"], entry[0][0], entry[0][1])):
        pair_payload.append({
            "a": a,
            "b": b,
            "code": int(item_by_id[a]["code"]),
            "image_a": item_by_id[a]["image_id"],
            "image_b": item_by_id[b]["image_id"],
            "same_group": float(st.get("same_group", 0.0)),
            "co_observed": int(st.get("co_observed", 0)),
            "ratio": float(st.get("ratio", 0.0)),
            "visual_ratio": float(st.get("visual_ratio", st.get("ratio", 0.0))),
            "cls_nbr_sim": float(st.get("cls_nbr_sim", 0.0)),
            "vggt_dist": st.get("vggt_dist"),
            "vggt_override": st.get("vggt_override"),
            "effective_affinity": _affinity(a, b, pair_stats),
        })

    groups_payload = {}
    scene_node_by_id = {sn.scene_node_id: sn for sn in scene_nodes}
    for group_id, group in sorted(groups.items()):
        member_ids = list(group["member_ids"])
        obs_per_image = _observation_image_counts(member_ids, item_by_id)
        groups_payload[group_id] = {
            "group_id": group_id,
            "code": int(group["code"]),
            "member_ids": member_ids,
            "support_count": len(member_ids),
            "support_images": sorted(obs_per_image.keys()),
            "observations_per_image": obs_per_image,
            "observation_budget_violations": _observation_budget_violations(member_ids, item_by_id),
            "fused_confidence": float(scene_node_by_id[group_id].fused_confidence) if group_id in scene_node_by_id else None,
        }

    cross_group_analysis = []
    for code, group_ids in sorted(groups_by_code.items()):
        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                gid_a = group_ids[i]
                gid_b = group_ids[j]
                group_a = groups[gid_a]
                group_b = groups[gid_b]
                obs_a = list(group_a["member_ids"])
                obs_b = list(group_b["member_ids"])
                merged_obs = obs_a + obs_b
                image_counts = Counter(item_by_id[obs_id]["image_id"] for obs_id in merged_obs)
                cross_group_analysis.append({
                    "code": int(code),
                    "group_a": gid_a,
                    "group_b": gid_b,
                    "support_a": len(obs_a),
                    "support_b": len(obs_b),
                    "images_a": sorted({item_by_id[obs_id]["image_id"] for obs_id in obs_a}),
                    "images_b": sorted({item_by_id[obs_id]["image_id"] for obs_id in obs_b}),
                    "merged_obs_count": len(merged_obs),
                    "merged_unique_images": len(image_counts),
                    "merged_duplicate_images": sorted(
                        image_id for image_id, count in image_counts.items() if count > 1
                    ),
                    "cross_pair_summary": _summarize_cross_pairs(obs_a, obs_b, item_by_id, pair_stats),
                    "soft_merge": _soft_merge_decision(group_a, group_b, item_by_id, pair_stats),
                })

    observation_budget_violations = []
    for scene_node in sorted(scene_nodes, key=lambda node: (node.fused_code, node.scene_node_id)):
        violations = _observation_budget_violations(scene_node.supporting_observed_ids, item_by_id)
        if not violations:
            continue
        observation_budget_violations.append({
            "scene_node_id": scene_node.scene_node_id,
            "code": int(scene_node.fused_code),
            "support_count": int(scene_node.support_count),
            "support_images": list(scene_node.support_images),
            "violations": violations,
        })

    return {
        "mode": "direct_pairwise" if REID_USE_DIRECT_PAIRWISE else "seed_coassoc",
        "thresholds": {
            "reid_match_threshold": REID_MATCH_THRESHOLD,
            "coassoc_min_runs": COASSOC_MIN_RUNS,
            "coassoc_min_ratio": COASSOC_MIN_RATIO,
            "final_group_min_affinity": FINAL_GROUP_MIN_AFFINITY,
            "same_image_repair_enabled": SAME_IMAGE_REPAIR_ENABLED,
            "same_image_repair_rounds": SAME_IMAGE_REPAIR_ROUNDS,
            "same_image_repair_min_gain": SAME_IMAGE_REPAIR_MIN_GAIN,
            "same_image_repair_min_score": SAME_IMAGE_REPAIR_MIN_SCORE,
            "soft_merge_enabled": SOFT_MERGE_ENABLED,
            "soft_merge_min_proto_sim": SOFT_MERGE_MIN_PROTO_SIM,
            "soft_merge_min_cross_coassoc": SOFT_MERGE_MIN_CROSS_COASSOC,
            "soft_merge_min_combined_score": SOFT_MERGE_MIN_COMBINED_SCORE,
            "soft_merge_allow_image_subset_repair": SOFT_MERGE_ALLOW_IMAGE_SUBSET_REPAIR,
            "soft_merge_max_subset_group_support": SOFT_MERGE_MAX_SUBSET_GROUP_SUPPORT,
            "soft_merge_repaired_min_cross_coassoc": SOFT_MERGE_REPAIRED_MIN_CROSS_COASSOC,
            "soft_merge_repaired_min_combined_score": SOFT_MERGE_REPAIRED_MIN_COMBINED_SCORE,
        },
        "n_seed_runs": int(n_seed_runs),
        "n_pair_stats": len(pair_stats),
        "n_scene_nodes": len(scene_nodes),
        "same_image_repair_log": repair_log,
        "items": item_payload,
        "pair_stats": pair_payload,
        "groups": groups_payload,
        "cross_group_analysis": cross_group_analysis,
        "observation_budget_violations": observation_budget_violations,
        "observed_to_scene": {
            iid: sn.scene_node_id
            for sn in scene_nodes
            for iid in sn.supporting_observed_ids
        },
    }


def identify_scene_nodes(
    observed_graphs: list,
    image_paths: list[Path] | None = None,
    search_roots: list[Path] | None = None,
) -> tuple[list[SceneNode], dict]:
    from config import IMAGES_DIR

    if image_paths is None:
        # Fallback: scan default images directory
        image_paths = list(IMAGES_DIR.glob("*.jpg"))

    all_items, items_by_image = _build_items(observed_graphs, image_paths)
    item_by_id = {item["observed_node_id"]: item for item in all_items}

    if not all_items:
        return [], {"error": "no_items"}

    image_ids = sorted(items_by_image.keys())
    n_images  = len(image_ids)

    # Adaptive min-runs: scale the evidence bar with view count.
    # In scenes with few images a detection can co-occur in at most
    # (n_images − 1) seed runs; requiring 2 runs would block all singletons.
    # Formula: max(1, min(BASE, n_images // 3))
    #   ≤5 views  → 1   (most nodes)
    #   6–8 views → 2   (unchanged from baseline)
    global COASSOC_MIN_RUNS
    COASSOC_MIN_RUNS = max(1, min(_COASSOC_MIN_RUNS_BASE, n_images // 3))

    print(f"  [ReID] building embeddings done: {len(all_items)} instances across {n_images} views  "
          f"(COASSOC_MIN_RUNS={COASSOC_MIN_RUNS})")

    if REID_USE_DIRECT_PAIRWISE:
        print(f"  [ReID] using direct pairwise cosine affinity (no seed runs)")
        pair_stats = _build_direct_pairwise_affinity(all_items, item_by_id)
    else:
        # Multi-seed runs (every image acts as seed once)
        single_seed_runs = [
            _run_single_seed(seed, image_ids, items_by_image, item_by_id)
            for seed in image_ids
        ]
        pair_stats = _build_coassoc(single_seed_runs, item_by_id)

    # VGGT 3D hard constraints: override visual affinities where 3D distance is decisive
    if VGGT_REID_ENABLED and search_roots:
        obs_geometry = _load_obs_geometry(observed_graphs, list(search_roots))
        if obs_geometry:
            pair_stats = _apply_vggt_3d_constraints(pair_stats, obs_geometry, all_items, item_by_id)

    # Iterative instance-level neighbourhood enhancement
    pair_stats = _enhance_with_instance_neighborhoods(
        pair_stats, all_items, item_by_id, observed_graphs
    )

    # Greedy cluster per component code
    items_by_code: dict[int, list[str]] = defaultdict(list)
    for item in all_items:
        items_by_code[item["code"]].append(item["observed_node_id"])

    groups:        dict[str, dict] = {}
    groups_by_code: dict[int, list[str]] = defaultdict(list)
    for code, code_ids in sorted(items_by_code.items()):
        clusters = _greedy_cluster_code(code, code_ids, item_by_id, pair_stats)
        for k, cluster in enumerate(clusters, start=1):
            gid = f"{code}#{k}"
            groups[gid] = {"code": code, "member_ids": cluster, "group_id": gid}
            groups_by_code[code].append(gid)

    groups, groups_by_code, repair_log = _repair_same_image_conflicts(
        groups, groups_by_code, item_by_id, pair_stats
    )

    # Soft post-merge
    groups, groups_by_code = _run_soft_post_merge(groups, groups_by_code, item_by_id, pair_stats)

    # Assemble SceneNode objects
    scene_nodes = _build_scene_nodes(groups, item_by_id, observed_graphs)

    n_seed_runs = 0 if REID_USE_DIRECT_PAIRWISE else len(single_seed_runs)
    debug = _serialize_reid_debug(
        all_items=all_items,
        item_by_id=item_by_id,
        pair_stats=pair_stats,
        groups=groups,
        groups_by_code=groups_by_code,
        scene_nodes=scene_nodes,
        n_seed_runs=n_seed_runs,
        repair_log=repair_log,
    )

    _print_reid_summary(scene_nodes, all_items, n_seed_runs, len(pair_stats))
    return scene_nodes, debug


def _print_reid_summary(scene_nodes: list, all_items: list, n_seeds: int, n_pairs: int) -> None:
    n_obs   = len(all_items)
    n_scene = len(scene_nodes)

    # Group scene nodes by code for a compact per-code table
    from collections import defaultdict as _dd
    by_code: dict = _dd(list)
    for sn in scene_nodes:
        by_code[sn.fused_code].append(sn)

    col = 18  # name column width
    print(f"\n  {'─' * 62}")
    print(f"  {'RE-ID SUMMARY':^62}")
    print(f"  {'─' * 62}")
    print(f"  {'code':<6}{'name':<{col}}{'groups':>7}{'obs':>6}{'views':>7}{'conf':>7}")
    print(f"  {'─' * 62}")
    for code in sorted(by_code):
        nodes      = by_code[code]
        name       = nodes[0].fused_name
        n_groups   = len(nodes)
        total_obs  = sum(sn.support_count for sn in nodes)
        all_imgs   = {img for sn in nodes for img in sn.support_images}
        avg_conf   = sum(sn.fused_confidence for sn in nodes) / n_groups
        print(f"  {code:<6}{name:<{col}}{n_groups:>7}{total_obs:>6}{len(all_imgs):>7}{avg_conf:>7.2f}")
    print(f"  {'─' * 62}")
    print(f"  total: {n_scene} scene nodes  |  {n_obs} observed  |  {n_seeds} seed runs  |  {n_pairs} pairs")
    print(f"  {'─' * 62}\n")


# ── Utilities used by downstream modules ─────────────────────────────────────
def build_observed_to_scene_map(scene_nodes: list[SceneNode]) -> dict[str, str]:
    return {
        obs_id: sn.scene_node_id
        for sn in scene_nodes
        for obs_id in sn.supporting_observed_ids
    }


def serialize_scene_nodes(scene_nodes: list[SceneNode]) -> list[dict]:
    return [
        {
            "scene_node_id":              sn.scene_node_id,
            "fused_code":                 sn.fused_code,
            "fused_name":                 sn.fused_name,
            "fused_confidence":           sn.fused_confidence,
            "support_count":              sn.support_count,
            "support_images":             sn.support_images,
            "supporting_observed_ids":    sn.supporting_observed_ids,
            "local_structural_signature": sn.local_structural_signature,
            "max_degree":                 sn.max_degree,
            "anchor_hint_score":          sn.anchor_hint_score,
            "representative_branch_roles":    sn.representative_branch_roles,
            "representative_triplet_hints":   sn.representative_triplet_hints,
        }
        for sn in scene_nodes
    ]


def merge_spurious_singletons(
    scene_nodes: list[SceneNode],
    observed_graphs: list,
    scene_geom: dict | None = None,
    scene_cutoff: float | None = None,
    pair_hints: list[dict] | None = None,
    pos_factor: float = 0.5,
    iou_threshold: float = 0.30,
) -> tuple[list[SceneNode], list[dict]]:
    """Merge singleton scene nodes that are likely spurious duplicates.

    Level 1 – 3D positional duplicate: centroid within pos_factor * scene_cutoff
    of a higher-support same-code node.

    Level 2 – 2D detection overlap: the singleton's sole detection has
    IoU >= iou_threshold with any detection of a higher-support same-code node
    in the same image.
    """
    obs_by_id: dict[str, object] = {}
    for og in observed_graphs:
        for obs_id, obs in og.nodes.items():
            obs_by_id[obs_id] = obs

    sn_by_id = {sn.scene_node_id: sn for sn in scene_nodes}
    parent = {sn.scene_node_id: sn.scene_node_id for sn in scene_nodes}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(dropped: str, kept: str) -> None:
        parent[_find(dropped)] = _find(kept)

    def _iou(b1: list, b2: list) -> float:
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter)

    strong_partners: dict[str, set[str]] = defaultdict(set)
    if isinstance(pair_hints, dict):
        pair_hint_entries = []
        for pair, meta in pair_hints.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                continue
            pair_hint_entries.append((pair, meta if isinstance(meta, dict) else {}))
    else:
        pair_hint_entries = []
        for hint in pair_hints or []:
            if isinstance(hint, dict):
                pair = hint.get("pair") or []
                meta = hint
            elif isinstance(hint, tuple) and len(hint) == 2 and isinstance(hint[1], dict):
                pair, meta = hint
            else:
                continue
            if len(pair) != 2:
                continue
            pair_hint_entries.append((tuple(pair), meta))

    for pair, meta in pair_hint_entries:
        if float(meta.get("score", 0.0)) < 0.75:
            continue
        a_id, b_id = pair
        strong_partners[a_id].add(b_id)
        strong_partners[b_id].add(a_id)

    merge_log: list[dict] = []

    for sn in scene_nodes:
        if sn.support_count != 1:
            continue
        candidates = sorted(
            [o for o in scene_nodes
             if o.scene_node_id != sn.scene_node_id
             and o.fused_code == sn.fused_code
             and o.support_count > 1],
            key=lambda o: -o.support_count,
        )
        same_code_candidates = sorted(
            [o for o in scene_nodes
             if o.scene_node_id != sn.scene_node_id
             and o.fused_code == sn.fused_code],
            key=lambda o: (-o.support_count, o.scene_node_id),
        )
        if not same_code_candidates:
            continue

        kept_id, reason = None, None

        # Level 1: 3D positional duplicate
        if scene_geom and scene_cutoff and sn.scene_node_id in scene_geom:
            ca = scene_geom[sn.scene_node_id]["centroid"]
            max_dist = scene_cutoff * pos_factor
            for other in candidates:
                if other.scene_node_id not in scene_geom:
                    continue
                cb = scene_geom[other.scene_node_id]["centroid"]
                dist = float(sum((x - y) ** 2 for x, y in zip(ca, cb)) ** 0.5)
                if dist <= max_dist:
                    kept_id = other.scene_node_id
                    reason = f"3d_pos_dup  dist={dist:.4f}m  ({dist/scene_cutoff:.2f}x)"
                    break

        # Level 2: 2D detection overlap
        if kept_id is None and sn.supporting_observed_ids:
            my_obs = obs_by_id.get(sn.supporting_observed_ids[0])
            if my_obs is not None:
                for other in candidates:
                    for oid in other.supporting_observed_ids:
                        o_obs = obs_by_id.get(oid)
                        if o_obs is None or o_obs.image_id != my_obs.image_id:
                            continue
                        iou = _iou(my_obs.bbox, o_obs.bbox)
                        if iou >= iou_threshold:
                            kept_id = other.scene_node_id
                            reason = f"2d_overlap  iou={iou:.3f}  img={my_obs.image_id}"
                            break
                    if kept_id:
                        break

        # Level 3: same-code singleton/low-support clusters sharing a strong 3D partner.
        if (
            kept_id is None
            and scene_geom and scene_cutoff
            and sn.scene_node_id in scene_geom
            and strong_partners.get(sn.scene_node_id)
        ):
            ca = scene_geom[sn.scene_node_id]["centroid"]
            best_shared = None
            for other in same_code_candidates:
                if other.scene_node_id not in scene_geom:
                    continue
                if set(sn.support_images) & set(other.support_images):
                    continue
                shared = sorted(
                    strong_partners.get(sn.scene_node_id, set())
                    & strong_partners.get(other.scene_node_id, set())
                )
                if not shared:
                    continue
                if not any(sn_by_id.get(pid) and sn_by_id[pid].support_count >= 2 for pid in shared):
                    continue
                cb = scene_geom[other.scene_node_id]["centroid"]
                dist = float(sum((x - y) ** 2 for x, y in zip(ca, cb)) ** 0.5)
                if dist > scene_cutoff:
                    continue
                candidate = (
                    len(shared),
                    other.support_count,
                    -dist,
                    other.scene_node_id,
                )
                if best_shared is None or candidate > best_shared[0]:
                    best_shared = (
                        candidate,
                        other.scene_node_id,
                        dist,
                        shared,
                    )
            if best_shared is not None:
                _, best_other_id, best_dist, shared = best_shared
                if sn_by_id[best_other_id].support_count >= sn.support_count:
                    kept_id = best_other_id
                    dropped_id = sn.scene_node_id
                else:
                    kept_id = sn.scene_node_id
                    dropped_id = best_other_id
                reason = (
                    f"3d_shared_partner  dist={best_dist:.4f}m  "
                    f"({best_dist/scene_cutoff:.2f}x)  shared={','.join(shared)}"
                )
                if _find(dropped_id) != _find(kept_id):
                    _union(dropped_id, kept_id)
                    merge_log.append({
                        "dropped": dropped_id,
                        "kept": kept_id,
                        "reason": reason,
                        "code": sn.fused_code,
                    })
                continue

        if kept_id is None or _find(sn.scene_node_id) == _find(kept_id):
            continue

        _union(sn.scene_node_id, kept_id)
        merge_log.append({"dropped": sn.scene_node_id, "kept": kept_id,
                          "reason": reason, "code": sn.fused_code})

    if not merge_log:
        return scene_nodes, []

    groups: dict[str, list[str]] = defaultdict(list)
    for sn in scene_nodes:
        groups[_find(sn.scene_node_id)].append(sn.scene_node_id)

    merged: list[SceneNode] = []
    for root, members in groups.items():
        if len(members) == 1:
            merged.append(sn_by_id[root])
            continue
        keeper = sn_by_id[root]
        all_obs = list(keeper.supporting_observed_ids)
        all_imgs = set(keeper.support_images)
        for mid in members:
            if mid == root:
                continue
            m = sn_by_id[mid]
            all_obs.extend(m.supporting_observed_ids)
            all_imgs.update(m.support_images)
        merged.append(SceneNode(
            scene_node_id=keeper.scene_node_id,
            fused_code=keeper.fused_code,
            fused_name=keeper.fused_name,
            fused_confidence=keeper.fused_confidence,
            support_count=len(set(all_obs)),
            support_images=sorted(all_imgs),
            supporting_observed_ids=list(set(all_obs)),
            local_structural_signature=keeper.local_structural_signature,
            max_degree=keeper.max_degree,
            anchor_hint_score=keeper.anchor_hint_score,
            representative_branch_roles=keeper.representative_branch_roles,
            representative_triplet_hints=keeper.representative_triplet_hints,
        ))

    n_merged = len(scene_nodes) - len(merged)
    print(f"  [SpuriousMerge] merged {n_merged} spurious singleton(s)")
    for e in merge_log:
        print(f"    {e['dropped']} -> {e['kept']}  {e['reason']}")
    return merged, merge_log
