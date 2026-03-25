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
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from scipy.optimize import linear_sum_assignment

from .branch_fragments import is_anchor, is_terminal
from config import COMPONENT_INFO


# ── RE-ID hyper-parameters ──────────────────────────────────────────────────
REID_MATCH_THRESHOLD      = 0.55   # min cosine sim for seed assignment acceptance
REID_ALLOW_UNMATCHED      = True   # skip weak matches instead of forcing them

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
                        S[r, c] = _cosine_sim(groups[gid]["prototype"], item_by_id[qid]["embedding"])

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
        imgs1 = {item_by_id[i]["image_id"] for i in ids1}
        imgs2 = {item_by_id[i]["image_id"] for i in ids2}
        if imgs1 & imgs2:
            return False, None
        p1, p2 = _proto(ids1), _proto(ids2)
        proto_sim   = _cosine_sim(p1, p2)
        cross_coass = _cross_coassoc(ids1, ids2)
        score = SOFT_MERGE_ALPHA * proto_sim + SOFT_MERGE_BETA * cross_coass
        if proto_sim   < SOFT_MERGE_MIN_PROTO_SIM:      return False, None
        if cross_coass < SOFT_MERGE_MIN_CROSS_COASSOC:  return False, None
        if score       < SOFT_MERGE_MIN_COMBINED_SCORE: return False, None
        return True, {"proto_sim": proto_sim, "cross_coassoc": cross_coass, "score": score}

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
        ids_union = list(dict.fromkeys(active[gid1]["member_ids"] + active[gid2]["member_ids"]))

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
def identify_scene_nodes(
    observed_graphs: list,
    image_paths: list[Path] | None = None,
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

    # Soft post-merge
    groups, groups_by_code = _run_soft_post_merge(groups, groups_by_code, item_by_id, pair_stats)

    # Assemble SceneNode objects
    scene_nodes = _build_scene_nodes(groups, item_by_id, observed_graphs)

    n_seed_runs = 0 if REID_USE_DIRECT_PAIRWISE else len(single_seed_runs)
    debug = {
        "mode":           "direct_pairwise" if REID_USE_DIRECT_PAIRWISE else "seed_coassoc",
        "n_seed_runs":    n_seed_runs,
        "n_pair_stats":   len(pair_stats),
        "n_scene_nodes":  len(scene_nodes),
        "observed_to_scene": {
            iid: sn.scene_node_id
            for sn in scene_nodes
            for iid in sn.supporting_observed_ids
        },
    }

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
