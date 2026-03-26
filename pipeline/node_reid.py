"""Step 2 – node re-identification.

For every component code the algorithm:
  1. Determines k (number of physical instances) = max detections of that code
     in any single image.  This is a structural observation: each image can
     see each physical component at most once (after NMS), so the densest
     single-image count equals the total number of instances.
  2. If k == 1: all detections belong to the one instance (trivial).
  3. If k > 1: cluster detections via K-Means on ResNet-50 patch embeddings,
     then apply the same-image hard constraint (two detections from the same
     image can never be the same instance).
  4. VGGT 3D distances (< VGGT_REID_CONFIDENT_DIST = 0.08 m) are used as a
     hard-merge bonus in the affinity matrix.

No dataset-tuned thresholds.  k comes directly from the detection structure;
the only physical constant is VGGT_REID_CONFIDENT_DIST.
"""

from __future__ import annotations

import cv2
import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .branch_fragments import is_anchor, is_terminal
from config import COMPONENT_INFO


# ── Patch extraction ──────────────────────────────────────────────────────────
PATCH_EXPAND_RATIO = 0.35
MIN_PATCH_SIZE     = 48

# ── VGGT 3D hard-merge threshold ─────────────────────────────────────────────
# Detections whose 3-D centroids are closer than this (metres) are the same
# physical object regardless of visual appearance.  This is a physical /
# sensor-noise constant, NOT a dataset-tuned hyperparameter.
VGGT_REID_ENABLED        = True
VGGT_REID_CONFIDENT_DIST = 0.08   # m


# ── SceneNode (downstream-compatible) ────────────────────────────────────────
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


# ── Embedding model (lazy singleton) ─────────────────────────────────────────
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


# ── Items: ObservedNode + visual embedding ────────────────────────────────────
def _build_items(
    observed_graphs: list,
    image_paths: list[Path],
) -> tuple[list[dict], dict[str, list[dict]]]:
    """Build one item per ObservedNode with a ResNet-50 visual embedding."""
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


# ── VGGT geometry ─────────────────────────────────────────────────────────────
def _load_obs_geometry(observed_graphs: list, search_roots: list) -> dict[str, dict]:
    """Extract per-detection 3D centroid from VGGT point maps."""
    try:
        from .vggt_3d_edges import (
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


# ── 3D cloud distance ────────────────────────────────────────────────────────
def _cloud_min_dist(ga: dict, gb: dict) -> float:
    """Minimum Euclidean distance between two point-cloud samples.

    Using the minimum rather than centroid-centroid distance is view-invariant:
    the same physical component seen from different angles exposes different
    surfaces, so centroids may differ by 0.3-0.6 m, but the closest points on
    the two clouds are always within the component's physical size (~5-15 cm).
    Falls back to centroid distance if no point-cloud sample is available.
    """
    pa = ga.get("pts_sample")
    pb = gb.get("pts_sample")
    if pa is None or pb is None:
        return float(np.linalg.norm(
            np.asarray(ga["centroid"]) - np.asarray(gb["centroid"])
        ))
    # Efficient batched min-distance: (m,1,3) - (1,n,3) → (m,n)
    diff = pa[:, None, :] - pb[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=2)).min())


# ── Per-code clustering ───────────────────────────────────────────────────────
def _enforce_same_image_split(
    labels: np.ndarray,
    code_ids: list[str],
    item_by_id: dict,
) -> np.ndarray:
    """If two detections from the same image share a label, reassign the
    lower-confidence one to a new singleton cluster."""
    labels = labels.copy()
    max_label = int(labels.max())
    seen: dict[tuple, int] = {}  # (image_id, label) → index

    for i, obs_id in enumerate(code_ids):
        img = item_by_id[obs_id]["image_id"]
        lbl = int(labels[i])
        key = (img, lbl)
        if key in seen:
            j = seen[key]
            conf_i = item_by_id[obs_id]["confidence"]
            conf_j = item_by_id[code_ids[j]]["confidence"]
            if conf_i >= conf_j:
                max_label += 1
                labels[j] = max_label
                seen[key] = i
            else:
                max_label += 1
                labels[i] = max_label
        else:
            seen[key] = i

    return labels


def _cluster_code(
    code: int,
    code_ids: list[str],
    item_by_id: dict,
    obs_geometry: dict[str, dict],
) -> list[list[str]]:
    """Cluster detections of one component code into scene-node groups.

    k = max detections of this code in any single image (structural estimate:
    each image sees each physical instance at most once after NMS, so the
    densest image count equals the scene's instance count).

    Affinity matrix (used by K-Means on spectral embedding):
      - same image              → 0.0  (hard constraint)
      - VGGT min-cloud < 0.08m → 0.98 (hard: certainly same instance)
      - both have embeddings    → max(0, cosine_similarity)
      - one/both missing emb    → 0.3  (neutral)
    """
    n = len(code_ids)
    if n == 1:
        return [list(code_ids)]

    # ── k from detection structure ────────────────────────────────────────────
    img_counts = Counter(item_by_id[oid]["image_id"] for oid in code_ids)
    k = max(img_counts.values())

    if k == 1:
        return [list(code_ids)]

    # ── Affinity matrix ───────────────────────────────────────────────────────
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = code_ids[i]
        ga = obs_geometry.get(a)
        for j in range(i + 1, n):
            b = code_ids[j]
            if item_by_id[a]["image_id"] == item_by_id[b]["image_id"]:
                continue  # same image → 0 (leave as zero)
            # VGGT hard merge
            val = 0.0
            if ga is not None:
                gb = obs_geometry.get(b)
                if gb is not None and _cloud_min_dist(ga, gb) <= VGGT_REID_CONFIDENT_DIST:
                    val = 0.98
            if val == 0.0:
                # Visual similarity
                ea = item_by_id[a].get("embedding")
                eb = item_by_id[b].get("embedding")
                if ea is not None and eb is not None:
                    val = max(0.0, float(np.dot(ea, eb)))
                else:
                    val = 0.3
            A[i, j] = A[j, i] = val

    # ── K-Means on normalised spectral embedding ──────────────────────────────
    # Normalised Laplacian → k smallest eigenvectors → K-Means
    d = A.sum(axis=1).clip(1e-10)
    D_isqrt = 1.0 / np.sqrt(d)
    L = np.eye(n) - (A * D_isqrt[:, None]) * D_isqrt[None, :]
    _, V = np.linalg.eigh(L)
    X = V[:, :k]
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(1e-10)
    X = X / norms

    from sklearn.cluster import KMeans
    labels = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(X)
    labels = _enforce_same_image_split(labels, code_ids, item_by_id)

    groups: dict[int, list[str]] = defaultdict(list)
    for idx, obs_id in enumerate(code_ids):
        groups[int(labels[idx])].append(obs_id)

    def _sort_key(cluster: list[str]) -> tuple:
        centers = [item_by_id[i]["center"] for i in cluster]
        return (
            -len(cluster),
            float(np.mean([c[1] for c in centers])),
            float(np.mean([c[0] for c in centers])),
        )

    return sorted(groups.values(), key=_sort_key)


# ── SceneNode assembly ────────────────────────────────────────────────────────
def _build_scene_nodes(groups: dict, item_by_id: dict, observed_graphs: list) -> list[SceneNode]:
    obs_node_by_id = {nid: node for og in observed_graphs for nid, node in og.nodes.items()}
    scene_nodes: list[SceneNode] = []

    for g in groups.values():
        gid        = g["group_id"]
        code       = g["code"]
        member_ids = g["member_ids"]
        valence, name = COMPONENT_INFO.get(code, (2, f"code_{code}"))

        support_images   = sorted({item_by_id[i]["image_id"] for i in member_ids})
        fused_confidence = float(np.mean([item_by_id[i]["confidence"] for i in member_ids]))

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
            "self_code":             code,
            "nearby_code_histogram": dict(Counter(all_nearby)),
            "anchor_hint_score":     anchor_hint_score,
            "terminal_hint_count":   terminal_count,
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


# ── Main entry point ──────────────────────────────────────────────────────────
def identify_scene_nodes(
    observed_graphs: list,
    image_paths: list[Path] | None = None,
    search_roots: list[Path] | None = None,
) -> tuple[list[SceneNode], dict]:
    from config import IMAGES_DIR

    if image_paths is None:
        image_paths = list(IMAGES_DIR.glob("*.jpg"))

    all_items, items_by_image = _build_items(observed_graphs, image_paths)
    item_by_id = {item["observed_node_id"]: item for item in all_items}

    if not all_items:
        return [], {"error": "no_items"}

    n_images = len(items_by_image)
    print(f"  [ReID] {len(all_items)} detections across {n_images} images")

    # VGGT 3D hard constraints
    obs_geometry: dict[str, dict] = {}
    if VGGT_REID_ENABLED and search_roots:
        obs_geometry = _load_obs_geometry(observed_graphs, list(search_roots))

    # Spectral clustering per component code
    items_by_code: dict[int, list[str]] = defaultdict(list)
    for item in all_items:
        items_by_code[item["code"]].append(item["observed_node_id"])

    groups:         dict[str, dict] = {}
    groups_by_code: dict[int, list[str]] = defaultdict(list)
    for code, code_ids in sorted(items_by_code.items()):
        clusters = _cluster_code(code, code_ids, item_by_id, obs_geometry)
        for k, cluster in enumerate(clusters, start=1):
            gid = f"{code}#{k}"
            groups[gid] = {"code": code, "member_ids": cluster, "group_id": gid}
            groups_by_code[code].append(gid)

    scene_nodes = _build_scene_nodes(groups, item_by_id, observed_graphs)

    _print_reid_summary(scene_nodes, all_items)
    return scene_nodes, {
        "n_scene_nodes": len(scene_nodes),
        "n_detections":  len(all_items),
        "n_images":      n_images,
        "n_vggt_geometries": len(obs_geometry),
    }


# ── Summary printer ───────────────────────────────────────────────────────────
def _print_reid_summary(scene_nodes: list, all_items: list) -> None:
    n_obs   = len(all_items)
    n_scene = len(scene_nodes)
    by_code: dict = defaultdict(list)
    for sn in scene_nodes:
        by_code[sn.fused_code].append(sn)

    col = 18
    print(f"\n  {'─' * 62}")
    print(f"  {'RE-ID SUMMARY':^62}")
    print(f"  {'─' * 62}")
    print(f"  {'code':<6}{'name':<{col}}{'groups':>7}{'obs':>6}{'views':>7}{'conf':>7}")
    print(f"  {'─' * 62}")
    for code in sorted(by_code):
        nodes    = by_code[code]
        name     = nodes[0].fused_name
        n_groups = len(nodes)
        total_obs = sum(sn.support_count for sn in nodes)
        all_imgs  = {img for sn in nodes for img in sn.support_images}
        avg_conf  = sum(sn.fused_confidence for sn in nodes) / n_groups
        print(f"  {code:<6}{name:<{col}}{n_groups:>7}{total_obs:>6}{len(all_imgs):>7}{avg_conf:>7.2f}")
    print(f"  {'─' * 62}")
    print(f"  total: {n_scene} scene nodes  |  {n_obs} detections")
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


# ── Post-VGGT spurious singleton removal ─────────────────────────────────────
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

    Level 3 – shared VGGT edge partner: two low-support nodes of the same code
    that share a strong VGGT-confirmed neighbour and are within scene_cutoff.
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

    # Build strong-partner index from VGGT pair hints
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

        # Level 3: shared VGGT edge partner
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
                candidate = (len(shared), other.support_count, -dist, other.scene_node_id)
                if best_shared is None or candidate > best_shared[0]:
                    best_shared = (candidate, other.scene_node_id, dist, shared)
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
                        "dropped": dropped_id, "kept": kept_id,
                        "reason": reason, "code": sn.fused_code,
                    })
                continue

        if kept_id is None or _find(sn.scene_node_id) == _find(kept_id):
            continue

        _union(sn.scene_node_id, kept_id)
        merge_log.append({
            "dropped": sn.scene_node_id, "kept": kept_id,
            "reason": reason, "code": sn.fused_code,
        })

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
