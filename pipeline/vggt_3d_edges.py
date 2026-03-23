"""Optional VGGT-backed 3D edge evidence for the direct graph builder.

This module is intentionally conservative:
- if no VGGT artefacts are found, it returns an empty payload
- if a point map cannot be parsed, that view is skipped
- 2D KNN adjacency remains the baseline signal

Supported point-map formats
---------------------------
- `.npy` / `.npz` with an array shaped `(H, W, 3)`
- `.json` containing one of:
  - `{"world_points": [[[x, y, z], ...], ...]}`
  - `{"points3d": ...}`
  - `{"points": ...}`
  - a raw nested list shaped `(H, W, 3)`
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from config import (
    FORBIDDEN_PAIRS,
    VGGT_3D_ADJACENCY_FACTOR,
    VGGT_3D_MAX_POINTS_PER_OBSERVATION,
    VGGT_3D_MIN_POINTS_PER_OBSERVATION,
    VGGT_3D_MIN_SCORE,
    VGGT_3D_MUTUAL_TOP_K,
    VGGT_EDGE_SEARCH_GLOB,
)


def _norm_scene_pair(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))


def _norm_code_pair(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def build_vggt_search_roots(node_id: int, output_dir: Path, work_dir: Path) -> list[Path]:
    """Return a conservative list of places where VGGT artefacts may live."""
    vggt_dir_name = VGGT_EDGE_SEARCH_GLOB
    roots = [
        output_dir / vggt_dir_name,
        output_dir,
        work_dir / vggt_dir_name / f"node{node_id}",
        work_dir / vggt_dir_name,
    ]
    return _unique_paths(roots)


def _candidate_files_for_image(image_id: str, search_roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        candidates.extend(root.rglob(f"{image_id}.npy"))
        candidates.extend(root.rglob(f"{image_id}.npz"))
        candidates.extend(root.rglob(f"{image_id}.json"))
        candidates.extend(root.rglob(f"{image_id}_world_points.npy"))
        candidates.extend(root.rglob(f"{image_id}_world_points.npz"))
        candidates.extend(root.rglob(f"{image_id}_world_points.json"))
        candidates.extend(root.rglob(f"{image_id}_pointmap.npy"))
        candidates.extend(root.rglob(f"{image_id}_pointmap.npz"))
        candidates.extend(root.rglob(f"{image_id}_pointmap.json"))
    return _unique_paths(candidates)


def _json_to_array(payload) -> np.ndarray | None:
    if isinstance(payload, dict):
        for key in ("world_points", "points3d", "points", "xyz", "pointmap"):
            if key in payload:
                return _json_to_array(payload[key])
        return None
    arr = np.asarray(payload, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    return None


def _load_pointmap(path: Path) -> np.ndarray | None:
    try:
        if path.suffix.lower() == ".npy":
            arr = np.load(path, allow_pickle=False)
        elif path.suffix.lower() == ".npz":
            archive = np.load(path, allow_pickle=False)
            arr = None
            for key in ("world_points", "points3d", "points", "xyz", "pointmap"):
                if key in archive:
                    arr = archive[key]
                    break
            if arr is None and archive.files:
                arr = archive[archive.files[0]]
        elif path.suffix.lower() == ".json":
            with path.open("r", encoding="utf-8") as fh:
                arr = _json_to_array(json.load(fh))
        else:
            return None
    except Exception:
        return None

    if arr is None:
        return None

    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        return None
    return arr


def _bbox_centroid_from_pointmap(
    pointmap: np.ndarray,
    bbox_norm: list[float],
) -> tuple[np.ndarray | None, int]:
    h, w = pointmap.shape[:2]
    x1, y1, x2, y2 = bbox_norm
    px1 = max(0, min(w - 1, int(np.floor(x1 * w))))
    py1 = max(0, min(h - 1, int(np.floor(y1 * h))))
    px2 = max(px1 + 1, min(w, int(np.ceil(x2 * w))))
    py2 = max(py1 + 1, min(h, int(np.ceil(y2 * h))))

    crop = pointmap[py1:py2, px1:px2]
    if crop.size == 0:
        return None, 0

    pts = crop.reshape(-1, 3)
    finite_mask = np.isfinite(pts).all(axis=1)
    pts = pts[finite_mask]
    if len(pts) < VGGT_3D_MIN_POINTS_PER_OBSERVATION:
        return None, int(len(pts))

    if len(pts) > VGGT_3D_MAX_POINTS_PER_OBSERVATION:
        step = max(1, len(pts) // VGGT_3D_MAX_POINTS_PER_OBSERVATION)
        pts = pts[::step][:VGGT_3D_MAX_POINTS_PER_OBSERVATION]

    centroid = np.median(pts, axis=0)
    return centroid.astype(np.float32), int(len(pts))


def _scene_centroids(obs_centroids: dict[str, np.ndarray], scene_nodes: list) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for sn in scene_nodes:
        pts = [
            obs_centroids[obs_id]
            for obs_id in sn.supporting_observed_ids
            if obs_id in obs_centroids
        ]
        if not pts:
            continue
        centroids[sn.scene_node_id] = np.median(np.stack(pts), axis=0).astype(np.float32)
    return centroids


def _scene_cutoff(scene_centroids: dict[str, np.ndarray]) -> float | None:
    if len(scene_centroids) < 2:
        return None
    pts = np.stack(list(scene_centroids.values()))
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)
    nn = nn[np.isfinite(nn)]
    if len(nn) == 0:
        return None
    return float(np.median(nn)) * VGGT_3D_ADJACENCY_FACTOR


def _knn_rank_map(scene_centroids: dict[str, np.ndarray]) -> dict[str, list[str]]:
    if len(scene_centroids) < 2:
        return {}
    ids = list(scene_centroids.keys())
    pts = np.stack([scene_centroids[sid] for sid in ids])
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    rank_map: dict[str, list[str]] = {}
    for idx, sid in enumerate(ids):
        order = np.argsort(dists[idx])
        rank_map[sid] = [ids[j] for j in order[:VGGT_3D_MUTUAL_TOP_K] if np.isfinite(dists[idx, j])]
    return rank_map


def build_vggt_edge_hints(
    scene_nodes: list,
    observed_graphs: list,
    search_roots: list[Path],
) -> dict:
    """Estimate scene-node 3D centroids and candidate edge hints from VGGT point maps."""
    obs_to_scene = {}
    sn_code = {}
    for sn in scene_nodes:
        sn_code[sn.scene_node_id] = sn.fused_code
        for obs_id in sn.supporting_observed_ids:
            obs_to_scene[obs_id] = sn.scene_node_id

    obs_centroids: dict[str, np.ndarray] = {}
    loaded_images: dict[str, str] = {}
    skipped_images: list[dict] = []

    for og in observed_graphs:
        image_id = og.image_id
        candidates = _candidate_files_for_image(image_id, search_roots)
        if not candidates:
            skipped_images.append({"image_id": image_id, "reason": "missing_pointmap"})
            continue

        pointmap = None
        chosen = None
        for path in candidates:
            pointmap = _load_pointmap(path)
            if pointmap is not None:
                chosen = path
                break
        if pointmap is None:
            skipped_images.append({"image_id": image_id, "reason": "unreadable_pointmap"})
            continue

        loaded_images[image_id] = str(chosen)
        for obs_id, obs_node in og.nodes.items():
            centroid, _ = _bbox_centroid_from_pointmap(pointmap, list(obs_node.bbox))
            if centroid is None:
                continue
            obs_centroids[obs_id] = centroid

    scene_centroids = _scene_centroids(obs_centroids, scene_nodes)
    cutoff = _scene_cutoff(scene_centroids)
    rank_map = _knn_rank_map(scene_centroids)

    pair_view_dists: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for og in observed_graphs:
        image_id = og.image_id
        present_scene_nodes: list[str] = []
        for obs_id in og.nodes:
            sn_id = obs_to_scene.get(obs_id)
            if sn_id is None or obs_id not in obs_centroids:
                continue
            present_scene_nodes.append(sn_id)
        present_scene_nodes = sorted(set(present_scene_nodes))
        for i in range(len(present_scene_nodes)):
            for j in range(i + 1, len(present_scene_nodes)):
                a = present_scene_nodes[i]
                b = present_scene_nodes[j]
                obs_a = next(
                    (oid for oid in og.nodes if obs_to_scene.get(oid) == a and oid in obs_centroids),
                    None,
                )
                obs_b = next(
                    (oid for oid in og.nodes if obs_to_scene.get(oid) == b and oid in obs_centroids),
                    None,
                )
                if obs_a is None or obs_b is None:
                    continue
                pair_view_dists[_norm_scene_pair(a, b)][image_id] = float(
                    np.linalg.norm(obs_centroids[obs_a] - obs_centroids[obs_b])
                )

    pair_hints: dict[tuple[str, str], dict] = {}
    if cutoff is not None:
        for pair, view_dists in pair_view_dists.items():
            a, b = pair
            code_pair = _norm_code_pair(sn_code.get(a, 0), sn_code.get(b, 0))
            if code_pair in FORBIDDEN_PAIRS:
                continue

            distances = list(view_dists.values())
            if not distances:
                continue

            close_views = [img_id for img_id, dist in view_dists.items() if dist <= cutoff]
            shared_views = len(view_dists)
            close_count = len(close_views)
            score = close_count / shared_views if shared_views > 0 else 0.0

            a_ranked = b in rank_map.get(a, [])
            b_ranked = a in rank_map.get(b, [])
            if score < VGGT_3D_MIN_SCORE and not (a_ranked and b_ranked and close_count > 0):
                continue
            if close_count == 0:
                continue

            pair_hints[pair] = {
                "score": float(score),
                "shared_views": int(shared_views),
                "vote_count": int(close_count),
                "supporting_images": sorted(close_views),
                "mean_dist": float(np.mean(distances)),
                "min_dist": float(np.min(distances)),
                "max_dist": float(np.max(distances)),
                "cutoff": float(cutoff),
                "mutual_topk": bool(a_ranked and b_ranked),
                "scene_centroid_a": scene_centroids.get(a).tolist() if a in scene_centroids else None,
                "scene_centroid_b": scene_centroids.get(b).tolist() if b in scene_centroids else None,
            }

    return {
        "enabled": True,
        "search_roots": [str(path) for path in search_roots],
        "loaded_images": loaded_images,
        "skipped_images": skipped_images,
        "n_loaded_images": len(loaded_images),
        "n_observation_centroids": len(obs_centroids),
        "n_scene_centroids": len(scene_centroids),
        "scene_cutoff": cutoff,
        "scene_centroids": {
            sid: centroid.tolist()
            for sid, centroid in scene_centroids.items()
        },
        "pair_hints": pair_hints,
        "pair_hints_serialized": [
            {
                "pair": [a, b],
                **hint,
            }
            for (a, b), hint in sorted(
                pair_hints.items(),
                key=lambda item: (-item[1]["vote_count"], item[1]["mean_dist"]),
            )
        ],
    }
