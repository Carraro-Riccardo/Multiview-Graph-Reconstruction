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
    COOC_COUNTS,
    COMPONENT_INFO,
    FORBIDDEN_PAIRS,
    VGGT_3D_ADJACENCY_FACTOR,
    VGGT_3D_AXIS_ALIGNMENT_MIN,
    VGGT_3D_MAX_POINTS_PER_OBSERVATION,
    VGGT_3D_MERGE_CONFIDENT_FACTOR,
    VGGT_3D_MERGE_LOW_OBS_THRESHOLD,
    VGGT_3D_MERGE_TENTATIVE_FACTOR,
    VGGT_3D_MIN_POINTS_PER_OBSERVATION,
    VGGT_3D_MIN_SCORE,
    VGGT_3D_MUTUAL_TOP_K,
    VGGT_3D_PORT_DISTANCE_FACTOR,
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


def build_vggt_artifact_request(image_ids: list[str]) -> dict:
    """Describe the minimal artefacts expected from a future VGGT export."""
    return {
        "format_version": 1,
        "description": "Per-view VGGT point maps required by the edge builder.",
        "supported_formats": [
            {
                "suffix": ".npy",
                "shape": "(H, W, 3)",
                "dtype": "float32/float64",
                "example_name": f"{image_ids[0]}.npy" if image_ids else "nodeX_0001.npy",
            },
            {
                "suffix": ".npz",
                "keys": ["world_points", "points3d", "points", "xyz", "pointmap"],
                "shape": "(H, W, 3)",
                "example_name": f"{image_ids[0]}.npz" if image_ids else "nodeX_0001.npz",
            },
            {
                "suffix": ".json",
                "keys": ["world_points", "points3d", "points", "xyz", "pointmap"],
                "shape": "(H, W, 3)",
                "example_name": f"{image_ids[0]}.json" if image_ids else "nodeX_0001.json",
            },
        ],
        "expected_images": [
            {
                "image_id": image_id,
                "accepted_names": [
                    f"{image_id}.npy",
                    f"{image_id}.npz",
                    f"{image_id}.json",
                    f"{image_id}_world_points.npy",
                    f"{image_id}_world_points.npz",
                    f"{image_id}_world_points.json",
                    f"{image_id}_pointmap.npy",
                    f"{image_id}_pointmap.npz",
                    f"{image_id}_pointmap.json",
                ],
            }
            for image_id in image_ids
        ],
    }


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


def _safe_unit(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return None
    return (vec / norm).astype(np.float32)


def _align_axis(axis: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    if reference is None:
        return axis
    if float(np.dot(axis, reference)) < 0.0:
        return -axis
    return axis


def _estimate_observation_geometry(
    pointmap: np.ndarray,
    bbox_norm: list[float],
) -> dict | None:
    h, w = pointmap.shape[:2]
    x1, y1, x2, y2 = bbox_norm
    px1 = max(0, min(w - 1, int(np.floor(x1 * w))))
    py1 = max(0, min(h - 1, int(np.floor(y1 * h))))
    px2 = max(px1 + 1, min(w, int(np.ceil(x2 * w))))
    py2 = max(py1 + 1, min(h, int(np.ceil(y2 * h))))

    crop = pointmap[py1:py2, px1:px2]
    if crop.size == 0:
        return None

    pts = crop.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < VGGT_3D_MIN_POINTS_PER_OBSERVATION:
        return None

    if len(pts) > VGGT_3D_MAX_POINTS_PER_OBSERVATION:
        step = max(1, len(pts) // VGGT_3D_MAX_POINTS_PER_OBSERVATION)
        pts = pts[::step][:VGGT_3D_MAX_POINTS_PER_OBSERVATION]

    centroid = np.median(pts, axis=0).astype(np.float32)
    centered = pts - centroid
    cov = centered.T @ centered / max(len(centered) - 1, 1)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        axis = None
    else:
        axis = _safe_unit(eigvecs[:, int(np.argmax(eigvals))].astype(np.float32))

    port_a = centroid.copy()
    port_b = centroid.copy()
    if axis is not None:
        projections = centered @ axis
        lo = float(np.quantile(projections, 0.10))
        hi = float(np.quantile(projections, 0.90))
        port_a = (centroid + axis * lo).astype(np.float32)
        port_b = (centroid + axis * hi).astype(np.float32)

    return {
        "centroid": centroid,
        "axis": axis,
        "ports": [port_a, port_b],
        "n_points": int(len(pts)),
    }


def _aggregate_scene_geometry(obs_geometry: dict[str, dict], scene_nodes: list) -> dict[str, dict]:
    scene_geom: dict[str, dict] = {}
    for sn in scene_nodes:
        geoms = [
            obs_geometry[obs_id]
            for obs_id in sn.supporting_observed_ids
            if obs_id in obs_geometry
        ]
        if not geoms:
            continue

        centroids = np.stack([g["centroid"] for g in geoms], axis=0)
        centroid = np.median(centroids, axis=0).astype(np.float32)

        ref_axis = None
        axes = []
        for geom in geoms:
            axis = geom.get("axis")
            if axis is None:
                continue
            if ref_axis is None:
                ref_axis = axis
            axes.append(_align_axis(axis, ref_axis))
        axis = _safe_unit(np.mean(np.stack(axes, axis=0), axis=0)) if axes else None

        ports = []
        if axis is not None:
            projected = []
            for geom in geoms:
                for port in geom["ports"]:
                    projected.append(float(np.dot(port - centroid, axis)))
            if projected:
                lo = float(np.quantile(projected, 0.15))
                hi = float(np.quantile(projected, 0.85))
                ports = [
                    (centroid + axis * lo).astype(np.float32),
                    (centroid + axis * hi).astype(np.float32),
                ]
        if not ports:
            ports = [centroid.copy(), centroid.copy()]

        scene_geom[sn.scene_node_id] = {
            "centroid": centroid,
            "axis": axis,
            "ports": ports,
            "n_observations": len(geoms),
        }
    return scene_geom


def _scene_cutoff(scene_geom: dict[str, dict]) -> float | None:
    if len(scene_geom) < 2:
        return None
    pts = np.stack([geom["centroid"] for geom in scene_geom.values()], axis=0)
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    nn = dists.min(axis=1)
    nn = nn[np.isfinite(nn)]
    if len(nn) == 0:
        return None
    return float(np.median(nn)) * VGGT_3D_ADJACENCY_FACTOR


def _knn_rank_map(scene_geom: dict[str, dict]) -> dict[str, list[str]]:
    if len(scene_geom) < 2:
        return {}
    ids = list(scene_geom.keys())
    pts = np.stack([scene_geom[sid]["centroid"] for sid in ids], axis=0)
    dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    rank_map: dict[str, list[str]] = {}
    for idx, sid in enumerate(ids):
        order = np.argsort(dists[idx])
        rank_map[sid] = [ids[j] for j in order[:VGGT_3D_MUTUAL_TOP_K] if np.isfinite(dists[idx, j])]
    return rank_map


def _port_distance(geom_a: dict, geom_b: dict) -> float:
    best = float("inf")
    for port_a in geom_a["ports"]:
        for port_b in geom_b["ports"]:
            best = min(best, float(np.linalg.norm(port_a - port_b)))
    return best


def _axis_alignment(geom_a: dict, geom_b: dict) -> float:
    axis_a = geom_a.get("axis")
    axis_b = geom_b.get("axis")
    if axis_a is None or axis_b is None:
        return 0.0
    return float(abs(np.dot(axis_a, axis_b)))


def _pair_geometry_stats(geom_a: dict, geom_b: dict) -> dict:
    centroid_dist = float(np.linalg.norm(geom_a["centroid"] - geom_b["centroid"]))
    port_dist = _port_distance(geom_a, geom_b)
    axis_alignment = _axis_alignment(geom_a, geom_b)
    return {
        "centroid_dist": centroid_dist,
        "port_dist": port_dist,
        "axis_alignment": axis_alignment,
    }


def _merge_duplicate_scene_nodes(
    scene_nodes: list,
    scene_geom: dict[str, dict],
    cutoff: float | None,
) -> tuple[list, list[dict]]:
    """Merge scene nodes of the same component type whose VGGT centroids are too close.

    Two merge criteria (both require identical fused_code):
    - Confident merge  : distance < CONFIDENT_FACTOR * cutoff  (any n_observations)
    - Tentative merge  : distance < TENTATIVE_FACTOR * cutoff
                         AND min(n_obs_a, n_obs_b) <= LOW_OBS_THRESHOLD

    Singleton nodes (n_obs=1) are allowed as merge candidates in both regimes,
    but only when their single image is NOT already present in the target cluster
    (image-conflict guard). This prevents merging two distinct components that
    happen to be spatially close but are both visible in the same view.

    Exception: if the singleton is *ultra-close* (dist < 2 cm and < 20% of
    confident_thr) it is almost certainly a duplicate detection of the same
    physical component in the same image, so the image-conflict guard is lifted.

    The node with more observations is kept; its supporting_observed_ids absorbs the other.
    Returns (updated_scene_nodes, merge_log).
    """
    if cutoff is None or len(scene_nodes) < 2:
        return scene_nodes, []

    confident_thr = cutoff * VGGT_3D_MERGE_CONFIDENT_FACTOR
    tentative_thr = cutoff * VGGT_3D_MERGE_TENTATIVE_FACTOR
    # Threshold below which an image-conflicted singleton is treated as a pure
    # duplicate detection (override the image-conflict guard).
    dup_override_thr = min(confident_thr * 0.20, 0.020)

    sn_by_id = {sn.scene_node_id: sn for sn in scene_nodes}

    by_code: dict[int, list[str]] = defaultdict(list)
    for sn in scene_nodes:
        if sn.scene_node_id in scene_geom:
            by_code[sn.fused_code].append(sn.scene_node_id)

    to_drop: set[str] = set()
    merge_log: list[dict] = []

    for node_ids in by_code.values():
        if len(node_ids) < 2:
            continue
        # Sort descending by n_observations so the most-observed node is always "a"
        node_ids_sorted = sorted(
            node_ids,
            key=lambda nid: scene_geom[nid]["n_observations"],
            reverse=True,
        )
        for i in range(len(node_ids_sorted)):
            a_id = node_ids_sorted[i]
            if a_id in to_drop:
                continue
            centroid_a = scene_geom[a_id]["centroid"]
            n_obs_a = scene_geom[a_id]["n_observations"]
            for j in range(i + 1, len(node_ids_sorted)):
                b_id = node_ids_sorted[j]
                if b_id in to_drop:
                    continue
                centroid_b = scene_geom[b_id]["centroid"]
                n_obs_b = scene_geom[b_id]["n_observations"]

                dist = float(np.linalg.norm(centroid_a - centroid_b))

                if dist <= confident_thr:
                    reason = "confident"
                elif dist <= tentative_thr and n_obs_b <= VGGT_3D_MERGE_LOW_OBS_THRESHOLD:
                    reason = "tentative_low_obs"
                else:
                    continue

                # For singleton nodes (n_obs=1) guard against image conflicts:
                # if the singleton's only image is already in the target cluster,
                # they are different detections in the same view → distinct components.
                # Exception: ultra-close distance → almost certainly a duplicate detection.
                if n_obs_b == 1:
                    singleton_img = sn_by_id[b_id].supporting_observed_ids[0].rsplit("_", 1)[0]
                    cluster_imgs  = {oid.rsplit("_", 1)[0]
                                     for oid in sn_by_id[a_id].supporting_observed_ids}
                    if singleton_img in cluster_imgs and dist > dup_override_thr:
                        continue
                    if singleton_img in cluster_imgs and dist <= dup_override_thr:
                        reason = "dup_override"

                sn_a = sn_by_id[a_id]
                sn_b = sn_by_id[b_id]
                # Absorb b's observations into a
                sn_a.supporting_observed_ids = list(
                    set(sn_a.supporting_observed_ids) | set(sn_b.supporting_observed_ids)
                )
                to_drop.add(b_id)
                merge_log.append({
                    "kept": a_id,
                    "dropped": b_id,
                    "dist_m": round(dist, 4),
                    "n_obs_kept": n_obs_a,
                    "n_obs_dropped": n_obs_b,
                    "reason": reason,
                })

    if not to_drop:
        return scene_nodes, []

    merged = [sn for sn in scene_nodes if sn.scene_node_id not in to_drop]
    return merged, merge_log


# ── Majority-vote code correction ─────────────────────────────────────────────

def _majority_vote_code(
    obs_ids: list[str],
    obs_to_code: dict[str, int],
    neighbor_codes: list[int],
) -> int | None:
    """Return the majority-vote hydraulic code for a set of observations.

    When votes are tied, use COOC_COUNTS compatibility with co-occurring
    neighbor codes to break the tie. Falls back to the first winner if no
    COOC data is available.
    """
    from collections import Counter
    votes: Counter = Counter(obs_to_code[oid] for oid in obs_ids if oid in obs_to_code)
    if not votes:
        return None
    max_count = max(votes.values())
    winners = [code for code, cnt in votes.items() if cnt == max_count]
    if len(winners) == 1:
        return winners[0]
    if neighbor_codes:
        scores = {
            code: sum(
                COOC_COUNTS.get((min(code, n), max(code, n)), 0)
                for n in neighbor_codes
            )
            for code in winners
        }
        return max(scores, key=lambda c: scores[c])
    return winners[0]


def _apply_majority_vote_codes(
    scene_nodes: list,
    observed_graphs: list,
) -> tuple[list, list[dict]]:
    """Recompute fused_code / fused_name for each scene node via majority vote.

    Returns (scene_nodes, vote_log) where vote_log records nodes whose code
    was corrected.  scene_nodes is the same list, mutated in place.
    """
    obs_to_code: dict[str, int] = {}
    for og in observed_graphs:
        for obs_id, obs_node in og.nodes.items():
            obs_to_code[obs_id] = obs_node.code

    # Build image → set-of-scene-nodes map for neighbour-code lookup
    img_to_sn_ids: dict[str, set[str]] = defaultdict(set)
    for sn in scene_nodes:
        for oid in sn.supporting_observed_ids:
            img = oid.rsplit("_", 1)[0]
            img_to_sn_ids[img].add(sn.scene_node_id)

    sn_by_id = {sn.scene_node_id: sn for sn in scene_nodes}

    vote_log: list[dict] = []
    for sn in scene_nodes:
        # Neighbour codes: codes of other scene nodes co-visible in the same images
        neighbour_codes: list[int] = []
        for oid in sn.supporting_observed_ids:
            img = oid.rsplit("_", 1)[0]
            for nid in img_to_sn_ids[img]:
                if nid != sn.scene_node_id and nid in sn_by_id:
                    neighbour_codes.append(sn_by_id[nid].fused_code)

        new_code = _majority_vote_code(sn.supporting_observed_ids, obs_to_code, neighbour_codes)
        if new_code is None or new_code == sn.fused_code:
            continue

        old_code = sn.fused_code
        sn.fused_code = new_code
        info = COMPONENT_INFO.get(new_code, {})
        sn.fused_name = info.get("name", str(new_code))
        vote_log.append({
            "scene_node_id": sn.scene_node_id,
            "old_code":      old_code,
            "new_code":      new_code,
            "n_obs":         len(sn.supporting_observed_ids),
        })

    return scene_nodes, vote_log


def replay_merge_log(
    scene_nodes: list,
    merge_log: list[dict],
    observed_graphs: list,
) -> list:
    """Re-apply a previously computed merge_log to a fresh set of SceneNodes.

    Used when loading VGGT results from cache: the merge decisions are stored in
    vggt_edge_debug.json and replayed here so that scene_nodes reflects the same
    merged state as if build_vggt_edge_hints had run from scratch.
    Entries whose IDs are not found in the current scene_nodes are silently skipped.
    """
    sn_by_id = {sn.scene_node_id: sn for sn in scene_nodes}
    obs_to_image: dict[str, str] = {
        obs_id: og.image_id
        for og in observed_graphs
        for obs_id in og.nodes
    }

    to_drop: set[str] = set()
    for entry in merge_log:
        kept_id = entry["kept"]
        dropped_id = entry["dropped"]
        if kept_id not in sn_by_id or dropped_id not in sn_by_id:
            continue
        sn_kept = sn_by_id[kept_id]
        sn_dropped = sn_by_id[dropped_id]
        sn_kept.supporting_observed_ids = list(
            set(sn_kept.supporting_observed_ids) | set(sn_dropped.supporting_observed_ids)
        )
        to_drop.add(dropped_id)

    if to_drop:
        for sn in scene_nodes:
            if sn.scene_node_id in to_drop:
                continue
            imgs = sorted({
                obs_to_image[oid]
                for oid in sn.supporting_observed_ids
                if oid in obs_to_image
            })
            sn.support_images = imgs
            sn.support_count = len(sn.supporting_observed_ids)

    return [sn for sn in scene_nodes if sn.scene_node_id not in to_drop]


def build_vggt_edge_hints(
    scene_nodes: list,
    observed_graphs: list,
    search_roots: list[Path],
) -> dict:
    """Estimate scene-node 3D geometry and candidate edge hints from VGGT point maps."""
    obs_to_scene: dict[str, str] = {}
    sn_code: dict[str, int] = {}
    for sn in scene_nodes:
        sn_code[sn.scene_node_id] = sn.fused_code
        for obs_id in sn.supporting_observed_ids:
            obs_to_scene[obs_id] = sn.scene_node_id

    image_ids = [og.image_id for og in observed_graphs]
    obs_geometry: dict[str, dict] = {}
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
            geom = _estimate_observation_geometry(pointmap, list(obs_node.bbox))
            if geom is None:
                continue
            obs_geometry[obs_id] = geom

    scene_geom = _aggregate_scene_geometry(obs_geometry, scene_nodes)
    cutoff = _scene_cutoff(scene_geom)

    # ── Duplicate-node merge via VGGT 3D centroids (pass 1) ─────────────────
    scene_nodes, merge_log = _merge_duplicate_scene_nodes(scene_nodes, scene_geom, cutoff)
    vote_log: list[dict] = []

    def _refresh_support(sns):
        obs_to_image: dict[str, str] = {
            obs_id: og.image_id
            for og in observed_graphs
            for obs_id in og.nodes
        }
        for sn in sns:
            imgs = sorted({
                obs_to_image[oid]
                for oid in sn.supporting_observed_ids
                if oid in obs_to_image
            })
            sn.support_images = imgs
            sn.support_count = len(sn.supporting_observed_ids)

    if merge_log:
        _refresh_support(scene_nodes)
        # Rebuild obs→scene and code maps on merged nodes
        obs_to_scene = {}
        sn_code = {}
        for sn in scene_nodes:
            sn_code[sn.scene_node_id] = sn.fused_code
            for obs_id in sn.supporting_observed_ids:
                obs_to_scene[obs_id] = sn.scene_node_id
        # Recompute geometry on merged nodes
        scene_geom = _aggregate_scene_geometry(obs_geometry, scene_nodes)
        cutoff = _scene_cutoff(scene_geom)

    # ── Majority-vote code correction ────────────────────────────────────────
    scene_nodes, vote_log = _apply_majority_vote_codes(scene_nodes, observed_graphs)

    # ── Second merge pass: picks up same-code nodes after code corrections ───
    merge_log_2: list[dict] = []
    if vote_log:
        scene_geom = _aggregate_scene_geometry(obs_geometry, scene_nodes)
        cutoff = _scene_cutoff(scene_geom)
        scene_nodes, merge_log_2 = _merge_duplicate_scene_nodes(scene_nodes, scene_geom, cutoff)
        if merge_log_2:
            _refresh_support(scene_nodes)
            scene_geom = _aggregate_scene_geometry(obs_geometry, scene_nodes)
            cutoff = _scene_cutoff(scene_geom)
        merge_log = merge_log + merge_log_2

    if merge_log or vote_log:
        # Rebuild obs→scene and code maps after all merges / corrections
        obs_to_scene = {}
        sn_code = {}
        for sn in scene_nodes:
            sn_code[sn.scene_node_id] = sn.fused_code
            for obs_id in sn.supporting_observed_ids:
                obs_to_scene[obs_id] = sn.scene_node_id
    # ────────────────────────────────────────────────────────────────────────

    rank_map = _knn_rank_map(scene_geom)
    port_cutoff = (cutoff * VGGT_3D_PORT_DISTANCE_FACTOR) if cutoff is not None else None

    pair_view_stats: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for og in observed_graphs:
        image_id = og.image_id
        per_scene_obs: dict[str, str] = {}
        for obs_id in og.nodes:
            sn_id = obs_to_scene.get(obs_id)
            if sn_id is None or obs_id not in obs_geometry:
                continue
            per_scene_obs.setdefault(sn_id, obs_id)

        present_scene_nodes = sorted(per_scene_obs.keys())
        for i in range(len(present_scene_nodes)):
            for j in range(i + 1, len(present_scene_nodes)):
                a = present_scene_nodes[i]
                b = present_scene_nodes[j]
                pair_view_stats[_norm_scene_pair(a, b)][image_id] = _pair_geometry_stats(
                    obs_geometry[per_scene_obs[a]],
                    obs_geometry[per_scene_obs[b]],
                )

    pair_hints: dict[tuple[str, str], dict] = {}
    if cutoff is not None and port_cutoff is not None:
        for pair, view_stats in pair_view_stats.items():
            a, b = pair
            code_pair = _norm_code_pair(sn_code.get(a, 0), sn_code.get(b, 0))
            if code_pair in FORBIDDEN_PAIRS:
                continue
            if a not in scene_geom or b not in scene_geom:
                continue

            centroid_distances = [stats["centroid_dist"] for stats in view_stats.values()]
            port_distances = [stats["port_dist"] for stats in view_stats.values()]
            alignments = [stats["axis_alignment"] for stats in view_stats.values()]
            if not centroid_distances or not port_distances:
                continue

            geom_scene_stats = _pair_geometry_stats(scene_geom[a], scene_geom[b])
            good_views = []
            for img_id, stats in view_stats.items():
                port_ok = stats["port_dist"] <= port_cutoff
                centroid_ok = stats["centroid_dist"] <= cutoff
                align_ok = stats["axis_alignment"] >= VGGT_3D_AXIS_ALIGNMENT_MIN
                if port_ok and (centroid_ok or align_ok):
                    good_views.append(img_id)

            shared_views = len(view_stats)
            vote_count = len(good_views)
            score = vote_count / shared_views if shared_views > 0 else 0.0

            a_ranked = b in rank_map.get(a, [])
            b_ranked = a in rank_map.get(b, [])
            mutual_topk = bool(a_ranked and b_ranked)
            geom_bonus = 0.0
            if geom_scene_stats["port_dist"] <= port_cutoff:
                geom_bonus += 0.15
            if geom_scene_stats["axis_alignment"] >= VGGT_3D_AXIS_ALIGNMENT_MIN:
                geom_bonus += 0.10
            final_score = min(1.0, score + geom_bonus)

            if final_score < VGGT_3D_MIN_SCORE and not (mutual_topk and vote_count > 0):
                continue
            if vote_count == 0:
                continue

            pair_hints[pair] = {
                "score": float(final_score),
                "raw_score": float(score),
                "shared_views": int(shared_views),
                "vote_count": int(vote_count),
                "supporting_images": sorted(good_views),
                "mean_dist": float(np.mean(centroid_distances)),
                "min_dist": float(np.min(centroid_distances)),
                "max_dist": float(np.max(centroid_distances)),
                "mean_port_dist": float(np.mean(port_distances)),
                "min_port_dist": float(np.min(port_distances)),
                "axis_alignment": float(np.mean(alignments)),
                "scene_port_dist": float(geom_scene_stats["port_dist"]),
                "scene_axis_alignment": float(geom_scene_stats["axis_alignment"]),
                "cutoff": float(cutoff),
                "port_cutoff": float(port_cutoff),
                "mutual_topk": mutual_topk,
                "scene_centroid_a": scene_geom[a]["centroid"].tolist(),
                "scene_centroid_b": scene_geom[b]["centroid"].tolist(),
                "scene_ports_a": [port.tolist() for port in scene_geom[a]["ports"]],
                "scene_ports_b": [port.tolist() for port in scene_geom[b]["ports"]],
                "scene_axis_a": scene_geom[a]["axis"].tolist() if scene_geom[a]["axis"] is not None else None,
                "scene_axis_b": scene_geom[b]["axis"].tolist() if scene_geom[b]["axis"] is not None else None,
            }

    return {
        "enabled": True,
        "search_roots": [str(path) for path in search_roots],
        "artifact_request": build_vggt_artifact_request(image_ids),
        "loaded_images": loaded_images,
        "skipped_images": skipped_images,
        "n_loaded_images": len(loaded_images),
        "n_observation_geometries": len(obs_geometry),
        "n_scene_geometries": len(scene_geom),
        "n_merged_nodes": len(merge_log),
        "merge_log": merge_log,
        "_scene_nodes_merged": scene_nodes,  # updated SceneNode list for orchestrator
        "scene_cutoff": cutoff,
        "port_cutoff": port_cutoff,
        "scene_geometry": {
            sid: {
                "centroid": geom["centroid"].tolist(),
                "ports": [port.tolist() for port in geom["ports"]],
                "axis": geom["axis"].tolist() if geom["axis"] is not None else None,
                "n_observations": geom["n_observations"],
            }
            for sid, geom in scene_geom.items()
        },
        "pair_hints": pair_hints,
        "pair_hints_serialized": [
            {
                "pair": [a, b],
                **hint,
            }
            for (a, b), hint in sorted(
                pair_hints.items(),
                key=lambda item: (-item[1]["score"], -item[1]["vote_count"], item[1]["mean_port_dist"]),
            )
        ],
    }
