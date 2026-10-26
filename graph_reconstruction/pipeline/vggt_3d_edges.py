"""Estrazione della geometria 3D e degli indizi di arco da VGGT.

Il modulo è intenzionalmente conservativo: una point map mancante o illeggibile
esclude soltanto la vista interessata. Se non rimane geometria utilizzabile,
restituisce un payload vuoto e lascia agli stadi successivi la gestione del
fallback. Sono accettati array ``.npy``/``.npz`` e JSON con forma ``(H, W, 3)``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from graph_reconstruction.config import (
    COOC_COUNTS,
    COMPONENT_INFO,
    FORBIDDEN_PAIRS,
    VGGT_3D_ADJACENCY_FACTOR,
    VGGT_3D_MAX_POINTS_PER_OBSERVATION,
    VGGT_3D_MIN_POINTS_PER_OBSERVATION,
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
    """Restituisce, senza duplicati, i percorsi compatibili con cache vecchie e nuove."""
    vggt_dir_name = VGGT_EDGE_SEARCH_GLOB
    roots = [
        output_dir / vggt_dir_name,
        output_dir,
        work_dir / vggt_dir_name / f"node{node_id}",
        work_dir / vggt_dir_name,
    ]
    return _unique_paths(roots)


def build_vggt_artifact_request(image_ids: list[str]) -> dict:
    """Descrive il contratto minimo dei file attesi dal processo di export."""
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

    # Il campione compatto serve soltanto alla distanza minima della ReID:
    # 128 punti mantengono il confronto leggero senza conservare l'intero crop.
    if len(pts) > 128:
        idx = np.linspace(0, len(pts) - 1, 128, dtype=int)
        pts_sample = pts[idx].astype(np.float32)
    else:
        pts_sample = pts.astype(np.float32)

    return {
        "centroid":  centroid,
        "axis":      axis,
        "ports":     [port_a, port_b],
        "n_points":  int(len(pts)),
        "pts_sample": pts_sample,
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


def _build_view_rank_maps(
    pair_view_stats: dict[tuple[str, str], dict[str, dict]],
) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """Ordina, per ogni vista e nodo, i vicini in base alla distanza tra porte.

    Il risultato alimenta un punteggio basato sui ranghi, evitando una soglia
    assoluta in metri che dipenderebbe dalla scala della scena.
    """
    view_neighbors: dict = defaultdict(lambda: defaultdict(list))
    for (a, b), view_stats in pair_view_stats.items():
        for img_id, stats in view_stats.items():
            pd = stats["port_dist"]
            view_neighbors[img_id][a].append((b, pd))
            view_neighbors[img_id][b].append((a, pd))
    for img_id in view_neighbors:
        for sid in view_neighbors[img_id]:
            view_neighbors[img_id][sid].sort(key=lambda x: x[1])
    return view_neighbors


# ── Correzione del codice per voto di maggioranza ─────────────────────────────

def _majority_vote_code(
    obs_ids: list[str],
    obs_to_code: dict[str, int],
    neighbor_codes: list[int],
) -> int | None:
    """Sceglie il codice idraulico più votato tra le osservazioni.

    In caso di parità usa la compatibilità con i codici co-visibili; se non
    esistono dati di co-occorrenza mantiene il primo vincitore.
    """
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
    """Ricalcola codice e nome fusi, registrando soltanto le correzioni.

    La lista viene modificata in-place; ``vote_log`` permette di riprodurre le
    stesse correzioni quando la geometria viene caricata dalla cache.
    """
    obs_to_code: dict[str, int] = {}
    for og in observed_graphs:
        for obs_id, obs_node in og.nodes.items():
            obs_to_code[obs_id] = obs_node.code

    # La co-visibilità per immagine fornisce il contesto usato negli spareggi.
    img_to_sn_ids: dict[str, set[str]] = defaultdict(set)
    for sn in scene_nodes:
        for oid in sn.supporting_observed_ids:
            img = oid.rsplit("_", 1)[0]
            img_to_sn_ids[img].add(sn.scene_node_id)

    sn_by_id = {sn.scene_node_id: sn for sn in scene_nodes}

    vote_log: list[dict] = []
    for sn in scene_nodes:
        # Codici degli altri nodi visibili nelle stesse immagini.
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
        _, sn.fused_name = COMPONENT_INFO.get(new_code, (2, str(new_code)))
        vote_log.append({
            "scene_node_id": sn.scene_node_id,
            "old_code":      old_code,
            "new_code":      new_code,
            "n_obs":         len(sn.supporting_observed_ids),
        })

    return scene_nodes, vote_log


def replay_vote_log(scene_nodes: list, vote_log: list[dict]) -> list:
    """Riapplica a una ReID corrente le correzioni salvate nella cache VGGT.

    Le voci riferite a ID assenti o a un codice iniziale differente vengono
    ignorate: in quel caso la cache non descrive più lo stesso raggruppamento.
    """
    sn_by_id = {sn.scene_node_id: sn for sn in scene_nodes}
    for entry in vote_log:
        sn = sn_by_id.get(entry["scene_node_id"])
        if sn is None:
            continue
        # Il codice iniziale è parte del contratto della cache, non solo l'ID.
        if sn.fused_code != entry["old_code"]:
            continue
        sn.fused_code = entry["new_code"]
        _, sn.fused_name = COMPONENT_INFO.get(
            entry["new_code"],
            (2, str(entry["new_code"])),
        )
    return scene_nodes


def build_vggt_edge_hints(
    scene_nodes: list,
    observed_graphs: list,
    search_roots: list[Path],
) -> dict:
    """Stima geometria delle istanze e indizi di arco dalle point map VGGT."""
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

    # La correzione del codice precede lo scoring degli archi, perché prior e
    # coppie proibite dipendono dal codice fuso aggiornato.
    scene_nodes, vote_log = _apply_majority_vote_codes(scene_nodes, observed_graphs)

    if vote_log:
        # Dopo una correzione si rigenerano le mappe derivate dai codici.
        obs_to_scene = {}
        sn_code = {}
        for sn in scene_nodes:
            sn_code[sn.scene_node_id] = sn.fused_code
            for obs_id in sn.supporting_observed_ids:
                obs_to_scene[obs_id] = sn.scene_node_id
        scene_geom = _aggregate_scene_geometry(obs_geometry, scene_nodes)
        cutoff = _scene_cutoff(scene_geom)

    # Statistiche geometriche delle coppie co-visibili, mantenute per vista.
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

    # Il voto di una vista è 1/(rango_AB * rango_BA): vale 1 quando i nodi sono
    # primi vicini reciproci. Le co-occorrenze premiano coppie plausibili, mentre
    # FORBIDDEN_PAIRS resta un veto assoluto.
    _max_cooc = max(COOC_COUNTS.values()) if COOC_COUNTS else 1
    view_rank_maps = _build_view_rank_maps(pair_view_stats)

    pair_hints: dict[tuple[str, str], dict] = {}
    for pair, view_stats in pair_view_stats.items():
        a, b = pair
        code_pair = _norm_code_pair(sn_code.get(a, 0), sn_code.get(b, 0))
        if code_pair in FORBIDDEN_PAIRS:
            continue
        if a not in scene_geom or b not in scene_geom:
            continue

        centroid_distances = [stats["centroid_dist"] for stats in view_stats.values()]
        port_distances     = [stats["port_dist"]     for stats in view_stats.values()]
        alignments         = [stats["axis_alignment"] for stats in view_stats.values()]

        shared_views  = len(view_stats)
        total_vote    = 0.0
        vote_count    = 0
        supporting_imgs: list[str] = []

        for img_id in view_stats:
            neighbors_a = [sid for sid, _ in view_rank_maps[img_id].get(a, [])]
            neighbors_b = [sid for sid, _ in view_rank_maps[img_id].get(b, [])]
            rank_ab = (neighbors_a.index(b) + 1) if b in neighbors_a else (len(neighbors_a) + 1)
            rank_ba = (neighbors_b.index(a) + 1) if a in neighbors_b else (len(neighbors_b) + 1)
            total_vote += 1.0 / (rank_ab * rank_ba)
            if rank_ab <= VGGT_3D_MUTUAL_TOP_K and rank_ba <= VGGT_3D_MUTUAL_TOP_K:
                vote_count += 1
                supporting_imgs.append(img_id)

        raw_score  = total_vote / shared_views
        cooc_bonus = COOC_COUNTS.get(code_pair, 0) / _max_cooc
        final_score = min(1.0, raw_score * (1 + cooc_bonus))

        geom_scene_stats = _pair_geometry_stats(scene_geom[a], scene_geom[b])

        pair_hints[pair] = {
            "score":             float(final_score),
            "raw_score":         float(raw_score),
            "shared_views":      int(shared_views),
            "vote_count":        int(vote_count),
            "supporting_images": sorted(supporting_imgs),
            "mean_dist":         float(np.mean(centroid_distances)),
            "min_dist":          float(np.min(centroid_distances)),
            "max_dist":          float(np.max(centroid_distances)),
            "mean_port_dist":    float(np.mean(port_distances)),
            "min_port_dist":     float(np.min(port_distances)),
            "axis_alignment":    float(np.mean(alignments)),
            "scene_port_dist":   float(geom_scene_stats["port_dist"]),
            "scene_axis_alignment": float(geom_scene_stats["axis_alignment"]),
            "scene_centroid_a":  scene_geom[a]["centroid"].tolist(),
            "scene_centroid_b":  scene_geom[b]["centroid"].tolist(),
            "scene_ports_a":     [port.tolist() for port in scene_geom[a]["ports"]],
            "scene_ports_b":     [port.tolist() for port in scene_geom[b]["ports"]],
            "scene_axis_a":      scene_geom[a]["axis"].tolist() if scene_geom[a]["axis"] is not None else None,
            "scene_axis_b":      scene_geom[b]["axis"].tolist() if scene_geom[b]["axis"] is not None else None,
        }

    return {
        "enabled":                  True,
        "search_roots":             [str(path) for path in search_roots],
        "artifact_request":         build_vggt_artifact_request(image_ids),
        "loaded_images":            loaded_images,
        "skipped_images":           skipped_images,
        "n_loaded_images":          len(loaded_images),
        "n_observation_geometries": len(obs_geometry),
        "n_scene_geometries":       len(scene_geom),
        "n_code_corrections":       len(vote_log),
        "vote_log":                 vote_log,
        "_scene_nodes_after_vote":  scene_nodes,
        "scene_cutoff":             cutoff,
        "scene_geometry": {
            sid: {
                "centroid":       geom["centroid"].tolist(),
                "ports":          [port.tolist() for port in geom["ports"]],
                "axis":           geom["axis"].tolist() if geom["axis"] is not None else None,
                "n_observations": geom["n_observations"],
            }
            for sid, geom in scene_geom.items()
        },
        "pair_hints": pair_hints,
        "pair_hints_serialized": [
            {"pair": [a, b], **hint}
            for (a, b), hint in sorted(
                pair_hints.items(),
                key=lambda item: (-item[1]["score"], -item[1]["vote_count"], item[1]["min_port_dist"]),
            )
        ],
    }
