"""
Instance-level multi-view spatial co-occurrence for edge candidacy.

Problema che risolve
--------------------
Il segnale knn_residual in candidate_scene_graph.py è limitato dagli archi
del grafo KNN locale (K_NEIGHBORS = 3).  Quando N istanze dello stesso tipo
si raggruppano (es. 3 Giunti), ciascuna usa tutti e K i propri slot KNN sulle
altre istanze dello stesso tipo, e i collegamenti verso tipi diversi vengono
tagliati fuori.  Risultato: nodi disconnessi → structural_fallback.

Soluzione
---------
Per ogni coppia di scene nodes (A, B) che compaiono in almeno una vista
comune, calcoliamo direttamente la distanza 2D tra le loro detection in ogni
vista condivisa — senza il limite K.

Score edge = n_viste_in_cui_sono_vicini / n_viste_condivise

"vicini" = distanza ≤ SPATIAL_COOC_CUTOFF_FACTOR × cutoff_della_vista

Dove cutoff_della_vista = mediana(nn_distances) × DIST_MULTIPLIER
(stesso calcolo di observed_graphs.py).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from config import (
    COOC_COUNTS,
    COMPONENT_INFO,
    DIAMETER_CODES,
    DIST_MULTIPLIER,
    FORBIDDEN_PAIRS,
    K_NEIGHBORS,
)

# ── Config ────────────────────────────────────────────────────────────────────
# Fattore moltiplicativo del cutoff KNN per considerare due nodi "vicini"
# nella vista.  1.5 = entro 1.5x la distanza mediana al NN.
SPATIAL_COOC_CUTOFF_FACTOR: float = 1.8

# Soglia minima di score per emettere un arco candidato.
# 0.33 = vicini in almeno 1/3 delle viste condivise.
MIN_SCORE: float = 0.20

# Numero minimo di viste condivise per calcolare lo score.
MIN_SHARED_VIEWS: int = 1
# ─────────────────────────────────────────────────────────────────────────────


def _norm_pair(a: str, b: str) -> tuple[str, str]:
    return (min(a, b), max(a, b))


def _code_pair(a: int, b: int) -> tuple[int, int]:
    return (min(a, b), max(a, b))


def _view_knn_cutoff(obs_nodes: dict) -> float:
    """Reproduce the same cutoff as observed_graphs._compute_knn."""
    centers = np.array([n.center for n in obs_nodes.values()])
    if len(centers) < 2:
        return float("inf")
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    nn_dists = distances.min(axis=1)
    return float(np.median(nn_dists)) * DIST_MULTIPLIER


def build_instance_spatial_cooccurrence(
    scene_nodes: list,
    observed_graphs: list,
    existing_pairs: set[tuple[str, str]] | None = None,
) -> list[dict]:
    """
    Calcola la co-occorrenza spaziale a livello di istanza tra scene nodes.

    Parameters
    ----------
    scene_nodes    : lista di SceneNode dal re-ID
    observed_graphs: lista di ObservedLocalGraph (Step 1)
    existing_pairs : coppie già presenti nel candidate graph (si aggiunge a
                     queste, non si sostituisce)

    Returns
    -------
    Lista di dict con:
        scene_node_a, scene_node_b, code_a, code_b,
        shared_views, n_adjacent, score,
        supporting_images, distances
    """
    if existing_pairs is None:
        existing_pairs = set()

    # ── Mappa: (scene_node_id, image_id) → (cx, cy) ────────────────────────
    obs_to_scene: dict[str, str] = {}
    for sn in scene_nodes:
        for obs_id in sn.supporting_observed_ids:
            obs_to_scene[obs_id] = sn.scene_node_id

    sn_pos_per_view: dict[tuple[str, str], tuple[float, float]] = {}
    for og in observed_graphs:
        for obs_id, obs_node in og.nodes.items():
            sn_id = obs_to_scene.get(obs_id)
            if sn_id is not None:
                # Se uno scene node ha più detection nella stessa vista
                # (non dovrebbe, ma per sicurezza) tieni la prima
                key = (sn_id, og.image_id)
                if key not in sn_pos_per_view:
                    sn_pos_per_view[key] = obs_node.center

    # ── Cutoff per vista ────────────────────────────────────────────────────
    view_cutoffs: dict[str, float] = {}
    for og in observed_graphs:
        view_cutoffs[og.image_id] = _view_knn_cutoff(og.nodes)

    # ── Per ogni vista, trova quali scene_nodes sono presenti ───────────────
    view_to_sn: dict[str, list[str]] = defaultdict(list)
    for (sn_id, img_id) in sn_pos_per_view:
        view_to_sn[img_id].append(sn_id)

    # ── Calcola co-occorrenza spaziale per ogni coppia ──────────────────────
    sn_ids = [sn.scene_node_id for sn in scene_nodes]
    sn_code: dict[str, int] = {sn.scene_node_id: sn.fused_code for sn in scene_nodes}

    # Accumula per ogni coppia: {pair → {img_id → distance}}
    pair_view_dists: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)

    for img_id, sn_list in view_to_sn.items():
        cutoff = view_cutoffs.get(img_id, float("inf"))
        for i in range(len(sn_list)):
            for j in range(i + 1, len(sn_list)):
                a, b = sn_list[i], sn_list[j]
                pair = _norm_pair(a, b)
                pos_a = sn_pos_per_view[(a, img_id)]
                pos_b = sn_pos_per_view[(b, img_id)]
                dist = math.sqrt(
                    (pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2
                )
                pair_view_dists[pair][img_id] = dist

    # ── Costruisci i candidati ───────────────────────────────────────────────
    results: list[dict] = []

    for pair, view_dists in pair_view_dists.items():
        a, b = pair

        # Salta coppie già ben coperte da altri segnali (knn_residual / branch)
        if pair in existing_pairs:
            continue

        n_shared = len(view_dists)
        if n_shared < MIN_SHARED_VIEWS:
            continue

        ca = sn_code.get(a, 0)
        cb = sn_code.get(b, 0)

        # Controllo coppie vietate
        code_p = _code_pair(ca, cb)
        if code_p in FORBIDDEN_PAIRS:
            continue
        # Stesso codice (eccetto Tee rid. 255)
        if ca == cb and ca != 255:
            continue

        # Calcola quante viste li vedono "vicini"
        n_adjacent = 0
        adjacent_images: list[str] = []
        distances: list[float] = []

        for img_id, dist in view_dists.items():
            cutoff = view_cutoffs.get(img_id, float("inf"))
            distances.append(dist)
            if dist <= cutoff * SPATIAL_COOC_CUTOFF_FACTOR:
                n_adjacent += 1
                adjacent_images.append(img_id)

        score = n_adjacent / n_shared if n_shared > 0 else 0.0

        if score < MIN_SCORE:
            continue

        cooc = COOC_COUNTS.get(code_p, 0)
        results.append({
            "scene_node_a":     a,
            "scene_node_b":     b,
            "code_a":           ca,
            "code_b":           cb,
            "shared_views":     n_shared,
            "n_adjacent":       n_adjacent,
            "score":            score,
            "supporting_images": sorted(adjacent_images),
            "all_shared_images": sorted(view_dists.keys()),
            "min_dist":         min(distances) if distances else 0.0,
            "mean_dist":        float(np.mean(distances)) if distances else 0.0,
            "cooc_count":       cooc,
        })

    # Ordina per score desc, poi distanza media asc
    results.sort(key=lambda r: (-r["score"], r["mean_dist"]))
    return results


def debug_summary(results: list[dict], top_k: int = 20) -> str:
    lines = [
        f"Instance-level spatial co-occurrence: {len(results)} new candidate edges",
        f"  {'pair':<30} {'shared':>7} {'adj':>5} {'score':>7} {'min_d':>8} {'cooc':>6}",
        f"  {'─' * 70}",
    ]
    for r in results[:top_k]:
        pair = f"{r['scene_node_a']} — {r['scene_node_b']}"
        lines.append(
            f"  {pair:<30} {r['shared_views']:>7} {r['n_adjacent']:>5} "
            f"{r['score']:>7.3f} {r['min_dist']:>8.4f} {r['cooc_count']:>6}"
        )
    return "\n".join(lines)
