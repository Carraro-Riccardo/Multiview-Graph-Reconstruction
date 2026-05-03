"""Step 2 – node re-identification via graph coloring + constrained agglomerative clustering.

For every component code the algorithm:
  1. Loads VGGT 3D geometry (centroid + point-cloud sample) for every detection.
  2. Computes k_min via greedy graph coloring of the incompatibility graph:
     two detections are incompatible iff they appear in the same image.
     k_min = minimum physical instances required to explain the co-occurrence pattern.
  3. Runs constrained single-linkage agglomerative clustering:
     - starts from N singleton clusters
     - iteratively merges the closest pair (by _cloud_min_dist) that shares no image
     - stops when exactly k_min clusters remain
  4. Zero tunable parameters — k is determined by scene structure (cannot-link graph),
     not thresholds.

Fallback when VGGT geometry is unavailable: use graph coloring assignment directly.
"""

from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .branch_fragments import is_anchor, is_terminal
from config import COMPONENT_INFO


VGGT_REID_ENABLED = True


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


# ── Items: ObservedNode metadata (no visual embedding) ───────────────────────
def _build_items(
    observed_graphs: list,
    image_paths: list[Path] | None = None,  # kept for API compat, unused
) -> tuple[list[dict], dict[str, list[dict]]]:
    all_items: list[dict] = []
    items_by_image: dict[str, list[dict]] = defaultdict(list)

    for og in observed_graphs:
        image_id = og.image_id
        for obs in og.nodes.values():
            item = {
                "observed_node_id": obs.observed_node_id,
                "image_id":         image_id,
                "code":             obs.code,
                "confidence":       obs.confidence,
                "center":           obs.center,
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
    diff = pa[:, None, :] - pb[None, :, :]
    return float(np.sqrt((diff * diff).sum(axis=2)).min())


# ── Agglomerative per-code clustering (zero thresholds) ──────────────────────
def _cluster_code_agglomerative(
    code: int,
    code_ids: list[str],
    item_by_id: dict,
    obs_geometry: dict[str, dict],
) -> list[list[str]]:
    """Cluster detections of one component code into scene-node groups.

    Algorithm:
      1. Graph coloring → k_min (hard lower bound from cannot-link constraints).
      2. Run constrained agglomerative to k_min using rank fusion of two signals:
           - rank_3d: position in ascending sort by _cloud_min_dist
           - rank_sim: position in descending sort by Jaccard similarity of
             co-detected codes (codes of other components in the same image)
         At each step: merge pair with max score = 1 / (rank_3d × rank_sim).
         Pairs with no neighborhood data use only rank_3d (rank_sim term omitted).

    Zero hard-coded weights or thresholds. Ranks are always relative to the
    current candidate set. Consistent with rank-based edge scoring.
    Fallback (no VGGT geometry): return graph coloring assignment directly.
    """
    n = len(code_ids)
    if n == 1:
        return [list(code_ids)]

    img_of = {oid: item_by_id[oid]["image_id"] for oid in code_ids}

    # Graph coloring → k_min ──────────────────────────────────────────────────
    img_to_obs: dict[str, list[str]] = defaultdict(list)
    for oid in code_ids:
        img_to_obs[img_of[oid]].append(oid)
    incompat: dict[str, set[str]] = {oid: set() for oid in code_ids}
    for peers in img_to_obs.values():
        for i in range(len(peers)):
            for j in range(i + 1, len(peers)):
                incompat[peers[i]].add(peers[j])
                incompat[peers[j]].add(peers[i])
    color: dict[str, int] = {}
    for oid in sorted(code_ids, key=lambda x: -len(incompat[x])):
        used = {color[nb] for nb in incompat[oid] if nb in color}
        color[oid] = next(c for c in range(n) if c not in used)
    k_min = len(set(color.values()))

    # Fallback: no VGGT geometry → return graph coloring assignment directly ──
    if not obs_geometry:
        groups: dict[int, list[str]] = defaultdict(list)
        for oid in code_ids:
            groups[color[oid]].append(oid)
        return list(groups.values())

    # Build per-observation co-detection sets:
    # for each obs, which other component codes appear in the same image?
    # Exclude the current code to avoid self-reference.
    codet_of: dict[str, frozenset[int]] = {}
    for oid in code_ids:
        img = img_of[oid]
        codet_of[oid] = frozenset(
            item_by_id[other]["code"]
            for other in img_to_obs[img]
            if other != oid and item_by_id[other]["code"] != code
        )

    # Cluster neighborhood = union of co-detection sets of its members
    cluster_neigh: dict[int, set[int]] = {
        i: set(codet_of[code_ids[i]]) for i in range(n)
    }

    # Constrained agglomerative with rank fusion to k_min ─────────────────────
    clusters: dict[int, list[str]] = {i: [code_ids[i]] for i in range(n)}
    cluster_images: dict[int, set[str]] = {i: {img_of[code_ids[i]]} for i in range(n)}

    while len(clusters) > k_min:
        cids = list(clusters.keys())

        # Collect all compatible pairs with their 3D distance and neigh sim
        candidates: list[tuple[int, int, float, float | None]] = []
        for ci_idx in range(len(cids)):
            for cj_idx in range(ci_idx + 1, len(cids)):
                ci, cj = cids[ci_idx], cids[cj_idx]
                if cluster_images[ci] & cluster_images[cj]:
                    continue  # cannot-link

                # Min 3D cloud distance between clusters
                best_d = float('inf')
                for oa in clusters[ci]:
                    ga = obs_geometry.get(oa)
                    if ga is None:
                        continue
                    for ob in clusters[cj]:
                        gb = obs_geometry.get(ob)
                        if gb is None:
                            continue
                        d = _cloud_min_dist(ga, gb)
                        if d < best_d:
                            best_d = d

                if not np.isfinite(best_d):
                    continue

                # Jaccard similarity of co-detection sets
                na, nb = cluster_neigh[ci], cluster_neigh[cj]
                union = na | nb
                sim: float | None = (len(na & nb) / len(union)) if union else None

                candidates.append((ci, cj, best_d, sim))

        if not candidates:
            break  # no compatible merges remain

        # Rank by 3D distance (ascending)
        sorted_3d = sorted(range(len(candidates)), key=lambda i: candidates[i][2])
        rank_3d = [0] * len(candidates)
        for r, i in enumerate(sorted_3d):
            rank_3d[i] = r + 1

        # Rank by neighborhood similarity (descending), only among pairs with data
        has_sim = [(i, candidates[i][3]) for i in range(len(candidates))
                   if candidates[i][3] is not None]
        rank_sim = [None] * len(candidates)
        if has_sim:
            sorted_sim = sorted(has_sim, key=lambda x: x[1], reverse=True)
            for r, (i, _) in enumerate(sorted_sim):
                rank_sim[i] = r + 1

        # Combined score: 1/(rank_3d * rank_sim) if both available, else 1/rank_3d
        def _score(i: int) -> float:
            r3 = rank_3d[i]
            rs = rank_sim[i]
            return 1.0 / (r3 * rs) if rs is not None else 1.0 / r3

        best_idx = max(range(len(candidates)), key=_score)
        best_i, best_j = candidates[best_idx][0], candidates[best_idx][1]

        clusters[best_i].extend(clusters.pop(best_j))
        cluster_images[best_i].update(cluster_images.pop(best_j))
        cluster_neigh[best_i].update(cluster_neigh.pop(best_j))

    def _sort_key(cluster: list[str]) -> tuple:
        centers = [item_by_id[oid]["center"] for oid in cluster]
        return (
            -len(cluster),
            float(np.mean([c[1] for c in centers])),
            float(np.mean([c[0] for c in centers])),
        )

    return sorted(clusters.values(), key=_sort_key)


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

    obs_geometry: dict[str, dict] = {}
    if VGGT_REID_ENABLED and search_roots:
        obs_geometry = _load_obs_geometry(observed_graphs, list(search_roots))

    print(f"  [ReID-Agglom] {len(obs_geometry)} detection geometries, zero-threshold clustering")

    items_by_code: dict[int, list[str]] = defaultdict(list)
    for item in all_items:
        items_by_code[item["code"]].append(item["observed_node_id"])

    groups: dict[str, dict] = {}
    for code, code_ids in sorted(items_by_code.items()):
        clusters = _cluster_code_agglomerative(code, code_ids, item_by_id, obs_geometry)
        for k, cluster in enumerate(clusters, start=1):
            gid = f"{code}#{k}"
            groups[gid] = {"code": code, "member_ids": cluster, "group_id": gid}

    scene_nodes = _build_scene_nodes(groups, item_by_id, observed_graphs)

    # Build obs→scene_node map for debug output
    obs_to_scene: dict[str, str] = {
        oid: sn.scene_node_id
        for sn in scene_nodes
        for oid in sn.supporting_observed_ids
    }

    _print_reid_summary(scene_nodes, all_items)
    return scene_nodes, {
        "n_scene_nodes":     len(scene_nodes),
        "n_detections":      len(all_items),
        "n_images":          n_images,
        "n_vggt_geometries": len(obs_geometry),
        # per-detection geometry + assignment — used by plot_reid_geometry.py
        "obs_geometry": {
            oid: {
                "centroid":   list(geom["centroid"]),
                "image_id":   item_by_id[oid]["image_id"],
                "code":       item_by_id[oid]["code"],
                "scene_node": obs_to_scene.get(oid, "?"),
            }
            for oid, geom in obs_geometry.items()
            if oid in item_by_id
        },
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
    iou_threshold: float = 0.30,  # kept for API compat, unused
) -> tuple[list[SceneNode], list[dict]]:
    """Merge singleton scene nodes that are likely spurious duplicates.

    Level 1 – 3D positional duplicate: centroid within pos_factor * scene_cutoff
    of a higher-support same-code node.

    Level 3 – shared VGGT edge partner: two low-support nodes of the same code
    that share a strong VGGT-confirmed neighbour and are within scene_cutoff.

    (Level 2 — 2D bounding-box IoU — removed in 3D-only pipeline.)
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
