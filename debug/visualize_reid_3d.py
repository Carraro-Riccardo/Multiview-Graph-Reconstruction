"""3D scatter visualisation of VGGT-ReID clusters.

For each node shows one point per detection coloured by pred scene-node
cluster.  Clusters are overlaid on a shared 3D axes so you can inspect:
  - how tight each cluster is in 3D space
  - which clusters map to the same GT instance (FP duplicates)
  - which GT instances are missing (FN)

Usage
-----
    python visualize_reid_3d.py              # nodes 7 and 9 side by side
    python visualize_reid_3d.py 7            # single node
    python visualize_reid_3d.py 4 7 9        # multiple nodes in one figure
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from config import WORK_DIR, IMAGES_DIR
from core.detection import discover_nodes
from evaluate import (
    load_detections_for_image,
    load_gt_for_node,
    match_detections_to_gt,
)
from pipeline.observed_graphs import build_all_observed_graphs
from pipeline.vggt_3d_edges import (
    _candidate_files_for_image,
    _estimate_observation_geometry,
    _load_pointmap,
    build_vggt_search_roots,
)

MULTIVIEW_DIR = WORK_DIR / "multiview"


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_obs_geometry(observed_graphs, search_roots) -> dict[str, dict]:
    obs_geometry: dict[str, dict] = {}
    for og in observed_graphs:
        candidates = _candidate_files_for_image(og.image_id, search_roots)
        pointmap = None
        for path in candidates:
            pointmap = _load_pointmap(path)
            if pointmap is not None:
                break
        if pointmap is None:
            continue
        for obs_id, obs_node in og.nodes.items():
            geom = _estimate_observation_geometry(pointmap, list(obs_node.bbox))
            if geom is not None:
                obs_geometry[obs_id] = geom
    return obs_geometry


def _load_scene_node_map(node_id: int) -> dict[str, str]:
    """Return {obs_id -> scene_node_id}."""
    import json
    p = MULTIVIEW_DIR / f"node{node_id}" / "scene_nodes.json"
    if not p.exists():
        return {}
    scene_nodes = json.loads(p.read_text())
    result: dict[str, str] = {}
    for sn in scene_nodes:
        for obs_id in sn["supporting_observed_ids"]:
            result[obs_id] = sn["scene_node_id"]
    return result


def _map_obs_to_gt(observed_graphs, gt_data) -> dict[str, int | None]:
    """Return {obs_id -> majority GT instance_id (or None)}."""
    obs_to_gt: dict[str, int | None] = {}
    for og in observed_graphs:
        image_stem = og.image_id
        dets       = load_detections_for_image(image_stem)
        gt_insts   = gt_data["instances_by_image"].get(image_stem, [])
        for obs_id, obs_node in og.nodes.items():
            parts = obs_id.rsplit("_", 1)
            if len(parts) != 2:
                obs_to_gt[obs_id] = None
                continue
            det_idx = int(parts[1])
            if det_idx >= len(dets):
                obs_to_gt[obs_id] = None
                continue
            det = dets[det_idx]
            iid, _ = match_detections_to_gt(det["bbox_norm"], det["code"], gt_insts)
            obs_to_gt[obs_id] = iid
    return obs_to_gt


def _gt_majority_per_cluster(obs_to_scene, obs_to_gt) -> dict[str, int | None]:
    votes: dict[str, Counter] = defaultdict(Counter)
    for obs_id, sn_id in obs_to_scene.items():
        gt = obs_to_gt.get(obs_id)
        if gt is not None:
            votes[sn_id][gt] += 1
    majority: dict[str, int | None] = {}
    for sn_id, ctr in votes.items():
        majority[sn_id] = ctr.most_common(1)[0][0] if ctr else None
    return majority


def _find_fp_clusters(majority_gt: dict[str, int | None]) -> set[str]:
    """Pred clusters whose majority GT is shared by another (higher-count) cluster.
    Also includes clusters with no GT match.
    """
    # Group clusters by their majority GT
    by_gt: dict[int, list[str]] = defaultdict(list)
    no_gt: set[str] = set()
    for sn_id, gt in majority_gt.items():
        if gt is None:
            no_gt.add(sn_id)
        else:
            by_gt[gt].append(sn_id)

    fps: set[str] = set(no_gt)
    for gt, sn_ids in by_gt.items():
        if len(sn_ids) > 1:
            # Keep the one with the most detections; mark the rest as FP
            fps.update(sn_ids[1:])  # simplified: all but first
    return fps


# ── per-node visualisation ───────────────────────────────────────────────────

def collect_node_data(node_id: int):
    """Return all data needed for the scatter plot of one node."""
    all_imgs = discover_nodes(IMAGES_DIR)
    if node_id not in all_imgs:
        print(f"[warn] node {node_id} not found in images dir")
        return None

    image_paths = all_imgs[node_id]
    observed_graphs = build_all_observed_graphs(image_paths)
    output_dir = MULTIVIEW_DIR / f"node{node_id}"
    search_roots = build_vggt_search_roots(node_id, output_dir, WORK_DIR)

    obs_geometry  = _load_obs_geometry(observed_graphs, search_roots)
    obs_to_scene  = _load_scene_node_map(node_id)
    gt_data       = load_gt_for_node(node_id)
    obs_to_gt     = _map_obs_to_gt(observed_graphs, gt_data)

    # Only keep observations with 3D geometry
    obs_ids = [oid for oid in obs_to_scene if oid in obs_geometry]

    if not obs_ids:
        print(f"[warn] node {node_id}: no observations with 3D geometry")
        return None

    pts       = np.array([obs_geometry[oid]["centroid"] for oid in obs_ids])
    cluster_labels = [obs_to_scene[oid] for oid in obs_ids]
    gt_labels = [obs_to_gt.get(oid) for oid in obs_ids]

    majority_gt = _gt_majority_per_cluster(obs_to_scene, obs_to_gt)
    fp_clusters = _find_fp_clusters(majority_gt)

    # GT instances not covered by any cluster
    all_gt_ids = {iid for iid in gt_data["instances_by_id"]}
    covered_gt = {v for v in majority_gt.values() if v is not None}
    fn_gt_ids  = all_gt_ids - covered_gt

    unique_sn_ids = sorted(set(cluster_labels))
    cmap = plt.colormaps["tab20"]
    color_map = {sn_id: cmap(i / max(len(unique_sn_ids) - 1, 1))
                 for i, sn_id in enumerate(unique_sn_ids)}

    return {
        "node_id":      node_id,
        "pts":          pts,
        "clusters":     cluster_labels,
        "gt_labels":    gt_labels,
        "color_map":    color_map,
        "fp_clusters":  fp_clusters,
        "fn_gt_ids":    fn_gt_ids,
        "majority_gt":  majority_gt,
        "n_gt":         len(all_gt_ids),
        "n_pred":       len(unique_sn_ids),
        "unique_sn":    unique_sn_ids,
    }


def plot_node(ax, data: dict):
    node_id    = data["node_id"]
    pts        = data["pts"]
    clusters   = data["clusters"]
    color_map  = data["color_map"]
    fp_clusters = data["fp_clusters"]
    majority_gt = data["majority_gt"]

    # Group points by cluster
    by_cluster: dict[str, list[int]] = defaultdict(list)
    for i, sn_id in enumerate(clusters):
        by_cluster[sn_id].append(i)

    for sn_id, idxs in sorted(by_cluster.items()):
        col = color_map[sn_id]
        sub = pts[idxs]
        is_fp = sn_id in fp_clusters

        # Cluster centroid
        ctr = sub.mean(axis=0)

        if is_fp:
            marker, edgecolors, linewidths, alpha = "X", "red", 1.5, 0.9
            label = f"{sn_id} [FP]"
        else:
            marker, edgecolors, linewidths, alpha = "o", col, 0.5, 0.75
            gt_id = majority_gt.get(sn_id)
            label = f"{sn_id}→GT{gt_id}"

        ax.scatter(
            sub[:, 0], sub[:, 1], sub[:, 2],
            c=[col] * len(sub),
            marker=marker,
            edgecolors=edgecolors,
            linewidths=linewidths,
            alpha=alpha,
            s=60,
            label=label,
            depthshade=True,
        )
        # Centroid annotation
        ax.text(ctr[0], ctr[1], ctr[2], sn_id.split("#")[-1],
                fontsize=7, ha="center", va="bottom", color=color_map[sn_id],
                fontweight="bold" if is_fp else "normal")

    fn_note = f"  FN_GT={data['fn_gt_ids']}" if data["fn_gt_ids"] else ""
    ax.set_title(
        f"node {node_id}  —  pred={data['n_pred']}  GT={data['n_gt']}"
        f"  FP={len(fp_clusters)}{fn_note}",
        fontsize=9,
    )
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="upper left", ncol=2, framealpha=0.6)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    node_ids = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [7, 9]

    datasets = []
    for nid in node_ids:
        print(f"Loading node {nid} ...")
        d = collect_node_data(nid)
        if d:
            datasets.append(d)

    if not datasets:
        print("No data to plot.")
        return

    n = len(datasets)
    fig = plt.figure(figsize=(7 * n, 6))
    fig.suptitle("VGGT-ReID 3D clusters  (X = FP cluster, o = TP cluster)", fontsize=11)

    for i, data in enumerate(datasets):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        plot_node(ax, data)
        print(
            f"  node {data['node_id']}: pred={data['n_pred']}  GT={data['n_gt']}"
            f"  FP={len(data['fp_clusters'])}  FN_GT={data['fn_gt_ids']}"
        )

    plt.tight_layout()
    out_path = Path(__file__).parent / "output" / "reid_3d_scatter.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
