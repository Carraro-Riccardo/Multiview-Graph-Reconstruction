from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from config import WORK_DIR, COMPONENT_INFO

DEBUG_ROOT = WORK_DIR / "debug"


def get_name(code: int) -> str:
    return COMPONENT_INFO.get(code, (2, str(code)))[1]


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Non serializable type: {type(value)!r}")


def init_node_debug(node_id: int) -> dict:
    root = DEBUG_ROOT / f"node{node_id}"
    dirs = {
        "root": root,
        "views": root / "views",
        "multiview": root / "multiview",
        "consolidation": root / "consolidation",
        "candidates": root / "candidates",
        "solver": root / "solver",
        "render": root / "render",
        "graphs": root / "graphs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)


def save_view_selection(debug_dirs: dict, all_views: list, scores: list, best_view_idx: int) -> None:
    payload = []
    for idx, view in enumerate(all_views):
        codes = [d["code"] for d in view["dets"] if d["code"] in COMPONENT_INFO]
        payload.append({
            "view_index": idx,
            "image": view["path"].name,
            "score": scores[idx],
            "is_best": idx == best_view_idx,
            "n_detections": len(view["dets"]),
            "codes": codes,
        })
    save_json(debug_dirs["views"] / "view_selection.json", payload)

    names = [item["image"] for item in payload]
    vals = [item["score"] for item in payload]
    colors = ["#00ff88" if item["is_best"] else "#4c78a8" for item in payload]
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 4.5))
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("view score")
    ax.set_title("Best view selection")
    fig.tight_layout()
    fig.savefig(debug_dirs["views"] / "view_selection_scores.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_type_consensus(debug_dirs: dict, type_consensus: dict) -> None:
    pairs = [
        {"pair": [int(a), int(b)], "names": [get_name(a), get_name(b)], "score": score}
        for (a, b), score in sorted(type_consensus.items(), key=lambda item: (-item[1], item[0]))
    ]
    save_json(debug_dirs["multiview"] / "type_consensus.json", pairs)

    codes = sorted({code for pair in type_consensus for code in pair})
    if not codes:
        return
    index = {code: i for i, code in enumerate(codes)}
    M = np.zeros((len(codes), len(codes)), dtype=float)
    for (a, b), score in type_consensus.items():
        ia, ib = index[a], index[b]
        M[ia, ib] = score
        M[ib, ia] = score

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(codes)))
    ax.set_yticks(range(len(codes)))
    ax.set_xticklabels([f"{c}" for c in codes], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([f"{c}" for c in codes], fontsize=8)
    ax.set_title("Multi-view type consensus")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(debug_dirs["multiview"] / "type_consensus_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_consolidation(debug_dirs: dict, payload: dict) -> None:
    save_json(debug_dirs["consolidation"] / "best_view_consolidation.json", payload)


def save_candidate_snapshot(debug_dirs: dict, best_view_path: Path, dets: list,
                            pair_features: dict, selected_pairs=None,
                            filename: str = "candidate_edges.png",
                            title: str = "Candidate edges") -> None:
    if not pair_features:
        return
    selected_pairs = {tuple(pair) for pair in (selected_pairs or [])}

    img_bgr = cv2.imread(str(best_view_path))
    if img_bgr is None:
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_rgb)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    scores = [feats["score"] for feats in pair_features.values()]
    smin, smax = min(scores), max(scores)

    for (i, j), feats in pair_features.items():
        xi, yi = dets[i]["cx"] * W, dets[i]["cy"] * H
        xj, yj = dets[j]["cx"] * W, dets[j]["cy"] * H
        if (i, j) in selected_pairs or (j, i) in selected_pairs:
            color = "#00ff88"
            lw = 2.7
            alpha = 0.95
        else:
            t = 0.5 if smax <= smin else (feats["score"] - smin) / (smax - smin)
            color = plt.cm.plasma(t)
            lw = 1.2
            alpha = 0.55
        ax.plot([xi, xj], [yi, yj], color=color, linewidth=lw, alpha=alpha, zorder=2)

    for idx, det in enumerate(dets):
        x, y = det["cx"] * W, det["cy"] * H
        ax.scatter([x], [y], s=130, color="#ffffff", edgecolors="#111111", zorder=3)
        ax.text(x + 6, y - 6, f"{det['code']}#{idx}", fontsize=7, color="white",
                bbox=dict(facecolor="black", alpha=0.6, pad=1), zorder=4)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(debug_dirs["candidates"] / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_solver_summary(debug_dirs: dict, payload: dict, name: str = "solver_summary.json") -> None:
    save_json(debug_dirs["solver"] / name, payload)


def save_graph_snapshot(debug_dirs: dict, G, stage: str) -> None:
    payload = {
        "stage": stage,
        "nodes": [],
        "edges": [],
    }
    for nid, attrs in G.nodes(data=True):
        item = {"nid": nid}
        item.update(attrs)
        payload["nodes"].append(item)
    for u, v, attrs in G.edges(data=True):
        item = {"u": u, "v": v}
        item.update(attrs)
        payload["edges"].append(item)
    save_json(debug_dirs["graphs"] / f"graph_{stage}.json", payload)


def save_render_summary(debug_dirs: dict, payload: list) -> None:
    save_json(debug_dirs["render"] / "edge_render_classes.json", payload)
