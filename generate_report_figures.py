"""Generate illustrative figures for the Re-ID section of the report.

Figures produced (in output/report_figures/):
  1_clustering.png        — spectral vs greedy clustering on affinity matrix
  2_vggt_mincloud.png     — min-cloud vs centroid distance explanation
  3_k_estimation.png      — incompatibility graph → chromatic number → k
  4_neighborhood_sig.png  — neighborhood signature per detection
  5_post_merge.png        — post-clustering VGGT merge pass
"""

from __future__ import annotations

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUT = Path("output/report_figures")
OUT.mkdir(parents=True, exist_ok=True)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

# ─────────────────────────────────────────────────────────────────────────────
# 1) CLUSTERING — show affinity matrix + spectral embedding for Saracinesca (node9)
# ─────────────────────────────────────────────────────────────────────────────
def fig_clustering():
    nodes = json.load(open("output/multiview/node8/scene_nodes_reid_raw.json"))
    # Collect Saracinesca scene nodes and their support images
    sara_nodes = [n for n in nodes if n["fused_code"] == 15]
    k = len(sara_nodes)  # 3 in node8

    # Build a synthetic affinity matrix that mirrors the real one:
    # same image → 0, different image → visual sim (high within true cluster, low across)
    rng = np.random.default_rng(42)
    n_det = sum(n["support_count"] for n in sara_nodes)

    # Assign detection indices to clusters
    cluster_of = []
    for ci, node in enumerate(sara_nodes):
        cluster_of.extend([ci] * node["support_count"])
    cluster_of = np.array(cluster_of)

    # Build image assignments (each row = one detection, each column = one image)
    n_images = 7
    img_of = []
    for ci, node in enumerate(sara_nodes):
        for img in node["support_images"]:
            img_of.append(int(img.split("_")[1]) - 1)
    img_of = np.array(img_of)

    # Build affinity matrix
    A = np.zeros((n_det, n_det))
    for i in range(n_det):
        for j in range(i + 1, n_det):
            if img_of[i] == img_of[j]:
                A[i, j] = A[j, i] = 0.0  # same image → hard 0
            elif cluster_of[i] == cluster_of[j]:
                A[i, j] = A[j, i] = 0.65 + rng.uniform(0, 0.25)
            else:
                A[i, j] = A[j, i] = rng.uniform(0.0, 0.35)

    # Spectral embedding (2D for visualisation)
    d = A.sum(axis=1).clip(1e-10)
    D_isqrt = 1.0 / np.sqrt(d)
    L = np.eye(n_det) - (A * D_isqrt[:, None]) * D_isqrt[None, :]
    _, V = np.linalg.eigh(L)
    X = V[:, 1:3]  # 2 smallest non-trivial eigenvectors
    norms = np.linalg.norm(X, axis=1, keepdims=True).clip(1e-10)
    X = X / norms

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: affinity matrix (sorted by true cluster)
    order = np.argsort(cluster_of)
    Am = A[np.ix_(order, order)]
    im = axes[0].imshow(Am, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Matrice di affinità\n(ordinata per cluster reale)", fontsize=11)
    axes[0].set_xlabel("Detection index")
    axes[0].set_ylabel("Detection index")
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    # Add cluster boundaries
    boundaries = np.cumsum([n["support_count"] for n in sara_nodes])[:-1]
    for b in boundaries:
        axes[0].axhline(b - 0.5, color="red", lw=0.8, ls="--")
        axes[0].axvline(b - 0.5, color="red", lw=0.8, ls="--")

    # Right: spectral embedding coloured by true cluster
    for ci in range(k):
        mask = cluster_of == ci
        axes[1].scatter(X[mask, 0], X[mask, 1],
                        color=COLORS[ci % len(COLORS)],
                        label=f"15#{ci+1}", s=60, edgecolors="white", linewidths=0.5)
    axes[1].set_title("Embedding spettrale (2D)\nK-Means applicato in questo spazio", fontsize=11)
    axes[1].set_xlabel("Autovettore 2")
    axes[1].set_ylabel("Autovettore 3")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].set_aspect("equal")

    fig.suptitle(f"Clustering spettrale — Saracinesca (Nodo 8, k={k})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "1_clustering.png", dpi=150)
    plt.close(fig)
    print("  1_clustering.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2) VGGT MIN-CLOUD vs CENTROID DISTANCE
# ─────────────────────────────────────────────────────────────────────────────
def fig_vggt_mincloud():
    # Load a real VGGT point map for node9 image 1
    npz = np.load("output/multiview/node8/vggt/node8_0001.npz")
    wp  = npz["world_points"]   # (H, W, 3)
    wpc = npz["world_points_conf"]  # (H, W)

    H, W = wp.shape[:2]

    # Simulate two bounding boxes for two Saracinesca detections
    # (roughly plausible pixel positions for node9_0001)
    rng = np.random.default_rng(7)
    def bbox_centroid_and_cloud(y1r, y2r, x1r, x2r, conf_thresh=0.3):
        y1, y2 = int(y1r * H), int(y2r * H)
        x1, x2 = int(x1r * W), int(x2r * W)
        mask = wpc[y1:y2, x1:x2] > conf_thresh
        pts  = wp[y1:y2, x1:x2][mask]
        if len(pts) < 5:
            pts = wp[y1:y2, x1:x2].reshape(-1, 3)
        centroid = pts.mean(axis=0)
        idx = rng.choice(len(pts), min(30, len(pts)), replace=False)
        return centroid, pts[idx]

    c1, pts1 = bbox_centroid_and_cloud(0.30, 0.55, 0.05, 0.28)
    c2, pts2 = bbox_centroid_and_cloud(0.30, 0.55, 0.30, 0.53)

    # Project to XZ for display
    def proj(p): return p[:, 0], p[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, title, show_min in zip(axes,
        ["Distanza centroide-centroide\n(sensibile al viewpoint)",
         "Distanza min-cloud\n(view-invariant)"],
        [False, True]):

        x1s, z1s = proj(pts1)
        x2s, z2s = proj(pts2)
        ax.scatter(x1s, z1s, c="#4C72B0", s=18, alpha=0.7, label="Detection A")
        ax.scatter(x2s, z2s, c="#DD8452", s=18, alpha=0.7, label="Detection B")

        # Centroids
        ax.scatter(*proj(c1[None])[0], *proj(c1[None])[1],
                   c="#4C72B0", s=120, marker="*", zorder=5)
        ax.scatter(*proj(c2[None])[0], *proj(c2[None])[1],
                   c="#DD8452", s=120, marker="*", zorder=5)

        if not show_min:
            # Draw centroid-centroid arrow
            ax.annotate("", xy=(proj(c2[None])[0][0], proj(c2[None])[1][0]),
                        xytext=(proj(c1[None])[0][0], proj(c1[None])[1][0]),
                        arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
            # Centroid distance label
            dist_cc = float(np.linalg.norm(c1 - c2))
            mx = (proj(c1[None])[0][0] + proj(c2[None])[0][0]) / 2
            mz = (proj(c1[None])[1][0] + proj(c2[None])[1][0]) / 2
            ax.text(mx, mz, f"d={dist_cc:.2f}m", fontsize=9,
                    ha="center", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
        else:
            # Find min-cloud pair
            diff = pts1[:, None, :] - pts2[None, :, :]
            dists = np.sqrt((diff**2).sum(axis=2))
            i_min, j_min = np.unravel_index(dists.argmin(), dists.shape)
            p_a = pts1[i_min]
            p_b = pts2[j_min]
            ax.annotate("", xy=(proj(p_b[None])[0][0], proj(p_b[None])[1][0]),
                        xytext=(proj(p_a[None])[0][0], proj(p_a[None])[1][0]),
                        arrowprops=dict(arrowstyle="<->", color="green", lw=2))
            dist_min = float(dists.min())
            mx = (proj(p_a[None])[0][0] + proj(p_b[None])[0][0]) / 2
            mz = (proj(p_a[None])[1][0] + proj(p_b[None])[1][0]) / 2
            ax.text(mx, mz, f"d_min={dist_min:.3f}m", fontsize=9,
                    ha="center", va="bottom", color="green",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.legend(fontsize=8)

    fig.suptitle("VGGT — Min-cloud distance vs distanza centroide", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "2_vggt_mincloud.png", dpi=150)
    plt.close(fig)
    print("  2_vggt_mincloud.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3) K ESTIMATION via incompatibility graph
# ─────────────────────────────────────────────────────────────────────────────
def fig_k_estimation():
    """Show the incompatibility graph for Saracinesca in node9 and its coloring."""
    nodes = json.load(open("output/multiview/node8/scene_nodes_reid_raw.json"))
    sara = [n for n in nodes if n["fused_code"] == 15]

    # Rebuild detection → image mapping
    det_ids, det_imgs, det_cluster = [], [], []
    for ci, node in enumerate(sara):
        for img in node["support_images"]:
            det_ids.append(f"d{len(det_ids)}")
            det_imgs.append(img)
            det_cluster.append(ci)

    n = len(det_ids)

    # Build incompatibility edges
    incompat_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if det_imgs[i] == det_imgs[j]:
                incompat_edges.append((i, j))

    # Greedy coloring (DSATUR-lite)
    incompat_adj = {i: set() for i in range(n)}
    for i, j in incompat_edges:
        incompat_adj[i].add(j)
        incompat_adj[j].add(i)

    color = {}
    for node_i in sorted(range(n), key=lambda x: -len(incompat_adj[x])):
        used = {color[nb] for nb in incompat_adj[node_i] if nb in color}
        color[node_i] = next(c for c in range(n) if c not in used)
    k_found = max(color.values()) + 1

    # Layout: circular
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(n)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def draw_graph(ax, node_colors, title, show_edges=True):
        if show_edges:
            for i, j in incompat_edges:
                xi, yi_ = pos[i]
                xj, yj_ = pos[j]
                ax.plot([xi, xj], [yi_, yj_], color="#cccccc", lw=0.6, zorder=1)
        for i in range(n):
            xi, yi_ = pos[i]
            ax.scatter(xi, yi_, s=180,
                       color=COLORS[node_colors[i] % len(COLORS)],
                       edgecolors="black", linewidths=0.8, zorder=3)
            ax.text(xi * 1.18, yi_ * 1.18, det_ids[i],
                    fontsize=6.5, ha="center", va="center")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11)

    # Left: uncolored graph (all same color)
    draw_graph(axes[0], [7] * n,
               f"Grafo di incompatibilità\n({n} detection, archi = stessa immagine)")

    # Right: colored (coloring = cluster assignment)
    draw_graph(axes[1], [color[i] for i in range(n)],
               f"Colorazione greedy → k = {k_found}\n"
               f"(max per immagine = {max(len([d for d in det_imgs if d==img]) for img in set(det_imgs))})")

    # Legend for right plot
    legend_handles = [
        mpatches.Patch(color=COLORS[c], label=f"Istanza 15#{c+1}")
        for c in range(k_found)
    ]
    axes[1].legend(handles=legend_handles, fontsize=7, loc="lower right",
                   ncol=2, framealpha=0.9)

    fig.suptitle("Stima strutturale di k — Saracinesca (Nodo 8)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "3_k_estimation.png", dpi=150)
    plt.close(fig)
    print("  3_k_estimation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4) NEIGHBORHOOD SIGNATURE
# ─────────────────────────────────────────────────────────────────────────────
def fig_neighborhood_signature():
    """Show two visually identical Saracinesca with DIFFERENT neighborhoods
    → Jaccard = 0 → hard exclusion (they are different physical instances)."""

    # Synthetic scene: two arms of a hydraulic node, each with a Saracinesca
    # Arm 1: Tee FL — Sarac. A — Giunto
    # Arm 2: Tee rid. — Sarac. B — Bout bic.
    component_positions = {
        "Tee FL":    np.array([-1.1,  0.0]),
        "Sarac. A":  np.array([ 0.0,  0.0]),
        "Giunto":    np.array([ 1.1,  0.0]),
        "Tee rid.":  np.array([-1.1, -1.8]),
        "Sarac. B":  np.array([ 0.0, -1.8]),
        "Bout bic.": np.array([ 1.1, -1.8]),
    }
    codes = {
        "Tee FL": 116, "Sarac. A": 15, "Giunto": 11,
        "Tee rid.": 255, "Sarac. B": 15, "Bout bic.": 95,
    }

    sig_A = {"Tee FL", "Giunto"}
    sig_B = {"Tee rid.", "Bout bic."}
    jaccard = len(sig_A & sig_B) / len(sig_A | sig_B)  # = 0.0

    fig, ax = plt.subplots(figsize=(10, 5))

    node_colors = {
        "Sarac. A":  "#4C72B0",
        "Sarac. B":  "#DD8452",
        "Tee FL":    "#aec6e8",
        "Giunto":    "#aec6e8",
        "Tee rid.":  "#f5c6a0",
        "Bout bic.": "#f5c6a0",
    }

    for name, pos in component_positions.items():
        ax.scatter(*pos, s=320, color=node_colors[name],
                   edgecolors="black", linewidths=1.2, zorder=3)
        ax.text(pos[0], pos[1] + 0.18, name, fontsize=9.5,
                ha="center", va="bottom", fontweight="bold")

    # Draw neighborhood arrows for Sarac. A
    for nb in ["Tee FL", "Giunto"]:
        ax.annotate("", xy=component_positions[nb],
                    xytext=component_positions["Sarac. A"],
                    arrowprops=dict(arrowstyle="-", color="#4C72B0",
                                    lw=2.2, linestyle="dashed"))
    ax.text(0.0, -0.38, "sig(A) = {Tee FL, Giunto}", fontsize=9,
            ha="center", color="#4C72B0", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.95))

    # Draw neighborhood arrows for Sarac. B
    for nb in ["Tee rid.", "Bout bic."]:
        ax.annotate("", xy=component_positions[nb],
                    xytext=component_positions["Sarac. B"],
                    arrowprops=dict(arrowstyle="-", color="#DD8452",
                                    lw=2.2, linestyle="dashed"))
    ax.text(0.0, -2.18, "sig(B) = {Tee rid., Bout bic.}", fontsize=9,
            ha="center", color="#DD8452", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.95))

    # Jaccard result box
    ax.text(0.0, -2.85,
            "Jaccard(sig_A, sig_B) = |∅| / |{Tee FL, Giunto, Tee rid., Bout bic.}| = 0.0\n"
            "→ esclusione hard: affinità = 0, istanze fisiche distinte",
            fontsize=9.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fdecea", ec="#C44E52",
                      linewidth=1.5, alpha=0.97))

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-3.4, 0.7)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle("Firma del vicinato 3D — esclusione hard tra istanze diverse",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "4_neighborhood_sig.png", dpi=150)
    plt.close(fig)
    print("  4_neighborhood_sig.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5) POST-CLUSTERING VGGT MERGE PASS
# ─────────────────────────────────────────────────────────────────────────────
def fig_post_merge():
    """Illustrate the post-clustering merge: two clusters of the same physical
    instance that K-Means split, merged because min-cloud distance < 0.25 m."""

    rng = np.random.default_rng(3)

    # True instance A: seen from 3 viewpoints → K-Means splits into A1 and A2
    # because visual embeddings vary (different angles)
    # A1: 3 detections around centroid_a with small spread
    centroid_a = np.array([1.0, 0.0, 2.0])
    pts_a1 = centroid_a + rng.normal(0, 0.03, (3, 3))
    pts_a2 = centroid_a + rng.normal(0, 0.03, (2, 3))

    # True instance B: different physical component, far away
    centroid_b = np.array([1.0, 0.0, 3.5])
    pts_b = centroid_b + rng.normal(0, 0.03, (4, 3))

    # Project to XZ
    def xz(pts): return pts[:, 0], pts[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, title, merged in zip(axes,
        ["Prima del merge\n(K-Means: A1 e A2 separati)",
         "Dopo il merge\n(min-cloud A1↔A2 < 0.25 m → uniti)"],
        [False, True]):

        # Draw clusters
        if not merged:
            ax.scatter(*xz(pts_a1), color="#4C72B0", s=80, label="Cluster A1", zorder=3)
            ax.scatter(*xz(pts_a2), color="#DD8452", s=80, label="Cluster A2", zorder=3)
        else:
            all_a = np.vstack([pts_a1, pts_a2])
            ax.scatter(*xz(all_a), color="#4C72B0", s=80, label="Cluster A (merged)", zorder=3)

        ax.scatter(*xz(pts_b), color="#55A868", s=80, label="Cluster B", zorder=3)

        # Centroids
        if not merged:
            for pts, c, lbl in [(pts_a1, "#4C72B0", "A1"), (pts_a2, "#DD8452", "A2"),
                                 (pts_b, "#55A868", "B")]:
                cen = pts.mean(axis=0)
                ax.scatter(*xz(cen[None]), color=c, s=200, marker="*", zorder=5)
                ax.text(xz(cen[None])[0][0] + 0.04, xz(cen[None])[1][0], lbl,
                        fontsize=10, fontweight="bold", color=c)

            # Show min-cloud distance between A1 and A2
            diff = pts_a1[:, None, :] - pts_a2[None, :, :]
            dists = np.sqrt((diff**2).sum(axis=2))
            i_m, j_m = np.unravel_index(dists.argmin(), dists.shape)
            pa, pb_ = pts_a1[i_m], pts_a2[j_m]
            ax.annotate("", xy=(xz(pb_[None])[0][0], xz(pb_[None])[1][0]),
                        xytext=(xz(pa[None])[0][0], xz(pa[None])[1][0]),
                        arrowprops=dict(arrowstyle="<->", color="red", lw=2))
            mx = (xz(pa[None])[0][0] + xz(pb_[None])[0][0]) / 2
            mz = (xz(pa[None])[1][0] + xz(pb_[None])[1][0]) / 2
            ax.text(mx + 0.05, mz, f"d_min={dists.min():.3f}m\n< 0.25 m → merge",
                    fontsize=8.5, color="red",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.9))

            # Show A1-B and A2-B distances (large → no merge)
            for pts_x, label_x in [(pts_a1, "A1↔B"), (pts_a2, "A2↔B")]:
                diff2 = pts_x[:, None, :] - pts_b[None, :, :]
                d2 = np.sqrt((diff2**2).sum(axis=2)).min()
                ax.text(1.0, 3.3 if label_x == "A1↔B" else 3.2,
                        f"{label_x}: {d2:.2f}m > 0.25 m", fontsize=7.5,
                        color="#888888")
        else:
            all_a = np.vstack([pts_a1, pts_a2])
            cen = all_a.mean(axis=0)
            ax.scatter(*xz(cen[None]), color="#4C72B0", s=200, marker="*", zorder=5)
            ax.text(xz(cen[None])[0][0] + 0.04, xz(cen[None])[1][0], "A",
                    fontsize=10, fontweight="bold", color="#4C72B0")
            cen_b = pts_b.mean(axis=0)
            ax.scatter(*xz(cen_b[None]), color="#55A868", s=200, marker="*", zorder=5)
            ax.text(xz(cen_b[None])[0][0] + 0.04, xz(cen_b[None])[1][0], "B",
                    fontsize=10, fontweight="bold", color="#55A868")

        ax.set_xlim(0.7, 1.4)
        ax.set_ylim(1.7, 3.8)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.legend(fontsize=8)
        ax.set_title(title, fontsize=11)

    fig.suptitle("Post-clustering VGGT merge pass", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "5_post_merge.png", dpi=150)
    plt.close(fig)
    print("  5_post_merge.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Generating figures in {OUT}/")
    fig_clustering()
    fig_vggt_mincloud()
    fig_k_estimation()
    fig_neighborhood_signature()
    fig_post_merge()
    print("Done.")
