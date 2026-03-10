"""
Funzioni di visualizzazione:
  - plot_node_detections: griglia immagine+scatter per Step 1
  - plot_graph_on_image:  overlay del grafo sull'immagine per Step 4
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import networkx as nx

from config import VALENCE_COLORS, WORK_DIR
from graph_engine import get_valence, get_name
from debug_utils import save_render_summary


# Step 1 - Detection & Decoding

def get_palette(codes: list) -> dict:
    unique = sorted(set(codes))
    colors = cm.tab20(np.linspace(0, 1, max(len(unique), 1)))
    return {c: colors[i] for i, c in enumerate(unique)}


def plot_node_detections(node_id: int, image_paths: list, all_dets: list) -> None:
    """Salva e mostra una griglia (immagine originale + scatter normalizzato) per ogni vista."""
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    all_codes = [d["code"] for dets in all_dets for d in dets]
    palette = get_palette(all_codes)

    for row, (img_path, dets) in enumerate(zip(image_paths, all_dets)):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_bgr.shape[:2]

        ax0 = axes[row, 0]
        ax0.imshow(img_rgb)
        for d in dets:
            cx_px = int(d["cx"] * W)
            cy_px = int(d["cy"] * H)
            col = palette[d["code"]]
            ax0.plot(cx_px, cy_px, "o", markersize=10,
                     color=col, markeredgecolor="white", linewidth=1.5, zorder=5)
            ax0.text(cx_px + 6, cy_px - 6, str(d["code"]), fontsize=7,
                     color="white", bbox=dict(facecolor="black", alpha=0.5, pad=1))
        ax0.set_title(f"{img_path.name}  ({len(dets)} detection)", fontsize=9)
        ax0.axis("off")

        ax1 = axes[row, 1]
        ax1.set_facecolor("#1a1a2e")
        for d in dets:
            col = palette[d["code"]]
            ax1.scatter(d["cx"], 1 - d["cy"], s=120, color=col,
                        edgecolors="white", linewidths=1.2, zorder=3)
            ax1.text(d["cx"] + 0.01, 1 - d["cy"] + 0.01,
                     str(d["code"]), fontsize=7, color="white")
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.2, color="white")
        ax1.set_title(f"Nodi rilevati - {img_path.name}", fontsize=9)

        seen = sorted(set(d["code"] for d in dets))
        legend = [mpatches.Patch(color=palette[c], label=str(c)) for c in seen]
        ax1.legend(handles=legend, fontsize=6, loc="lower right",
                   framealpha=0.7, ncol=3, title="codice")

    fig.suptitle(f"Node {node_id} - Detection & Decoding", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = WORK_DIR / f"node{node_id}_detections.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {out}")


# Step 4 - Graph Overlay

def plot_graph_on_image(node_id: int, G: nx.Graph, best_view: dict,
                        pos_2d: dict, cutoff: float,
                        debug_dirs: dict | None = None) -> None:
    """Sovrappone il grafo ricostruito sull'immagine della best view e salva."""
    img_bgr = cv2.imread(str(best_view["path"]))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    fig.patch.set_facecolor("#0d1117")
    pos_px = {nid: (xy[0] * W, xy[1] * H) for nid, xy in pos_2d.items()}
    render_payload = []

    for ax_idx, ax in enumerate(axes):
        ax.imshow(img_rgb)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")

        for u, v in G.edges():
            if u not in pos_px or v not in pos_px:
                continue
            ux, uy = pos_px[u]
            vx, vy = pos_px[v]
            ratio = G[u][v].get("confirm_ratio", 0)
            cooc = G[u][v].get("cooc", 0)
            is_post = G[u][v].get("postprocess", False)
            is_forced = G[u][v].get("forced", False)
            is_recov = G[u][v].get("recovery", False)
            is_supported_reconnect = G[u][v].get("supported_reconnect", False)

            if is_forced and is_supported_reconnect:
                color, lw, ls = "#66e0ff", 2.4, "-"
                render_class = "supported_reconnect"
            elif is_forced:
                color, lw, ls = "#00ccff", 2.0, "-."
                render_class = "forced"
            elif is_recov:
                color, lw, ls = "#cc44ff", 1.5, ":"
                render_class = "recovery"
            elif is_post:
                color, lw, ls = "#ff66cc", 2.0, "--"
                render_class = "postprocess"
            elif ratio >= 0.5:
                color, lw, ls = "#00ff88", 3.0, "-"
                render_class = "confirmed"
            elif cooc > 0.3:
                color, lw, ls = "#ffdd00", 2.0, "-"
                render_class = "prior_strong"
            else:
                color, lw, ls = "#ff8800", 1.8, "--"
                render_class = "dubious"

            if ax_idx == 1:
                render_payload.append({
                    "u": u,
                    "v": v,
                    "class": render_class,
                    "ratio": ratio,
                    "cooc": cooc,
                    "forced": bool(is_forced),
                    "supported_reconnect": bool(is_supported_reconnect),
                    "recovery": bool(is_recov),
                    "postprocess": bool(is_post),
                    "confirmations": G[u][v].get("confirmations", 0),
                })
                print(
                    f"    [EdgeRender] {u}->{v} class={render_class} "
                    f"ratio={ratio:.2f} cooc={cooc:.3f} "
                    f"forced={int(is_forced)} supported={int(is_supported_reconnect)} recov={int(is_recov)} post={int(is_post)}"
                )

            ax.plot([ux, vx], [uy, vy], color=color, linewidth=lw,
                    linestyle=ls, zorder=3, alpha=0.85)

            if ax_idx == 1:
                mx, my = (ux + vx) / 2, (uy + vy) / 2
                ax.text(mx, my, f"ok{G[u][v].get('confirmations', 0)}",
                        fontsize=7, color="#aaffaa",
                        ha="center", va="center", zorder=6,
                        bbox=dict(facecolor="black", alpha=0.4, pad=1))

        for nid, (px, py) in pos_px.items():
            code = G.nodes[nid]["code"]
            val = get_valence(code)
            col = VALENCE_COLORS.get(val, "#aaa")
            r = W * 0.013 if val >= 3 else W * 0.009
            circle = plt.Circle((px, py), r, color=col, zorder=4,
                                linewidth=2,
                                edgecolor="white" if val >= 3 else "#aaaaaa")
            ax.add_patch(circle)
            if ax_idx == 1:
                ax.text(px, py - r * 1.4,
                        f"{get_name(code)}\n({nid})",
                        fontsize=6, color="white", fontweight="bold",
                        ha="center", va="bottom", zorder=5,
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.5))
            else:
                ax.text(px, py - r * 1.4, str(code),
                        fontsize=7, color="white", fontweight="bold",
                        ha="center", va="bottom", zorder=5)

        for nid in [n for n in G.nodes() if G.degree(n) == 0 and n in pos_px]:
            px, py = pos_px[nid]
            ax.plot(px, py, "x", markersize=14, color="red",
                    markeredgewidth=2.5, zorder=7)

    axes[0].set_title("Overview", color="white", fontsize=11, pad=6)

    n_forced = sum(1 for u, v in G.edges() if G[u][v].get("forced"))
    n_recov = sum(1 for u, v in G.edges() if G[u][v].get("recovery"))
    n_post = sum(1 for u, v in G.edges() if G[u][v].get("postprocess"))
    n_iso = sum(1 for n in G.nodes() if G.degree(n) == 0)
    connected = nx.is_connected(G) if G.number_of_nodes() > 0 else True
    axes[1].set_title(
        f"Dettaglio | cutoff={cutoff:.3f} | "
        f"forced={n_forced} recov={n_recov} post={n_post} "
        f"{'connesso' if connected else 'DISCONNESSO'}",
        color="#00ff88" if connected else "#ff4444",
        fontsize=10, pad=6,
    )

    axes[1].legend(handles=[
        mpatches.Patch(color=VALENCE_COLORS[1], label="Val.1 terminale"),
        mpatches.Patch(color=VALENCE_COLORS[2], label="Val.2 in-linea"),
        mpatches.Patch(color=VALENCE_COLORS[3], label="Val.3 Tee"),
        mpatches.Patch(color=VALENCE_COLORS[4], label="Val.4 Croce"),
        mpatches.Patch(color="#00ff88", label="confermato >=50%"),
        mpatches.Patch(color="#ffdd00", label="prior forte"),
        mpatches.Patch(color="#ff8800", label="dubbio"),
        mpatches.Patch(color="#66e0ff", label="reconnect supportato"),
        mpatches.Patch(color="#cc44ff", label="recovery"),
        mpatches.Patch(color="#00ccff", label="forzato (connettivita)"),
        mpatches.Patch(color="#ff0000", label="isolato"),
        mpatches.Patch(color="#ff66cc", label="post-process"),
    ], loc="lower right", framealpha=0.75, facecolor="#1a1a2e",
       labelcolor="white", fontsize=8)

    fig.suptitle(
        f"Node {node_id} - {best_view['path'].name} "
        f"({G.number_of_nodes()} nodi · {G.number_of_edges()} edge)",
        color="white", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = WORK_DIR / f"node{node_id}_graph_overlay.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    if debug_dirs is not None:
        save_render_summary(debug_dirs, render_payload)
    print(
        f"  Salvato: {out}  "
        f"[{'CONNESSO' if connected else 'DISCONNESSO'} | "
        f"forced={n_forced} recov={n_recov} post={n_post} isolati={n_iso}]"
    )
