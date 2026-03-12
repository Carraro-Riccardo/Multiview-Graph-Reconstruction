"""
Pipeline idraulica - entry point.

  run_step1():              Detection + Decoding + Plot per ogni nodo
  run_graph_reconstruction(): Constraint Engine + Graph Reconstruction per ogni nodo
  run_fusion_all():         Multi-view graph fusion (reference prior + grafi ricostruiti)
"""

import numpy as np
import networkx as nx
from pathlib import Path

from config import (
    IMAGES_DIR, WORK_DIR, SKIP_NODES, MIN_IMAGES, DIAMETER_CODES,
    PDF_DIR, PDF_PRIOR_ENABLED, PDF_DIAGRAM_CANDIDATES_ENABLED, PDF_REFERENCE_GRAPH_ENABLED,
)
from core.detection import (
    discover_nodes,
    detect_and_decode_simple,
    detect_and_decode_with_conf,
    score_view,
)
from core.graph_engine import (
    aggregate_type_consensus,
    consolidate_best_view_codes,
    build_graph_ilp_soft,
    confirm_edges_multivista,
    break_cycles,
    get_name,
)
from core.visualization import plot_node_detections, plot_graph_on_image
from core.debug_utils import (
    init_node_debug,
    save_json,
    save_view_selection,
    save_type_consensus,
    save_graph_snapshot,
)
from project_prior import (
    extract_pdf_prior,
    extract_diagram_candidates,
    build_reference_graph_from_pdf_prior,
)
from graph_fusion import run_fusion


def _eligible_nodes(images_dir):
    all_nodes = discover_nodes(images_dir)
    return {nid: imgs for nid, imgs in all_nodes.items()
            if nid not in SKIP_NODES and len(imgs) >= MIN_IMAGES}


def _maybe_extract_pdf_prior(node_id: int):
    if not PDF_PRIOR_ENABLED:
        return None

    pdf_path = PDF_DIR / f"node{node_id}.pdf"
    if not pdf_path.exists():
        return None

    print(f"  [PDFPrior] trovato {pdf_path.name} - avvio estrazione prior")
    extraction = extract_pdf_prior(pdf_path)
    diagram_candidates = (
        extract_diagram_candidates(extraction)
        if PDF_DIAGRAM_CANDIDATES_ENABLED
        else None
    )
    reference_graph = (
        build_reference_graph_from_pdf_prior(
            extraction,
            diagram_result=diagram_candidates,
        )
        if PDF_REFERENCE_GRAPH_ENABLED
        else None
    )
    print(
        f"  [PDFPrior] status={extraction.status} backend={extraction.backend_used} "
        f"pages={len(extraction.pages)} debug={extraction.debug_paths.get('root')}"
    )
    if diagram_candidates is not None:
        top_total = sum(len(items) for items in diagram_candidates.page_candidates.values())
        print(
            f"  [PDFPrior] diagram_candidates={top_total} "
            f"debug={diagram_candidates.debug_paths.get('root')}"
        )
    if reference_graph is not None:
        print(
            f"  [PDFPrior] reference_graph={reference_graph.status} "
            f"nodes={len(reference_graph.nodes)} edges={len(reference_graph.edges)} "
            f"debug={reference_graph.debug_paths.get('root')}"
        )
        for warning in reference_graph.warnings:
            print(f"  [PDFPrior][warn] {warning}")
    return extraction, diagram_candidates, reference_graph


# Step 1 - Detection + Decoding + Plot

def run_step1():
    eligible = _eligible_nodes(IMAGES_DIR)

    for nid, image_paths in sorted(eligible.items()):
        print(f"\n{'=' * 60}")
        print(f"  NODE {nid}  ({len(image_paths)} immagini)")
        print(f"{'=' * 60}")

        all_dets = []
        for img_path in image_paths:
            dets, _ = detect_and_decode_simple(img_path)
            all_dets.append(dets)
            codes = [d["code"] for d in dets]
            print(f"  {img_path.name}: {len(dets)} det -> codici: {codes}")

        plot_node_detections(nid, image_paths, all_dets)


# Graph Reconstruction - Constraint Engine + Multi-view Graph Reconstruction

def run_graph_reconstruction():
    eligible = _eligible_nodes(IMAGES_DIR)

    for node_id, image_paths in sorted(eligible.items()):
        print(f"\n{'=' * 60}\n  NODE {node_id}\n{'=' * 60}")
        pdf_bundle = _maybe_extract_pdf_prior(node_id)
        debug_dirs = init_node_debug(node_id)
        if pdf_bundle is not None:
            pdf_prior, diagram_candidates, reference_graph = pdf_bundle
            save_json(
                debug_dirs["views"] / "pdf_prior_bridge.json",
                {
                    "pdf_path": pdf_prior.pdf_path,
                    "status": pdf_prior.status,
                    "backend_used": pdf_prior.backend_used,
                    "n_pages": len(pdf_prior.pages),
                    "debug_root": pdf_prior.debug_paths.get("root"),
                    "diagram_candidate_counts": (
                        {
                            str(page_idx): len(cands)
                            for page_idx, cands in diagram_candidates.page_candidates.items()
                        }
                        if diagram_candidates is not None else {}
                    ),
                    "diagram_debug_root": (
                        diagram_candidates.debug_paths.get("root")
                        if diagram_candidates is not None else None
                    ),
                    "reference_graph_status": (
                        reference_graph.status if reference_graph is not None else "disabled"
                    ),
                    "reference_graph_provider": (
                        reference_graph.provider if reference_graph is not None else None
                    ),
                    "reference_graph_model": (
                        reference_graph.model if reference_graph is not None else None
                    ),
                    "reference_graph_debug_root": (
                        reference_graph.debug_paths.get("root")
                        if reference_graph is not None else None
                    ),
                    "reference_graph_summary": (
                        reference_graph.summary_stats if reference_graph is not None else {}
                    ),
                    "reference_graph_nodes": (
                        len(reference_graph.nodes) if reference_graph is not None else 0
                    ),
                    "reference_graph_edges": (
                        len(reference_graph.edges) if reference_graph is not None else 0
                    ),
                    "reference_graph_warnings": (
                        reference_graph.warnings if reference_graph is not None else []
                    ),
                    "todo": [
                        "align_image_graph_to_pdf_prior",
                        "scoring observed vs prior nodes",
                        "prior-guided multi-view fusion",
                    ],
                },
            )

        all_views = []
        for view_idx, img_path in enumerate(sorted(image_paths)):
            dets, _ = detect_and_decode_with_conf(img_path)
            all_views.append({"path": img_path, "dets": dets})
            topo_codes = [d["code"] for d in dets if d["code"] not in DIAMETER_CODES]
            confs = [d.get("conf", 0.0) for d in dets if d["code"] not in DIAMETER_CODES]
            mean_conf = float(np.mean(confs)) if confs else 0.0
            print(
                f"    [View {view_idx}] {img_path.name} | det={len(dets)} "
                f"topo={len(topo_codes)} mean_conf={mean_conf:.3f} codes={topo_codes}"
            )

        scores = [score_view(v["dets"]) for v in all_views]
        best_view_idx = int(np.argmax(scores))
        best_view = all_views[best_view_idx]
        best_dets = best_view["dets"]
        n_topo = len([d for d in best_dets if d["code"] not in DIAMETER_CODES])
        print(
            f"  Vista best: {best_view['path'].name}"
            f"  (score={scores[best_view_idx]:.2f}, {n_topo} nodi topo)"
        )

        save_view_selection(debug_dirs, all_views, scores, best_view_idx)
        save_json(
            debug_dirs["views"] / "all_views_detections.json",
            [{"image": view["path"].name, "detections": view["dets"]} for view in all_views],
        )
        save_json(debug_dirs["views"] / "best_view_detections_raw.json", best_dets)

        all_views_dets = [v["dets"] for v in all_views]
        type_consensus = aggregate_type_consensus(all_views_dets)
        save_type_consensus(debug_dirs, type_consensus)

        top = sorted(type_consensus.items(), key=lambda x: -x[1])[:10]
        print("  Multi-view consensus top pairs:")
        for (ca, cb), frac in top:
            print(f"    mv [{frac:.2f}] {get_name(ca)} ↔ {get_name(cb)}")

        best_dets = consolidate_best_view_codes(
            best_dets,
            all_views_dets,
            type_consensus,
            debug_dirs=debug_dirs,
        )
        save_json(debug_dirs["consolidation"] / "best_view_detections_consolidated.json", best_dets)

        G, pos_2d, cutoff = build_graph_ilp_soft(
            best_dets,
            type_consensus,
            debug_dirs=debug_dirs,
            best_view_path=best_view["path"],
        )
        save_graph_snapshot(debug_dirs, G, "after_build")

        print("\n  Grafo dopo build:")
        for nid in sorted(G.nodes()):
            deg = G.degree(nid)
            val = G.nodes[nid]["valence"]
            nbs = [G.nodes[nb]["name"] for nb in G.neighbors(nid)]
            flag = " ⚠ iso" if deg == 0 else (f" ⚠ rem={val-deg}" if deg < val else "")
            print(f"    {nid:12s} val={val} deg={deg}{flag:12s} → {', '.join(nbs) or '—'}")

        G = confirm_edges_multivista(G, all_views_dets, best_view_idx, debug_dirs=debug_dirs)
        save_graph_snapshot(debug_dirs, G, "after_multiview_confirm")

        G = break_cycles(G, pos_2d, cutoff)
        save_graph_snapshot(debug_dirs, G, "final")

        n_iso = sum(1 for n in G.nodes() if G.degree(n) == 0)
        n_forced = sum(1 for u, v in G.edges() if G[u][v].get("forced"))
        connected = nx.is_connected(G) if G.number_of_nodes() > 0 else True
        print(
            f"\n  Finale: {G.number_of_nodes()} nodi · "
            f"{G.number_of_edges()} edge · "
            f"forced={n_forced} · iso={n_iso} · "
            f"{'CONNESSO' if connected else 'DISCONNESSO'}"
        )

        for nid in sorted(G.nodes()):
            nbs = [G.nodes[nb]["name"] for nb in G.neighbors(nid)]
            rv = G.nodes[nid]["valence"] - G.degree(nid)
            flag = (" ⚠ isolato" if G.degree(nid) == 0 else
                    " ⚠ slot liberi" if rv > 0 else "")
            print(f"    {nid:12s} → {', '.join(nbs) or '—'}{flag}")

        plot_graph_on_image(node_id, G, best_view, pos_2d, cutoff, debug_dirs=debug_dirs)

        # ── Graph Fusion (reference prior vs. grafi ricostruiti) ──
        if pdf_bundle is not None:
            _, _, reference_graph = pdf_bundle
            if reference_graph is not None:
                ref_graph_json = (
                    Path(reference_graph.debug_paths["root"]) / "graph" / "reference_graph.json"
                )
                if ref_graph_json.exists():
                    fusion_out = WORK_DIR / "debug" / f"node{node_id}" / "fusion"
                    fused, fusion_state = run_fusion(
                        reference_graph_path=ref_graph_json,
                        observed_graphs_folder=debug_dirs["graphs"],
                        output_dir=fusion_out,
                    )
                    n_supported = sum(1 for v in fusion_state.node_support.values() if v > 0)
                    n_edges_sup = sum(1 for v in fusion_state.edge_support.values() if v > 0)
                    print(
                        f"  [Fusion] nodi supportati={n_supported}/{len(fused.nodes)} "
                        f"archi supportati={n_edges_sup}/{len(fused.edges)} "
                        f"contraddizioni={len(fusion_state.contradictions)} "
                        f"output={fusion_out}"
                    )


# Fusion standalone - esegue run_fusion su tutti i nodi che hanno
# sia un reference_graph.json (da PDF) sia grafi ricostruiti.

def run_fusion_all():
    eligible = _eligible_nodes(IMAGES_DIR)

    for node_id in sorted(eligible):
        ref_graph_json = next(
            (WORK_DIR / "pdf_prior" / f"node{node_id}").rglob("reference_graph.json"),
            None,
        )
        graphs_dir = WORK_DIR / "debug" / f"node{node_id}" / "graphs"

        if ref_graph_json is None:
            print(f"  [Fusion] node{node_id}: nessun reference_graph.json trovato — skip")
            continue
        if not graphs_dir.exists():
            print(f"  [Fusion] node{node_id}: cartella grafi non trovata — skip")
            continue

        print(f"\n{'=' * 60}\n  FUSION NODE {node_id}\n{'=' * 60}")
        fusion_out = WORK_DIR / "debug" / f"node{node_id}" / "fusion"
        fused, fusion_state = run_fusion(
            reference_graph_path=ref_graph_json,
            observed_graphs_folder=graphs_dir,
            output_dir=fusion_out,
        )
        n_supported = sum(1 for v in fusion_state.node_support.values() if v > 0)
        n_edges_sup = sum(1 for v in fusion_state.edge_support.values() if v > 0)
        print(
            f"  nodi supportati={n_supported}/{len(fused.nodes)} "
            f"archi supportati={n_edges_sup}/{len(fused.edges)} "
            f"contraddizioni={len(fusion_state.contradictions)}"
        )


# ── Debug: visualizza fused_graph.json ────────────────────────────────────────

def plot_fused_graph(
    fused_graph_path: str | Path | None = None,
    reference_graph_path: str | Path | None = None,
) -> None:
    """
    Visualizzazione debug di un fused_graph.json con layout ispirato al reference.

    Layout:
      Se il reference_graph.json è disponibile (parametro esplicito o rilevato
      automaticamente in output/pdf_prior/<nodeN>/), i nodi vengono posizionati
      usando le coordinate `center` del diagramma PDF come seed, poi raffinate
      con poche iterazioni di spring layout per distanziare eventuali overlap.
      Il risultato è una birdview topologicamente coerente con il diagramma
      originale, ma non una copia esatta.
      Fallback: spring layout standard.

    Codifica visiva:
      - Nodo verde   → observed_confidence == 1.0  (supportato da tutte le immagini)
      - Nodo arancio → 0 < observed_confidence < 1.0  (supporto parziale)
      - Nodo grigio  → observed_confidence == 0.0  (solo prior, mai osservato)
      - Arco verde solido        → support_count > 0  (confermato)
      - Arco rosso tratteggiato  → support_count == 0  (prior non confermato)
      - Label arco: "k/N" (immagini che confermano / totale)

    Salva fused_graph_debug.png nella stessa cartella del JSON.
    """
    import json
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    if fused_graph_path is None:
        fused_graph_path = WORK_DIR / "debug" / "node9" / "fusion" / "fused_graph.json"
    fused_graph_path = Path(fused_graph_path)

    data  = json.loads(fused_graph_path.read_text(encoding="utf-8"))
    total = data["total_images"]

    # ── Costruisci il grafo NetworkX ──
    G = nx.Graph()
    for n in data["nodes"]:
        G.add_node(n["id"], **n)
    for e in data["edges"]:
        G.add_edge(e["node_a"], e["node_b"], **e)

    # Layout seeded dalle coordinate PDF del reference graph (se disponibile)
    pos = _layout_from_reference(G, fused_graph_path, reference_graph_path)

    # ── Colori nodi ──
    def _node_color(nid: str) -> str:
        c = G.nodes[nid]["observed_confidence"]
        if c >= 1.0:
            return "#2ecc71"   # verde — pienamente supportato
        if c > 0.0:
            return "#f39c12"   # arancio — supporto parziale
        return "#666677"       # grigio — nessuna osservazione

    node_colors = [_node_color(n) for n in G.nodes()]
    node_sizes  = [400 + 150 * G.nodes[n]["degree"] for n in G.nodes()]

    # ── Separa archi confermati da non confermati ──
    edges_ok  = [(u, v) for u, v in G.edges() if G[u][v]["support_count"] > 0]
    edges_bad = [(u, v) for u, v in G.edges() if G[u][v]["support_count"] == 0]

    # ── Etichette nodi: id corto + tipo + DN ──
    labels = {}
    for nid in G.nodes():
        nd = G.nodes[nid]
        dn_str = f"DN{int(nd['dn_main'])}" if nd.get("dn_main") else ""
        labels[nid] = f"{nid}\n{nd['type']}\n{dn_str}"

    # ── Etichette archi: support_count / total ──
    edge_labels = {
        (e["node_a"], e["node_b"]): f"{e['support_count']}/{total}"
        for e in data["edges"]
    }

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(18, 13), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    nx.draw_networkx_edges(
        G, pos, edgelist=edges_ok, ax=ax,
        edge_color="#2ecc71", width=2.5, alpha=0.85,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=edges_bad, ax=ax,
        edge_color="#e74c3c", width=1.5, alpha=0.75, style="dashed",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=node_sizes, alpha=0.95,
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax, font_size=7, font_color="white",
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_size=6, font_color="#bbbbbb", bbox={"alpha": 0},
    )

    # ── Legenda ──
    ax.legend(
        handles=[
            mpatches.Patch(color="#2ecc71", label=f"Nodo supportato ({total}/{total} img)"),
            mpatches.Patch(color="#f39c12", label="Nodo parzialmente supportato"),
            mpatches.Patch(color="#666677", label="Nodo non osservato (solo prior)"),
            mpatches.Patch(color="#e74c3c", label="Arco prior non confermato"),
        ],
        loc="lower right", framealpha=0.7,
        facecolor="#2a2a4e", labelcolor="white", fontsize=9,
    )

    # ── Titolo ──
    n_sup = sum(1 for n in data["nodes"] if n["observed_confidence"] > 0)
    e_sup = sum(1 for e in data["edges"] if e["support_count"] > 0)
    node_label = fused_graph_path.parts[-3] if len(fused_graph_path.parts) >= 3 else ""
    ax.set_title(
        f"Fused Graph — {node_label}  |  "
        f"nodi {n_sup}/{len(data['nodes'])} supportati  ·  "
        f"archi {e_sup}/{len(data['edges'])} supportati  "
        f"({total} immagini)",
        color="white", fontsize=12, pad=14,
    )
    ax.axis("off")
    plt.tight_layout()

    out = fused_graph_path.parent / "fused_graph_debug.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [FusedGraph] salvato: {out}")


def _layout_from_reference(
    G: nx.Graph,
    fused_graph_path: Path,
    reference_graph_path: "str | Path | None",
) -> dict:
    """
    Calcola il layout per plot_fused_graph usando le coordinate PDF del reference.

    1. Cerca reference_graph.json (parametro esplicito → auto-detect in
       output/pdf_prior/<nodeN>/ → fallback spring puro).
    2. Estrae `center` [x, y] di ogni componente (pixel PDF, y verso il basso).
    3. Normalizza in [-1, 1] e inverte y per ottenere l'orientamento corretto.
    4. Usa queste coordinate come seed per spring_layout con poche iterazioni:
       mantiene la topologia del diagramma, distanzia overlap minimi.
    """
    import json

    ref_path: Path | None = None

    if reference_graph_path is not None:
        ref_path = Path(reference_graph_path)

    if ref_path is None:
        # Auto-detect: output/pdf_prior/<nodeN>/**/reference_graph.json
        node_label = fused_graph_path.parts[-3] if len(fused_graph_path.parts) >= 3 else ""
        pdf_prior_dir = WORK_DIR / "pdf_prior" / node_label
        if pdf_prior_dir.exists():
            candidates = list(pdf_prior_dir.rglob("reference_graph.json"))
            if candidates:
                ref_path = candidates[0]

    seed_pos: dict[str, tuple[float, float]] = {}

    if ref_path is not None and ref_path.exists():
        try:
            ref_data = json.loads(ref_path.read_text(encoding="utf-8"))
            raw: dict[str, tuple[float, float]] = {}
            for comp in ref_data.get("components", []):
                center = comp.get("center")
                if center and len(center) == 2 and comp["id"] in G.nodes:
                    raw[comp["id"]] = (float(center[0]), float(center[1]))

            if raw:
                all_x = [v[0] for v in raw.values()]
                all_y = [v[1] for v in raw.values()]
                x_min, x_rng = min(all_x), max(max(all_x) - min(all_x), 1.0)
                y_min, y_rng = min(all_y), max(max(all_y) - min(all_y), 1.0)

                for nid, (x, y) in raw.items():
                    seed_pos[nid] = (
                        2.0 * (x - x_min) / x_rng - 1.0,
                        1.0 - 2.0 * (y - y_min) / y_rng,  # flip y: PDF down → plot up
                    )
        except Exception:
            seed_pos = {}

    if seed_pos:
        # Seed dalle posizioni reference + poche iterazioni spring per overlap
        return nx.spring_layout(G, pos=seed_pos, k=0.4, iterations=25, seed=42)

    return nx.spring_layout(G, seed=42, k=2.0)


if __name__ == "__main__":
    WORK_DIR.mkdir(exist_ok=True)
    #run_graph_reconstruction()
    plot_fused_graph()
