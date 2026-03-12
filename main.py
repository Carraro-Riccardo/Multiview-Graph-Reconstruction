"""
Pipeline idraulica - entry point.

  run_step1():  Detection + Decoding + Plot per ogni nodo
  run_step4():  Constraint Engine + Graph Reconstruction per ogni nodo
"""

import numpy as np
import networkx as nx

from config import (
    IMAGES_DIR, WORK_DIR, SKIP_NODES, MIN_IMAGES, DIAMETER_CODES,
    PDF_DIR, PDF_PRIOR_ENABLED, PDF_DIAGRAM_CANDIDATES_ENABLED, PDF_REFERENCE_GRAPH_ENABLED,
)
from detection import (
    discover_nodes,
    detect_and_decode_simple,
    detect_and_decode_with_conf,
    score_view,
)
from graph_engine import (
    aggregate_type_consensus,
    consolidate_best_view_codes,
    build_graph_ilp_soft,
    confirm_edges_multivista,
    break_cycles,
    get_name,
)
from visualization import plot_node_detections, plot_graph_on_image
from debug_utils import (
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


# Step 4 - Constraint Engine + Graph Reconstruction

def run_step4():
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


if __name__ == "__main__":
    WORK_DIR.mkdir(exist_ok=True)
    run_step4()
