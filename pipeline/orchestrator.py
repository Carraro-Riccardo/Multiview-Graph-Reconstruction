"""Image-only multi-view hydraulic reconstruction - direct adjacency pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from config import IMAGES_DIR, MIN_IMAGES, SKIP_NODES, VGGT_EDGE_ENABLED, VGGT_SKIP_NODES, WORK_DIR
from core.debug_utils import save_json
from core.detection import discover_nodes
from .diameter_annotation import annotate_graph_diameters
from .direct_graph_builder import build_direct_graph
from .graph_refinement import force_spanning_tree, serialize_fused_graph
from .node_reid import identify_scene_nodes, merge_spurious_singletons, serialize_scene_nodes
from .observed_graphs import build_all_observed_graphs
from .vggt_runtime import ensure_vggt_artifacts
from .vggt_3d_edges import build_vggt_edge_hints, build_vggt_search_roots, replay_vote_log

_W = 72
_VGGT_CACHE_VERSION = 3


def _sep(char="-"):
    fill = "-" if char == "-" else "="
    print(f"  {fill * _W}")


def _output_dir(node_id: str) -> Path:
    return WORK_DIR / "multiview" / f"node{node_id}"


def run_multiview_pipeline(node_id: str, image_paths: list, output_dir: Path | None = None) -> dict:
    if output_dir is None:
        output_dir = _output_dir(node_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * (_W + 4)}")
    print(f"  NODE {node_id}  -  {len(image_paths)} views  -  {output_dir}")
    print(f"{'=' * (_W + 4)}")

    print("\n  [Step 1] Per-view detection & local graph hypotheses")
    _sep()
    observed_graphs = build_all_observed_graphs(image_paths)
    if not observed_graphs:
        print(f"  [node {node_id}] ERROR: no valid detections in any view - skipping")
        return {"status": "error", "reason": "no_detections", "node_id": node_id}

    total_obs = sum(len(og.nodes) for og in observed_graphs)
    print(f"  views accepted : {len(observed_graphs)}/{len(image_paths)}")
    print(f"  total detections (post-filter) : {total_obs}")

    # ── VGGT artefacts (needed for both re-id 3D constraints and edge hints) ───
    vggt_edge_payload = {"enabled": False, "pair_hints": {}}
    vggt_generation = {"status": "not_requested", "generated": False}
    search_roots: list = []
    _vggt_active = VGGT_EDGE_ENABLED and node_id not in VGGT_SKIP_NODES
    if node_id in VGGT_SKIP_NODES:
        print(f"  [VGGT] skipped for node {node_id} (non-overlapping views)")
    if _vggt_active:
        vggt_generation = ensure_vggt_artifacts(output_dir, image_paths)
        search_roots = build_vggt_search_roots(node_id, output_dir, WORK_DIR)

    print("\n  [Step 2] Node re-identification (graph coloring + agglomerative)")
    _sep()
    scene_nodes, reid_debug = identify_scene_nodes(
        observed_graphs, image_paths=image_paths, search_roots=search_roots
    )
    save_json(output_dir / "scene_nodes_reid_raw.json", serialize_scene_nodes(scene_nodes))
    _obs_geom_payload = reid_debug.pop("obs_geometry", {})
    save_json(output_dir / "reid_debug.json", reid_debug)
    save_json(output_dir / "reid_geometry.json", _obs_geom_payload)
    save_json(output_dir / "scene_nodes.json", serialize_scene_nodes(scene_nodes))

    col = 16
    print(f"  {'scene_node_id':<18}{'name':<{col}}{'code':>6}{'supp':>6}{'views':>7}{'conf':>7}{'anchor':>8}")
    _sep()
    for sn in sorted(scene_nodes, key=lambda s: (s.fused_code, s.scene_node_id)):
        print(
            f"  {sn.scene_node_id:<18}{sn.fused_name:<{col}}"
            f"{sn.fused_code:>6}{sn.support_count:>6}{len(sn.support_images):>7}"
            f"{sn.fused_confidence:>7.2f}{sn.anchor_hint_score:>8.2f}"
        )
    _sep()
    print(f"  {len(scene_nodes)} scene nodes from {total_obs} observations")

    if _vggt_active:

        # ── Cache check ────────────────────────────────────────────────────────
        # If all .npz files were already present AND vggt_edge_debug.json exists
        # with the same set of scene-node IDs as the current ReID output, we can
        # skip the expensive build_vggt_edge_hints() call entirely.
        debug_path = output_dir / "vggt_edge_debug.json"
        _cache_hit = False
        if vggt_generation.get("status") == "already_present" and debug_path.exists():
            try:
                _cached = json.loads(debug_path.read_text(encoding="utf-8"))
                if _cached.get("cache_version") != _VGGT_CACHE_VERSION:
                    raise ValueError("stale VGGT cache version")
                if "vote_log" not in _cached:
                    raise ValueError("legacy cache format — missing vote_log")
                _cached_ids = set(_cached.get("scene_geometry", {}).keys())
                _current_ids = {sn.scene_node_id for sn in scene_nodes}
                if _current_ids == _cached_ids:
                    _cache_hit = True
            except Exception:
                pass  # malformed or legacy cache → fall through to full recompute

        if _cache_hit:
            _vote_log = _cached.get("vote_log", [])
            _pair_hints = {
                (entry["pair"][0], entry["pair"][1]): {k: v for k, v in entry.items() if k != "pair"}
                for entry in _cached.get("pair_hints", [])
            }
            n_corrected = len(_vote_log)
            if n_corrected > 0:
                scene_nodes = replay_vote_log(scene_nodes, _vote_log)
                save_json(output_dir / "scene_nodes.json", serialize_scene_nodes(scene_nodes))
                print(f"  [VGGT] cache hit — replayed {n_corrected} code correction(s)")
                for entry in _vote_log:
                    print(f"    {entry['scene_node_id']}: {entry['old_code']} → {entry['new_code']}")
            vggt_edge_payload = {
                "enabled":        True,
                "pair_hints":     _pair_hints,
                "scene_geometry": _cached.get("scene_geometry", {}),
                "scene_cutoff":   _cached.get("scene_cutoff"),
            }
            print(
                f"  [VGGT] generation={vggt_generation.get('status')}  "
                f"cache=hit  corrected={n_corrected}  "
                f"edge_hints={len(_pair_hints)}"
            )
        # ── Full recompute ─────────────────────────────────────────────────────
        else:
            vggt_edge_payload = build_vggt_edge_hints(scene_nodes, observed_graphs, search_roots)

            # Apply VGGT-based code corrections (no merges — re-id handles duplicates)
            n_corrected = vggt_edge_payload.get("n_code_corrections", 0)
            if n_corrected > 0:
                scene_nodes = vggt_edge_payload["_scene_nodes_after_vote"]
                save_json(output_dir / "scene_nodes.json", serialize_scene_nodes(scene_nodes))
                vote_log = vggt_edge_payload.get("vote_log", [])
                print(f"  [VGGT] corrected {n_corrected} scene node code(s) via majority vote:")
                for entry in vote_log:
                    print(
                        f"    {entry['scene_node_id']}: "
                        f"{entry['old_code']} → {entry['new_code']}"
                        f"  (n_obs={entry['n_obs']})"
                    )

            save_json(
                output_dir / "vggt_edge_debug.json",
                {
                    "cache_version":        _VGGT_CACHE_VERSION,
                    "generation":           vggt_generation,
                    "enabled":              vggt_edge_payload.get("enabled", False),
                    "search_roots":         vggt_edge_payload.get("search_roots", []),
                    "artifact_request":     vggt_edge_payload.get("artifact_request", {}),
                    "n_loaded_images":      vggt_edge_payload.get("n_loaded_images", 0),
                    "n_scene_geometries":   vggt_edge_payload.get("n_scene_geometries", 0),
                    "n_code_corrections":   n_corrected,
                    "vote_log":             vggt_edge_payload.get("vote_log", []),
                    "scene_cutoff":         vggt_edge_payload.get("scene_cutoff"),
                    "loaded_images":        vggt_edge_payload.get("loaded_images", {}),
                    "skipped_images":       vggt_edge_payload.get("skipped_images", []),
                    "scene_geometry":       vggt_edge_payload.get("scene_geometry", {}),
                    "pair_hints":           vggt_edge_payload.get("pair_hints_serialized", []),
                },
            )
            save_json(
                output_dir / "vggt_artifact_request.json",
                vggt_edge_payload.get("artifact_request", {}),
            )
            print(
                f"  [VGGT] generation={vggt_generation.get('status')}  "
                f"cache=miss  "
                f"loaded_views={vggt_edge_payload.get('n_loaded_images', 0)}/"
                f"{len(observed_graphs)}  "
                f"scene_geometries={vggt_edge_payload.get('n_scene_geometries', 0)}  "
                f"corrected={n_corrected}  "
                f"edge_hints={len(vggt_edge_payload.get('pair_hints', {}))}"
            )

    # ── Spurious singleton merge (post-VGGT, uses 3D geometry when available) ──
    _scene_geom   = vggt_edge_payload.get("scene_geometry") or {}
    _scene_cutoff = vggt_edge_payload.get("scene_cutoff")
    scene_nodes, _spur_log = merge_spurious_singletons(
        scene_nodes, observed_graphs,
        scene_geom=_scene_geom or None,
        scene_cutoff=_scene_cutoff,
        pair_hints=vggt_edge_payload.get("pair_hints"),
    )
    if _spur_log:
        save_json(output_dir / "scene_nodes.json", serialize_scene_nodes(scene_nodes))

    print("\n  [Step 3] 3D graph build + spanning tree")
    _sep()
    fused_graph, bridge_edges, direct_debug = build_direct_graph(
        scene_nodes,
        observed_graphs,
        len(observed_graphs),
        vggt_edge_hints=vggt_edge_payload.get("pair_hints", {}),
    )
    n_vggt_pairs = direct_debug["n_vggt_pairs"]
    print(f"  vggt_pairs={n_vggt_pairs}  edges={fused_graph.number_of_edges()}")

    fused_graph, span_debug = force_spanning_tree(fused_graph, bridge_edges)
    n_forced = span_debug["n_forced"]
    print(f"  forced edges added : {n_forced}")
    for fe in span_debug["forced_edges"]:
        print(f"    {fe['a']}  --({fe['via']})->  {fe['b']}")

    fused_graph, diameter_debug = annotate_graph_diameters(
        fused_graph,
        scene_nodes,
        image_paths,
    )
    save_json(output_dir / "diameter_debug.json", diameter_debug)
    print(
        "  diameter labels: "
        f"{diameter_debug['n_diameter_observations']} observations, "
        f"{diameter_debug['n_seeded_edges']} seeded edge(s)"
    )

    save_json(output_dir / "fused_graph.json", serialize_fused_graph(fused_graph, len(observed_graphs)))

    n_isolated = sum(1 for nid in fused_graph.nodes() if fused_graph.degree(nid) == 0)
    connected = nx.is_connected(fused_graph) if fused_graph.number_of_nodes() > 0 else True

    print(f"\n  {'-' * _W}")
    print(f"  FINAL GRAPH  -  node {node_id}")
    print(f"  {'-' * _W}")
    print(
        f"  nodes: {fused_graph.number_of_nodes()}   edges: {fused_graph.number_of_edges()}   "
        f"isolated: {n_isolated}   {'CONNECTED' if connected else 'DISCONNECTED'}"
    )
    print(f"  {'-' * _W}")
    print(f"  {'node_id':<18}{'name':<16}{'code':>6}{'deg':>5}{'val':>5}  support_images")
    print(f"  {'-' * _W}")
    for nid in sorted(fused_graph.nodes(), key=lambda x: (fused_graph.nodes[x].get('code', 0), x)):
        nd = fused_graph.nodes[nid]
        deg = fused_graph.degree(nid)
        val = nd.get("valence", nd.get("max_degree", "?"))
        name = nd.get("name", nd.get("fused_name", "?"))
        code = nd.get("code", nd.get("fused_code", "?"))
        imgs = ", ".join(nd.get("support_images", []))
        flag = " !" if deg == 0 else ""
        print(f"  {nid:<18}{name:<16}{str(code):>6}{deg:>5}{str(val):>5}  {imgs}{flag}")
    print(f"  {'-' * _W}")
    print(f"  output: {output_dir}")

    # ── Bird's-eye view ───────────────────────────────────────────────────────
    from bird_view import render_bird_view
    bv_path = render_bird_view(output_dir)
    if bv_path:
        print(f"  [BirdView] saved → {bv_path.name}")

    return {
        "status": "ok",
        "node_id": node_id,
        "n_views": len(observed_graphs),
        "n_scene_nodes": len(scene_nodes),
        "n_vggt_pairs": n_vggt_pairs,
        "n_final_nodes": fused_graph.number_of_nodes(),
        "n_final_edges": fused_graph.number_of_edges(),
        "n_forced_edges": n_forced,
        "n_isolated": n_isolated,
        "connected": connected,
        "output_dir": str(output_dir),
    }


def run_multiview_all(images_dir: Path | None = None) -> dict:
    if images_dir is None:
        images_dir = IMAGES_DIR

    eligible = {
        node_id: image_paths
        for node_id, image_paths in discover_nodes(images_dir).items()
        if node_id not in SKIP_NODES and len(image_paths) >= MIN_IMAGES
    }
    def _sort_key(node_id: str):
        return [int(p) if p.isdigit() else p for p in node_id.split("_")]

    node_ids = sorted(eligible, key=_sort_key)
    print(f"\nNodes to process: {node_ids}")

    results = {}
    for node_id, image_paths in sorted(eligible.items(), key=lambda kv: _sort_key(kv[0])):
        results[node_id] = run_multiview_pipeline(node_id, image_paths)

    ok_results = [r for r in results.values() if r.get("status") == "ok"]
    if ok_results:
        print(f"\n\n{'=' * (_W + 4)}")
        print(f"  OVERALL SUMMARY  -  {len(ok_results)}/{len(results)} nodes completed")
        print(f"{'=' * (_W + 4)}")
        hdr = f"  {'node':>5}{'views':>7}{'scene_n':>9}{'vggt_p':>9}{'final_n':>9}{'final_e':>9}{'forced':>8}{'iso':>5}  conn"
        print(hdr)
        print(f"  {'-' * (_W + 2)}")
        for r in sorted(ok_results, key=lambda x: [int(p) if p.isdigit() else p for p in str(x["node_id"]).split("_")]):
            conn = "Y" if r["connected"] else "N"
            print(
                f"  {r['node_id']:>5}"
                f"{r['n_views']:>7}"
                f"{r['n_scene_nodes']:>9}"
                f"{r.get('n_vggt_pairs', 0):>9}"
                f"{r['n_final_nodes']:>9}"
                f"{r['n_final_edges']:>9}"
                f"{r.get('n_forced_edges', 0):>8}"
                f"{r['n_isolated']:>5}  {conn}"
            )
        print(f"  {'-' * (_W + 2)}")

    return results


if __name__ == "__main__":
    WORK_DIR.mkdir(exist_ok=True)
    run_multiview_all()
