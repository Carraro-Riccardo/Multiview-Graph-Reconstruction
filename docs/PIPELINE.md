# Hydraulic Graph Reconstruction Pipeline

## Scope

This document describes the current end-to-end pipeline used in this repository to reconstruct a hydraulic scene graph from multi-view image detections.

The pipeline takes multiple images of the same hydraulic node and produces:

- a set of detected/decoded hydraulic components
- a selected best view for geometric layout
- a connected acyclic graph
- node-level metadata and edge-level confidence annotations
- an overlay visualization on the best view

The intended graph assumptions are:

- nodes are hydraulic components
- edges are physical hydraulic connections
- the graph for one scene should be connected
- the graph should be acyclic
- node valence must be respected
- geometry should gate candidate edges
- multi-view and semantic priors should rerank local options, not justify long jumps

## Code Map

### Entry point

- `main.py`

Relevant functions:

- `run_step1()`
- `run_step4()`
- `_eligible_nodes()`

### Detection and decoding

- `detection.py`

Relevant functions:

- `discover_nodes()`
- `detect_and_decode()`
- `detect_and_decode_cached()`
- `detect_and_decode_with_conf()`
- `detect_and_decode_simple()`
- `score_view()`

### Graph construction

- `graph_engine.py`

Relevant functions:

- `aggregate_type_consensus()`
- `consolidate_best_view_codes()`
- `build_per_view_type_edges()`
- `_build_local_candidate_pairs()`
- `build_graph_ilp_soft()`
- `_postprocess_tree_edges()`
- `confirm_edges_multivista()`
- `break_cycles()`

### Visualization

- `visualization.py`

Relevant functions:

- `plot_node_detections()`
- `plot_graph_on_image()`

### Configuration / priors

- `config.py`

Relevant data:

- `COMPONENT_INFO`
- `COOC_COUNTS`
- `TRIPLET_COUNTS`
- `DEGREE_SATURATION`
- `FORBIDDEN_PAIRS`
- thresholds such as `K_NEIGHBORS`, `DIST_MULTIPLIER`, `CONF_THRESHOLD`

## End-to-End Flow

### 1. Image grouping

`discover_nodes()` scans the image directory and groups files by the pattern `node<id>_*.jpg`.

`_eligible_nodes()` then filters out:

- nodes listed in `SKIP_NODES`
- nodes with fewer than `MIN_IMAGES`

This means the reconstruction always runs on a multi-view bundle, not on a single image in isolation.

### 2. Detection and decoding

For each image, `detect_and_decode()` performs:

1. image load
2. augmentation / preprocessing through `augmentor`
3. YOLO detection on the augmented image
4. patch extraction on the original image
5. Shurikode decoding through `decoder_engine`
6. assembly of normalized detections with:
   - bounding box
   - decoded component code
   - decode confidence
   - normalized center `(cx, cy)`
   - name
   - physical valence
   - diameter / non-diameter flag

#### Cache

`detect_and_decode_cached()` stores one JSON per image under:

- `step1_detection/detection_cache/`

This allows `run_step4()` to start directly from cached detections/decodings without rerunning YOLO and patch decoding every time.

The two public wrappers are:

- `detect_and_decode_with_conf()`: used by graph reconstruction
- `detect_and_decode_simple()`: used by the step-1 visualization

Both reuse the same cache.

### 3. Best-view selection

`run_step4()` scores each view with `score_view()`:

- count of topological detections
- plus mean confidence over non-diameter detections

The best view is the one with the highest score. This best view provides:

- the node instances used for final graph construction
- the 2D positions used for edge geometry
- the final overlay image

Important: multi-view is still used after best-view selection, but only as supporting evidence.

### 4. Best-view code consolidation

`consolidate_best_view_codes()` now applies a conservative relabeling pass on the best view before graph construction.

Purpose:

- keep the best view as geometric reference
- use other views as structural support
- fix suspicious codes that are weakly supported in the best view
- avoid changing stable, well-supported detections

The relabeler does not add or remove nodes. It only considers replacing a best-view code when:

- the current code has weak support across views
- an alternative code is observed in the scene bundle
- the alternative fits the local neighbor context better
- the alternative improves semantic and triplet consistency by a clear margin

When a correction is applied, the pipeline logs:

- `[ViewFuse] ...`

### 5. Multi-view type consensus

`aggregate_type_consensus()` builds a type-level adjacency prior from all views.

For each view:

- `build_per_view_type_edges()` builds a local KNN graph on topological detections
- it converts instance proximity into type-pair adjacency votes
- it keeps only type pairs allowed by `edge_allowed()`

The final output is:

- `type_consensus[(code_a, code_b)] = fraction_of_views_supporting_that_pair`

Interpretation:

- this is not an instance-level edge
- it is evidence that two component types are often adjacent in the current scene bundle
- it is used as a reranking signal, not as permission for long-distance edges

### 6. Local candidate-edge generation

The most important upstream filter is `_build_local_candidate_pairs()`.

Inputs:

- best-view detections
- full pairwise distance matrix
- type consensus
- local geometric cutoff

#### 5.1 Base geometric cutoff

`compute_dist_cutoff()` computes a scene-specific cutoff from the median nearest-neighbor distance multiplied by `DIST_MULTIPLIER`.

This yields a scale estimate for the current best view.

#### 5.2 Hard locality rules

The candidate builder enforces locality through:

- `LOCAL_TOP_K`
- `LOCAL_RANK_K`
- `HARD_MAX_DIST_FACTOR`
- mutual-rank filtering
- top-k neighborhood restriction

Normal candidate admission requires:

- type pair allowed by `edge_allowed()`
- local neighbor status
- mutual-rank compatibility
- distance below hard cap

#### 5.3 Hub-aware exceptions

For high-valence nodes, the builder allows some controlled flexibility:

- extra local neighbors for hubs
- multi-view hub rescue candidates

This is used for tee/cross cases where perspective can make the physically correct branch appear farther away in the best view.

The rescue logic is still bounded:

- only for hub-like nodes
- only for preferred semantic neighbors
- only when multi-view support is strong enough
- still under an extended but finite distance cap

#### 5.4 Pair features

For each admitted candidate, the builder computes:

- `dist`
- `dist_norm`
- `mutual_rank`
- `mv`
- `cooc`
- `hub_bonus`
- `hub_penalty`
- `source`
- `score`

The candidate score is geometry-first:

- positive weight on mutual-rank agreement
- positive weight on multi-view support
- positive weight on empirical co-occurrence
- positive hub bonus for semantically plausible hub neighborhoods
- explicit penalty for hub-to-disfavored types
- strong quadratic penalty for long edges

The output of `_build_local_candidate_pairs()` is the candidate set that effectively defines the optimization problem.

## 7. Tree optimization with ILP

`build_graph_ilp_soft()` is the current graph-construction core.

Despite the historical name `soft`, the current formulation is much stricter than before.

### 6.1 Objective

The ILP maximizes the sum of candidate edge scores.

The final edge score includes:

- local pair score from `_build_local_candidate_pairs()`
- optional triplet bonus derived from `TRIPLET_COUNTS`

### 6.2 Hard constraints

The solver enforces:

- degree upper bounds from physical valence `get_valence()`
- minimum degree `>= 1` for nodes that have incident candidates
- full degree saturation for nodes with valence `>= 3` when enough local candidates exist
- exactly `n - 1` selected edges

Because the graph has `n - 1` edges and connectivity is enforced, the result is a tree:

- connected
- acyclic

### 6.3 Connectivity constraints

The solver starts from the tree-size constraint and then iteratively adds connectivity cuts if the provisional solution is disconnected.

This avoids the older behavior where the solver could stop on a forest and rely heavily on late rescue edges.

### 6.4 Fallback

If the ILP becomes infeasible, the code falls back to `build_graph_consensus_first()`.

That fallback is older and less desirable than the ILP path, but it remains as a safety net.

## 8. Hub post-processing

After the ILP, `_postprocess_tree_edges()` may apply a small local rewiring around tee/cross-like nodes.

Purpose:

- replace a hub edge toward a semantically disfavored neighbor
- with a more plausible preferred neighbor
- while preserving connectivity and degree constraints

This step is conservative:

- it only considers edges that already exist in the local candidate set
- it performs only local swaps
- it does not rewrite the whole tree

When it triggers, it logs:

- `[PostHub] ...`

## 9. Multi-view confirmation annotation

`confirm_edges_multivista()` does not change the graph topology.

It annotates each selected edge with:

- `confirmations`
- `confirm_ratio`

This is computed by checking, in each non-best view, whether the opposite endpoint type appears within the local KNN neighborhood of the source type.

This value is currently used only for visualization and diagnostics.

Important:

- an edge can be correct and still have low `confirm_ratio`
- low `confirm_ratio` does not mean the solver rejected the edge
- it only means that the edge is weakly reconfirmed by this specific KNN-based multi-view check

## 10. Cycle breaking

`break_cycles()` is a final safety stage.

The ILP path should already return a tree, but this function remains as a guardrail for fallback or post-process cases.

If a cycle exists, it removes the weakest edge according to a score based on:

- `mv_score`
- `confirm_ratio`
- `triplet_score`
- `cooc`
- penalties for `forced` and `recovery` edges

It tries not to isolate nodes while removing cycle edges.

## 11. Visualization semantics

`plot_graph_on_image()` renders both:

- overview image
- detailed overlay

The edge color/style is purely a visualization layer:

- light-cyan solid: `supported_reconnect`
- cyan `-.`: `forced`
- magenta `:`: `recovery`
- pink `--`: `postprocess`
- green solid: `confirm_ratio >= 0.5`
- yellow solid: `confirm_ratio < 0.5` but `cooc > 0.3`
- orange dashed: everything else, labeled `dubbio`

### Why an edge can be marked `dubbio`

An edge is rendered as `dubbio` when:

- it is not `forced` or `supported_reconnect`
- it is not `recovery`
- it is not `postprocess`
- `confirm_ratio < 0.5`
- `cooc <= 0.3`

This is a rendering rule, not a graph-validity rule.

So an edge can be:

- topologically correct
- chosen by the ILP
- geometrically plausible
- semantically allowed

and still appear orange/dashed only because its visualization evidence is weak under this narrow overlay criterion.

## 12. Meaning of the debug logs

### `mv [x.xx] A â†” B`

Type-level multi-view support for pair `(A, B)`.

### `[HubDebug]`

Printed for high-valence nodes during ILP setup.

Shows:

- ranked local candidates
- candidate score
- multi-view support
- mutual-rank score
- co-occurrence
- hub bonus / penalty
- distance
- source of admission

Useful for diagnosing:

- perspective problems
- missing branches around tees/crosses
- whether a correct edge was absent from the candidate set or simply lost in ranking

### `[HubChosen]`

Final neighbors selected for each high-valence node.

### `[ViewFuse]`

Best-view code corrected using support from other views.

### `supported_reconnect`

Connectivity edge added late, but with strong semantic and multi-view support.

### `[Soft-Connect]`

Local reconnection edge added by the older soft connectivity helper.

### `[PostHub]`

Post-ILP hub rewiring.

### `[CycleBreak]`

Cycle removal at the final cleanup stage.

## 13. Priors and constraints

The current priors live in `config.py`.

### `COOC_COUNTS`

Empirical type-pair adjacency counts from annotations.

Used through `cooc_score()` as a normalized semantic prior.

### `TRIPLET_COUNTS`

Empirical local patterns of the form:

- neighbor A
- center node
- neighbor C

Used to reward locally consistent neighborhoods, especially around valence-2 and valence-3 components.

### `DEGREE_SATURATION`

Observed fraction of times a component type reaches full degree in the annotated dataset.

This affects:

- `effective_valence()`
- some older fallback behavior
- interpretation of which components are commonly terminal in the visible scene

### `FORBIDDEN_PAIRS`

Empirical never-seen type pairs.

These are used in `edge_allowed()` as hard exclusions.

## 14. Why the current system works better than the original one

The original failure mode was a combination of:

- overly permissive candidate generation
- semantic priors that could justify long-range edges
- soft connectivity repaired later with implausible rescue edges

The current system improves this by:

- aggressively pruning the candidate set first
- making geometry a hard gate
- forcing a connected tree directly in the optimizer
- keeping hard degree constraints
- using multi-view mainly as local reranking support
- reserving post-processing for small, hub-centered repairs

## 15. Known limitations

The pipeline is much more stable now, but some limitations remain.

### Perspective distortion

A single best view can still distort local distances enough to make a correct branch look weaker than an incorrect closer branch.

This is why hub-rescue and post-hub rewiring exist.

### Type-only multi-view confirmation

`confirm_edges_multivista()` checks type adjacency, not instance identity across views.

As a result:

- it is useful for diagnostics
- it is not a full multi-view instance matcher

### Fallback path quality

`build_graph_consensus_first()` still exists as a fallback and is less strict than the ILP path.

If `pulp` is unavailable or the ILP becomes infeasible, quality may degrade.

### Visualization confidence is conservative

The overlay labels are intentionally simple and can understate confidence for correct edges.


## 16. Debug outputs and how to read them

Every `run_step4()` execution now creates a per-node debug folder under:

- `step1_detection/debug/node<id>/`

Subfolders:

- `views/`
- `multiview/`
- `consolidation/`
- `candidates/`
- `solver/`
- `render/`
- `graphs/`

### `views/`

Contents:

- `view_selection.json`: one entry per image with score, number of detections, chosen best view
- `view_selection_scores.png`: bar chart of best-view selection
- `all_views_detections.json`: raw decoded detections for every image
- `best_view_detections_raw.json`: raw best-view detections before consolidation
- `best_view_detections_consolidated.json`: best-view detections after multi-view relabeling

Use this folder to answer:

- which image won and why?
- how many detections are present in each view?
- is the wrong code already present before graph construction?

### `multiview/`

Contents:

- `type_consensus.json`: type-pair support across views
- `type_consensus_heatmap.png`: matrix view of the same information
- `edge_confirmations.json`: per-final-edge confirmation report with per-view evidence

Use this folder to answer:

- which type pairs are repeatedly visible across views?
- which final edges are confirmed strongly or weakly?
- which views support each edge and which do not?

### `consolidation/`

Contents:

- `best_view_consolidation.json`

This file records:

- view support per code
- observed alternative codes in the full bundle
- relabeling checks for each suspicious detection
- accepted relabel events

Use this folder to answer:

- why was a best-view code kept?
- why was it replaced?
- which alternative hypotheses lost and by how much?

### `candidates/`

Contents:

- `candidate_pairs.json`: all admitted local candidate edges with geometry and semantic features
- `pair_scores_after_triplets.json`: candidate scores after triplet bonuses
- `candidate_edges_all.png`: overlay of all local candidate edges
- `candidate_edges_selected.png`: overlay of the final selected tree edges

Use this folder to answer:

- did the correct edge enter the candidate set?
- was the correct edge penalized by distance or mutual-rank?
- did triplet support change the ranking enough?

### `solver/`

Contents:

- `solver_iter_*.json`: ILP iteration snapshots with active edges and connected components
- `solver_selected_edges.json`: final selected tree edges and their features

Use this folder to answer:

- was the ILP already connected in the first iteration?
- which component cuts were added?
- which edge scores drove the final tree selection?

### `render/`

Contents:

- `edge_render_classes.json`: final rendering class for each edge

This records the exact visualization semantics:

- `confirmed`
- `prior_strong`
- `dubious`
- `forced`
- `supported_reconnect`
- `recovery`
- `postprocess`

Use this folder to answer:

- why is an edge green/yellow/orange/blue/pink?
- is the visual uncertainty coming from the solver or just from the overlay thresholds?

### `graphs/`

Contents:

- `graph_after_build.json`
- `graph_after_multiview_confirm.json`
- `graph_final.json`

Use this folder to answer:

- how does the graph evolve stage by stage?
- which metadata gets added after multi-view confirmation?
- did cycle-breaking or post-processing modify the structure?

### Console logs now emitted

The console output now exposes the main reasoning chain:

- `[DetectCache]`: whether detections were loaded from cache or recomputed
- `[DetectRun]`: how many patches were decoded and accepted
- `[View ...]`: per-image summary for best-view selection
- `Multi-view consensus top pairs`: strongest type-level cross-view supports
- `[ViewFuseCheck]`: per-node relabel diagnostic before best-view consolidation
- `[ViewFuse]`: accepted code replacement in the best view
- `[CandidateSet]`: total local candidate edges admitted to the optimizer
- `[Cand#k]`: top-ranked candidate edges before optimization
- `[HubDebug]`: detailed local ranking around high-valence nodes
- `[ILPIter]`: ILP solve iterations and connected-component count
- `[HubChosen]`: final neighbors chosen for tee/cross nodes
- `[MVEdge]`: per-final-edge multi-view confirmation summary
- `[EdgeRender]`: final overlay class assigned to every drawn edge
- `[CycleBreak]`: any edge removed to maintain acyclicity

Recommended reading order when debugging one node:

1. `views/`
2. `consolidation/`
3. `multiview/type_consensus.json`
4. `candidates/`
5. `solver/`
6. `graphs/`
7. `render/`
## 17. Practical debugging checklist

If a graph looks wrong, inspect in this order:

1. Are the detections/decoded codes correct in the best view?
2. Did the correct edge enter `_build_local_candidate_pairs()`?
3. In `[HubDebug]`, is the correct neighbor present but ranked too low?
4. Did the ILP choose a semantically weaker but geometrically shorter edge?
5. Did post-processing trigger?
6. Is the edge only marked `dubbio` because of visualization thresholds?

This order helps separate:

- detection errors
- candidate-set pollution
- ranking mistakes
- optimizer mistakes
- visualization-only artifacts

## 18. Current status

With the latest candidate pruning, ILP tree constraints, hub saturation logic, updated priors, and cache reuse:

- all current validation nodes are reconstructing correctly at graph level
- node connectivity is no longer being recovered mainly through long rescue edges
- residual issues are now mostly about diagnostic confidence labels or perspective-specific ambiguity, not catastrophic topology failures

