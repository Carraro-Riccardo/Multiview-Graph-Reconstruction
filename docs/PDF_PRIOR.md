# PDF Prior Extraction

## Goal

This module introduces the first PDF-based prior workflow without breaking the current image-first pipeline.

Current scope:

- load a project PDF such as `folder_with_pdf/node9.pdf`
- inspect whether the PDF contains extractable vector information
- extract vector/text primitives in a deterministic way
- cluster geometric groups into diagram candidates
- rank the most plausible diagram regions
- build a first semantic reference graph from the best diagram candidate
- save an intermediate representation and diagnostic previews
- expose a stable API for later graph-prior construction

Still out of scope:

- full graph-to-photo alignment
- solver changes driven directly by the PDF prior
- mismatch handling between project and as-built scene

## Files

- `project_prior/pdf_graph_extraction.py`
- `project_prior/reference_graph_builder.py`
- `project_prior/__init__.py`
- `run_pdf_prior.py`
- `main.py` integration hook

## Entry points

### Standalone

Run only the PDF parsing stage:

```bash
python run_pdf_prior.py
```

or

```bash
python run_pdf_prior.py folder_with_pdf/node9.pdf
```

### Integrated pipeline

`main.py` now checks, for each node, whether a matching PDF exists in:

- `folder_with_pdf/node<id>.pdf`

If found and `PDF_PRIOR_ENABLED=True`, the extraction runs before the image graph pipeline.

For now this is an extraction-and-logging step only.

## API

Main API:

```python
extract_pdf_prior(pdf_path: str | Path) -> PdfPriorExtractionResult
extract_diagram_candidates(parsed_pdf: PdfPriorExtractionResult) -> DiagramCandidateExtractionResult
build_reference_graph(diagram_candidate: DiagramCandidate, parsed_pdf: PdfPriorExtractionResult) -> ReferenceGraphResult
build_reference_graph_from_pdf_prior(parsed_pdf: PdfPriorExtractionResult, ...) -> ReferenceGraphResult
```

The result object contains:

- `pdf_path`
- `status`
- `backend_used`
- `pages`
- `warnings`
- `debug_paths`

Per page:

- `segments`
- `curves`
- `rects`
- `words`
- `candidate_components`
- `summary`
- `preview_path`

Diagram candidate result:

- `page_candidates`
- `warnings`
- `debug_paths`
- `parameters`

## Backends

The extraction strategy is conservative and layered:

1. try `PyMuPDF` / `fitz`
2. fallback to `pdfplumber`
3. if neither works, return a declared failure with warnings

This keeps the module transparent:

- no opaque OCR-first logic
- no manual crop heuristics
- no center-of-page assumptions

## Candidate components

At this stage, `candidate_components` are not hydraulic semantic nodes yet.

They are deterministic geometric groupings derived from extracted drawing objects:

- with `fitz`: one component per drawing object/path group
- with `pdfplumber`: one component per primitive object fallback

This is intentionally simple and inspectable.

## Diagnostic outputs

The default output root is:

- `step1_detection/pdf_prior/<pdf_stem>/`

Subfolders:

- `json/`
- `previews/`
- `logs/`
- `clusters/`
- `reference_graph/`

Saved files include:

- `pdf_prior_extraction.json`
- `pdf_prior_summary.json`
- page previews with vector/text overlay when available
- `extraction_log.txt`
- `diagram_candidates.json`
- `clusters/page0_clusters.json`
- `clusters/page0_clusters_overlay.png`
- `clusters/page0_cluster_rank_01.png`
- additional top-k cluster previews per page
- `reference_graph/json/primitive_cleanup.json`
- `reference_graph/json/component_candidates.json`
- `reference_graph/json/text_matches.json`
- `reference_graph/json/reference_graph.json`
- `reference_graph/plots/page0_raw_diagram.png`
- `reference_graph/plots/page0_cleaned_primitives.png`
- `reference_graph/plots/page0_component_candidates_overlay.png`
- `reference_graph/plots/page0_component_candidates_ranked.png`
- `reference_graph/plots/page0_text_matches_overlay.png`
- `reference_graph/plots/page0_reference_graph_overlay.png`
- `reference_graph/plots/page0_reference_graph_numbered.png`
- `reference_graph/plots/page0_reference_graph_labels.png`
- `reference_graph/logs/summary.txt`

## Diagram candidate extraction

The new phase does not work directly on all raw segments pairwise.
Instead it reuses the deterministic `candidate_components` already produced by the PDF parser.

Current strategy:

1. filter micro-components with too few primitives
2. expand component bounding boxes by a small scale-relative tolerance
3. build spatial connected components through a hashed grid
4. compute cluster features:
   - bbox
   - area
   - component count
   - primitive count
   - segment count
   - curve count
   - nearby word count
   - segment density
   - curve density
5. rank clusters by a transparent geometric/textual score

This keeps the method:

- deterministic
- inspectable
- independent from hardcoded page coordinates
- scalable enough for dense vector PDFs

## Reuse of existing project pieces

This first PDF step intentionally reuses current repo patterns where possible:

- `config.py` for path/feature flags
- `debug_utils.save_json()` for serializable outputs
- `main.py` as the main integration point
- existing graph pipeline left untouched unless PDF exists

## TODO for next steps
The new module already exposes named hooks for the next refactors:

- `align_image_graph_to_pdf_prior(...)`

Planned progression:

1. refine `reference_graph_builder` with axis extraction from pipe-like primitives
2. improve symbol grouping and text-to-component association
3. compare image-derived graph candidates against the reference graph
4. use the PDF prior as structural support for:
   - missing nodes
   - ambiguous codes
   - branch disambiguation
   - graph completion
