# PDF Prior Extraction

## Goal

This module introduces the first PDF-based prior workflow without breaking the current image-first pipeline.

Current scope:

- load a project PDF such as `folder_with_pdf/node9.pdf`
- inspect whether the PDF contains extractable vector information
- extract vector/text primitives in a deterministic way
- cluster geometric groups into diagram candidates
- rank the most plausible diagram regions
- save an intermediate representation and diagnostic previews
- expose a stable API for later graph-prior construction

Out of scope for this step:

- full prior graph reconstruction from the PDF
- node/edge alignment between PDF and image graph
- solver changes driven by the PDF prior

## Files

- `project_prior/pdf_graph_extraction.py`
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

- `build_reference_graph_from_pdf_prior(...)`
- `align_image_graph_to_pdf_prior(...)`

Planned progression:

1. convert PDF primitives/components into a reference graph
2. use the best diagram candidate as the base region of interest
3. compare image-derived graph candidates against that reference
4. use the PDF prior as structural support for:
   - missing nodes
   - ambiguous codes
   - branch disambiguation
   - graph completion
