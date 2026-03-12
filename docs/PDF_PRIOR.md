# PDF Prior Flow

## Goal

The active PDF-side flow is now:

1. parse the vector PDF
2. extract and rank diagram candidates
3. select the best cluster
4. render that cluster as an image
5. call a multimodal LLM provider
6. validate the returned JSON
7. save a reusable `reference_graph.json` plus visual/debug artifacts

This keeps the deterministic part where it helps:

- PDF parsing
- primitive extraction
- diagram candidate extraction
- cluster ranking
- debug rendering

And moves the semantic diagram understanding to the multimodal API step.

## Main files

- `project_prior/pdf_graph_extraction.py`
- `project_prior/llm_reference_graph.py`
- `project_prior/__init__.py`
- `run_pdf_prior.py`
- `run_pdf_to_reference_graph.py`
- `main.py`

## Provider configuration

Create a local `.env` file in the repo root.

Example:

```env
PDF_REFERENCE_GRAPH_PROVIDER=gemini
OPENAI_API_KEY=...
GEMINI_API_KEY=...
PDF_REFERENCE_GRAPH_OPENAI_MODEL=gpt-5.1
PDF_REFERENCE_GRAPH_GEMINI_MODEL=gemini-2.5-flash
```

Supported providers:

- `gemini`
- `openai`

The provider is selected through:

- `PDF_REFERENCE_GRAPH_PROVIDER`

API key env names can also be overridden through config:

- `PDF_REFERENCE_GRAPH_OPENAI_API_KEY_ENV`
- `PDF_REFERENCE_GRAPH_GEMINI_API_KEY_ENV`

## Entry points

### Full PDF -> reference graph flow

```bash
python run_pdf_to_reference_graph.py --pdf folder_with_pdf/node9.pdf
```

### Existing PDF runner

```bash
python run_pdf_prior.py
```

This now runs:

- parse PDF
- extract diagram candidates
- call the LLM-based reference graph stage

## Main API

```python
extract_pdf_prior(pdf_path) -> PdfPriorExtractionResult
extract_diagram_candidates(parsed_pdf) -> DiagramCandidateExtractionResult
generate_reference_graph_from_diagram(parsed_pdf, diagram_result) -> ReferenceGraphResult
build_reference_graph_from_pdf_prior(parsed_pdf, diagram_result=None) -> ReferenceGraphResult
```

## Inputs passed to the model

Minimum mandatory input:

- best cluster image

Additional support passed when available:

- nearby vector words with PDF bboxes
- candidate regions derived from the ranked cluster
- rendered pages from `components.pdf`
- strict JSON schema requested from the model

## Output folder

Default root:

- `step1_detection/pdf_prior/<pdf_stem>/llm_reference_graph/`

Structured layout:

- `inputs/`
- `prompts/`
- `responses/`
- `graph/`
- `visuals/`
- `summaries/`

Expected artifacts:

- `inputs/page0_best_cluster.png`
- `inputs/page0_best_cluster_with_bbox.png`
- `inputs/best_candidate_words.json`
- `inputs/best_candidate_regions.json`
- `prompts/llm_prompt.txt`
- `responses/llm_request_payload.json`
- `responses/llm_raw_response.txt`
- `responses/llm_extracted_text.txt`
- `responses/llm_parsed_response.json`
- `graph/reference_graph.json`
- `graph/reference_graph_result.json`
- `visuals/reference_graph_overlay.png`
- `visuals/reference_graph_numbered.png`
- `visuals/reference_graph_labels.png`
- `visuals/reference_graph_schema.png`
- `visuals/reference_graph_schema_numbered.png`
- `summaries/llm_request_summary.txt`
- `summaries/run_summary.txt`

## Visualization-only rerun

If you already have a `reference_graph.json` and want to regenerate only the
visual artifacts without re-calling the LLM:

```bash
python run_reference_graph_visualization.py
```

or:

```bash
python run_reference_graph_visualization.py --graph step1_detection/pdf_prior/node9/llm_reference_graph/graph/reference_graph.json --image step1_detection/pdf_prior/node9/llm_reference_graph/inputs/page0_best_cluster.png
```

This is useful when the JSON is semantically correct but the overlay is not
precise enough geometrically. In that case, prefer the standalone schema views.

## Validation

The LLM response is not trusted blindly.

Current validation includes:

- unique component ids
- allowed component types only
- bbox normalization and clamping to image size
- connection endpoints must exist
- self-loops removed
- unknown types remapped to `unknown`

If parsing or validation fails:

- raw response is still saved
- a parse/validation report is saved
- the pipeline returns a readable status instead of crashing opaquely

## Notes

The previous geometry-heavy semantic graph builder is no longer the main path.
It can remain in the repo as an experimental helper, but the active PDF-side reference graph generation is LLM-first.

## Next steps

- align image-derived graph candidates to the PDF reference graph
- score observed node vs prior component
- handle project vs as-built mismatches
- guide multi-view fusion with the prior graph
