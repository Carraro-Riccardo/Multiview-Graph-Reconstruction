"""
Utilities per prior strutturali esterni al flusso immagini.
"""

from .pdf_graph_extraction import (
    DiagramCandidate,
    DiagramCandidateExtractionResult,
    PdfPriorExtractionResult,
    extract_pdf_prior,
    extract_diagram_candidates,
    run_pdf_prior_extraction,
)
from .llm_reference_graph import (
    ReferenceGraphResult,
    ReferenceGraphComponent,
    ReferenceGraphConnection,
    generate_reference_graph_from_diagram,
    build_reference_graph_from_pdf_prior,
)
from .reference_graph_visualization import (
    save_reference_graph_overlay,
    save_reference_graph_schema_view,
)

__all__ = [
    "DiagramCandidate",
    "DiagramCandidateExtractionResult",
    "PdfPriorExtractionResult",
    "ReferenceGraphResult",
    "ReferenceGraphComponent",
    "ReferenceGraphConnection",
    "extract_pdf_prior",
    "extract_diagram_candidates",
    "generate_reference_graph_from_diagram",
    "build_reference_graph_from_pdf_prior",
    "save_reference_graph_overlay",
    "save_reference_graph_schema_view",
    "run_pdf_prior_extraction",
]
