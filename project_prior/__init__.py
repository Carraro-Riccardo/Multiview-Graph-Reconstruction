"""
Utilities per prior strutturali esterni al flusso immagini.
"""

from .pdf_graph_extraction import (
    DiagramCandidate,
    DiagramCandidateExtractionResult,
    PdfPriorExtractionResult,
    extract_pdf_prior,
    extract_diagram_candidates,
    build_reference_graph_from_pdf_prior,
    align_image_graph_to_pdf_prior,
    run_pdf_prior_extraction,
)

__all__ = [
    "DiagramCandidate",
    "DiagramCandidateExtractionResult",
    "PdfPriorExtractionResult",
    "extract_pdf_prior",
    "extract_diagram_candidates",
    "build_reference_graph_from_pdf_prior",
    "align_image_graph_to_pdf_prior",
    "run_pdf_prior_extraction",
]
