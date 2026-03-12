"""
Piccolo entrypoint per lanciare il flusso PDF:
  parsing -> diagram candidates -> reference graph.

Uso:
  python run_pdf_prior.py
oppure:
  python run_pdf_prior.py folder_with_pdf/node9.pdf
"""

from pathlib import Path
import sys

from config import PDF_DIR
from project_prior import run_pdf_prior_extraction


def main():
    pdf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (PDF_DIR / "node9.pdf")
    run_pdf_prior_extraction(pdf_path)


if __name__ == "__main__":
    main()
