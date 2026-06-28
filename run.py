"""Entry point — run the multi-view hydraulic graph reconstruction pipeline.

Usage:
    python run.py                  # process all nodes in IMAGES_DIR
    python run.py 2 7 9            # process specific node IDs
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_reconstruction.pipeline.orchestrator import run_multiview_all, run_multiview_pipeline
from graph_reconstruction.config import IMAGES_DIR
from graph_reconstruction.shared.detection import discover_nodes


def main() -> None:
    if len(sys.argv) > 1:
        node_ids = sys.argv[1:]
        all_nodes = discover_nodes(IMAGES_DIR)
        for nid in node_ids:
            if nid not in all_nodes:
                print(f"[run] node {nid} not found in {IMAGES_DIR}")
                continue
            run_multiview_pipeline(nid, all_nodes[nid])
    else:
        run_multiview_all()


if __name__ == "__main__":
    main()
