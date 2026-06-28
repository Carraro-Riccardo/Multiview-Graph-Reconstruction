"""Runtime helpers to generate VGGT artefacts with the embedded Python env."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from graph_reconstruction.config import VGGT_EMBEDDED_PYTHON, VGGT_GENERATE_IF_MISSING, VGGT_MODEL_ID, VGGT_PREPROCESS_MODE


def _artifact_dir(output_dir: Path) -> Path:
    return output_dir / "vggt"


def _expected_files(output_dir: Path, image_paths: list[Path]) -> list[Path]:
    base = _artifact_dir(output_dir)
    return [base / f"{Path(image_path).stem}.npz" for image_path in image_paths]


def ensure_vggt_artifacts(output_dir: Path, image_paths: list[Path], force: bool = False) -> dict:
    """Generate per-view VGGT files if they are missing and generation is enabled."""
    output_dir = Path(output_dir)
    image_paths = [Path(p) for p in image_paths]
    artifact_dir = _artifact_dir(output_dir)
    expected = _expected_files(output_dir, image_paths)
    missing = [str(path) for path in expected if not path.exists()]

    if not force and not missing:
        return {
            "status": "already_present",
            "artifact_dir": str(artifact_dir),
            "generated": False,
            "missing_files": [],
        }

    if not VGGT_GENERATE_IF_MISSING and not force:
        return {
            "status": "generation_disabled",
            "artifact_dir": str(artifact_dir),
            "generated": False,
            "missing_files": missing,
        }

    python_exe = Path(VGGT_EMBEDDED_PYTHON)
    script_path = Path(__file__).resolve().parent / "vggt_export_cli.py"
    if not python_exe.exists():
        return {
            "status": "missing_embedded_python",
            "artifact_dir": str(artifact_dir),
            "generated": False,
            "missing_files": missing,
            "python_exe": str(python_exe),
        }

    payload = {
        "output_dir": str(artifact_dir),
        "image_paths": [str(path.resolve()) for path in image_paths],
        "model_id": VGGT_MODEL_ID,
        "preprocess_mode": VGGT_PREPROCESS_MODE,
        "force": force,
    }
    request_path = artifact_dir / "generation_request.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cmd = [str(python_exe), str(script_path), "--request", str(request_path)]
    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
    )

    summary_path = artifact_dir / "generation_summary.json"
    summary = {
        "status": "ok" if completed.returncode == 0 else "error",
        "artifact_dir": str(artifact_dir),
        "generated": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
        "missing_files_before": missing,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
