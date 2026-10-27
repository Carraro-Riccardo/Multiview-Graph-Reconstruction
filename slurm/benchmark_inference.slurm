#!/bin/bash
#SBATCH --job-name=ETRA_inference_bench
#SBATCH --error=/home/rcarraro/projects/Multiview-Graph-Reconstruction/slurm/logs/benchmarks_errors_%j.err
#SBATCH --output=/home/rcarraro/projects/Multiview-Graph-Reconstruction/slurm/logs/benchmarks_output_%j.out
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --gres=mps:30
#SBATCH --account=thesis

set -euo pipefail

PROJECT_ROOT="/home/rcarraro/projects/Multiview-Graph-Reconstruction"
BENCHMARK_DIR="${PROJECT_ROOT}/output/benchmarks/${SLURM_JOB_ID}"
mkdir -p "${BENCHMARK_DIR}"

# Input paths used by the existing university-cluster job.
export IMAGES_DIR=/home/rcarraro/datasets/ds_operativo/images/val
export LABELS_EXT_DIR=/home/rcarraro/datasets/ds_operativo/labels_extended

# YOLO_MODEL_PATH must either be present in the cluster .env or exported when
# submitting, for example:
#   sbatch --export=ALL,YOLO_MODEL_PATH=/path/to/model.pt slurm/benchmark_inference.slurm

if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "ERRORE: progetto non trovato: ${PROJECT_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${PROJECT_ROOT}/tools/benchmark/benchmark_inference.py" ]]; then
    echo "ERRORE: benchmark_inference.py non trovato" >&2
    exit 1
fi
if [[ ! -d "${IMAGES_DIR}" ]]; then
    echo "ERRORE: IMAGES_DIR non trovata: ${IMAGES_DIR}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

echo "========================================================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "Started:       $(date --iso-8601=seconds)"
echo "Project root:  ${PROJECT_ROOT}"
echo "Python:        $(which python)"
echo "Benchmark dir: ${BENCHMARK_DIR}"
echo "MPS percent:   ${CUDA_MPS_ACTIVE_THREAD_PERCENTAGE-<scheduler managed>}"
echo "========================================================================"

python --version
python -m pip --version
nvidia-smi

python - <<'PY'
import importlib
import sys
import torch

required = ["cv2", "numpy", "PIL", "scipy", "ultralytics", "shurikodecontext", "vggt"]
missing = []
for name in required:
    try:
        importlib.import_module(name)
        print(f"dependency {name}: OK")
    except Exception as exc:
        missing.append(f"{name}: {exc}")

print("Python executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
if missing:
    raise RuntimeError("Dipendenze mancanti: " + " | ".join(missing))
if not torch.cuda.is_available():
    raise RuntimeError("CUDA non disponibile nel job")
PY

# With no node IDs the benchmark automatically chooses node9 (seven views),
# the largest non-skipped VGGT case in the current dataset. WORK_DIR is unique,
# so both YOLO and VGGT run without inference artefact caches.
python tools/benchmark/benchmark_inference.py \
    --require-cuda \
    --sample-interval 0.25 \
    --output-dir "${BENCHMARK_DIR}"

echo "========================================================================"
echo "Finished: $(date --iso-8601=seconds)"
echo "Machine-readable report: ${BENCHMARK_DIR}/benchmark_report.json"
echo "Raw samples:              ${BENCHMARK_DIR}/benchmark_samples.csv"
echo
echo "After job completion, verify Slurm's independent RAM accounting with:"
echo "sacct -j ${SLURM_JOB_ID} --format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocTRES%80"
echo "========================================================================"
