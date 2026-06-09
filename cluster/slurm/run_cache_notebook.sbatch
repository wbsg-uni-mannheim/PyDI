#!/bin/bash
#SBATCH --job-name=cache-notebook
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Execute a domain's human-baseline workflow notebook headless to cache its
# fused output (the SR silver standard), then compute the SR composite for that
# domain's BoB BASELINE run via compare_to_human_baseline.py.
#
#   DOMAIN=music BASELINE_RUN=pipelines/music/run_slurm_baseline_<jid> \
#       sbatch cluster/slurm/run_cache_notebook.sbatch
#
# Output: pipelines/<domain>/baselines/notebook_fused.csv  (one-time cache)
#         pipelines/<domain>/<baseline_run>/comparison.md   (SR panel)

mkdir -p logs
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi

PYDI_VENV="${PYDI_VENV:-pydi-dev}"
set +u
# shellcheck source=/dev/null
source "${PYDI_VENV}/bin/activate"
set -u

# Notebooks call ChatOpenAI; export OPENAI_API_KEY from .env (they also
# load_dotenv themselves, this is belt-and-suspenders).
if [ -f .env ]; then
    set -a
    # shellcheck source=/dev/null
    . ./.env
    set +a
fi

: "${DOMAIN:?DOMAIN must be set, e.g. DOMAIN=music}"

case "${DOMAIN}" in
    products) NB="usecases/products/products_workflow_minimal.ipynb" ;;
    papers)   NB="usecases/papers/papers_workflow_minimal.ipynb" ;;
    *)        NB="usecases/${DOMAIN}/${DOMAIN}_workflow.ipynb" ;;
esac
CACHE="pipelines/${DOMAIN}/baselines"

echo "==== cache-notebook ===="
echo "job_id  : ${SLURM_JOB_ID:-<none>}"
echo "node    : $(hostname)"
echo "domain  : ${DOMAIN}"
echo "notebook: ${NB}"
echo "cache   : ${CACHE}"
echo "baseline: ${BASELINE_RUN:-<none — SR comparison skipped>}"
nvidia-smi || echo "(no nvidia-smi)"
echo "========================"

# Register the venv as an ipykernel so nbclient runs the notebook with PyDI
# importable. Guard against the rmtree+recreate race when several cache jobs
# start at once (concurrent `ipykernel install` to the shared user kernel dir
# clobber each other -> FileNotFoundError); only install when absent.
KSPEC="${HOME}/.local/share/jupyter/kernels/pydi-dev/kernel.json"
if [ ! -f "${KSPEC}" ]; then
    python -m ipykernel install --user --name pydi-dev --display-name pydi-dev || true
fi
[ -f "${KSPEC}" ] || { echo "pydi-dev kernel missing and install failed" >&2; exit 1; }

echo "==== [1/2] execute notebook -> ${CACHE}/notebook_fused.csv ===="
python pipelines/scripts/cache_notebook.py \
    --notebook "${NB}" \
    --cache-dir "${CACHE}" \
    --kernel pydi-dev

if [ -n "${BASELINE_RUN:-}" ]; then
    echo "==== [2/2] SR comparison: compare_to_human_baseline --pipeline-run ${BASELINE_RUN} ===="
    python pipelines/scripts/compare_to_human_baseline.py \
        --domain "${DOMAIN}" \
        --pipeline-run "${BASELINE_RUN}"
fi

echo "==== cache-notebook DONE for ${DOMAIN} ===="
echo "SR silver : ${CACHE}/notebook_fused.csv"
[ -n "${BASELINE_RUN:-}" ] && echo "SR panel  : ${BASELINE_RUN}/comparison.md"
