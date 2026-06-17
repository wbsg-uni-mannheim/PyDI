#!/bin/bash
#SBATCH --job-name=validate-level
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Validate ONE variant level (Phase 3 only) for a domain, as a standalone job
# so easy/medium/hard can run in PARALLEL instead of sequentially in a single
# job. Requires Phase 1 (measure_baseline) to have already written
#   usecases_synthetic/baselines/<domain>/baseline_metrics.json
# (run_measure_validate.sbatch fans these out automatically; this script can
# also be submitted by hand to re-run a single level).
#
# Submit (all three levels in parallel):
#   for lvl in easy medium hard; do
#       DOMAIN=papers LEVEL=$lvl WITH_LLM=true sbatch cluster/slurm/run_validate_level.sbatch
#   done
#
# Env:
#   DOMAIN    REQUIRED  {companies,games,music,products,papers}
#   LEVEL     REQUIRED  {easy,medium,hard,baseline}
#   WITH_LLM  false     true -> pass --with-llm
#   STAGES    <all>     e.g. "sm,norm,fusion" to skip EM
#   PYDI_VENV pydi-dev  path to the uv-managed venv
#
# Each level writes its own usecases_synthetic/validation/<domain>/<level>/ dir
# (no output races). The shared fusion_committee_selection.json is written once
# by Phase 1 and only READ here, and the LLM prompt caches are content-addressed
# (concurrent-safe + shared), so parallel level jobs do not collide.

mkdir -p logs

# ---------------------------------------------------------------------------
# Ensure we're at the repo root regardless of where sbatch was launched from.
# ---------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

# ---------------------------------------------------------------------------
# Activate the uv-managed pydi-dev venv.
# ---------------------------------------------------------------------------
PYDI_VENV="${PYDI_VENV:-pydi-dev}"
if [ ! -f "${PYDI_VENV}/bin/activate" ]; then
    echo "pydi-dev venv not found at ${PYDI_VENV}/bin/activate" >&2
    echo "Bootstrap with:  uv pip install -e \".[dev]\" --python ${PYDI_VENV}/bin/python" >&2
    exit 1
fi
set +u
# shellcheck source=/dev/null
source "${PYDI_VENV}/bin/activate"
set -u

# ---------------------------------------------------------------------------
# Required + default args
# ---------------------------------------------------------------------------
: "${DOMAIN:?DOMAIN must be set, e.g. DOMAIN=papers}"
: "${LEVEL:?LEVEL must be set, e.g. LEVEL=easy}"
WITH_LLM="${WITH_LLM:-false}"

LLM_FLAG=()
WITH_LLM_LC="$(printf '%s' "${WITH_LLM}" | tr '[:upper:]' '[:lower:]')"
case "${WITH_LLM_LC}" in
    true|1|yes|on) LLM_FLAG=(--with-llm) ;;
    false|0|no|off|"") : ;;
    *) echo "Unrecognized WITH_LLM value: ${WITH_LLM}" >&2; exit 3 ;;
esac

STAGES_FLAG=()
if [ -n "${STAGES:-}" ]; then
    STAGES_FLAG=(--stages "${STAGES}")
fi

# ---------------------------------------------------------------------------
# Provenance banner
# ---------------------------------------------------------------------------
echo "==== validate-level ===="
echo "job_id      : ${SLURM_JOB_ID:-<no slurm>}"
echo "node        : $(hostname)"
echo "cwd         : $(pwd)"
echo "venv        : ${PYDI_VENV}"
echo "python      : $(which python)  ($(python --version 2>&1))"
echo "domain      : ${DOMAIN}"
echo "level       : ${LEVEL}"
echo "with_llm    : ${WITH_LLM}"
echo "stages      : ${STAGES:-<all>}"
nvidia-smi || echo "(no nvidia-smi available)"
echo "========================"

echo "==== validate_variant domain=${DOMAIN} level=${LEVEL} ===="
python usecases_synthetic/scripts/validate_variant.py \
    --domain "${DOMAIN}" \
    --level "${LEVEL}" \
    "${LLM_FLAG[@]}" \
    "${STAGES_FLAG[@]}"

echo "==== validate-level DONE for ${DOMAIN}/${LEVEL} ===="
echo "validation : usecases_synthetic/validation/${DOMAIN}/${LEVEL}/"
