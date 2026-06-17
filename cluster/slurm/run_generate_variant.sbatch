#!/bin/bash
#SBATCH --job-name=generate-variant
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Generate the S1 augmented variants (all knobs, in canonical order, then
# package) for ONE domain on a Slurm cluster. CPU-only: K2 niche embeddings are
# read from usecases_synthetic/cache/knob_02_embeddings/ and the SentenceTransformer
# is only instantiated on a cache miss (and falls back to CPU), so no GPU is
# requested. LLM operators (K1 paraphrase, K2 interpolation / non-corner refill)
# read their per-cell caches under usecases_synthetic/cache/knob_0{1,2}_* and only
# hit the API on a cache miss.
#
# Submit from the repo root (one job per domain):
#
#   DOMAIN=papers sbatch cluster/slurm/run_generate_variant.sbatch
#   DOMAIN=papers LEVEL=easy sbatch cluster/slurm/run_generate_variant.sbatch
#
# Optional env overrides (before sbatch, or via --export=ALL,DOMAIN=...):
#   DOMAIN=papers                    # REQUIRED: {companies,games,music,products,papers}
#   LEVEL=all                        # easy|medium|hard|all (default all -> + cross-level monotonicity)
#   SEED=                            # master RNG seed (default: domain config master_seed)
#   PROTECTION_SOURCE=silver         # gold|silver (K1/K6 closeness target universe; default
#                                    # silver per project policy — ALWAYS silver. Requires the
#                                    # per-domain usecases_synthetic/baselines/<domain>/fusion_silver_standard.*)
#   SKIP_MONOTONICITY=false          # true -> pass --skip-monotonicity (only meaningful with LEVEL=all)
#   PYDI_VENV=pydi-dev               # path to the uv-managed venv
#
# Output: usecases/<domain>-augmented/{easy,medium,hard}/ (final packaged variant),
# including the regenerated EM gold
#   input/entitymatching/<pair>_{train,val,test}_{baseline_pruned,corner_filled}.csv

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
# `activate` references PS1 / VIRTUAL_ENV_PROMPT under set -u; unset is safer.
set +u
# shellcheck source=/dev/null
source "${PYDI_VENV}/bin/activate"
set -u

# ---------------------------------------------------------------------------
# Required + default args
# ---------------------------------------------------------------------------
: "${DOMAIN:?DOMAIN must be set, e.g. DOMAIN=papers}"
LEVEL="${LEVEL:-all}"
PROTECTION_SOURCE="${PROTECTION_SOURCE:-silver}"

SEED_FLAG=()
if [ -n "${SEED:-}" ]; then
    SEED_FLAG=(--seed "${SEED}")
fi

SKIP_MONO_FLAG=()
SKIP_MONO_LC="$(printf '%s' "${SKIP_MONOTONICITY:-false}" | tr '[:upper:]' '[:lower:]')"
case "${SKIP_MONO_LC}" in
    true|1|yes|on) SKIP_MONO_FLAG=(--skip-monotonicity) ;;
    false|0|no|off|"") : ;;
    *) echo "Unrecognized SKIP_MONOTONICITY value: ${SKIP_MONOTONICITY}" >&2; exit 3 ;;
esac

# ---------------------------------------------------------------------------
# Provenance banner
# ---------------------------------------------------------------------------
echo "==== generate-variant ===="
echo "job_id      : ${SLURM_JOB_ID:-<no slurm>}"
echo "node        : $(hostname)"
echo "cwd         : $(pwd)"
echo "venv        : ${PYDI_VENV}"
echo "python      : $(which python)  ($(python --version 2>&1))"
echo "domain      : ${DOMAIN}"
echo "level       : ${LEVEL}"
echo "seed        : ${SEED:-<domain master_seed>}"
echo "protection  : ${PROTECTION_SOURCE}"
echo "=========================="

python usecases_synthetic/scripts/generate_variant.py \
    --domain "${DOMAIN}" \
    --level "${LEVEL}" \
    --protection-source "${PROTECTION_SOURCE}" \
    "${SEED_FLAG[@]}" \
    "${SKIP_MONO_FLAG[@]}"

echo "==== generate-variant DONE for ${DOMAIN} (level=${LEVEL}) ===="
echo "output: usecases/${DOMAIN}-augmented/"
