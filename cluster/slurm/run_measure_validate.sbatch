#!/bin/bash
#SBATCH --job-name=measure-validate
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

# Run the synthetic-pipeline Phase-1 baseline measurement + Phase-3 per-variant
# validation for ONE domain on a Slurm cluster.
#
# Submit from the repo root (one job per domain):
#
#   DOMAIN=companies sbatch cluster/slurm/run_measure_validate.sbatch
#   for d in companies games music products papers; do
#       DOMAIN=$d sbatch cluster/slurm/run_measure_validate.sbatch
#   done
#
# Phase 1 (measure_baseline) writes
#   usecases_synthetic/baselines/<domain>/baseline_metrics.json
# The Normalization stage is scored against the JSON-Schema + x-pydi-consistency
# constraints in usecases/<domain>/input/schemamatching/target_schema.json
# (NORM_SURFACE=schema_constraints, the default). The surface is recorded in
# meta["scoring_surface"] so Phase 3 inherits it.
#
# Phase 3 (validate_variant) scores each variant level (easy/medium/hard)
# against that baseline and writes
#   usecases_synthetic/validation/<domain>/<level>/{metrics.json,level_report.md,...}
#
# Optional env overrides (before sbatch, or via --export=ALL,DOMAIN=...):
#   DOMAIN=companies                 # REQUIRED: {companies,games,music,products,papers}
#   LEVELS="easy medium hard"        # variant levels to validate (default all three)
#   STAGES=                          # e.g. "sm,norm,fusion" to skip EM; default = all stages
#   WITH_LLM=false                   # true -> pass --with-llm to both scripts
#   NORM_SURFACE=schema_constraints  # measure_baseline norm scoring surface (default)
#   PYDI_VENV=pydi-dev               # path to the uv-managed venv
#
# NOTE on EM: the em_blocking / em_matching committees reference Ditto / SC-block
# checkpoints under usecases_synthetic/cache/ditto_checkpoints/<domain>/best
# (gitignored). When a checkpoint is missing the committee "loudly disables" that
# member and the run still completes with the remaining members. Train first with
# cluster/slurm/run_em_training.sbatch (writing into the cache path the committee
# YAML points at) for full EM, or pass STAGES="sm,norm,fusion" to skip EM here.

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
: "${DOMAIN:?DOMAIN must be set, e.g. DOMAIN=companies}"
LEVELS="${LEVELS:-easy medium hard}"
WITH_LLM="${WITH_LLM:-false}"
NORM_SURFACE="${NORM_SURFACE:-schema_constraints}"

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
echo "==== measure-validate ===="
echo "job_id      : ${SLURM_JOB_ID:-<no slurm>}"
echo "node        : $(hostname)"
echo "cwd         : $(pwd)"
echo "venv        : ${PYDI_VENV}"
echo "python      : $(which python)  ($(python --version 2>&1))"
echo "domain      : ${DOMAIN}"
echo "levels      : ${LEVELS}"
echo "stages      : ${STAGES:-<all>}"
echo "with_llm    : ${WITH_LLM}"
echo "norm_surface: ${NORM_SURFACE}"
nvidia-smi || echo "(no nvidia-smi available)"
echo "=========================="

# ---------------------------------------------------------------------------
# Phase 1 — baseline measurement (norm scored vs target_schema constraints)
# ---------------------------------------------------------------------------
echo "==== [1/2] measure_baseline domain=${DOMAIN} (norm_surface=${NORM_SURFACE}) ===="
python usecases_synthetic/scripts/measure_baseline.py \
    --domain "${DOMAIN}" \
    --norm-scoring-surface "${NORM_SURFACE}" \
    "${LLM_FLAG[@]}" \
    "${STAGES_FLAG[@]}"

# ---------------------------------------------------------------------------
# Phase 3 — per-variant validation (inherits the baseline scoring surface)
# ---------------------------------------------------------------------------
for level in ${LEVELS}; do
    echo "==== [2/2] validate_variant domain=${DOMAIN} level=${level} ===="
    python usecases_synthetic/scripts/validate_variant.py \
        --domain "${DOMAIN}" \
        --level "${level}" \
        "${LLM_FLAG[@]}" \
        "${STAGES_FLAG[@]}"
done

echo "==== measure-validate DONE for ${DOMAIN} ===="
echo "baseline   : usecases_synthetic/baselines/${DOMAIN}/baseline_metrics.json"
echo "validation : usecases_synthetic/validation/${DOMAIN}/{${LEVELS// /,}}/"
