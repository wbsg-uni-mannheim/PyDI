#!/bin/bash
#SBATCH --job-name=em-per-pair
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

# Recompute the winning matcher's PER-PAIR EM test F1 for a baseline
# best-of-breed run, by re-running ONLY the EM stage (blocking +
# matching committees) against the effective committee YAMLs the
# original run persisted. Writes <RUN_DIR>/em_per_pair_test_f1.json.
#
# The recomputed cross-pair AVERAGE must match the persisted
# stage_4_em_matching_selection.json test_score (deterministic
# matchers) — the driver asserts this and flags a mismatch.
#
# Submit from the repo root, one job per domain:
#
#   for d in games companies music products papers; do
#       DOMAIN=$d sbatch cluster/slurm/run_em_per_pair.sbatch
#   done
#
# Optional env overrides (before sbatch, or via --export):
#   DOMAIN=papers                 # REQUIRED: {games,companies,music,products,papers}
#   RUN_DIR=pipelines/<d>/run_... # default: latest run_slurm_baseline_* for DOMAIN
#   PYDI_VENV=pydi-dev            # path to the uv-managed venv

mkdir -p logs

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

PYDI_VENV="${PYDI_VENV:-pydi-dev}"
if [ ! -f "${PYDI_VENV}/bin/activate" ]; then
    echo "pydi-dev venv not found at ${PYDI_VENV}/bin/activate" >&2
    exit 1
fi
set +u
# shellcheck source=/dev/null
source "${PYDI_VENV}/bin/activate"
set -u

: "${DOMAIN:?DOMAIN must be set, e.g. DOMAIN=papers}"

# Resolve the baseline run dir (latest by job-id suffix) unless given.
if [ -z "${RUN_DIR:-}" ]; then
    RUN_DIR="$(ls -d pipelines/${DOMAIN}/run_slurm_baseline_* 2>/dev/null | sort | tail -1)"
fi
if [ -z "${RUN_DIR:-}" ] || [ ! -d "${RUN_DIR}" ]; then
    echo "FATAL: no baseline run dir for ${DOMAIN} (RUN_DIR='${RUN_DIR:-}')" >&2
    exit 2
fi

echo "==== em-per-pair ===="
echo "job_id   : ${SLURM_JOB_ID:-<no slurm>}"
echo "node     : $(hostname)"
echo "cwd      : $(pwd)"
echo "venv     : ${PYDI_VENV}"
echo "python   : $(which python)  ($(python --version 2>&1))"
echo "domain   : ${DOMAIN}"
echo "run_dir  : ${RUN_DIR}"
nvidia-smi || echo "(no nvidia-smi available)"
echo "===================="

OUT_FLAG=()
if [ -n "${OUT_JSON:-}" ]; then
    OUT_FLAG=(--out "${OUT_JSON}")
fi

python pipelines/scripts/recompute_em_per_pair.py \
    --domain "${DOMAIN}" \
    --run-dir "${RUN_DIR}" \
    "${OUT_FLAG[@]}"

echo "==== em-per-pair DONE for ${DOMAIN} ===="
echo "Result: ${RUN_DIR}/em_per_pair_test_f1.json"
