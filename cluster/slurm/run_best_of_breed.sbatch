#!/bin/bash
#SBATCH --job-name=bob-pipeline
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

# Run the best-of-breed data-integration pipeline on a Slurm cluster.
#
# Submit from the login node, from the repo root:
#
#   DOMAIN=products sbatch cluster/slurm/run_best_of_breed.sbatch
#   DOMAIN=music    sbatch cluster/slurm/run_best_of_breed.sbatch
#   DOMAIN=games    sbatch cluster/slurm/run_best_of_breed.sbatch
#   DOMAIN=companies sbatch cluster/slurm/run_best_of_breed.sbatch
#
# Optional env overrides before sbatch (also: --export=ALL,DOMAIN=...):
#   export DOMAIN=products                    # REQUIRED, one of {products,music,games,companies,papers}
#   export VARIANT=baseline                    # baseline (default) | easy | medium | hard
#   export MODE=replay                         # replay (default) | sweep
#   export OUT_DIR=pipelines/$DOMAIN/run_slurm_${VARIANT}_$SLURM_JOB_ID/
#   export DITTO_CHECKPOINT_OVERRIDE=pipelines/$DOMAIN/checkpoints/em_matching/ditto/runs/best
#   export SC_BLOCK_CHECKPOINT_OVERRIDE=pipelines/$DOMAIN/checkpoints/em_blocking/sc_block/best
#   export FUSION_MEMBERS=voting_only,prefer_higher_trust_only
#   export LLM_SM=true                         # default true  (keep LLM in schema matching)
#   export LLM_EM=false                        # default false (drop LLM from entity matching)
#   export LLM_FUSION=false                    # default false (drop LLM from fusion)
#   export PYDI_VENV=pydi-dev                  # path to the uv-managed venv (default pydi-dev)
#   export EXTRA_ARGS="--log-level DEBUG"      # appended to the python invocation
#
# Example: submit all five domains in one go (each as its own job):
#   for d in products music games companies papers; do
#       DOMAIN=$d sbatch cluster/slurm/run_best_of_breed.sbatch
#   done
#
# Example: submit a domain x variant grid (use the helper):
#   bash cluster/slurm/submit_all_variants.sh

mkdir -p logs

# ---------------------------------------------------------------------------
# Ensure we're at the repo root regardless of where sbatch was launched from.
# ---------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

# ---------------------------------------------------------------------------
# Activate the uv-managed pydi-dev venv (drop-in replacement for the conda
# activation in the example). ⁠ set -u ⁠ is fine: activate handles unset vars.
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
: "${DOMAIN:?DOMAIN must be set, e.g. DOMAIN=products}"
VARIANT="${VARIANT:-baseline}"
MODE="${MODE:-replay}"
OUT_DIR="${OUT_DIR:-pipelines/${DOMAIN}/run_slurm_${VARIANT}_${SLURM_JOB_ID}/}"
CONFIG="pipelines/configs/${DOMAIN}.yaml"

if [ ! -f "${CONFIG}" ]; then
    echo "Missing pipeline config: ${CONFIG}" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Pipeline-isolated EM checkpoints (no committee-cache reuse).
#
# ditto_plm + sc_block are LOUDLY disabled by the pipeline when their YAML
# default checkpoint_path sits under usecases_synthetic/cache/ and no override
# is supplied. To honour "do not silently drop any breed members", we derive
# the BoB-isolated checkpoints from DOMAIN + VARIANT (trained by
# cluster/slurm/run_bob_em_train.sbatch into pipelines/<domain>/checkpoints/)
# and ASSERT they exist before running — a missing checkpoint fails the job
# loudly rather than letting the member be dropped. Explicit overrides set in
# the environment win; SKIP_EM_CHECKPOINT_CHECK=1 bypasses the assertion.
# ---------------------------------------------------------------------------
case "${VARIANT}" in
    baseline)          LEVELKEY="baseline" ;;
    easy|medium|hard)  LEVELKEY="variant_${VARIANT}" ;;
    *) echo "Unrecognized VARIANT: ${VARIANT}" >&2; exit 3 ;;
esac
DITTO_CHECKPOINT_OVERRIDE="${DITTO_CHECKPOINT_OVERRIDE:-pipelines/${DOMAIN}/checkpoints/em_matching/ditto/${LEVELKEY}/best}"
SC_BLOCK_CHECKPOINT_OVERRIDE="${SC_BLOCK_CHECKPOINT_OVERRIDE:-pipelines/${DOMAIN}/checkpoints/em_blocking/sc_block/${LEVELKEY}/best}"

if [ -z "${SKIP_EM_CHECKPOINT_CHECK:-}" ]; then
    require_ckpt () {
        # require_ckpt <member-label> <best-dir> <required-file>
        local label="$1" dir="$2" must="$3"
        if [ ! -e "${dir}/${must}" ]; then
            echo "FATAL: ${label} checkpoint missing: ${dir}/${must}" >&2
            echo "Refusing to run: a neural EM breed member would be dropped " \
                 "(no silent dropping directive)." >&2
            echo "Train the isolated checkpoints first:" >&2
            echo "  DOMAIN=${DOMAIN} sbatch cluster/slurm/run_bob_em_train.sbatch" >&2
            echo "(or set SKIP_EM_CHECKPOINT_CHECK=1 to bypass — drops the member)." >&2
            exit 4
        fi
    }
    require_ckpt "ditto_plm" "${DITTO_CHECKPOINT_OVERRIDE}" "model.pt"
    require_ckpt "sc_block"  "${SC_BLOCK_CHECKPOINT_OVERRIDE}" "model.safetensors"
fi

# ---------------------------------------------------------------------------
# Per-stage LLM toggles (defaults reflect the 2026-06-01 directive:
# SM on, EM off, Fusion off). Env vars must be literal "true" / "false".
# ---------------------------------------------------------------------------
llm_flag () {
    # llm_flag <env_value> <enable_flag> <disable_flag>
    local v="${1:-}" on="$2" off="$3"
    case "${v,,}" in
        true|1|yes|on) echo "${on}" ;;
        false|0|no|off) echo "${off}" ;;
        "") echo "" ;;
        *) echo "Unrecognized boolean value: ${v}" >&2; exit 3 ;;
    esac
}
SM_FLAG="$(llm_flag "${LLM_SM:-true}" --llm-sm --no-llm-sm)"
EM_FLAG="$(llm_flag "${LLM_EM:-false}" --llm-em --no-llm-em)"
FUSION_FLAG="$(llm_flag "${LLM_FUSION:-false}" --llm-fusion --no-llm-fusion)"

# ---------------------------------------------------------------------------
# Optional pipeline-isolated checkpoint overrides + fusion-members allowlist
# ---------------------------------------------------------------------------
EXTRA_FLAGS=()
if [ -n "${DITTO_CHECKPOINT_OVERRIDE:-}" ]; then
    EXTRA_FLAGS+=(--ditto-checkpoint-override "${DITTO_CHECKPOINT_OVERRIDE}")
fi
if [ -n "${SC_BLOCK_CHECKPOINT_OVERRIDE:-}" ]; then
    EXTRA_FLAGS+=(--sc-block-checkpoint-override "${SC_BLOCK_CHECKPOINT_OVERRIDE}")
fi
if [ -n "${FUSION_MEMBERS:-}" ]; then
    EXTRA_FLAGS+=(--fusion-members "${FUSION_MEMBERS}")
fi

# ---------------------------------------------------------------------------
# Provenance banner
# ---------------------------------------------------------------------------
echo "==== bob-pipeline ===="
echo "job_id   : ${SLURM_JOB_ID:-<no slurm>}"
echo "node     : $(hostname)"
echo "cwd      : $(pwd)"
echo "venv     : ${PYDI_VENV}"
echo "python   : $(which python)  ($(python --version 2>&1))"
echo "domain   : ${DOMAIN}"
echo "variant  : ${VARIANT}"
echo "config   : ${CONFIG}"
echo "mode     : ${MODE}"
echo "out_dir  : ${OUT_DIR}"
echo "llm_sm   : ${LLM_SM:-true}   (-> ${SM_FLAG})"
echo "llm_em   : ${LLM_EM:-false}  (-> ${EM_FLAG})"
echo "llm_fusion: ${LLM_FUSION:-false} (-> ${FUSION_FLAG})"
echo "extra    : ${EXTRA_FLAGS[*]:-} ${EXTRA_ARGS:-}"
nvidia-smi || echo "(no nvidia-smi available)"
echo "===================="

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "${OUT_DIR}"

# ⁠ ${EXTRA_ARGS} ⁠ is intentionally word-split so callers can pass multi-arg
# strings like  ⁠ EXTRA_ARGS="--log-level DEBUG" ⁠.
# shellcheck disable=SC2086
python pipelines/scripts/run_best_of_breed.py \
    --config "${CONFIG}" \
    --variant "${VARIANT}" \
    --out "${OUT_DIR}" \
    --mode "${MODE}" \
    ${SM_FLAG} ${EM_FLAG} ${FUSION_FLAG} \
    "${EXTRA_FLAGS[@]}" \
    ${EXTRA_ARGS:-}

echo "==== bob-pipeline DONE ===="
echo "Run artifacts: ${OUT_DIR}"
echo
# Notebook paths: products uses the *_minimal.ipynb variant; others use <domain>_workflow.ipynb.
case "${DOMAIN}" in
    products) NB_PATH="usecases/products/products_workflow_minimal.ipynb" ;;
    *)        NB_PATH="usecases/${DOMAIN}/${DOMAIN}_workflow.ipynb" ;;
esac
echo "To produce the comparison report against the human-baseline notebook:"
echo "  # (1) one-time per domain — cache the notebook output:"
echo "  python pipelines/scripts/compare_to_human_baseline.py \\"
echo "      --domain ${DOMAIN} --cache-from-notebook ${NB_PATH}"
echo "  # (2) write comparison.md (per-stage table + apples-to-apples fusion eval + e2e panel):"
echo "  python pipelines/scripts/compare_to_human_baseline.py \\"
echo "      --domain ${DOMAIN} --pipeline-run ${OUT_DIR}"
