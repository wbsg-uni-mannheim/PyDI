#!/bin/bash
#SBATCH --job-name=bob-besttime
#SBATCH --partition=gpu-vram-48gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Time the final best pipeline INCLUDING training of its winning neural members
# (Option B). The best-of-breed pipeline only LOADS ditto/sc_block checkpoints
# (inference-only), so a fair "time the best pipeline" must also TRAIN the
# winning neural members. This:
#   1. builds single-member committees from the reference run's winners,
#   2. TRAINS (timed) the winning neural members fresh into a temp checkpoint
#      dir -- ditto when the matcher winner is ditto_plm, sc_block when the
#      blocker winner is sc_block (magellan/bm25/etc. train inline during the
#      run, so no separate step),
#   3. RUNS (timed) the best pipeline against those fresh checkpoints.
# Reports train_seconds + infer_seconds + total_seconds to a SEPARATE
# pipelines/<domain>/run_besttime_*/best_pipeline_time.txt.
#
#   DOMAIN=games REF_RUN=pipelines/games/run_slurm_baseline_255979 \
#     sbatch cluster/slurm/run_best_pipeline_timing.sbatch

set -euo pipefail
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then cd "${SLURM_SUBMIT_DIR}"; fi
set +u
# shellcheck source=/dev/null
source pydi-dev/bin/activate
set -u

: "${DOMAIN:?DOMAIN must be set, e.g. DOMAIN=games}"
: "${REF_RUN:?REF_RUN must be set (a run dir with per_stage_summary.csv)}"
JOB="${SLURM_JOB_ID:-local}"
WORK="pipelines/${DOMAIN}/best_only_${JOB}"
OUT="pipelines/${DOMAIN}/run_besttime_${JOB}"
TMPCKPT="pipelines/${DOMAIN}/besttime_ckpt_${JOB}"
mkdir -p "${OUT}" "${TMPCKPT}" logs

BLOCK_WINNER="$(awk -F, '$1=="em_blocking"{print $2}' "${REF_RUN}/per_stage_summary.csv")"
MATCH_WINNER="$(awk -F, '$1=="em_matching"{print $2}' "${REF_RUN}/per_stage_summary.csv")"
echo "blocking winner=${BLOCK_WINNER} | matching winner=${MATCH_WINNER}"

echo "==== build best-only committees ===="
python pipelines/scripts/build_best_only_committees.py \
    --domain "${DOMAIN}" --ref-run "${REF_RUN}" \
    --out-committee-dir "${WORK}/committees" --out-config "${WORK}/${DOMAIN}.yaml"

# ---------------------------------------------------------------------------
# TRAIN phase (timed): fresh training of the winning neural members only.
# ---------------------------------------------------------------------------
DITTO_DIR="${TMPCKPT}/em_matching/ditto/baseline"
SCB_DIR="${TMPCKPT}/em_blocking/sc_block/baseline"
EXTRA=()
TRAIN_START=$(date +%s.%N)

if [ "${MATCH_WINNER}" = "ditto_plm" ]; then
    case "${DOMAIN}" in
        companies) PREP_SUB="companies_pydi" ;;
        games)     PREP_SUB="games_pydi_raw" ;;
        *)         PREP_SUB="${DOMAIN}" ;;
    esac
    PREP_DIR="usecases_synthetic/output/ditto/${PREP_SUB}"
    if [ ! -f "${PREP_DIR}/train.json.gz" ]; then
        echo "==== [ditto prep] ${PREP_DIR} missing -> ditto/_prep_${DOMAIN}.py ===="
        python "usecases_synthetic/scripts/ditto/_prep_${DOMAIN}.py"
    fi
    TESTFLAG=()
    [ -f "${PREP_DIR}/test.json.gz" ] && TESTFLAG=(--test-json-gz "${PREP_DIR}/test.json.gz")
    mkdir -p "${DITTO_DIR}"
    echo "==== [TRAIN ditto] domain=${DOMAIN} -> ${DITTO_DIR} ===="
    python usecases_synthetic/scripts/ditto/train.py \
        --domain "${DOMAIN}" \
        --train-json-gz "${PREP_DIR}/train.json.gz" \
        --val-json-gz "${PREP_DIR}/val.json.gz" \
        "${TESTFLAG[@]}" \
        --config usecases_synthetic/config/ditto/default_train.yaml \
        --output-dir "${DITTO_DIR}"
    ln -sfn "$(basename "$(cat "${DITTO_DIR}/LATEST_RUN")")/checkpoints/best" "${DITTO_DIR}/best"
    EXTRA+=(--ditto-checkpoint-override "${DITTO_DIR}/best")
fi

if [ "${BLOCK_WINNER}" = "sc_block" ]; then
    mkdir -p "${SCB_DIR}"
    echo "==== [TRAIN sc_block] domain=${DOMAIN} -> ${SCB_DIR} ===="
    python usecases_synthetic/scripts/sc_block/train.py \
        --domain "${DOMAIN}" --output-dir "${SCB_DIR}"
    EXTRA+=(--sc-block-checkpoint-override "${SCB_DIR}/best")
fi

TRAIN_END=$(date +%s.%N)
TRAIN_SEC=$(python3 -c "print(f'{${TRAIN_END}-${TRAIN_START}:.2f}')")
echo "==== TRAIN phase done: ${TRAIN_SEC}s (overrides: ${EXTRA[*]:-none}) ===="

# ---------------------------------------------------------------------------
# INFER phase (timed): run the best pipeline against the fresh checkpoints.
# ---------------------------------------------------------------------------
echo "==== TIMED best-pipeline run -> ${OUT} ===="
INFER_START=$(date +%s.%N)
python pipelines/scripts/run_best_of_breed.py \
    --config "${WORK}/${DOMAIN}.yaml" \
    --committee-dir "${WORK}/committees" \
    --variant baseline --mode replay --out "${OUT}" \
    --llm-sm --no-llm-em --no-llm-fusion \
    "${EXTRA[@]}"
INFER_END=$(date +%s.%N)
INFER_SEC=$(python3 -c "print(f'{${INFER_END}-${INFER_START}:.2f}')")
TOTAL_SEC=$(python3 -c "print(f'{${TRAIN_SEC}+${INFER_SEC}:.2f}')")

echo ""
echo "Total execution time: ${TOTAL_SEC} seconds"
{
    echo "domain=${DOMAIN}"
    echo "train_seconds=${TRAIN_SEC}"
    echo "infer_seconds=${INFER_SEC}"
    echo "total_seconds=${TOTAL_SEC}"
    echo "blocking_winner=${BLOCK_WINNER}"
    echo "matching_winner=${MATCH_WINNER}"
    echo "ref_run=${REF_RUN}"
    echo "out=${OUT}"
} | tee "${OUT}/best_pipeline_time.txt"
echo "==== done ===="
