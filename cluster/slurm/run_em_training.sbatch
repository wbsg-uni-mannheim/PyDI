#!/bin/bash
#SBATCH --job-name=em-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Train a neural EM member (ditto_plm or sc_block) into a *pipeline-isolated*
# checkpoint dir, so run_best_of_breed.py can include it via its
# --ditto-checkpoint-override / --sc-block-checkpoint-override flags (the
# no-model-reuse policy refuses checkpoints living under usecases_synthetic/cache/).
#
# Submit from the repo root (job name distinguishes the two models):
#
#   MODEL=ditto    DOMAIN=music sbatch --partition=gpu-vram-32gb -J em-train-ditto-music   cluster/slurm/run_em_training.sbatch
#   MODEL=sc_block DOMAIN=music sbatch --partition=gpu-vram-12gb -J em-train-scblock-music cluster/slurm/run_em_training.sbatch
#
# Optional env overrides:
#   export MODEL=ditto|sc_block      # REQUIRED
#   export DOMAIN=music              # default music
#   export PYDI_VENV=pydi-dev        # path to the uv-managed venv
#   export DITTO_OUT=...             # ditto output root (default pipelines/<domain>/checkpoints/em_matching/ditto/runs)
#   export SCB_OUT=...               # sc_block output root (default pipelines/<domain>/checkpoints/em_blocking/sc_block)

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

: "${MODEL:?MODEL must be set to ditto or sc_block}"
DOMAIN="${DOMAIN:-music}"

echo "==== em-training ===="
echo "job_id : ${SLURM_JOB_ID:-<no slurm>}"
echo "node   : $(hostname)"
echo "cwd    : $(pwd)"
echo "venv   : ${PYDI_VENV}"
echo "python : $(which python)  ($(python --version 2>&1))"
echo "model  : ${MODEL}"
echo "domain : ${DOMAIN}"
nvidia-smi || echo "(no nvidia-smi available)"
echo "===================="

case "${MODEL}" in
    ditto)
        DITTO_OUT="${DITTO_OUT:-pipelines/${DOMAIN}/checkpoints/em_matching/ditto/runs}"
        # Prep dir defaults to .../ditto/<domain>/ but some prep scripts write
        # to domain-specific names (e.g. companies_pydi, games_pydi_raw); allow
        # an override.
        PREP_DIR="${DITTO_PREP_DIR:-usecases_synthetic/output/ditto/${DOMAIN}}"
        if [ ! -f "${PREP_DIR}/train.json.gz" ]; then
            echo "Ditto prep data missing; running _prep_${DOMAIN}.py"
            python "usecases_synthetic/scripts/ditto/_prep_${DOMAIN}.py"
        fi
        mkdir -p "${DITTO_OUT}"
        # --domain sources the canonical wide committee field scope (matches
        # what the prep serialized and what inference serializes); the config's
        # default 'fields' (products) is overridden.
        # DITTO_BATCH optionally overrides batch size (e.g. to fit a 12GB GPU).
        python usecases_synthetic/scripts/ditto/train.py \
            --domain "${DOMAIN}" \
            --train-json-gz "${PREP_DIR}/train.json.gz" \
            --val-json-gz "${PREP_DIR}/val.json.gz" \
            --test-json-gz "${PREP_DIR}/test.json.gz" \
            --config usecases_synthetic/config/ditto/default_train.yaml \
            ${DITTO_BATCH:+--batch-size ${DITTO_BATCH}} \
            --output-dir "${DITTO_OUT}"
        echo "DITTO training done. Checkpoints under: ${DITTO_OUT}"
        echo "Override flag: --ditto-checkpoint-override <run_dir>/checkpoints/best"
        ;;
    sc_block)
        SCB_OUT="${SCB_OUT:-pipelines/${DOMAIN}/checkpoints/em_blocking/sc_block}"
        mkdir -p "${SCB_OUT}"
        # eval-pair defaults to the domain's canonical pair (music=musicbrainz,discogs).
        python usecases_synthetic/scripts/sc_block/train.py \
            --domain "${DOMAIN}" \
            --output-dir "${SCB_OUT}"
        echo "SC_BLOCK training done. Checkpoints under: ${SCB_OUT}"
        echo "Override flag: --sc-block-checkpoint-override ${SCB_OUT}/best"
        ;;
    *)
        echo "Unknown MODEL: ${MODEL} (expected ditto or sc_block)" >&2
        exit 2
        ;;
esac

echo "==== em-training DONE ===="
