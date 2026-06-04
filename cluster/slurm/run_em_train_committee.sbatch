#!/bin/bash
#SBATCH --job-name=em-train-committee
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

# Train the Ditto (matching) + SC-Block (blocking) checkpoints the SYNTHETIC
# committees load, for ONE domain, landing them at the exact paths
# measure_baseline / validate_variant read:
#
#   usecases_synthetic/cache/ditto_checkpoints/<domain>/best
#   usecases_synthetic/cache/sc_block_checkpoints/<domain>/best
#   usecases_synthetic/cache/{ditto,sc_block}_checkpoints/<domain>/variant_<level>/best
#
# (NOT the pipelines/<domain>/checkpoints/... paths that run_em_training.sbatch
# targets — that one is for the best-of-breed pipeline's --checkpoint-override.)
#
# Submit from the repo root (one job per domain):
#
#   DOMAIN=companies sbatch cluster/slurm/run_em_train_committee.sbatch
#   for d in companies games music products papers; do
#       DOMAIN=$d sbatch cluster/slurm/run_em_train_committee.sbatch
#   done
#
# Then run cluster/slurm/run_measure_validate.sbatch for the same domain and the
# committees will pick up these checkpoints with no overrides.
#
# Steps (all on by default; comma-list to subset / resume):
#   STEPS="prep,ditto,scblock,variants"
#     prep     -> generate Ditto training data (ditto/_prep_<domain>.py)
#     ditto    -> train baseline Ditto -> cache/ditto_checkpoints/<domain>/best
#     scblock  -> train baseline SC-Block -> cache/sc_block_checkpoints/<domain>/best
#     variants -> retrain_variant_cascade.py -> {ditto,sc_block}/<domain>/variant_<level>/best
#                 (both models x LEVELS; independent of the baseline checkpoints)
#
# Optional env:
#   DOMAIN=companies              # REQUIRED: {companies,games,music,products,papers}
#   STEPS="prep,ditto,scblock,variants"
#   LEVELS="easy medium hard"     # variant levels for the cascade (default all 3)
#   PYDI_VENV=pydi-dev
#
# GPU: Ditto + SC-Block are RoBERTa fine-tunes. Baseline + 3 variant levels x 2
# models = 8 trainings; budget several GPU-hours per domain (bump --time if a
# large domain like products/papers needs it). Single-GPU / single-process
# (plain `python`); for multi-GPU, launch ditto via torchrun (it auto-enables
# DDP when WORLD_SIZE>1).

mkdir -p logs
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "${SLURM_SUBMIT_DIR}"
fi

PYDI_VENV="${PYDI_VENV:-pydi-dev}"
if [ ! -f "${PYDI_VENV}/bin/activate" ]; then
    echo "pydi-dev venv not found at ${PYDI_VENV}/bin/activate" >&2
    echo "Bootstrap with:  uv pip install -e \".[dev,embedding,plm,faiss,features,synthetic]\" --python ${PYDI_VENV}/bin/python" >&2
    exit 1
fi
set +u
# shellcheck source=/dev/null
source "${PYDI_VENV}/bin/activate"
set -u

: "${DOMAIN:?DOMAIN must be set, e.g. DOMAIN=companies}"
STEPS="${STEPS:-prep,ditto,scblock,variants}"
LEVELS="${LEVELS:-easy medium hard}"

step_enabled () { case ",${STEPS}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# The Ditto prep output dir is domain-specific (the committee-correct
# --train-source pydi default writes to <domain>_pydi / <domain>_pydi_raw for
# companies/games; the others write to <domain>).
case "${DOMAIN}" in
    companies) DITTO_PREP_SUBDIR="companies_pydi" ;;
    games)     DITTO_PREP_SUBDIR="games_pydi_raw" ;;
    *)         DITTO_PREP_SUBDIR="${DOMAIN}" ;;
esac
PREP_DIR="usecases_synthetic/output/ditto/${DITTO_PREP_SUBDIR}"
DITTO_OUT="usecases_synthetic/cache/ditto_checkpoints/${DOMAIN}"

echo "==== em-train-committee ===="
echo "job_id   : ${SLURM_JOB_ID:-<no slurm>}"
echo "node     : $(hostname)"
echo "cwd      : $(pwd)"
echo "venv     : ${PYDI_VENV}"
echo "python   : $(which python)  ($(python --version 2>&1))"
echo "domain   : ${DOMAIN}"
echo "steps    : ${STEPS}"
echo "levels   : ${LEVELS}"
echo "ditto_prep_dir : ${PREP_DIR}"
nvidia-smi || echo "(no nvidia-smi available)"
echo "============================"

# ---------------------------------------------------------------------------
# Step: prep — generate Ditto training data (idempotent: skip if present)
# ---------------------------------------------------------------------------
if step_enabled prep; then
    if [ -f "${PREP_DIR}/train.json.gz" ] && [ -f "${PREP_DIR}/val.json.gz" ]; then
        echo "==== [prep] ${PREP_DIR}/train.json.gz already present — skipping ===="
    else
        echo "==== [prep] ditto/_prep_${DOMAIN}.py (-> ${PREP_DIR}) ===="
        python "usecases_synthetic/scripts/ditto/_prep_${DOMAIN}.py"
    fi
fi

# ---------------------------------------------------------------------------
# Step: ditto baseline — train + create the <domain>/best symlink the committee
# reads (ditto/train.py does NOT make it; it writes run_<ts>/checkpoints/best
# and a LATEST_RUN marker).
# ---------------------------------------------------------------------------
if step_enabled ditto; then
    echo "==== [ditto] baseline train domain=${DOMAIN} -> ${DITTO_OUT}/best ===="
    if [ ! -f "${PREP_DIR}/train.json.gz" ]; then
        echo "Ditto prep data missing at ${PREP_DIR}/train.json.gz (run STEPS=prep first)" >&2
        exit 2
    fi
    DITTO_TEST_FLAG=()
    [ -f "${PREP_DIR}/test.json.gz" ] && DITTO_TEST_FLAG=(--test-json-gz "${PREP_DIR}/test.json.gz")
    mkdir -p "${DITTO_OUT}"
    python usecases_synthetic/scripts/ditto/train.py \
        --domain "${DOMAIN}" \
        --train-json-gz "${PREP_DIR}/train.json.gz" \
        --val-json-gz "${PREP_DIR}/val.json.gz" \
        "${DITTO_TEST_FLAG[@]}" \
        --output-dir "${DITTO_OUT}"
    # Point <domain>/best at the run just produced (relative symlink, matching
    # the sc_block convention). LATEST_RUN holds the run dir path.
    RUN_DIR="$(cat "${DITTO_OUT}/LATEST_RUN")"
    ln -sfn "$(basename "${RUN_DIR}")/checkpoints/best" "${DITTO_OUT}/best"
    echo "[ditto] ${DITTO_OUT}/best -> $(readlink "${DITTO_OUT}/best")"
fi

# ---------------------------------------------------------------------------
# Step: sc_block baseline — train.py defaults --output-dir to the cache path
# AND auto-creates the <domain>/best symlink, and picks the correct per-domain
# --eval-pair default, so a plain invocation lands correctly.
# ---------------------------------------------------------------------------
if step_enabled scblock; then
    echo "==== [scblock] baseline train domain=${DOMAIN} -> usecases_synthetic/cache/sc_block_checkpoints/${DOMAIN}/best ===="
    python usecases_synthetic/scripts/sc_block/train.py --domain "${DOMAIN}"
fi

# ---------------------------------------------------------------------------
# Step: variants — retrain both matchers for every level in one cascade. Each
# retrain auto-creates its variant_<level>/best symlink; independent of the
# baseline checkpoints.
# ---------------------------------------------------------------------------
if step_enabled variants; then
    echo "==== [variants] retrain_variant_cascade domain=${DOMAIN} levels='${LEVELS}' (ditto + sc_block) ===="
    python usecases_synthetic/scripts/retrain_variant_cascade.py \
        --domain "${DOMAIN}" \
        --levels ${LEVELS}
fi

echo "==== em-train-committee DONE for ${DOMAIN} ===="
echo "ditto baseline   : ${DITTO_OUT}/best"
echo "sc_block baseline: usecases_synthetic/cache/sc_block_checkpoints/${DOMAIN}/best"
echo "variants         : usecases_synthetic/cache/{ditto,sc_block}_checkpoints/${DOMAIN}/variant_{${LEVELS// /,}}/best"
echo "Next: DOMAIN=${DOMAIN} sbatch cluster/slurm/run_measure_validate.sbatch"
