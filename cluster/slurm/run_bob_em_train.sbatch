#!/bin/bash
#SBATCH --job-name=bob-em-train
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

# Train the *pipeline-isolated* Ditto (EM matching) + SC-Block (EM blocking)
# checkpoints the BEST-OF-BREED pipeline reads via its
# --ditto-checkpoint-override / --sc-block-checkpoint-override flags. These are
# DISTINCT from the committee checkpoints under usecases_synthetic/cache/ that
# run_em_train_committee.sbatch produces: the BoB pipeline's no-model-reuse
# policy refuses committee-cache checkpoints, so it trains its own under
#
#   pipelines/<domain>/checkpoints/em_matching/ditto/<levelkey>/best
#   pipelines/<domain>/checkpoints/em_blocking/sc_block/<levelkey>/best
#
# where <levelkey> is one of: baseline, variant_easy, variant_medium,
# variant_hard. run_best_of_breed.sbatch derives exactly these paths from
# DOMAIN + VARIANT and asserts they exist before running (no silent EM-member
# dropping).
#
# Submit from the repo root (one job per domain):
#
#   DOMAIN=companies sbatch cluster/slurm/run_bob_em_train.sbatch
#   # papers ships no variant corner_filled splits -> baseline only:
#   DOMAIN=papers LEVELS="" sbatch cluster/slurm/run_bob_em_train.sbatch
#
# Steps (all on by default; comma-list to subset / resume):
#   STEPS="prep,ditto,scblock,variants"
#     prep     -> ensure baseline Ditto training data (ditto/_prep_<domain>.py)
#     ditto    -> train baseline Ditto    -> .../ditto/baseline/best
#     scblock  -> train baseline SC-Block -> .../sc_block/baseline/best
#     variants -> retrain_variant_cascade.py --out-root pipelines/<d>/checkpoints
#                 -> .../{ditto,sc_block}/variant_<level>/best  (per LEVELS)
#
# Optional env:
#   DOMAIN=companies              # REQUIRED: {companies,games,music,products,papers}
#   LEVELS="easy medium hard"     # variant levels (empty "" => baseline only)
#   STEPS="prep,ditto,scblock,variants"
#   PYDI_VENV=pydi-dev

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

: "${DOMAIN:?DOMAIN must be set, e.g. DOMAIN=companies}"
STEPS="${STEPS:-prep,ditto,scblock,variants}"
LEVELS="${LEVELS-easy medium hard}"   # note: LEVELS="" (set-but-empty) => baseline only

step_enabled () { case ",${STEPS}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

# Baseline Ditto prep output dir is domain-specific (the committee-correct
# --train-source pydi default writes to <domain>_pydi / <domain>_pydi_raw for
# companies/games; the others write to <domain>).
case "${DOMAIN}" in
    companies) DITTO_PREP_SUBDIR="companies_pydi" ;;
    games)     DITTO_PREP_SUBDIR="games_pydi_raw" ;;
    *)         DITTO_PREP_SUBDIR="${DOMAIN}" ;;
esac
PREP_DIR="usecases_synthetic/output/ditto/${DITTO_PREP_SUBDIR}"

CKPT_ROOT="pipelines/${DOMAIN}/checkpoints"
DITTO_BASE_DIR="${CKPT_ROOT}/em_matching/ditto/baseline"
SCB_BASE_DIR="${CKPT_ROOT}/em_blocking/sc_block/baseline"

echo "==== bob-em-train ===="
echo "job_id   : ${SLURM_JOB_ID:-<no slurm>}"
echo "node     : $(hostname)"
echo "cwd      : $(pwd)"
echo "venv     : ${PYDI_VENV}"
echo "python   : $(which python)  ($(python --version 2>&1))"
echo "domain   : ${DOMAIN}"
echo "steps    : ${STEPS}"
echo "levels   : ${LEVELS:-<none — baseline only>}"
echo "ckpt_root: ${CKPT_ROOT}"
echo "prep_dir : ${PREP_DIR}"
nvidia-smi || echo "(no nvidia-smi available)"
echo "======================"

# ---------------------------------------------------------------------------
# Step: prep — ensure baseline Ditto training data exists (idempotent).
# ---------------------------------------------------------------------------
if step_enabled prep; then
    if [ -f "${PREP_DIR}/train.json.gz" ] && [ -f "${PREP_DIR}/val.json.gz" ]; then
        echo "==== [prep] ${PREP_DIR}/train.json.gz present — skipping ===="
    else
        echo "==== [prep] ditto/_prep_${DOMAIN}.py (-> ${PREP_DIR}) ===="
        python "usecases_synthetic/scripts/ditto/_prep_${DOMAIN}.py"
    fi
fi

# ---------------------------------------------------------------------------
# Step: baseline Ditto — train into the pipeline-isolated baseline dir, then
# point <baseline>/best at the produced run (relative symlink).
# ---------------------------------------------------------------------------
if step_enabled ditto; then
    echo "==== [ditto] baseline train domain=${DOMAIN} -> ${DITTO_BASE_DIR}/best ===="
    if [ ! -f "${PREP_DIR}/train.json.gz" ]; then
        echo "Ditto prep data missing at ${PREP_DIR}/train.json.gz (run STEPS=prep first)" >&2
        exit 2
    fi
    DITTO_TEST_FLAG=()
    [ -f "${PREP_DIR}/test.json.gz" ] && DITTO_TEST_FLAG=(--test-json-gz "${PREP_DIR}/test.json.gz")
    mkdir -p "${DITTO_BASE_DIR}"
    python usecases_synthetic/scripts/ditto/train.py \
        --domain "${DOMAIN}" \
        --train-json-gz "${PREP_DIR}/train.json.gz" \
        --val-json-gz "${PREP_DIR}/val.json.gz" \
        "${DITTO_TEST_FLAG[@]}" \
        --config usecases_synthetic/config/ditto/default_train.yaml \
        --output-dir "${DITTO_BASE_DIR}"
    RUN_DIR="$(cat "${DITTO_BASE_DIR}/LATEST_RUN")"
    ln -sfn "$(basename "${RUN_DIR}")/checkpoints/best" "${DITTO_BASE_DIR}/best"
    echo "[ditto] ${DITTO_BASE_DIR}/best -> $(readlink "${DITTO_BASE_DIR}/best")"
fi

# ---------------------------------------------------------------------------
# Step: baseline SC-Block — train.py auto-creates the <output-dir>/best symlink
# and picks the correct per-domain --eval-pair default.
# ---------------------------------------------------------------------------
if step_enabled scblock; then
    echo "==== [scblock] baseline train domain=${DOMAIN} -> ${SCB_BASE_DIR}/best ===="
    mkdir -p "${SCB_BASE_DIR}"
    python usecases_synthetic/scripts/sc_block/train.py \
        --domain "${DOMAIN}" \
        --output-dir "${SCB_BASE_DIR}"
fi

# ---------------------------------------------------------------------------
# Step: variants — retrain both matchers for each level into the isolated tree.
# Skipped entirely when LEVELS is empty (e.g. papers, which ships no variant
# corner_filled splits).
# ---------------------------------------------------------------------------
if step_enabled variants; then
    if [ -z "${LEVELS}" ]; then
        echo "==== [variants] LEVELS empty — skipping variant training (baseline only) ===="
    else
        echo "==== [variants] retrain_variant_cascade domain=${DOMAIN} levels='${LEVELS}' -> ${CKPT_ROOT} ===="
        python usecases_synthetic/scripts/retrain_variant_cascade.py \
            --domain "${DOMAIN}" \
            --levels ${LEVELS} \
            --out-root "${CKPT_ROOT}"
    fi
fi

echo "==== bob-em-train DONE for ${DOMAIN} ===="
echo "ditto baseline   : ${DITTO_BASE_DIR}/best"
echo "sc_block baseline: ${SCB_BASE_DIR}/best"
if [ -n "${LEVELS}" ]; then
    echo "variants         : ${CKPT_ROOT}/em_{matching,blocking}/<model>/variant_{${LEVELS// /,}}/best"
fi
echo "Next: DOMAIN=${DOMAIN} bash cluster/slurm/submit_all_variants.sh   (BoB jobs read these via --*-checkpoint-override)"
