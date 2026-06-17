#!/bin/bash
# Submit the (domain x variant) grid of best-of-breed pipeline jobs to Slurm --
# each combination as its own job, wrapping cluster/slurm/run_best_of_breed.sbatch
# with DOMAIN + VARIANT env vars.
#
# Per the directives baked into this grid:
#   * LLM only for schema matching (LLM_SM=true, LLM_EM=false, LLM_FUSION=false).
#   * No breed members silently dropped: each job derives its pipeline-isolated
#     ditto_plm + sc_block checkpoints (pipelines/<domain>/checkpoints/...,
#     trained by cluster/slurm/run_bob_em_train.sbatch) and ASSERTS they exist.
#   * papers has no variant corner_filled splits -> baseline only.
#
# Each BoB job optionally waits (afterok) on its domain's bob-em-train-<domain>
# job so the isolated checkpoints are ready first. Discovery is automatic from
# squeue by job name; override per domain with DEP_<domain>=<jobid>, or disable
# with NO_DEPS=1.
#
# Usage (from the repo root):
#
#   bash cluster/slurm/submit_all_variants.sh
#
# Override which domains / variants to sweep (VARIANTS overrides ALL domains;
# omit it to use the per-domain defaults, i.e. papers=baseline only):
#
#   DOMAINS="products music" VARIANTS="medium hard" \
#     bash cluster/slurm/submit_all_variants.sh
#
# Dry-run (print sbatch commands without submitting):
#
#   DRY_RUN=1 bash cluster/slurm/submit_all_variants.sh
#
# Progress: per-domain logs at logs/bob_progress/<domain>.log + a manifest at
# logs/bob_progress/manifest.tsv. Refresh statuses with
# `bash cluster/slurm/track_bob_progress.sh`.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DOMAINS="${DOMAINS:-companies games music products papers}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-cluster/slurm/run_best_of_breed.sbatch}"
PROGRESS_DIR="${PROGRESS_DIR:-logs/bob_progress}"
MANIFEST="${PROGRESS_DIR}/manifest.tsv"

# LLM directive (explicit, not relying on the wrapper defaults): LLM in schema
# matching only.
LLM_SM="${LLM_SM:-true}"
LLM_EM="${LLM_EM:-false}"
LLM_FUSION="${LLM_FUSION:-false}"

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "Missing sbatch wrapper: ${SBATCH_SCRIPT}" >&2
    echo "Run this from the repo root." >&2
    exit 2
fi

# Per-domain variant list. papers ships no variant corner_filled EM splits, so
# its isolated variant checkpoints can't be trained -> baseline only.
variants_for () {
    if [ -n "${VARIANTS:-}" ]; then
        echo "${VARIANTS}"
        return
    fi
    case "$1" in
        papers) echo "baseline" ;;
        *)      echo "baseline easy medium hard" ;;
    esac
}

# Resolve the afterok dependency job id for a domain's BoB jobs: the
# bob-em-train-<domain> training job (if still queued/running). Explicit
# DEP_<domain> wins; NO_DEPS=1 disables.
dep_for () {
    local d="$1"
    [ -n "${NO_DEPS:-}" ] && { echo ""; return; }
    local explicit
    explicit="$(eval "echo \${DEP_${d}:-}")"
    if [ -n "${explicit}" ]; then echo "${explicit}"; return; fi
    # Most-recent job with this exact name in our queue (PD or R).
    squeue --me -h -n "bob-em-train-${d}" -o "%i" 2>/dev/null | sort -n | tail -1
}

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
mkdir -p "${PROGRESS_DIR}"
NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ -z "${DRY_RUN:-}" ] && [ ! -f "${MANIFEST}" ]; then
    printf "domain\tvariant\tjobid\tdep_jobid\tsubmitted_utc\n" > "${MANIFEST}"
fi

echo "Submitting BoB grid: domains=[${DOMAINS}]"
echo "LLM toggles: sm=${LLM_SM} em=${LLM_EM} fusion=${LLM_FUSION}"
echo

submitted=0
skipped=0
for d in ${DOMAINS}; do
    dep="$(dep_for "${d}")"
    dep_flag=()
    [ -n "${dep}" ] && dep_flag=(--dependency="afterok:${dep}")
    domain_log="${PROGRESS_DIR}/${d}.log"
    if [ -z "${DRY_RUN:-}" ]; then
        {
            echo "===== ${d} — submitted ${NOW} ====="
            echo "dependency (afterok bob-em-train): ${dep:-<none, checkpoints assumed ready>}"
        } >> "${domain_log}"
    fi
    for v in $(variants_for "${d}"); do
        if [ -n "${DRY_RUN:-}" ]; then
            echo "DRY: env DOMAIN=${d} VARIANT=${v} LLM_SM=${LLM_SM} LLM_EM=${LLM_EM} LLM_FUSION=${LLM_FUSION} sbatch -J bob-${d}-${v} ${dep_flag[*]:-} ${SBATCH_SCRIPT}"
            skipped=$((skipped + 1))
            continue
        fi
        printf "  %-10s %-9s -> " "${d}" "${v}"
        jobid="$(env \
            "DOMAIN=${d}" "VARIANT=${v}" \
            "LLM_SM=${LLM_SM}" "LLM_EM=${LLM_EM}" "LLM_FUSION=${LLM_FUSION}" \
            sbatch --parsable -J "bob-${d}-${v}" "${dep_flag[@]}" "${SBATCH_SCRIPT}")"
        echo "job ${jobid}${dep:+  (afterok:${dep})}"
        printf "%s\t%s\t%s\t%s\t%s\n" "${d}" "${v}" "${jobid}" "${dep:-}" "${NOW}" >> "${MANIFEST}"
        printf "  %-9s job=%s  out=logs/bob-%s-%s-%s.out  run=pipelines/%s/run_slurm_%s_%s/\n" \
            "${v}" "${jobid}" "${d}" "${v}" "${jobid}" "${d}" "${v}" "${jobid}" >> "${domain_log}"
        submitted=$((submitted + 1))
    done
done

echo
if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY_RUN: ${skipped} commands shown, none submitted."
else
    echo "Submitted ${submitted} jobs. Manifest: ${MANIFEST}"
    echo "Track with: bash cluster/slurm/track_bob_progress.sh   (or: squeue --me)"
fi
