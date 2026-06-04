#!/bin/bash
# Submit the full (domain x variant) grid of best-of-breed pipeline jobs to
# the Slurm cluster -- each combination as its own job. Wraps
# ``cluster/slurm/run_best_of_breed.sbatch`` with both ``DOMAIN`` and
# ``VARIANT`` env vars.
#
# Usage (from the repo root):
#
#   bash cluster/slurm/submit_all_variants.sh
#
# Override which domains / variants to sweep:
#
#   DOMAINS="papers products"  VARIANTS="medium hard" \
#     bash cluster/slurm/submit_all_variants.sh
#
# Dry-run (print sbatch commands without submitting):
#
#   DRY_RUN=1 bash cluster/slurm/submit_all_variants.sh
#
# Extra flags passed through to the wrapper survive (e.g.
# ``MODE=sweep bash submit_all_variants.sh``).
#
# Each job writes to
# ``pipelines/<domain>/run_slurm_<variant>_<jobid>/``; tail with
# ``squeue --me``.

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DOMAINS="${DOMAINS:-companies games music papers products}"
VARIANTS="${VARIANTS:-baseline easy medium hard}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-cluster/slurm/run_best_of_breed.sbatch}"

if [ ! -f "${SBATCH_SCRIPT}" ]; then
    echo "Missing sbatch wrapper: ${SBATCH_SCRIPT}" >&2
    echo "Run this from the repo root." >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
echo "Submitting grid: domains=[${DOMAINS}] x variants=[${VARIANTS}]"
echo

submitted=0
skipped=0
for d in ${DOMAINS}; do
    for v in ${VARIANTS}; do
        cmd=(env "DOMAIN=${d}" "VARIANT=${v}" sbatch "${SBATCH_SCRIPT}")
        if [ -n "${DRY_RUN:-}" ]; then
            echo "DRY: ${cmd[*]}"
            skipped=$((skipped + 1))
        else
            printf "  %-10s %-9s -> " "${d}" "${v}"
            "${cmd[@]}"
            submitted=$((submitted + 1))
        fi
    done
done

echo
if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY_RUN: ${skipped} commands shown, none submitted."
else
    echo "Submitted ${submitted} jobs. Track with: squeue --me"
fi
