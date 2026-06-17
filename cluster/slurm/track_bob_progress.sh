#!/bin/bash
# Refresh best-of-breed pipeline progress from the submission manifest.
#
# Reads logs/bob_progress/manifest.tsv (written by submit_all_variants.sh),
# queries each job's Slurm state (sacct, falling back to squeue), pulls the
# composite_score from finished runs, and writes:
#   * logs/bob_progress/SUMMARY.md          — one table across all jobs
#   * logs/bob_progress/<domain>.log        — appends a timestamped status block
#
# Usage (from the repo root):  bash cluster/slurm/track_bob_progress.sh
# Re-run any time; safe to repeat. Pair with `watch` or the /loop skill.

set -uo pipefail

PROGRESS_DIR="${PROGRESS_DIR:-logs/bob_progress}"
MANIFEST="${MANIFEST:-${PROGRESS_DIR}/manifest.tsv}"
SUMMARY="${PROGRESS_DIR}/SUMMARY.md"

if [ ! -f "${MANIFEST}" ]; then
    echo "No manifest at ${MANIFEST} — submit jobs first with submit_all_variants.sh" >&2
    exit 1
fi

NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

job_state () {
    # Prefer sacct (covers running + terminal states); fall back to squeue.
    local jid="$1" st=""
    st="$(sacct -j "${jid}" -n -X -o State 2>/dev/null | head -1 | tr -d ' ')"
    if [ -z "${st}" ]; then
        st="$(squeue -j "${jid}" -h -o '%T' 2>/dev/null | head -1)"
    fi
    [ -z "${st}" ] && st="UNKNOWN"
    echo "${st}"
}

composite_for () {
    # Pull composite_score from the slurm .out, else the run's summary.md.
    local d="$1" v="$2" jid="$3" out score=""
    out="logs/bob-${d}-${v}-${jid}.out"
    if [ -f "${out}" ]; then
        score="$(grep -aoE 'composite_score:[[:space:]]*[0-9.]+' "${out}" | tail -1 | grep -oE '[0-9.]+$')"
    fi
    if [ -z "${score}" ]; then
        local rundir="pipelines/${d}/run_slurm_${v}_${jid}"
        if [ -f "${rundir}/summary.md" ]; then
            score="$(grep -aoE 'composite[_ ]score[^0-9]*[0-9.]+' "${rundir}/summary.md" | tail -1 | grep -oE '[0-9.]+$')"
        fi
    fi
    echo "${score:-—}"
}

# ---------------------------------------------------------------------------
# Build the SUMMARY table + per-domain snapshots.
# ---------------------------------------------------------------------------
{
    echo "# Best-of-breed pipeline progress"
    echo
    echo "_Updated: ${NOW}_"
    echo
    echo "| domain | variant | job | state | composite | run dir |"
    echo "|---|---|---|---|---|---|"
} > "${SUMMARY}"

declare -A DOMAIN_ROWS DOMAIN_COUNTS
# tail -n +2 skips the header row.
while IFS=$'\t' read -r d v jid dep sub; do
    [ -z "${jid:-}" ] && continue
    st="$(job_state "${jid}")"
    score="—"
    case "${st}" in
        COMPLETED) score="$(composite_for "${d}" "${v}" "${jid}")" ;;
    esac
    rundir="pipelines/${d}/run_slurm_${v}_${jid}"
    echo "| ${d} | ${v} | ${jid} | ${st} | ${score} | ${rundir} |" >> "${SUMMARY}"
    DOMAIN_ROWS["${d}"]+="  ${v}: job ${jid} -> ${st} (composite ${score})"$'\n'
    DOMAIN_COUNTS["${d}"]="$(( ${DOMAIN_COUNTS["${d}"]:-0} + 1 ))"
done < <(tail -n +2 "${MANIFEST}")

for d in "${!DOMAIN_ROWS[@]}"; do
    {
        echo "----- STATUS ${NOW} (${DOMAIN_COUNTS[$d]} jobs) -----"
        printf '%s' "${DOMAIN_ROWS[$d]}"
    } >> "${PROGRESS_DIR}/${d}.log"
done

echo "Updated ${SUMMARY} and per-domain logs under ${PROGRESS_DIR}/"
echo
cat "${SUMMARY}"
