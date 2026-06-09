#!/bin/bash
# One-off verification of the eval-target fixes (norm entity-scoping + val
# surfaces for norm/em/fusion). Runs measure_baseline on companies with
# --with-llm false (exercises the changed non-LLM scoring paths, no API cost)
# into a throwaway out-dir, then inspects the JSON for the new val keys.
set -uo pipefail
cd /ceph/rpeeters/projects/PyDI
export HF_HOME=/ceph/rpeeters/cache
source pydi-dev/bin/activate
OUT=/tmp/verify_companies_eval
rm -rf "$OUT"; mkdir -p "$OUT"
echo "==== running measure_baseline companies (with_llm=false) -> $OUT ===="
# --with-llm is a boolean flag; OMITTING it = LLM members disabled (free, fast).
python usecases_synthetic/scripts/measure_baseline.py \
    --domain companies --out-dir "$OUT" 2>&1 | tail -25
echo "==== EXIT=$? ===="
echo "==== inspect output JSON for val surfaces + norm entity-scoping ===="
python - "$OUT/baseline_metrics.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
ps = m.get("per_stage", {})
def show(stage):
    s = ps.get(stage) or {}
    pm = s.get("per_member") or {}
    print(f"\n## {stage}: members={list(pm)[:8]}")
    for name, mem in list(pm.items())[:4]:
        met = mem.get("metrics", mem) if isinstance(mem, dict) else {}
        keys = {k: round(v,4) for k,v in met.items() if isinstance(v,(int,float)) and (
            k in ("f1","f1_val","f1_test","precision","recall","pair_recall","pair_recall_val",
                  "macro_accuracy","macro_accuracy_val","scoring_surface"))}
        print(f"   {name}: {keys}")
for st in ("sm","normalization","norm","em_blocking","em_matching","fusion"):
    if st in ps: show(st)
print("\n## meta.scoring_surface:", m.get("meta",{}).get("scoring_surface"))
PY
