#!/bin/bash
set -uo pipefail
cd /ceph/rpeeters/projects/PyDI
export HF_HOME=/ceph/rpeeters/cache
source pydi-dev/bin/activate
OUT=/tmp/verify_em_only; rm -rf "$OUT"; mkdir -p "$OUT"
python usecases_synthetic/scripts/measure_baseline.py \
    --domain companies --stages em_blocking,em_matching --out-dir "$OUT" 2>&1 | tail -8
python - "$OUT/baseline_metrics.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); ps=m.get("per_stage",{})
for st in ("em_blocking","em_matching"):
    pm=(ps.get(st) or {}).get("per_member") or {}
    print(f"\n## {st}")
    for name,mem in list(pm.items())[:6]:
        met=mem.get("metrics",{})
        print(f"  {name}: f1={met.get('f1')} f1_val={met.get('f1_val')} "
              f"pair_recall={met.get('pair_recall')} pair_recall_val={met.get('pair_recall_val')}")
PY
