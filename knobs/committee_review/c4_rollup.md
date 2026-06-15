# C4 — Cross-committee consistency review

Rollup for [plan_committee_finalization.md C4](../../plans/plan_committee_finalization.md) — the
last row in the committee-finalization tracker and the final item inside
[plan_s1_scale.md S4a](../../plans/plan_s1_scale.md#phase-a--generator--measurement-fixes-must-be-x-before-any-full-domain-generation).

## Scope

C4 verifies that the three committees (SM / EM-blocking / EM-matching / Fusion)
agree on the properties that span file boundaries, and that the measurement-
provenance machinery still pins the files runtime actually reads after the
C2.4b split. Specifically:

1. `column_mapping` blocks stay in sync across EM + Fusion.
2. `trust_scores` direction is consistent with the learned TD ranking.
3. Committee-version pinning hashes the YAMLs that drive runtime (not a dead file).
4. Phase A S1 / S2 did not invalidate inputs that committee members consume.

## Findings

### 1. `column_mapping` agreement — already green

`em_blocking_committee.yaml`, `em_matching_committee.yaml`, and
`fusion_committee.yaml` have **byte-identical** `column_mapping` blocks for
the companies domain (dbpedia + forbes + fullcontact renames onto the
canonical schema). Verified by direct YAML diff during C4 review.

`TestEMMatchingCommitteeConfig::test_column_mapping_matches_blocking` already
enforced the EM-blocking↔EM-matching pair. Added
`TestCrossCommitteeInvariants::test_column_mapping_blocking_matching_fusion_agree`
in [test_committee_configs.py](../../usecases_synthetic/tests/test_committee_configs.py)
to extend the invariant to Fusion, so future edits to any one of the three
must land in the other two or the test breaks.

### 2. `trust_scores` vs learned TD ranking — direction-consistent

Manual priors in `fusion_committee.yaml`: `{forbes: 3, fullcontact: 2, dbpedia: 1}`.
The C3.4.11 smoke test established that TruthFinder / FusionQuery / AccuSim
all learn `forbes > fullcontact > dbpedia` on the companies fixture (dbpedia
is consistently wrong on 3 of 5 entities in the test corpus). The manual
prior agrees with the learned direction.

Manual priors only feed `favour_sources` and `prefer_higher_trust` (the
non-TD trust-weighted strategies); the TD methods learn their own vector
from the data and ignore the manual prior. Keeping the manual prior
aligned with the learned direction avoids two classes of strategy voting
in opposite directions on the same source.

Added `TestCrossCommitteeInvariants::test_trust_scores_agree_with_td_learned_ordering`
as a regression guard on the direction (not the exact values — the prior
is a tie-breaker, not a quantitative calibration).

### 3. Committee-version pinning — fixed

**Pre-C4 state (stale):** [measure_baseline.py](../../usecases_synthetic/scripts/measure_baseline.py)
and [validate_variant.py](../../usecases_synthetic/scripts/validate_variant.py)
both hardcoded the EM-stage version string as `em_committee.yaml@<sha12>`.
That file was the pre-C2.4b combined roster; runtime hasn't read it since
the split. Edits to `em_blocking_committee.yaml` or
`em_matching_committee.yaml` therefore would NOT trigger the drift guard —
`validate_variant.py` would happily re-run against a stale baseline and
silently produce misleading deltas.

**Fix:** both scripts now hash **both** split YAMLs and emit a combined
stage string `em_blocking_committee.yaml@<sha>+em_matching_committee.yaml@<sha>`
under the single `em` stage key (so the Stage `Literal["sm", "em", "fusion"]`
type alias stays unchanged and downstream readers don't need a schema
bump). Both files share a `_STAGE_YAML_FILES` constant defining the
same ordered tuple, so the two scripts deterministically emit identical
strings for identical on-disk content. See:

- [measure_baseline.py lines 83-124](../../usecases_synthetic/scripts/measure_baseline.py#L83-L124)
- [validate_variant.py lines 99-127](../../usecases_synthetic/scripts/validate_variant.py#L99-L127)

**Consequence for downstream validation:** existing `baseline_metrics.json`
files under [baselines/companies-small/](../../usecases_synthetic/baselines/companies-small/)
and [baselines/companies/](../../usecases_synthetic/baselines/companies/) still
carry the old `em_committee.yaml@<sha>` marker. `validate_variant.py` will
correctly refuse to run against those baselines with a `RuntimeError:
Committee YAML drift detected vs baseline_metrics.json` — the baseline is
stale and S5's full-companies re-run checklist already schedules a fresh
`measure_baseline.py --domain companies` run. For `companies-small`, S4b
is gated on the same re-baselining. Both are expected; this rollup does
not retroactively rewrite the validation artefacts.

### 4. Phase A S1 / S2 impact on committee inputs — no retrain required

S1 fixed [apply_knob_02_niche.py](../../usecases_synthetic/scripts/apply_knob_02_niche.py)
so `test_gold_regenerated.csv` contains real positives instead of an all-
negatives artefact. S2 promoted that regenerated CSV to the primary EM
measurement surface inside [committee_em.py](../../usecases_synthetic/lib/committee_em.py)
and fell back to pool / original gold only when it isn't present.

Neither change touched the **training** gold that Phase A0 D8's Ditto
checkpoint consumed (`forbes_2_dbpedia_{train,val,test}.csv` — the original
PyDI EM gold). The regenerated CSV is variant-specific and exists only
under `output/<variant>/...` paths produced by `generate_variant.py`; the
Ditto checkpoint was trained off the baseline gold before any variant
generation occurs. Verified the training-time path hasn't moved: see D8
in [plan_s1_scale.md](../../plans/plan_s1_scale.md) and the actual checkpoint
under [usecases_synthetic/cache/ditto_checkpoints/companies/best/](../../usecases_synthetic/cache/ditto_checkpoints/companies/best/).

Therefore no committee member needs retraining as a consequence of S1 / S2.
The committee-version-pinning drift check handles everything else: if a
future fix does change a training surface, the YAML edit (e.g. a new
`checkpoint_path` in `ditto_plm`) will re-hash and force re-baselining.

## Artefacts retired as part of C4

- [usecases_synthetic/config/committees/em_committee.yaml](../../usecases_synthetic/config/committees/)
  — deleted. Pre-C2.4b combined EM roster; nothing reads it at runtime.
  A regression-guard test (`test_retired_em_committee_yaml_absent`) ensures
  it doesn't get resurrected accidentally.
- `TestEMCommitteeConfig` — deleted from
  [test_committee_configs.py](../../usecases_synthetic/tests/test_committee_configs.py).
  Replaced by `TestEMBlockingCommitteeConfig` +
  `TestEMMatchingCommitteeConfig` (both introduced in C2.4b).

## Stale-doc cleanups

References to the retired YAML updated in:

- [usecases_synthetic/config/ditto/README.md](../../usecases_synthetic/config/ditto/README.md)
  — now points at `em_matching_committee.yaml`.
- [usecases_synthetic/lib/ditto_matcher.py](../../usecases_synthetic/lib/ditto_matcher.py)
  — docstring updated.
- [usecases_synthetic/lib/magellan_em_matcher.py](../../usecases_synthetic/lib/magellan_em_matcher.py)
  — docstring updated.
- [usecases_synthetic/lib/variant_loader.py](../../usecases_synthetic/lib/variant_loader.py)
  — `resolve_column_mapping` docstring updated to mention both split files.
- [usecases_synthetic/config/committees/fusion_committee.yaml](../../usecases_synthetic/config/committees/fusion_committee.yaml)
  — comment above `column_mapping` now names the two split files and
  references the cross-committee invariant test.

The historical shortlist / portfolio documents
([em_shortlist.md](em_shortlist.md), [em_portfolio.md](em_portfolio.md))
are deliberately left pointing at the old YAML — they record the pre-split
state and the design decision that led to C2.4b.

## Exit criteria

All three committee YAMLs pass
[test_committee_configs.py](../../usecases_synthetic/tests/test_committee_configs.py)
(43 tests green, 1 pre-existing skip). The three new cross-committee
invariant tests pin the properties reviewed above. C4 is done; S4a in
[plan_s1_scale.md](../../plans/plan_s1_scale.md) is unblocked and S4b can
proceed once C4's downstream `measure_baseline.py` re-run against the
new version-pinning scheme has landed for `companies-small`.
