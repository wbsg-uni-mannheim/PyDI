# Module 3: EM Committee Runner (+ Pool Diagnostic)

## Purpose

Concrete `EMCommitteeRunner` that runs the M1 `em_committee.yaml` roster against a `VariantBundle`, measures per-source-pair and macro-averaged F1 against the test gold, and **additionally** computes the committee-vs-pool agreement diagnostic required by [cross_cutting.md § Protection set semantics](../../knobs/cross_cutting.md#protection-set-semantics-not-replacement-gold) — point 3: *"reported F1 stays on the test gold; additionally compute a secondary committee-vs-pool agreement rate as a diagnostic"*.

This is the most complex committee runner. It must support multiple blocker/matcher combinations, three source-pair partitions (companies: forbes↔dbpedia, forbes↔fullcontact, dbpedia↔fullcontact), and both the test-gold metric and the pool-agreement diagnostic side by side.

## Spec References

- **Target stage knobs:**
  - [knob_01_surface_augmentation.md](../../knobs/knob_01_surface_augmentation.md) § "Committee expectations" — lexical vs embedding blocker spread
  - [knob_02_niche_density.md](../../knobs/knob_02_niche_density.md) § "Committee expectations" — similarity-threshold vs learned matchers
  - [knob_03_attribute_drop.md](../../knobs/knob_03_attribute_drop.md) § "Committee expectations" — rule-based vs missing-value-tolerant
  - [knob_06_value_noise.md](../../knobs/knob_06_value_noise.md) § "Committee expectations" — n-gram blockers vs embedding blockers; rule vs learned comparators
- **Pool diagnostic requirement:** [cross_cutting.md § Protection set semantics](../../knobs/cross_cutting.md#protection-set-semantics-not-replacement-gold) — "committee-vs-pool agreement rate … never a reported number … used to disambiguate test-gold collapse"
- **PyDI EM module:** [../../PyDI/entitymatching/](../../PyDI/entitymatching/) — `StandardBlocker`, `EmbeddingBlocker`, `RuleBasedMatcher`, `StringComparator`, `MaximumBipartiteMatching`, `EntityMatchingEvaluator`
- **Reference companies workflow:** [../../tests/companies_test/test_workflow_companies.py](../../tests/companies_test/test_workflow_companies.py) — validated blocker/matcher stack for companies; use as one committee member and the starting point for others
- **Pool files:** [../../usecases_synthetic/pools/companies/pooled_positives.csv](../../usecases_synthetic/pools/companies/pooled_positives.csv)
- **EM gold files:** `usecases/companies/input/entitymatching/<src1>_2_<src2>_{train,val,test,all}.csv`
- **M0 infrastructure:** [module_00_infrastructure.md](module_00_infrastructure.md)
- **M1 roster:** [module_01_committee_spec.md](module_01_committee_spec.md) — `em_committee.yaml`

## Files to Create

### Library (`usecases_synthetic/lib/`)

| File | Responsibility |
|---|---|
| `committee_em.py` | `EMCommitteeRunner(CommitteeRunner)`. For each source pair and each roster member, run `blocker.materialize() → matcher.match() → (optional) MaximumBipartiteMatching.cluster()`, score against test gold, and compute pool-agreement diagnostic against pooled positives |
| `committee_em_scoring.py` | `score_em_correspondences(pred, gold) -> dict[str, float]` (P/R/F1 over `{(id1, id2)}` sets); `pool_agreement(pred, pool) -> dict[str, float]` (fraction of pred pairs that appear in the pool, fraction of pool pairs covered by pred) |

### Tests (`usecases_synthetic/tests/`)

| File | What it tests |
|---|---|
| `test_committee_em.py` | `EMCommitteeRunner` runs a tiny roster (1 standard+rule, 1 embedding+rule) on a 2-source fixture with synthetic gold; produces per-pair F1, macro F1, and pool-agreement numbers; identical pred→gold gives F1=1.0; random pred gives F1 ≈ 0 |

## Interfaces

```python
# usecases_synthetic/lib/committee_em.py
class EMCommitteeRunner(CommitteeRunner):
    stage: Literal["em"] = "em"

    def __init__(self, roster_path: Path, with_llm: bool = False,
                 clustering: Literal["none", "mbm"] = "mbm"):
        """Load roster; filter LLM members unless with_llm."""

    def run(self, bundle: VariantBundle) -> CommitteeResult:
        """
        For each source pair (src1, src2) in bundle.em_gold:
            for each roster member (blocker_spec, matcher_spec):
                candidates = build_blocker(blocker_spec, ...).materialize()
                preds = build_matcher(matcher_spec, ...).match(...)
                if clustering == "mbm":
                    preds = MaximumBipartiteMatching().cluster(preds)
                f1 = score_em_correspondences(preds, bundle.em_gold[(src1, src2)])
                pool_agree = pool_agreement(preds, bundle.pooled_positives)
                record member + per-pair metrics

        Aggregated:
            macro F1 = mean over pairs of (mean over members of per-pair F1)
            per-member F1 = mean over pairs
            per-pair F1 = best/mean/worst across members
        Returns CommitteeResult(
            stage="em",
            per_member={member_name: MemberResult with metrics},
            aggregated={"macro_f1", "min_member_f1", "max_member_f1",
                        "macro_pool_agreement"},
            per_partition={f"{src1}_{src2}": {metric: value}},
            per_attribute={},  # not applicable at EM stage; left empty
            ...)
        """
```

## Metric semantics — reported vs diagnostic

Per [cross_cutting.md](../../knobs/cross_cutting.md#protection-set-semantics-not-replacement-gold), the runner must produce **two** numbers per committee member:

1. **Reported F1** (goes into `metrics.json`): precision/recall/F1 against the human-annotated `<src1>_2_<src2>_test.csv` gold. This is the number that appears in all downstream reports.
2. **Pool agreement** (diagnostic, goes into `metrics.json` under a separate key): fraction of predicted pairs that appear in `pooled_positives.csv`, and fraction of pool pairs recovered by predictions. Never reported as "the F1"; only used by M8 to disambiguate collapse.

If reported F1 collapses but pool agreement stays high: collapse is hidden-positive noise, not difficulty. M8 flags this and M10 triages.

## Source pair handling for companies

Companies baseline has only `forbes↔dbpedia` and `forbes↔fullcontact` test golds ([usecases/companies/input/entitymatching/](../../usecases/companies/input/entitymatching/)). There is **no** `dbpedia↔fullcontact` gold. M3 must NOT synthesize one — if a pair is missing, skip it and report the absence in `CommitteeResult.per_partition` with an `f1=None` marker. M1's `em_committee.yaml` reflects this (only two pairs are enabled for companies).

For variants, K2 regenerates test sets per level. The loader (M0) reads the regenerated `<src1>_2_<src2>_test.csv` from `usecases/companies-augmented/<level>/input/entitymatching/` so this is transparent.

## Threshold and seed policy

- **Threshold per matcher** is fixed at baseline measurement time and stored in the roster YAML. Variants use the same threshold. Do NOT re-tune.
- **Blocker batch_size** identical across runs for reproducibility.
- **Seeds** for embedding-based blockers/matchers taken from a committee-wide seed stored in the roster.
- **Torch determinism:** set `torch.use_deterministic_algorithms(True)` + seed `numpy`/`torch` before any embedding model call — mirror the pattern in [test_workflow_companies.py:33-35](../../tests/companies_test/test_workflow_companies.py#L33-L35).

## Acceptance Criteria

1. `EMCommitteeRunner` runs a 2-member fixture roster on companies `forbes↔dbpedia` (both baseline and M6-produced hard variant) end-to-end.
2. `CommitteeResult.per_member` contains per-pair F1 and per-pair pool-agreement for each enabled member.
3. Identical-pred-to-gold smoke test: F1 == 1.0, pool_agreement precision == 1.0.
4. Missing-pair graceful degradation: `dbpedia↔fullcontact` (no gold) produces a `per_partition` entry with `f1=None` and no crash.
5. Torch determinism: two sequential runs produce identical F1 numbers (to machine epsilon).
6. `pydi-dev/bin/pytest usecases_synthetic/tests/test_committee_em.py -v` passes.

## Dependencies

M0 (infrastructure), M1 (roster). Must not depend on M4 (fusion).

## Notes

- Do NOT reinvent `EntityMatchingEvaluator` — use it where it gives us the metric shape we need. If it only exposes a single F1 per run rather than (P, R, F1) tuple, wrap it or compute directly; do not force downstream code to re-implement F1.
- Pool agreement is deliberately an *asymmetric* diagnostic: pool-precision (what fraction of preds match the pool) and pool-recall (what fraction of the pool is covered). Both numbers go into metrics.json.
- Companies pool size is 2803 ([PIPELINE.md line 46](../../usecases_synthetic/PIPELINE.md#L46)). Pool-recall across all pairs will be low because test gold is much smaller; this is expected.
- Budget-wise, EM with embedding blockers on companies (thousands of entities per source) is the most expensive step per run. Caching embeddings to disk is mandatory — reuse [usecases_synthetic/cache/](../../usecases_synthetic/cache/) conventions from knob 2 embeddings if a pattern already exists; otherwise add `usecases_synthetic/cache/committee_em_embeddings/`.
