# Module 1: Committee Composition Spec + YAML Configs

## Purpose

Design-only module (no executable code beyond YAML loaders). Produces three committee roster specs for companies that will be consumed by M2/M3/M4. Answers the open decisions in [plan_s1_validation.md](plan_s1_validation.md) § "Open decisions to lock in during M1".

The deliverable is a frozen choice per stage: which concrete algorithms, with which parameters, constitute "the committee" for this benchmark. Future-you must be able to justify each choice from the knob cards and existing PyDI code — not from convenience.

## Spec References

- **Committee mechanism:** [../../knobs/cross_cutting.md](../../knobs/cross_cutting.md) § "Committee-validated augmentation" and § "Committee composition (fusion, draft)"
- **Per-knob expectations** (each card § "Committee expectations"):
  - [knob_01_surface_augmentation.md](../../knobs/knob_01_surface_augmentation.md) — lexical vs embedding spread expected on EM
  - [knob_02_niche_density.md](../../knobs/knob_02_niche_density.md) — similarity-threshold vs learned matchers on EM
  - [knob_03_attribute_drop.md](../../knobs/knob_03_attribute_drop.md) — rule-based vs missing-value-tolerant matchers on EM
  - [knob_05_format_unit.md](../../knobs/knob_05_format_unit.md) — naive vs canonicalizing spread across EM and Fusion; primary target is Normalization
  - [knob_06_value_noise.md](../../knobs/knob_06_value_noise.md) — lexical/n-gram blockers vs embedding blockers; rule vs learned comparators on EM
  - [knob_08_schema_naming.md](../../knobs/knob_08_schema_naming.md) — string similarity vs instance/embedding SM matchers (primary SM target)
  - [knob_10_source_reliability.md](../../knobs/knob_10_source_reliability.md) — per-attribute voting vs per-source-trust vs entity-level provenance on Fusion
- **Algorithm tier framework:** [../../plan_algorithmselection.md](../../plan_algorithmselection.md) — tier A/B/C is a guide for implementation cost vs validation strength
- **Existing companies workflow** (ground truth of what "standard PyDI" looks like): [../../tests/companies_test/test_workflow_companies.py](../../tests/companies_test/test_workflow_companies.py)
- **PyDI matchers inventory:**
  - SM: [../../PyDI/schemamatching/](../../PyDI/schemamatching/) — `label_based`, `instance_based`, `duplicate_based`, `llm_based`
  - EM: [../../PyDI/entitymatching/](../../PyDI/entitymatching/) — `rule_based`, `ml_based`, `plm_based`, `llm_based`, `StandardBlocker`, `EmbeddingBlocker`, `StringComparator`
  - Fusion: [../../PyDI/fusion/](../../PyDI/fusion/) — `longest_string`, `shortest_string`, `voting`, `union`, `intersection`, `prefer_higher_trust`, etc.

## Files to Create

### Config (`usecases_synthetic/config/committees/`)

| File | Contents |
|---|---|
| `sm_committee.yaml` | List of SM matchers + init params. At minimum: label-based (string similarity), instance-based (value overlap / distribution), optional LLM. Each entry: `{name, class, module, params, enabled_by_default}` |
| `em_committee.yaml` | Per-source-pair list of `{blocker, matcher, comparators, threshold}` tuples. At minimum: one lexical-blocker + rule-based comparator path, one embedding-blocker + rule-based path, one embedding-blocker + embedding/LLM-matcher path. Companies has three pairs: `forbes↔dbpedia`, `forbes↔fullcontact`, `dbpedia↔fullcontact` |
| `fusion_committee.yaml` | Per-attribute strategy list. For companies: `name`, `assets`, `revenue`, `keypeople_name`, `founded`, `country`, `city`, `industry`. Each attribute: list of strategies to try. Extends the draft in [cross_cutting.md § Committee composition](../../knobs/cross_cutting.md#committee-composition-fusion-draft) |

### Design doc (`plans/validation/committee_design_notes.md`)

Narrative justification — one paragraph per stage — explaining why each committee member is in the roster, which knob cards predict it will differentiate, and which alternatives were rejected. This is the artifact we point at when a future reviewer asks "why this committee?".

### Tests (`usecases_synthetic/tests/`)

| File | What it tests |
|---|---|
| `test_committee_configs.py` | Each YAML loads; each entry references an importable class; enabled-by-default members cover the axes required by the knob cards (e.g. EM must have at least one lexical and one embedding blocker) |

## Committee axes (decision matrix)

For each stage, M1 must pick members that span these axes. This is the criterion test_committee_configs.py enforces.

### SM axes
- **Signal type:** label (column name similarity) vs instance (value distribution) vs duplicate (known correspondences)
- **Robustness to K8:** string-similarity matchers collapse on anonymized headers; instance-based matchers degrade gracefully. The spread *is* the Knob 8 difficulty signal.

### EM axes
- **Blocking:** lexical/n-gram (standard blocker on a tokenized key) vs embedding
- **Matching:** rule-based comparator aggregation vs learned/embedding matcher
- **Missing-value tolerance:** at least one matcher that silently handles NaN (needed for Knob 3's fix-on-collapse signal)

### Fusion axes
- **Cell-local vs entity-aware:** per-cell voting vs per-source trust vs per-entity provenance reasoning (Knob 10)
- **String attribute canonicalization:** at least one naive strategy and one canonicalizing strategy for each string attribute (Knob 5 spread)
- **Conflict absorption:** `longest_string` / `shortest_string` / `voting` / `most_frequent` coverage (Knob 1 spread)

## Decisions to make and record

| Decision | Default recommendation | Rationale |
|---|---|---|
| SM: include LLM matcher? | **No** by default; opt-in flag | Knob cards do call out LLM spread, but cost + non-determinism make every-run inclusion expensive. Opt-in means `validate_variant.py --with-llm` |
| EM: include PLM matcher? | **Yes** (`plm_based`) | Already used in pool construction; low extra cost since embeddings are cached |
| EM: include LLM matcher? | **No** by default; opt-in flag | Same reasoning as SM |
| Fusion: per-attribute strategy lists | Companies: name=voting+longest_string+most_frequent; numerics (assets, revenue)=median+prefer_higher_trust+mean; keypeople_name=union+voting; country=voting+most_frequent; city=voting+shortest_string+most_frequent; founded=year_only_match+voting | Spans cell-local vs trust-weighted; matches K10 expectations |
| Macro vs micro F1 for EM across source pairs | **Macro** as primary, micro as secondary | Three pairs, small pairs skew micro; macro gives each pair equal weight |
| Threshold policy | Per-matcher threshold fixed at baseline measurement time; same threshold used on all variants | Moving the threshold per variant hides the difficulty signal |
| Random seeds | All non-deterministic committee members seeded from a committee-wide seed recorded in the roster YAML | Reproducibility |

## Acceptance Criteria

1. Three YAMLs exist under `usecases_synthetic/config/committees/` and load without error via `lib/domain_config.py`-style loaders.
2. `committee_design_notes.md` has one paragraph per stage, each citing at least two knob cards.
3. Each YAML passes `test_committee_configs.py`: every class reference is importable, axis coverage is satisfied.
4. No executable committee runner exists yet (that's M2/M3/M4). M1 is design-only apart from the YAML schema and its validator.
5. `pydi-dev/bin/pytest usecases_synthetic/tests/test_committee_configs.py -v` passes.

## Dependencies

M0 (uses `CommitteeRunner` ABC and roster-loading conventions from infrastructure).

## Notes

- This is the module where you stop and check decisions with the user before M2/M3/M4 start. Getting the roster wrong wastes all subsequent modules.
- Prefer algorithms already exercised by [test_workflow_companies.py](../../tests/companies_test/test_workflow_companies.py) where possible — they're known-working on companies and give us a grounded baseline.
- The LLM opt-in flag must propagate through `validate_variant.py` and `measure_baseline.py` uniformly, so both baseline and per-level runs use the same roster.
