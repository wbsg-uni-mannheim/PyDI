# Best-of-breed pipeline — run commands (2026-06-01)

Per the 2026-06-01 directive:

- **Schema matching keeps LLM members** (llm_matcher).
- **Entity matching drops LLM members** (llm_matcher / matchgpt / comem).
- **Fusion drops LLM members** (llm_only / llm_judge).
- **Fusion evaluation in the comparison report** uses the SAME
  `DataFusionEvaluator` + per-attribute strategy as each human-baseline
  workflow notebook (per-attribute rules captured in
  [pipelines/lib/notebook_fusion_eval.py](lib/notebook_fusion_eval.py)).

The CLI exposes per-stage LLM toggles. Defaults reflect the directive
so no extra flags are needed:

```
--llm-sm / --no-llm-sm        (default: --llm-sm)
--llm-em / --no-llm-em        (default: --no-llm-em)
--llm-fusion / --no-llm-fusion (default: --no-llm-fusion)
--no-llm                      (legacy global kill-switch)
```

## Prerequisites

Activate the dev env and make sure `OPENAI_API_KEY` is in `.env`
(loaded automatically by the run script):

```bash
source pydi-dev/bin/activate
```

The pipeline reads per-domain checkpoints from
`pipelines/<domain>/checkpoints/...` (no reuse of
`usecases_synthetic/cache/`). When the YAML default checkpoint points
at the synthetic cache, the member is loudly disabled — pass
`--ditto-checkpoint-override` / `--sc-block-checkpoint-override` to
keep it competing.

## Products

```bash
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/products.yaml \
    --out pipelines/products/run_$(date -u +%Y%m%d_%H%M%S)/ \
    --mode replay \
    --ditto-checkpoint-override pipelines/products/checkpoints/em_matching/ditto/runs/run_20260528_232623/checkpoints/best \
    --sc-block-checkpoint-override pipelines/products/checkpoints/em_blocking/sc_block/best
```

Cache the notebook baseline once (already done for products if
`pipelines/products/baselines/notebook_fused.csv` exists):

```bash
python pipelines/scripts/compare_to_human_baseline.py \
    --domain products \
    --cache-from-notebook usecases/products/products_workflow_minimal.ipynb
```

Compare:

```bash
python pipelines/scripts/compare_to_human_baseline.py \
    --domain products \
    --pipeline-run pipelines/products/run_<id>/
```

## Music

Notebook anchor source: musicbrainz (prefix `mbrainz_`).

```bash
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/music.yaml \
    --out pipelines/music/run_$(date -u +%Y%m%d_%H%M%S)/ \
    --mode replay
```

If Ditto / sc_block checkpoints have been retrained under
`pipelines/music/checkpoints/...`, add `--ditto-checkpoint-override`
+ `--sc-block-checkpoint-override`. Otherwise those members are
loudly disabled (the run still completes with the other committee
members).

Cache + compare:

```bash
python pipelines/scripts/compare_to_human_baseline.py \
    --domain music \
    --cache-from-notebook usecases/music/music_workflow.ipynb

python pipelines/scripts/compare_to_human_baseline.py \
    --domain music \
    --pipeline-run pipelines/music/run_<id>/
```

## Games

Notebook anchor source: metacritic. Caveat: the EM gold under
`usecases/games/input/entitymatching/` ships only `_train` and `_test`
splits (no `_val` or `_all`). Variant_loader serves `_test` as the
blocking gold and trains the matchers on `_train`; any committee
member that needs a held-out val surface re-thresholds against train.

```bash
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/games.yaml \
    --out pipelines/games/run_$(date -u +%Y%m%d_%H%M%S)/ \
    --mode replay
```

Cache + compare:

```bash
python pipelines/scripts/compare_to_human_baseline.py \
    --domain games \
    --cache-from-notebook usecases/games/games_workflow.ipynb

python pipelines/scripts/compare_to_human_baseline.py \
    --domain games \
    --pipeline-run pipelines/games/run_<id>/
```

## Companies

Notebook anchor source: forbes. Forbes + dbpedia ids are FULL URIs
(`http://www.forbes.com/...`, `http://dbpedia.org/...`);
fullcontact ids are short (`fullcontact_<id>`). The `source_prefix_map`
in [pipelines/configs/companies.yaml](configs/companies.yaml) reflects
this.

Companies has no per-domain committee YAMLs — the resolver falls back
to the unsuffixed `em_blocking_committee.yaml`,
`em_matching_committee.yaml`, `fusion_committee.yaml` in
`usecases_synthetic/config/committees/` (the companies-canonical roster).

```bash
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/companies.yaml \
    --out pipelines/companies/run_$(date -u +%Y%m%d_%H%M%S)/ \
    --mode replay
```

Cache + compare:

```bash
python pipelines/scripts/compare_to_human_baseline.py \
    --domain companies \
    --cache-from-notebook usecases/companies/companies_workflow.ipynb

python pipelines/scripts/compare_to_human_baseline.py \
    --domain companies \
    --pipeline-run pipelines/companies/run_<id>/
```

## What lands in `comparison.md`

Each domain's `comparison.md` now carries:

- **Part 1, Stage 1–6**: per-stage test metrics with the winning
  algorithm on each side.
- **Part 1, Stage 6 supplement (NEW)**: apples-to-apples per-attribute
  fusion accuracy. Both the best-of-breed fused frame and the cached
  notebook fused frame are scored with the SAME notebook
  `DataFusionStrategy` — exact_match, numeric_tolerance_match(tol=…),
  year_only_match, tokenized_match, set_equality_match, intersection,
  and the products-specific `hardware_strict_spec_match`. Δ column =
  best-of-breed − notebook.
- **Part 2**: the e2e composite + four-tier panel (entity_coverage,
  column_shape, cluster_correctness, value_correctness).

## Notebook-style fusion eval rules per domain

| Domain | Rules | Notable kwargs |
|---|---|---|
| products | 14 | brand/product_type=`exact_match`; 8 numerical attrs `numeric_tolerance_match(tolerance=0.15)`; 4 hardware strings `hardware_strict_spec_match` (custom) |
| music | 7 | name/artist/release-country/label/tracks=`tokenized_match`; duration=`numeric_tolerance_match(tolerance=10)`; release-date=`year_only_match` |
| games | 8 | name/platform/developer/ESRB=`exact_match`; releaseYear=`year_only_match`; criticScore=`numeric_tolerance_match(tolerance=2)`; userScore=`numeric_tolerance_match(tolerance=0.2)`; genres=`intersection` |
| companies | 8 | name/country/city=`tokenized_match`; assets=`numeric_tolerance_match(tolerance=0.1)` (after a tokenized_match registration the notebook overrides — preserved); revenue=`numeric_tolerance_match(tolerance=0.1)`; founders=`set_equality_match`; founded=`year_only_match` |

Source: [pipelines/lib/notebook_fusion_eval.py](lib/notebook_fusion_eval.py).
