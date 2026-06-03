# Best-of-breed pipeline — run commands (2026-06-01)

Per the 2026-06-01 directive:

- **Schema matching keeps LLM members** (llm_matcher).
- **Entity matching drops LLM members** (llm_matcher / matchgpt / comem).
- **Fusion drops LLM members** (llm_only / llm_judge).
- **Fusion evaluation in the comparison report** uses the SAME
  `DataFusionEvaluator` + per-attribute strategy as each human-baseline
  workflow notebook (per-attribute rules captured in
  [pipelines/lib/notebook_fusion_eval.py](lib/notebook_fusion_eval.py)).

The CLI exposes per-stage LLM toggles + a `--variant` flag for
augmented bundles. Defaults reflect the directive so no extra flags
are needed for a baseline run:

```
--llm-sm / --no-llm-sm        (default: --llm-sm)
--llm-em / --no-llm-em        (default: --no-llm-em)
--llm-fusion / --no-llm-fusion (default: --no-llm-fusion)
--no-llm                      (legacy global kill-switch)
--variant {baseline,easy,medium,hard}   (default: baseline)
```

## Data sources (2026-06-01)

All five domains read evaluation gold from canonical
`usecases/<domain>/` (NOT `usecases_synthetic/`). Verify with:

```bash
python pipelines/scripts/audit_data_sources.py
```

| Domain    | Bundle source | Variants available |
|-----------|---------------|--------------------|
| products  | canonical     | baseline + easy / medium / hard (via `usecases/products-augmented/`) |
| music     | canonical     | baseline + easy / medium / hard (via `usecases/music-augmented/`) |
| games     | canonical     | baseline only |
| companies | canonical     | baseline only |
| papers    | canonical     | baseline only (no augmented tree yet) |

The products + papers canonical loaders are domain-specific
(`pipelines/lib/canonical_loader.py`); other domains route through
`variant_loader.load_variant` (since they have no synthetic
`data_root` override).

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

## Papers (new 2026 domain)

Sources: dblp (anchor for fusion), crossref, open_alex. IDs are
auto-generated by `PyDI.io.load_json(add_index=True)` as
`<source>-NNNNN`. Source pairs: (dblp, crossref) and (dblp, open_alex).
Fusion gold uses `doi` as the join key.

Papers is not in `usecases_synthetic`'s `VALID_DOMAINS`; the bundle is
loaded via a dedicated `canonical_loader` entry. SM gold is not yet
authored — the SM committee will skip its scoring step until
`usecases/papers/input/schemamatching/sm_mapping_gold.csv` is written.
Committee YAMLs fall back to the unsuffixed canonical
`em_blocking_committee.yaml` / `em_matching_committee.yaml` /
`fusion_committee.yaml`; author papers-specific YAMLs for a serious
run.

```bash
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/papers.yaml \
    --out pipelines/papers/run_$(date -u +%Y%m%d_%H%M%S)/ \
    --mode replay
```

Cache + compare:

```bash
python pipelines/scripts/compare_to_human_baseline.py \
    --domain papers \
    --cache-from-notebook usecases/papers/papers_workflow_minimal.ipynb

python pipelines/scripts/compare_to_human_baseline.py \
    --domain papers \
    --pipeline-run pipelines/papers/run_<id>/
```

## Variant runs (easy / medium / hard)

Variants live under `usecases/<domain>-augmented/<level>/`. Only the
domains with an augmented tree (`music`, `products` as of 2026-06-01)
support non-baseline levels. Pass `--variant <level>` on the CLI:

```bash
python pipelines/scripts/run_best_of_breed.py \
    --config pipelines/configs/products.yaml \
    --variant medium \
    --out pipelines/products/run_medium_$(date -u +%Y%m%d_%H%M%S)/ \
    --mode replay
```

The default `--out` already includes the variant in the directory name
(`pipelines/<domain>/run_<variant>_<ts>/`), so you can omit `--out`.

`games`, `companies`, `papers` have NO augmented tree yet; passing
`--variant easy/medium/hard` for these will fail at bundle-load time
with a clear error.

Slurm: pass `VARIANT=medium` to the sbatch wrapper:

```bash
DOMAIN=products VARIANT=medium sbatch cluster/slurm/run_best_of_breed.sbatch
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
| companies | 8 | name/country/city=`tokenized_match`; assets=`numeric_tolerance_match(tolerance=0.1)` (after a tokenized_match registration the notebook overrides — preserved); revenue=`numeric_tolerance_match(tolerance=0.1)`; keypeople=`set_equality_match` (target_schema renamed `founders → keypeople` in the 2026 refresh); founded=`year_only_match` |
| papers | 14 | doi/type/volume/issue/first_page/last_page=`exact_match`; title/journal/publisher/keywords=`tokenized_match`; authors=`set_equality_match`; publication_year=`year_only_match`; referenced_works_count/cited_by_count=`numeric_tolerance_match(tolerance=0.1)`. **Note**: the papers notebook does not register `add_evaluation_function` calls — these are sensible defaults aligned with target_schema column types + the notebook's fuser choices |

Source: [pipelines/lib/notebook_fusion_eval.py](lib/notebook_fusion_eval.py).
