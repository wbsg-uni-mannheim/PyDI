# statistics/

Central XLSX reporting for the variant pipeline — one workbook per domain. Re-generate after every new baseline / variant / validate run.

## Regenerate

```
source pydi-dev/bin/activate
PYTHONPATH=. python usecases_synthetic/scripts/build_statistics.py
```

Or for a specific subset of domains:

```
python usecases_synthetic/scripts/build_statistics.py --domain music --domain products
```

## File layout

Each workbook has six sheets:

1. **sizes** — source row counts across `baseline` + `easy` + `medium` + `hard`.
2. **splits** — combined train / val / test breakdown: per-EM-pair `total` / `positive` / `negative` / `pos_rate` plus a fusion (validation / test) entity-count block at the bottom. Variants use the regenerated EM gold copies; missing files (e.g. games has no EM val splits authored at any level) emit blanks.
3. **examples** — 10 entity clusters per domain, selected by **value-set drift** between baseline and hard (Jaccard distance over record values, ignoring column names — so K8 column renames do not dominate selection). Each cluster expands to every configured source-pair edge × 4 levels, rendered as `k=v; k=v; ...` strings. Reading down a record column makes K1/K5/K6 value mutations + K10 corruption + K3 `<dropped>` records directly visible.
4. **transformations** — same 10 clusters as `examples`, but rendered per-record and per-field. For each cluster member, every field is one row with values across baseline / easy / medium / hard side-by-side. Field alignment is position-based against the baseline column order (K8 renames preserve position), so the canonical baseline column name labels each row regardless of how the level renames it. `<dropped>` fills the row when a record is K3-dropped at a level.
5. **committee_summary** — per-stage (SM, Norm, EM-blocking, EM-matching, Fusion) committee macro headline + best-member value + best-member name, across all 4 levels.
6. **per_member** — every committee member's headline metric value across all 4 levels. Grouped by stage with a section header row.

## Where the data comes from

- Baselines: `usecases_synthetic/baselines/<domain>/baseline_metrics.json`
- Per-level metrics: `usecases_synthetic/validation/<domain>/<level>/metrics.json`
- Source / EM / fusion files: `usecases/<domain>/input/` for baseline, `usecases/<domain>-augmented/<level>/input/` for variants. Products honors `data_root: usecases_synthetic/usecases` per [config/domains/products.yaml](../config/domains/products.yaml).

## Source code

[scripts/build_statistics.py](../scripts/build_statistics.py).
