# Products Data Integration Workflow

End-to-end data integration pipeline for hardware product listings (GPUs, SSDs, HDDs, USB sticks) across four source datasets using the **PyDI** framework. Covers information extraction, normalization, schema matching, entity matching, and data fusion.

---

## Part 1: Information Extraction

**Datasets:** Four JSON source files (`products_1–4`) loaded via `load_json`. Each record has a free-text `title` and `description`. Dataset sizes printed and summed before extraction.

**Schema definition:** A Pydantic `ProductSchema` with ~50 typed optional fields, covering all four product types (GPU, SSD, HDD, USB_STICK). Fields grouped into: identity (`brand`, `model`, `model_number`, `product_type`, `chipset_name`), capacity (`vram`, `storage_size`), performance (`read_speed`, `write_speed`), connectivity (`bus_type`, `interface_type`), and physical attributes (`weight`, `height`, `width`, `length`, `form_factor`). Source datasets were explored (USB / SSD / HDD / RTX example CSVs) to inform schema design.

**LLM extraction (few-shot):** `LLMExtractor` with `gpt-5.2` (temperature=0). System prompt includes 4 manually crafted examples — one per product type — each pairing a raw `title + description` string with a complete ground-truth JSON extraction. Examples cover: SanDisk USB stick, Corsair Force SSD, Gigabyte RTX 3080, WD Blue HDD. Inputs formed by concatenating `title` and `description` with `. Description: `. Token counts tracked per dataset (input + output). Results saved to `output/informationextraction/LLM_extracted_products_data/`.

---

## Part 2: Data Loading and Profiling

Extracted JSON files reloaded with `load_json` and profiled with `DataProfiler.summary()` and `profiler.analyze_coverage()`.

**Dropping sparse columns:** Global density computed across all four datasets combined. A **15% threshold** was applied, with explicit exceptions for generic physical/identity attributes (`weight`, `height`, `width`, `length`, `price`, `brand`, `model`, `model_number`, `id`, `url`, `cluster_id`) kept regardless of density. Per-category coverage (GPU / HDD / SSD / USB_STICK) was also analyzed to verify type-specific attribute strengths before committing to drops.

**Dropped columns** (below 15% global coverage or below 1% overall): `L1_cache`, `L2_cache`, `L3_cache`, `ai_cores`, `dual_bios`, `upscaling_technology`, `shader_model_version`, `ray_tracing_units`, `adaptive_sync`, `blocked_slots`, `graphics_interface_speed`, `power_adapter`, `opengl_version`, `manufacturing_process`, `max_monitor_support`, `operating_temperature`, `power_draw_w`, `case_material`, `directx_version`, `controller`, `MTBF_hours`, `encryption_type`, `fan_count`, `cooling_type`, `max_resolution`, `memory_clock`, `base_core_clock`, `random_4k_read_iops`, `random_4k_write_iops`, `overclocked`, `delivery_scope`, `core_shader_count`, `chipset_series`, `protocol`, `boost_core_clock`, `memory_interface_width`, `video_out_interface`, `TBW_rating`, `storage_technology`.

Datasets saved to `output/informationextraction_then_dropped_columns/` before normalization.

---

## Part 3: Normalization

All normalization applied uniformly across all four datasets. Each step included visual distribution checks and sample inspection before committing transformations.

- **Brand** — Lowercased, stripped, mapped to canonical names via hand-built dictionary (e.g. `"wd"` → `"Western Digital"`, `"aorus"` → `"Gigabyte"`, `"hyperx"` → `"Kingston"`, `"barracuda"` → `"Seagate"`). Acronyms (`MSI`, `EVGA`, `AMD`, `ADATA`) kept uppercase via explicit list. `None`/`NaN` string artefacts from `.astype(str)` cleaned up.

- **Weight** — Per-type valid gram ranges (USB: 3–60 g, SSD: 5–300 g, HDD: 100–2000 g, GPU: 150–2000 g). Values < 10 treated as kg (×1000); values in per-type oz range converted (×28.35). Values still outside valid range after all conversion attempts nulled.

- **Storage size** — Unified to GB. HDD/SSD values < 64 multiplied by 1000 (TB → GB); USB sizes unchanged. Residual "limbo" values (HDD/SSD between 16–119 GB) inspected manually row-by-row.

- **Dimensions (height, width, length)** — Per-type unit correction: GPU cm-range values converted ×10 to mm (length < 60, height/width < 20). SSDs corrected with subtype-aware ranges (M.2 vs 2.5-inch). HDD/USB values < 5 treated as inches (×25.4); values 5–15 treated as cm (×10). Post-pass: swapped GPU height/width pairs corrected; GPU heights > 55 mm where height < length nulled; GPU lengths > 400 mm nulled; SSD heights > 15 mm nulled; second pass catches GPU rows still in cm after step 1.

- **VRAM** — Values ≥ 128 treated as MB → divided by 1024 to GB. Edge case `1.024` rounded to `1.0`.

- **Chipset name** — Mapped to canonical GPU model strings via large hand-built dictionary covering GeForce GT, GTX 1050–1660, RTX 2060–3090, Radeon RX 550–6900 series. Vague entries (`"GeForce GT"`, `"Radeon RX"`) resolved by manual lookup or set to `None`.

- **Bus type** — Normalised to canonical labels (e.g. `"PCIe 3.0 x16"` → `"PCI Express x16"`, `"SATA III"` → `"SATA"`, `"USB 3.1 Gen 1"` → `"USB 3.0"`). Unmapped residuals nulled. Applied in a late-stage comprehensive cleaning pass that also normalised `form_factor`, `storage_connection_type`, `interface_type`, and stripped encoding artefacts from `title`/`description`.

- **Price** — Scientific notation cleaned; non-numeric coerced to NaN; zero/negative values nulled; rounded to 2 decimal places.

Normalized datasets saved to `output/normalized_products_after_drp_cols_and_extractions/`.

---

## Part 4: LLM-Based Schema Matching

Target schema (`products_target_schema.json`) loaded and used with `LLMBasedSchemaMatcher` (`gpt-5.2`, 40 sample rows) to produce per-dataset column mappings. `SchemaTranslator` applied without re-normalization (already complete). Second profiling pass with coverage analysis and per-dataset HTML reports generated post-translation.

---

## Part 5: Entity Matching

### Blocking

**Topology: Star schema** — products_1 as hub, linked pairwise to products_2, products_3, and products_4. Consistent with all reference notebooks.

**Blocker: `StandardBlocker` on `['product_type']`** — candidates generated only within the same category (GPU / SSD / HDD / USB_STICK). Achieving Pair Completeness: 1.000 but sacrificing a bit on 'reduction_ratio' which is fine s o grouping errors doesnt cascade downards. Evaluated with `EntityMatchingEvaluator.evaluate_blocking_batched()` against validation ground truth splits.

Three blockers: `standard_blocker_p1_p2`, `standard_blocker_p1_p3`, `standard_blocker_p1_p4`.

### Comparators

Four comparators shared across all matchers:

| Comparator | Column | Method | Preprocessing |
|---|---|---|---|
| `StringComparator` | `title` | Sørensen-Dice, word tokens | Strip units (GB, TB, SSD, NVMe, SATA, HDD), remove punctuation, lowercase |
| `StringComparator` | `brand` | Jaccard, word tokens | `str.lower` |
| `StringComparator` | `product_type` | Jaccard, word tokens | `str.lower` |
| `NumericComparator` | `storage_gb` | Relative difference | max_difference=0.1 |

Sørensen-Dice on `title` outperformed Jaccard after experimentation — better at handling word-order variation between listings from different sources.

### Rule-Based Matcher

`RuleBasedMatcher` with weights `[0.30, 0.30, 0.30, 0.10]` (title, brand, product_type, storage_gb) and threshold `0.70`, applied to all three pairs. Multiple weight configurations tested; this gave the best F1 balance across all pairs. Evaluated with `EntityMatchingEvaluator.evaluate_matching()`. Cluster distributions written per pair via `write_cluster_details()`.

### Post-Processing (1:1 Enforcement)

Both `GreedyOneToOneMatchingAlgorithm` and `MaximumBipartiteMatching` applied and compared on all three pairs. Greedy raised precision with minor recall cost. Refined correspondences used for fusion.

### ML-Based Matcher

`MLBasedMatcher` + `RandomForestClassifier` (100 trees, `random_state=42`). One classifier trained per pair on `prod1_to_prodX_train.csv` using `FeatureExtractor` over the same four comparators. Probability threshold: 0.45. Evaluated on `_test.csv` splits.

---

## Part 6: Data Fusion

### Trust Score Determination


The audit strategy uses:
- `exact_match` for identity attributes (`brand`, `product_type`, `model_number`)
- `numeric_tolerance_match` (15% tolerance) for numeric specs and dimensions
- A custom `hardware_strict_spec_match` for technical strings — extracts and compares digit sequences first (prevents `"PCIe x8"` matching `"PCIe x16"`), then checks cleaned substring equality. it improves scores a bit but not too much

Each source evaluated on the subset of validation rows where it appears (`source_left == name` or `source_right == name`). Since P1 is always `id_left` in the validation set (star schema with P1 as hub), the ID column mapping is stable: P1 uses `id_left`, all others use `id_right`.

Trust scores assigned based on overall accuracy ranking:

```python
products_1_cleaned.attrs["trust_score"] = 3  # most reliable
products_2_cleaned.attrs["trust_score"] = 2
products_3_cleaned.attrs["trust_score"] = 1  # least reliable
products_4_cleaned.attrs["trust_score"] = 2
```

P1 ID anchored as master: `products_1_cleaned["p1_id"] = products_1_cleaned["id"]`

### Correspondences

```python
all_correspondences = pd.concat([
    correspondences_p1_p2_refined,
    correspondences_p1_p3_refined,
    correspondences_p1_p4_refined
], ignore_index=True)
```

### Fusion Strategy

```python
strategy = DataFusionStrategy('hardware_fusion_strategy')

# Identity — vote across sources
for attr in ['brand', 'product_type', 'model_number']:
    strategy.add_attribute_fuser(attr, voting)

# Performance specs — take minimum (conservative; avoids inflated marketing numbers)
for attr in ['vram_gb', 'storage_gb', 'read_speed_mb_s', 'write_speed_mb_s']:
    strategy.add_attribute_fuser(attr, minimum)

# Technical strings and dimensions — defer to highest-trust source
for attr in ['chipset_name', 'bus_type', 'interface_type', 'storage_connection_type',
             'memory_type', 'form_factor', 'width_mm', 'length_mm', 'height_mm', 'weight_g']:
    strategy.add_attribute_fuser(attr, prefer_higher_trust, trust_key="trust_score")
```

### Fusion Execution

```python
engine = DataFusionEngine(strategy, debug=True, debug_format='json',
                          debug_file=OUTPUT_DIR / "data_fusion" / "hardware_fusion_debug.jsonl")

fused = engine.run(
    datasets=[products_1_cleaned, products_2_cleaned, products_3_cleaned, products_4_cleaned],
    correspondences=all_correspondences,
    id_column="id",
    include_singletons=False,
)
```

### Evaluation

Evaluation run on **manually verified rows only** (`filled == 'y'`). Numeric columns in the gold set explicitly cast to numeric before evaluation.

Evaluation functions mirror the audit strategy:
- `exact_match` — `brand`, `product_type`
- `numeric_tolerance_match` (15%) — all numeric specs and dimensions
- `hardware_strict_spec_match` — `chipset_name`, `bus_type`, `interface_type`, `memory_type`

```python
evaluator = DataFusionEvaluator(strategy, debug=True,
    debug_file=OUTPUT_DIR / "data_fusion" / "hardware_eval_debug.jsonl", debug_format="json")

evaluation_results = evaluator.evaluate(
    fused_df=fused,
    fused_id_column='p1_id',
    gold_df=test_set_filled,
    gold_id_column='id_left'
)
```

**Validation vs Test discipline:** Validation set (`fusion_validation_set.csv`) used throughout development for trust score determination and strategy tuning. Test set (`fusion_test_set.csv`) reserved for the single final evaluation run only.

---

## File Structure

```
products/
├── input/
│   ├── data/
│   │   ├── products_1.json
│   │   ├── products_2.json
│   │   ├── products_3.json
│   │   └── products_4.json
│   ├── few_shot_examples.json
│   ├── schemamatching/
│   │   └── products_target_schema.json
│   ├── entitymatching/
│   │   └── per_file_splits/
│   │       ├── prod1_to_prod2_{train,val,test}.csv
│   │       ├── prod1_to_prod3_{train,val,test}.csv
│   │       └── prod1_to_prod4_{train,val,test}.csv
│   └── fusion/
│       ├── fusion_validation_set.csv
│       └── fusion_test_set.csv
└── output/
    ├── informationextraction/
    │   └── LLM_extracted_products_data/
    ├── informationextraction_then_dropped_columns/
    ├── normalized_products_after_drp_cols_and_extractions/
    ├── dataset-profiles/
    ├── Blocking/
    │   ├── standard_blocker_on_product_type/
    │   └── blocking_eval_prod1_prod{2,3,4}/
    ├── debug_results_entity_matching/
    ├── cluster_analysis/
    ├── logs/
    └── data_fusion/
        ├── hardware_fusion_debug.jsonl
        └── hardware_eval_debug.jsonl
```