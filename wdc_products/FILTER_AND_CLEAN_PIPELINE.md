# WDC Products Filtering Pipeline

This document summarizes the data cleaning and filtering steps implemented in
`wdc_products/*.py`, along with the current output counts from the generated
artifacts in `wdc_products/`.

## Inputs

- JSONL inputs: 46 files in `wdc_products/data/*.jsonl`
    - All files from WDC Products Multi-class
- Total JSONL lines: 21,981,131

## Steps and outputs

### Step 1: `clean_data_step1.py`

Purpose:
- Load all JSONL files in `wdc_products/data/`.
- Drop rows with missing `id` or `description`.
- De-duplicate by `id`.
- Drop clusters with fewer than 4 entries (and `NULL` cluster_id).

Outputs:
- SQLite cache at `wdc_products/data/dedup.sqlite`.

Observed counts (via `get_counts.py`):
- `dedup.sqlite`: 324,110 records; 61,085 clusters

### Step 2: `filter_clusters_step2.py`

Purpose:
- Read the deduped SQLite DB.
- Identify clusters for categories (gpu, ssd, hdd, sticks) using specific keyowrds with FTS or LIKE.
- Keep clusters that match exactly one category.

Outputs:
- Filtered SQLite DB at `wdc_products/data/dedup_filtered.sqlite`.

Observed counts:
- `dedup_filtered.sqlite`: 14,364 records; 2,376 clusters

### Step 3: `sqlite_to_json_step3.py`

Purpose:
- Export cluster-grouped rows from SQLite into JSON.

Outputs:
- `wdc_products/clusters_filtered_1.json`

Counts:
- Clusters: 2,376
- Offers: 14,364

### Step 4: `filter_clusters_step4.py`

Purpose:
- Rule-based scoring and filtering by category-specific keywords.
- Reject clusters with too few positives or low positive ratio.
- Keep top 4 offers per cluster.

Outputs:
- `wdc_products/clusters_filtered_2.json`
- `wdc_products/clusters_filtered_2_rejected.json`

Counts:
- Kept: 887 clusters, 3,238 offers
- Rejected: 1,489 clusters, 8,563 offers

### Step 5: `filter_clusters_llm_step5.py`

Purpose:
- LLM-based binary filter (KEEP/DROP) per cluster.
- Uses OpenAI Responses API with strict instruction to keep only true clusters.

Outputs:
- `wdc_products/clusters_filtered_3.json`
- `wdc_products/clusters_filtered_3_rejected.json`

Counts:
- Kept: 812 clusters, 3,012 offers
- Rejected: 75 clusters, 226 offers

### Step 6: `distribute_offers_round_robin_step6.py`

Purpose:
- Distribute offers across 4 buckets (round-robin per cluster) for extraction.

Outputs:
- `wdc_products/offers_1.json`
- `wdc_products/offers_2.json`
- `wdc_products/offers_3.json`
- `wdc_products/offers_4.json`

Counts:
- offers_1: 812
- offers_2: 812
- offers_3: 762
- offers_4: 626

### Step 7: `extract_offers_open_schema_step7.py`

Purpose:
- LLM extraction into open-schema attributes (flat snake_case).
- Writes structured offers and key frequency suggestions.

Outputs:
- `wdc_products/structured_offers_1.json` .. `structured_offers_4.json`
- `wdc_products/schema_suggestions_offers_1.json` .. `schema_suggestions_offers_4.json`

Counts (with default `--limit 20`):
- structured_offers_1: 20
- structured_offers_2: 20
- structured_offers_3: 20
- structured_offers_4: 20
