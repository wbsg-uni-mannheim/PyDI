# PyDI Entity Matching Curator (Streamlit)

The entity matching curator provides an interactive workflow for sampling and labelling
candidate record pairs directly on top of PyDI's blocking and matching components. It is
packaged as a Streamlit application so that the full stack remains Python-first.

## Prerequisites

- Install PyDI together with the optional UI dependencies:

  ```bash
  pip install -e .[ui]
  ```

  Alternatively install Streamlit manually (`pip install streamlit`) if you are working
  inside an existing environment.

- Ensure that the optional embedding dependencies are present (already part of the base
  PyDI installation). If you plan to use the embedding blocker, having `torch` with CPU
  support is sufficient.

## Starting the app

From the repository root run:

```bash
streamlit run PyDI/entitymatching/streamlit_app.py
```

The app runs locally on `http://localhost:8501` and supports Streamlit's standard CLI
flags (for instance `--server.headless true` for remote servers or `--server.port 8503`
to avoid port conflicts).

## Workflow overview

1. **Load data**  
   Use the sidebar to upload “left” and “right” datasets (CSV/TSV/JSON/Parquet) or start
   with the bundled movie demo. Choose the shared identifier column that uniquely names
   records on both sides.

2. **Configure blocking**  
   Combine any of the existing PyDI blockers: standard key-based blocking, token
   overlap, sorted neighbourhood, and the ANN-based embedding blocker. Multiple
   blockers are unioned automatically and the epsilon random explorer keeps recall high
   by sampling a tiny fraction of the Cartesian space.

3. **Configure comparators**  
   Pick the string, numeric, and/or date columns that should feed into the
   `RuleBasedMatcher`. The UI exposes comparator weights so you can steer the final
   similarity score without editing code.

4. **Tune thresholds**  
   - `tau_low` → lower bound for “easy non-matches”
   - `tau_high` → upper bound for “easy matches”
   - `corner share` → fraction of the batch reserved for corner cases (default 30%)
   - `matcher threshold` → minimum score returned by the rule matcher (defaults to 0 to
     keep every scored pair, which is recommended when iterating interactively)

5. **Run the sampler**  
   The curated batch combines 35/35/30 bins by default (easy positives / easy negatives /
   corner cases), ensures at least 40% of the corner pool are blocker disagreements, and
   up-weights rare entities to avoid popularity bias. Each row includes:

   - provenance (`source_blockers`)
   - matchness score (weighted union of embedding similarity, key overlap, and rule
     score)
   - entropy-based uncertainty
   - rarity score (`1 / (degree_left + degree_right)`)
   - short previews of the records on both sides

6. **Label & export**  
   Use the editor in the “Curated Batch” table to assign labels (`match`, `non_match`,
   `unsure`). The “Download labels” button exports the current annotations as CSV for
   downstream training or QA.

## Extensibility

- The sampling backend lives in `PyDI/entitymatching/interactive.py`. You can import the
  `run_interactive_matching` function directly from Python notebooks or services to
  reproduce the same logic without the Streamlit UI.
- The Streamlit app is intentionally modular. If you want to add bespoke block sources
  or new comparator types, extend the helper functions in `streamlit_app.py`:
  - `_build_blocking_config` – surface new candidate generation strategies
  - `_build_comparator_config` – expose additional comparator families
- For automation, take the `MatchingSessionResult.sampled_batch` DataFrame and push it to
  your own task queues or persisted stores (the object already contains the full scoring
  metadata).

## Troubleshooting

- **No candidates returned**  
  Double-check the blocking keys/thresholds. Try loosening the embedding threshold or
  enabling epsilon random exploration (`0.0005` is a safe starting value).

- **Missing dependencies**  
  If the app raises `ImportError: streamlit`, install the optional extras
  (`pip install -e .[ui]`). For embedding-based blocking ensure that
  `sentence-transformers` and `torch` are available.

- **Large datasets**  
  The app is optimised for interactive curation (tens of thousands of candidate pairs).
  For larger workloads consider running the pipeline headlessly via
  `run_interactive_matching` and persisting results to a database or Parquet store.

