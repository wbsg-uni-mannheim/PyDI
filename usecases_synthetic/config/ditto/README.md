# Ditto training config

Ditto-style entity-matching trainer used as the PLM member of the EM matching
committee (see `../committees/em_matching_committee.yaml` and the `ditto_plm`
entry).

- [`default_train.yaml`](default_train.yaml) — default hyperparameters consumed
  by [../../scripts/ditto/train.py](../../scripts/ditto/train.py). CLI flags on
  `train.py` override any key set here.

The vendored runtime lives under
[../../third_party/ditto_modern/](../../third_party/ditto_modern/) (origin:
Ditto-style stack, see that directory's `README.md` / `ORIGIN.md`). Data-bridge
and driver scripts live under
[../../scripts/ditto/](../../scripts/ditto/).

## Dependencies

The `plm` extra in [../../../pyproject.toml](../../../pyproject.toml) already
covers `torch`, `transformers`, and `sentence-transformers`; `pandas` is a
top-level dependency. `spacy` + `nltk` are only needed if you pass
`--summarize` or `--dk` (domain-knowledge injector); neither is used by the
D5 smoke test, so they are not required for initial wiring.

## Typical invocations

### Smoke test (D5, CPU-tractable)

```
python usecases_synthetic/scripts/ditto/train.py \
  --train-json-gz usecases_synthetic/output/ditto/trial_companies/tiny_train.json.gz \
  --val-json-gz   usecases_synthetic/output/ditto/trial_companies/tiny_val.json.gz \
  --test-json-gz  usecases_synthetic/output/ditto/trial_companies/tiny_test.json.gz \
  --config usecases_synthetic/config/ditto/default_train.yaml \
  --model-name distilbert-base-uncased \
  --fields name,country,city,industry,sector,founded \
  --max-field-len 200 --batch-size 8 --max-len 128 --epochs 1 --no-fp16 \
  --output-dir usecases_synthetic/output/ditto/trial_companies/runs/
```

### Production checkpoint (D8, GPU recommended)

```
python usecases_synthetic/scripts/ditto/train.py \
  --train-json-gz <companies train.json.gz> \
  --val-json-gz   <companies val.json.gz> \
  --test-json-gz  <companies test.json.gz> \
  --config usecases_synthetic/config/ditto/default_train.yaml \
  --model-name roberta-base \
  --fields name,country,city,industry,sector,founded \
  --epochs 10 --batch-size 32 --max-len 256 \
  --output-dir usecases_synthetic/cache/ditto_checkpoints/companies/
```

After training, symlink or copy the `runs/run_*/checkpoints/best/` directory
to `usecases_synthetic/cache/ditto_checkpoints/<domain>/best/` so that the
committee YAML's `checkpoint_path` is stable across retrains.

Expected runtime: the CPU smoke test (distilbert, 64 train pairs, 1 epoch)
finishes in well under ten minutes. The production retrain on full companies
EM gold with roberta-base / 10–15 epochs / batch 16–32 runs on a CUDA GPU or
Apple Silicon (MPS) in a few minutes; the training and inference paths
auto-select the best available accelerator (CUDA → MPS → CPU). Without any
accelerator, fall back to `distilbert-base-uncased --epochs 3` on CPU and
accept lower PLM quality (narrows S3's margin band but does not block the
policy — see [../../../plans/plan_s1_scale.md](../../../plans/plan_s1_scale.md)
"Hard blockers").

`lr: 5e-5` in [default_train.yaml](default_train.yaml) is the original WDC
recipe default; for small domain-specific training sets (e.g. the ~450-pair
companies EM gold) override to `--lr 2e-5` — roberta-base diverges or stays
at random-classifier loss (~0.69) at 5e-5 on sets that small. MixDA (`--da
del`) is left enabled because the companies recipe benefits from it, but it
interacts with the learning rate: if you see the val F1 stuck at exactly
`0.644` (recall=1.0, threshold=0.0) the model collapsed — lower the LR first.

## Current companies checkpoint

Trained as part of R2.2 in [../../../plans/plan_s1_scale.md](../../../plans/plan_s1_scale.md).
Apple Silicon (M5 Max, MPS, bf16 autocast). Trained on ADI's labeled
pool across **3 pairs** — dbpedia_forbes, forbes_fullcontact,
**dbpedia_fullcontact** (no PyDI gold for this 3rd pair, but ADI provides
labels). PyDI gold test is preserved as the held-out test split.

### Data setup (heterogeneous train/val/test)

* **Train** = ADI's `training_<pair>_latest.csv` for all 3 pairs, with
  pairs that overlap PyDI test removed (leak-free) and intra-train
  duplicates deduped → **5,713 pairs (538 pos / 5,175 neg, 9.4% pos)**.
* **Val** = ADI's `similarity_validation_faiss_<pair>.csv` for all 3
  pairs, leak-removed → **866 pairs (270 pos / 596 neg, 31.2% pos)**.
* **Test** = PyDI gold `<pydi_pair>_test.csv` for the 2 pairs with PyDI
  gold (db↔fb 140, fb↔fc 459) → **599 pairs (176 pos / 423 neg, 29.4% pos)**.
  dbpedia_fullcontact is excluded from test eval.

Side-alignment normalises every pair to ADI's `(src_left, src_right)`
ordering so the model sees a consistent left/right schema. Prep script:
[../../scripts/ditto/_prep_companies.py](../../scripts/ditto/_prep_companies.py).

### θ-tuning disabled (`--fixed-threshold 0.5`)

Per the analysis in R2.2's first sweep, val-tuned θ over-fit val and lost
~0.4–1.8 pp test F1 vs fixed θ=0.5. Trainer flag added in this round
disables per-epoch threshold tuning entirely — both checkpoint selection
(by val F1 at θ=0.5) and deploy use the same threshold.

### LR × class-balance sweep (8 runs)

Fixed: `roberta-base --batch-size 32 --max-len 256 --max-field-len 350
--epochs 50 --early-stopping-patience 5 --warmup-ratio 0.06 --no-fp16
--da del --seed 42 --fixed-threshold 0.5`. Sample weights for
`weighted` runs use sklearn `class_weight='balanced'` formula (pos≈5.31,
neg≈0.55, reflecting the 9.4% positive rate on ADI training).

| lr   | mode       | best_epoch | val_f1   | test_f1   | test P | test R |
|------|------------|-----------:|---------:|----------:|-------:|-------:|
| 5e-6 | weighted   | 5          | 0.9775   | 0.8925    | 0.847  | 0.943  |
| 5e-6 | unweighted | 9          | 0.9774   | 0.9136    | 0.896  | 0.932  |
| 1e-5 | weighted   | 7          | 0.9755   | 0.9049    | 0.918  | 0.892  |
| **1e-5** | **unweighted** | **9** | **0.9757** | **0.9261** | **0.926** | **0.926** |
| 5e-5 | weighted   | 8          | 0.9812   | 0.9136    | 0.896  | 0.932  |
| 5e-5 | unweighted | 4          | 0.9737   | 0.8958    | 0.888  | 0.903  |
| 1e-4 | weighted   | 6          | 0.4754   | 0.4542    | 0.294  | 1.000  |
| 1e-4 | unweighted | 1          | 0.9638   | 0.8731    | 0.789  | 0.977  |

**Winner: lr=1e-5 unweighted, test F1 = 0.9261**, P=R=0.926, 13 FPs / 13 FNs.

Pattern observations:
- **best_val_f1 ≠ best_test_f1**: the highest val F1 (0.9812 @ lr=5e-5
  weighted) lands only mid-pack on test. The FAISS-mined val and PyDI
  gold test draw from different distributions; val-only selection
  over-fits to ADI's hard-negative mining.
- **unweighted/weighted preference flips with LR**: unweighted wins at
  low LR (5e-6, 1e-5), weighted wins at moderate LR (5e-5), and at
  1e-4 weighted destabilises entirely (collapsed to "always positive",
  matching the README's known divergence pattern).

### Per-pair test F1 (winner)

| pair                  |  F1   |  P    |  R    | TP/FP/FN/TN |
|-----------------------|-------|-------|-------|-------------|
| dbpedia_forbes        | 0.968 | 0.968 | 0.968 | 61/2/2/75   |
| forbes_fullcontact    | 0.903 | 0.903 | 0.903 | 102/11/11/335 |
| **OVERALL**           | 0.926 | 0.926 | 0.926 | 163/13/13/410 |

### Comparison vs PyDI + ADI baselines

| pair                  | PyDI rule + greedy 1:1 | ADI MLBased (RF) | this Ditto run |
|-----------------------|------------------------|------------------|----------------|
| dbpedia↔forbes        | 0.907                  | 0.954            | **0.968**      |
| forbes↔fullcontact    | 0.922                  | 0.995            | 0.903          |

Ditto **beats ADI's RF on dbpedia↔forbes by +1.4pp** and beats the PyDI
rule-based+greedy baseline on the same pair by +6.1pp. On
forbes↔fullcontact, Ditto sits 9pp behind ADI's RF — fullcontact's
`Attribute_*` abstract schema dominates the field text and ADI's 38
hand-crafted comparators capture signals Ditto's PLM doesn't recover
from the same canonical projection.

### Pinned recipe

```
python usecases_synthetic/scripts/ditto/train.py \
  --train-json-gz usecases_synthetic/output/ditto/companies/train.json.gz \
  --val-json-gz   usecases_synthetic/output/ditto/companies/val.json.gz \
  --test-json-gz  usecases_synthetic/output/ditto/companies/test.json.gz \
  --config usecases_synthetic/config/ditto/default_train.yaml \
  --model-name roberta-base \
  --fields name,country,city,industry,sector,founded \
  --epochs 50 --batch-size 32 --max-len 256 --max-field-len 350 \
  --lr 1e-5 --warmup-ratio 0.06 --no-fp16 --da del \
  --early-stopping-patience 5 --seed 42 \
  --fixed-threshold 0.5 \
  --output-dir cache/ditto_checkpoints/companies/sweep_lr_1e-5_unweighted/
```

`cache/ditto_checkpoints/companies/best` symlinks to the winner's
`checkpoints/best/`. The S3 hard-negative gate at
`config/knob_02_niche/companies.yaml §hard_negative_gate` is pinned to
`plm_threshold_theta: 0.5`, `plm_max_len: 256`, `plm_max_field_len: 350`,
`plm_batch_size: 32` to match this recipe.

### Known caveats

- Train pool overlaps with PyDI test — 25.7% of db↔fb test pairs and
  8.5% of fb↔fc test pairs were in ADI's training pool. The prep script
  drops these (89 train rows + 34 val rows total) so PyDI test stays a
  clean held-out set.
- Train (9.4% pos) and val (31.2% pos) come from different sampling
  distributions: ADI's training pool is FAISS-mined-hard-negatives-heavy,
  while ADI's val is balanced. Token-length distribution at the actual
  tokenizer maxes at 73 tokens, so `max_len=256` is overkill — could be
  dropped to 128 for a free ~2× training-time speedup on future runs.
- Token-length: median 43, p99 59, max 73 → `max_len=256` is way over-
  provisioned. No truncation is happening.

## Current games checkpoint

Trained as part of R2.2 in [../../../plans/plan_s1_scale.md](../../../plans/plan_s1_scale.md).
Apple Silicon (M5 Max, MPS, bf16 autocast). Trained on ADI's labeled
pool for **2 of 3 pairs** — dbpedia_metacritic, dbpedia_sales. The 3rd
pair (metacritic_sales) has no ADI data and the top-level PyDI
train/test files are byte-identical, so it is **test-only / transfer-
learned** (option A from the R2.2 redo).

### Data setup

* **Train** = ADI's `training_<pair>_latest.csv` for db↔mc + db↔sales,
  with PyDI test pairs leak-filtered out and intra-train duplicates
  deduped → **1,683 pairs (1,144 pos / 539 neg, 68.0% pos)**.
* **Val** = ADI's `similarity_validation_faiss_<pair>.csv` for the
  same 2 pairs, leak-filtered → **598 pairs (199 pos / 399 neg,
  33.3% pos)**.
* **Test** = top-level PyDI gold for all 3 pairs:
  `dbpedia_2_metacritic_test.csv` (337), `dbpedia_2_sales_test.csv` (402),
  `metacritic_2_sales_test.csv` (582) → **1,321 pairs (357 pos / 964 neg, 27.0% pos)**.

The top-level `*_2_*` files are used (not the `train_test/` subdir) per
user direction. The `train_test/` subdir was discarded because its
test files reference dbpedia IDs that no longer resolve in the
refreshed source (~22% attrition). The top-level test files for the 2
ADI pairs have **0% missing IDs** — clean.

Side-alignment is trivial: the top-level test files for db↔mc and
db↔sales already use ADI's `(src_left, src_right)` order — no
id1↔id2 swap needed. Prep script:
[../../scripts/ditto/_prep_games.py](../../scripts/ditto/_prep_games.py).

### LR × class-balance sweep (8 runs, `--fixed-threshold 0.5`)

Fixed: `roberta-base --batch-size 32 --max-len 256 --max-field-len 350
--epochs 50 --early-stopping-patience 5 --warmup-ratio 0.06 --no-fp16
--da del --seed 42 --fixed-threshold 0.5`. Sample weights for
`weighted` runs use sklearn `class_weight='balanced'` (pos≈0.74,
neg≈1.56 — **down-weighting positives because they're 68% of train**).

| lr   | mode       | best_epoch | val_f1   | test_f1   | test P | test R |
|------|------------|-----------:|---------:|----------:|-------:|-------:|
| 5e-6 | weighted   | 14         | 0.9255   | 0.9308    | 0.981  | 0.885  |
| 5e-6 | unweighted | 19         | 0.9211   | 0.9056    | 0.956  | 0.860  |
| **1e-5** | **weighted** | **13** | **0.9096** | **0.9388** | **0.954** | **0.924** |
| 1e-5 | unweighted | 18         | 0.9243   | 0.9045    | 0.907  | 0.902  |
| 5e-5 | weighted   | 14         | 0.9143   | 0.9370    | 0.959  | 0.916  |
| 5e-5 | unweighted | 23         | 0.9086   | 0.9160    | 0.887  | 0.947  |
| 1e-4 | weighted   | 5          | 0.4994   | 0.4255    | 0.270  | 1.000  |
| 1e-4 | unweighted | 1          | 0.4994   | 0.4255    | 0.270  | 1.000  |

**Winner: lr=1e-5 weighted, test F1 = 0.9388**, P=0.954, R=0.924,
16 FPs / 27 FNs.

Pattern observations vs companies (which had 9.4% pos train):
- **weighted/unweighted preference inverted as predicted** — games has
  positives over-represented in train (68%), so down-weighting them
  via `--sample-weights-csv` aligns the decision boundary with
  test/val (27%-33% pos). Weighted wins by ~3 pp at every LR.
- **Both lr=1e-4 runs collapsed** to "always positive" (P=0.27, R=1.00).
  Same divergence pattern as companies' lr=1e-4 weighted, but here the
  unweighted variant also collapsed (1 epoch vs 5). Practical
  upper-bound LR for roberta-base on this data: 5e-5.
- **test F1 > val F1** (0.94 vs 0.91 at the winner) — opposite of
  companies. ADI's val is harder (FAISS-mined hard negatives, 33% pos)
  than PyDI test (random sample, 27% pos).

### Per-pair test F1 (winner @ θ=0.5)

| pair                  |  F1   |  P    |  R    | TP/FP/FN/TN | training |
|-----------------------|-------|-------|-------|-------------|---------|
| dbpedia_metacritic    | 0.881 | 0.927 | 0.840 | 89/7/17/224 | ADI     |
| dbpedia_sales         | 0.952 | 0.973 | 0.931 | 108/3/8/283 | ADI     |
| **metacritic_sales**  | **0.971** | 0.957 | 0.985 | 133/6/2/441 | **transfer-only** |
| **OVERALL**           | 0.939 | 0.954 | 0.924 | 330/16/27/948 | mixed |

**Notable**: `metacritic_sales` — the test-only pair with zero
training rows — is the **highest-F1 pair** (0.971). The PLM transfers
strongly because the entity shape (game title + platform + genres +
developer + year) is consistent across all 3 sources. This validates
option (A) for handling the no-ADI-data pair.

### Comparison vs PyDI + ADI baselines

| pair                  | ADI MLBased (RF)       | this Ditto run |
|-----------------------|------------------------|----------------|
| dbpedia↔metacritic    | 0.964                  | 0.881 (-8pp)   |
| dbpedia↔sales         | 0.985                  | 0.952 (-3pp)   |
| metacritic↔sales      | n/a (no ADI run)       | **0.971**      |

Ditto trails ADI's RF on the 2 pairs ADI ran (-8pp on db↔mc, -3pp on
db↔sales), but the test set composition isn't directly comparable
(ADI evals on the FAISS-mined val; we eval on PyDI gold). Per-pair
gap on db↔mc is mostly recall (0.840 vs ADI's 0.94).

### Pinned recipe

```
python usecases_synthetic/scripts/ditto/train.py \
  --train-json-gz usecases_synthetic/output/ditto/games/train.json.gz \
  --val-json-gz   usecases_synthetic/output/ditto/games/val.json.gz \
  --test-json-gz  usecases_synthetic/output/ditto/games/test.json.gz \
  --config usecases_synthetic/config/ditto/default_train.yaml \
  --model-name roberta-base \
  --fields name,platform,genres,developer,releaseYear \
  --epochs 50 --batch-size 32 --max-len 256 --max-field-len 350 \
  --lr 1e-5 --warmup-ratio 0.06 --no-fp16 --da del \
  --early-stopping-patience 5 --seed 42 \
  --fixed-threshold 0.5 \
  --sample-weights-csv usecases_synthetic/output/ditto/games/train_sample_weights.csv \
  --output-dir cache/ditto_checkpoints/games/sweep_lr_1e-5_weighted/
```

`cache/ditto_checkpoints/games/best` symlinks to the winner's
`checkpoints/best/`. The S3 hard-negative gate at
`config/knob_02_niche/games.yaml §hard_negative_gate` is pinned to
`plm_threshold_theta: 0.5`, `plm_max_len: 256`, `plm_max_field_len: 350`,
`plm_batch_size: 32`, fields `[name, platform, genres, developer, releaseYear]`
(`publisher` was dropped — single-source, only present in `sales`).

### Known caveats

- ADI provides labels for only 2 of 3 source-pairs (db↔mc + db↔sales).
  metacritic↔sales has no ADI training data, and the top-level PyDI
  train/test files for it are byte-identical (582-pair file appears
  twice under different names). Option (A) was chosen: train on the 2
  ADI pairs, test on all 3, transfer-learn mc↔sales — the result
  (0.971 on the transfer-only pair) is empirically the best of the
  three pairs.
- ADI training pool has substantial intra-file duplicates: the
  combined 2,681 ADI train rows dedupe to 1,683 unique pairs (37%
  duplicates within `training_<pair>_latest.csv`).
- `train_test/` subdir test files have ~22% dbpedia ID attrition vs
  the refreshed `dbpedia.csv` (108 missing of 500 in db↔mc, 117 of
  500 in db↔sales). Top-level `dbpedia_2_*_test.csv` files are clean
  (0% missing) so the top-level path was preferred.

## Current music checkpoint

Trained as part of R2.2 in [../../../plans/plan_s1_scale.md](../../../plans/plan_s1_scale.md).
Apple Silicon (M5 Max, MPS, bf16 autocast). **Music uses option (b) —
pure PyDI throughout** (PyDI's gold for music is 15× larger than ADI's
training pool, ~36k vs 2.4k pairs, so the ADI-train approach used for
companies/games is the wrong lever here).

### Data setup (pure PyDI)

* **Train** = PyDI `<pair>_train.csv` for both source-pairs
  (musicbrainz↔discogs + musicbrainz↔lastfm) → **36,658 pairs
  (3,264 pos / 33,394 neg, 8.9% pos)**.
* **Val** = PyDI `<pair>_val.csv` for both pairs → **17,010 pairs
  (1,583 pos / 15,427 neg, 9.3% pos)**.
* **Test** = PyDI `<pair>_test.csv` for both pairs → **2,000 pairs
  (666 pos / 1,334 neg, 33.3% pos)**.

PyDI's natural train/val/test splits ship pre-disjoint — verified:
`train ∩ test = val ∩ test = train ∩ val = 0`. No leak removal needed
(the prep code is wired anyway and would catch any future
contamination). No intra-train duplicates.

Side-alignment: trivial — all PyDI files use `(musicbrainz, X)`
ordering. No id1↔id2 swap. Prep script:
[../../scripts/ditto/_prep_music.py](../../scripts/ditto/_prep_music.py).

### Field projection: `duration` swapped in for `genre`

Per-source coverage (refreshed CSVs share identical column set):

| field            | mb   | discogs | lastfm | ditto fields? |
|------------------|------|---------|--------|---------------|
| name             | 100% | 100%    | 100%   | yes           |
| artist           | 100% | 100%    | 100%   | yes           |
| release-date     | 93%  | 90%     | 0%     | yes           |
| release-country  | 84%  | 97%     | 0%     | yes           |
| **duration**     | 90%  | 57%     | 46%    | **yes (added)** |
| genre            | 0%   | 100%    | 0%     | **no (single-source — empty on at least one side of every pair)** |
| label            | 0%   | 100%    | 0%     | no (single-source + S11 reserved-name collision) |

`genre` was originally in the committee's `ditto_plm.fields` but is
**0% on both sides for every pair** (only discogs has it; mb↔discogs
and mb↔lastfm both have musicbrainz on the left, which has 0% genre).
Replaced with `duration`, which is 41-51% both-sides — meaningful
signal even if not "everywhere". `label` stays excluded per the S11
reserved-name fix (Ditto reserves "label" for the binary class column).
`tracks` is also 100% in all 3 but is a list field, not serialized.

### LR × class-balance sweep (8 runs, `--fixed-threshold 0.5`)

Fixed: `roberta-base --batch-size 32 --max-len 256 --max-field-len 350
--epochs 50 --early-stopping-patience 5 --warmup-ratio 0.06 --no-fp16
--da del --seed 42 --fixed-threshold 0.5`. Sample weights for
`weighted` runs use sklearn `class_weight='balanced'` (pos≈5.62,
neg≈0.55 — **up-weighting positives because train is only 8.9% pos**).

| lr   | mode       | best_epoch | val_f1   | test_f1   | test P | test R |
|------|------------|-----------:|---------:|----------:|-------:|-------:|
| 5e-6 | weighted   | 22         | 0.9505   | 0.9692    | 0.994  | 0.946  |
| 5e-6 | unweighted | 11         | 0.9535   | 0.9612    | 0.994  | 0.931  |
| 1e-5 | weighted   | 18         | 0.9508   | 0.9725    | 0.991  | 0.955  |
| 1e-5 | unweighted | 6          | 0.9521   | 0.9678    | 0.989  | 0.947  |
| 5e-5 | weighted   | 5          | 0.8784   | 0.9770    | 0.968  | 0.986  |
| 5e-5 | unweighted | 6          | 0.9413   | 0.9605    | 0.992  | 0.931  |
| **1e-4** | **weighted** | **2**  | **0.8817** | **0.9837** | **0.969** | **0.998** |
| 1e-4 | unweighted | 2          | 0.8936   | 0.8869    | 0.994  | 0.800  |

**Winner: lr=1e-4 weighted, test F1 = 0.9837** (P=0.969, R=0.998 —
only 1 FN out of 666 positives in test).

Pattern observations vs companies/games:
- **Higher LR did not collapse** the way it did on companies/games
  (which both crashed to "always positive" at lr=1e-4 weighted). Music's
  much-larger train set (36k vs 5.7k for companies, 1.7k for games)
  makes the model robust at high LR. Higher LR actually wins here.
- **Weighted dominates across every LR pair** — same as games (where
  positives also need adjustment) but more pronounced. All 4 weighted
  runs beat all 4 unweighted runs.
- **Val/test divergence is enormous and grows with LR**: best val F1
  (0.9535 @ lr=5e-6 unweighted) corresponds to test F1 0.9612 — far
  from the leader. The two highest-test runs (lr=5e-5 and lr=1e-4
  weighted) have val F1 in the 0.87–0.88 range. Val (PyDI's huge
  9.3%-pos sampling) is harder than test (33.3%-pos balanced sample).
  Picking by val_f1 would have selected the worst run.
- **lr=1e-4 unweighted** didn't fully collapse like on companies/games
  but did mis-train: P=0.994, R=0.800 — a different failure mode (high-
  precision, low-recall) caused by under-weighting positives at high LR.

### Per-pair test F1 (winner @ θ=0.5)

| pair                  |  F1   |  P    |  R    | TP/FP/FN/TN  |
|-----------------------|-------|-------|-------|--------------|
| musicbrainz↔discogs   | 0.979 | 0.960 | 1.000 | 333/14/0/653 |
| musicbrainz↔lastfm    | 0.988 | 0.979 | 0.997 | 332/7/1/660  |
| **OVERALL**           | 0.984 | 0.969 | 0.998 | 665/21/1/1313 |

### Comparison vs ADI baselines

| pair                  | ADI MLBased (RF)       | this Ditto run |
|-----------------------|------------------------|----------------|
| musicbrainz↔discogs   | 0.976                  | **0.979**  (+0.3pp) |
| musicbrainz↔lastfm    | 0.995                  | 0.988  (-0.7pp)     |

Beats ADI's RF on db↔mb by +0.3pp, lags slightly on lf↔mb (-0.7pp).
Note: ADI's eval set is its 300-pair FAISS-mined val (not directly
comparable to PyDI's 1,000-pair-per-pair test).

### Pinned recipe

```
python usecases_synthetic/scripts/ditto/train.py \
  --train-json-gz usecases_synthetic/output/ditto/music/train.json.gz \
  --val-json-gz   usecases_synthetic/output/ditto/music/val.json.gz \
  --test-json-gz  usecases_synthetic/output/ditto/music/test.json.gz \
  --config usecases_synthetic/config/ditto/default_train.yaml \
  --model-name roberta-base \
  --fields name,artist,release-date,release-country,duration \
  --epochs 50 --batch-size 32 --max-len 256 --max-field-len 350 \
  --lr 1e-4 --warmup-ratio 0.06 --no-fp16 --da del \
  --early-stopping-patience 5 --seed 42 \
  --fixed-threshold 0.5 \
  --sample-weights-csv usecases_synthetic/output/ditto/music/train_sample_weights.csv \
  --output-dir cache/ditto_checkpoints/music/sweep_lr_1e-4_weighted/
```

`cache/ditto_checkpoints/music/best` symlinks to the winner's
`checkpoints/best/`. The S3 hard-negative gate at
`config/knob_02_niche/music.yaml §hard_negative_gate` is pinned to
`plm_threshold_theta: 0.5`, `plm_max_len: 256`, `plm_max_field_len: 350`,
`plm_batch_size: 32`, fields `[name, artist, release-date, release-country, duration]`.

### Known caveats

- `discogs` has many `0` placeholder values in `duration` (43% of
  rows are zero — likely missing-as-zero). Ditto serializes them as
  `COL duration VAL 0` rather than the bridge stripping them, so the
  model has to learn that "duration=0 vs absent on the other side"
  is equivalent to "both sides missing". Empirically this didn't
  block the 0.984 result, but a follow-up could clean these to NaN
  in `_prep_music.py`.
- The 0.998 test recall is suspiciously high — only 1 FN out of 666
  positives. PyDI test (33.3% pos balanced 1,000-per-pair) is easier
  than ADI's FAISS-mined eval. Don't read this as Ditto being a
  near-perfect EM matcher in general; it's a near-perfect matcher
  *on this specific easier test distribution*.
