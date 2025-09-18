# Blocking

Generate candidate pairs efficiently before entity matching. All blockers stream batches and avoid full cross‑product materialization. Good blocking drastically reduces comparisons while preserving most true matches.

Modules
- `PyDI.entitymatching.blocking.base.BaseBlocker`
- `PyDI.entitymatching.blocking.standard.StandardBlocking` (keys on columns)
- `PyDI.entitymatching.blocking.sorted_neighbourhood.SortedNeighbourhood` (sliding window on a key)
- `PyDI.entitymatching.blocking.token_blocking.TokenBlocking` (token overlap on a text column)
- `PyDI.entitymatching.blocking.embedding.EmbeddingBlocking` (text embeddings)
- `PyDI.entitymatching.blocking.noblocking.NoBlocking`
- `PyDI.entitymatching.blocking.blocking_evaluation.BlockingEvaluator`

Choosing a Strategy
- Standard: when you have reliable join keys or key composites.
- Sorted Neighbourhood: when a sortable key exists but exact equality is too strict.
- Token: when entity names are multi‑token and you expect partial overlaps.
- Embedding: when textual fields vary semantically (synonyms, paraphrases).

Parameters That Matter
- `batch_size`: memory/perf trade‑off for streaming.
- Key selection: prefer normalized, informative attributes; combine columns if necessary.
- Tokenization: adjust `min_token_len`/custom tokenizer to reduce noise.

Example
```python
from PyDI.entitymatching.blocking import StandardBlocking

blocker = StandardBlocking(df_left, df_right, on=["title","year"], batch_size=100_000, out_dir="output/matching/blocking")
for batch in blocker:  # yields DataFrames with [id1, id2, block_key]
    process(batch)
```

Evaluation
```python
from PyDI.entitymatching.blocking import BlockingEvaluator

report = BlockingEvaluator.evaluate(candidates=blocker, gold=test_pairs, out_dir="output/matching/blocking_eval")
```

Reading the Metrics
- Candidate recall: fraction of true matches present in candidates (should be high).
- Reduction ratio: how much the cross‑product was reduced (higher is better, but not at the expense of recall).

Artifacts
- Blocking stats, debug files, and batch summaries under `out_dir`.
