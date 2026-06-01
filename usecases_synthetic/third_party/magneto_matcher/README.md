# magneto_matcher

Vendored copy of the Magneto schema-matching framework
(VIDA-NYU, VLDB 2025). See [ORIGIN.md](ORIGIN.md) for the upstream
reference.

Used by the synthetic-benchmark schema-matching committee —
[usecases_synthetic/lib/magneto_sm_matcher.py](../../lib/magneto_sm_matcher.py)
adapts the upstream `Magneto` class to PyDI's
`BaseSchemaMatcher` contract.

## Entry point

```python
from usecases_synthetic.third_party.magneto_matcher.magneto import Magneto

mag = Magneto(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    encoding_mode="header_values_verbose",
    topk=20,
    use_bp_reranker=True,
    use_gpt_reranker=True,
    llm_model="openai/gpt-4o-mini",
    llm_model_kwargs={"temperature": 0.0},
)
matches = mag.get_matches(source_df, target_df)
```

The returned `matches` is a Valentine-format dict keyed by
`((source_table_name, source_col), (target_table_name, target_col))`
with float scores in `[0, 1]`.

## Dependencies

Installed via PyDI's `magneto` extras group (see
[pyproject.toml](../../../pyproject.toml)):

- `fuzzywuzzy` — string similarity
- `mmh3` — priority-sampling hash
- `valentine` — `MatcherResults` return type
- `litellm` — uniform LLM client for the LLM reranker
- `json_repair` — lenient JSON parsing of LLM responses

`sentence-transformers`, `torch`, `transformers`, `scikit-learn`,
`scipy`, `pandas`, `numpy` are already core PyDI dependencies.

## Files touched by the vendoring rewrite

Absolute imports (`from magneto.xxx import ...`) rewritten to relative
imports (`from .xxx import ...`) in:

- `magneto/__init__.py`
- `magneto/basic_matcher.py`
- `magneto/column_encoder.py`
- `magneto/embedding_matcher.py`
- `magneto/llm_reranker.py` (no rewrite needed — only external deps)
- `magneto/magneto.py`
- `magneto/retriever.py`
- `magneto/utils/base_table.py`
- `magneto/utils/dataframe_column.py`
- `magneto/utils/dataframe_table.py`
- `magneto/utils/retriever_utils.py`
- `magneto/utils/utils.py`
