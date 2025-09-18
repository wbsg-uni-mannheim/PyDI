# Information Extraction

Extract normalized attributes from free text or semi‑structured columns using regex, code, or LLMs. IE complements schema matching by deriving clearer, typed fields from messy sources.

Modules
- `PyDI.informationextraction.base.BaseExtractor`, `ExtractorPipeline`
- `PyDI.informationextraction.regex.RegexExtractor`
- `PyDI.informationextraction.code` — function‑based extraction
- `PyDI.informationextraction.llm` — optional LLM extractor
- `PyDI.informationextraction.evaluation.InformationExtractionEvaluator`

Rule Writing Guidelines
- Keep patterns conservative to reduce false positives; chain postprocessors for normalization.
- Use named rules per attribute; prefer one clear rule over many overlapping ones.
- For dates/amounts, route through `normalization.values` or `normalization.units` for consistent output.

Postprocessing
- `postprocess` may reference built‑in helpers (e.g., `parse_money`, `parse_date`) or a custom callable.
- Normalize to canonical types (e.g., ISO dates, floats) for easier downstream use.

LLM Extractor Notes
- Useful for complex patterns; slower and cost‑bearing.
- PyDI logs prompts/responses and errors to `out_dir` for auditability.
- Consider throttling and caching for large‑scale use.

Pipelines
- Compose multiple extractors via `ExtractorPipeline` to incrementally enrich a dataset.

Example (RegexExtractor)
```python
from PyDI.informationextraction import RegexExtractor

rules = {
    "price": {"pattern": r"[$€]\s?(\d+[\.,]?\d*)", "postprocess": "parse_money"},
    "release_date": {"pattern": r"\b\d{4}-\d{2}-\d{2}\b"}
}
ex = RegexExtractor(source_column="description", rules=rules, out_dir="output/ie")
df_out = ex.extract(df)
```

Evaluation
```python
from PyDI.informationextraction.evaluation import InformationExtractionEvaluator

evaluator = InformationExtractionEvaluator(id_column="_id")
evaluator.set_evaluation_function("price", lambda y_true, y_pred: float(y_true == y_pred))
metrics = evaluator.evaluate(predicted=df_pred, gold=df_gold, attributes=["price","release_date"], out_dir="output/ie/eval")
```

Artifacts
- Extracted datasets, debug logs, and evaluation reports under `out_dir`.
