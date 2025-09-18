# Information Extraction

The Information Extraction module derives structured, typed attributes from text‑heavy or semi‑structured columns. It supports pattern‑based extraction with regular expressions, custom Python functions for dataset‑specific logic, and model‑based extraction using large language models. Extracted fields are appended as new columns and can be evaluated against a gold standard when available.

Supported Approaches (module `PyDI.informationextraction`)
- Regex‑based extraction: `RegexExtractor` applies rule sets of patterns with optional post‑processing to normalize results (e.g., amounts, dates, units).
- Code‑based extraction: `CodeExtractor` runs user‑defined Python functions on text columns or full rows for flexible, domain‑specific logic.
- LLM‑based extraction: `LLMExtractor` performs schema‑guided extraction via LangChain, supporting many providers and models through a unified chat‑model interface (e.g., OpenAI, Anthropic). Prompts, responses, and errors are persisted for transparency.

Additional Notes
- Normalize outputs to canonical types (e.g., ISO dates, floats) to aid downstream matching and fusion.
- Multiple extractors can be combined using `ExtractorPipeline` to incrementally enrich a dataset.

Evaluation of Extraction
- Module: `PyDI.informationextraction.evaluation.InformationExtractionEvaluator` compares predicted columns with a gold standard aligned by record IDs.
- Metrics: reports attribute‑level results and aggregate micro/macro precision, recall, F1, and non‑null accuracy; supports attribute‑specific rules (e.g., exact match, tokenized text, numeric tolerance, set equality).
- Diagnostics: optional mismatch logs (text or JSONL) to inspect errors and refine rules or prompts.
