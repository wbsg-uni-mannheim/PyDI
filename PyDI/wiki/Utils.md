# Utils

Convenience utilities for comparators, similarity functions, normalization helpers, and LLM logging. These utilities reduce boilerplate and standardize common tasks.

Comparators (quick functions)
- `PyDI.utils.jaccard(column)` — token Jaccard similarity over string tokens
- `PyDI.utils.date_within_years(column, years)` — date proximity returning 1.0 if within window

Similarity Registry
- `PyDI.utils.similarity_registry.SimilarityRegistry`
  - Discover and validate available similarity functions for schema/entity matching
  - `get_function(name)`, `list_available_functions(category=None)`, `get_function_info(name)`, `get_recommended_functions(use_case, matcher_type=None)`
- Use this to centralize comparator choice across projects and ensure reproducibility.

Normalization Helpers
- `PyDI.utils.normalization` — text cleaning, remove HTML, phone normalization, currency/percentage parsing, encoding fixes.
- Apply these before matching to reduce noise and improve similarity scores.

LLM Usage Logging
- `PyDI.utils.llm.LLMCallLogger` — record prompts/responses/tokens and flush artifacts via callback, enabling audit trails for LLM‑based matchers/extractors.

Example
```python
from PyDI.utils import jaccard
from PyDI.utils.similarity_registry import list_similarity_functions

sim = jaccard("title")
print(list_similarity_functions())
```
