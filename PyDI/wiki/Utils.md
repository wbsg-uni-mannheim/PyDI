# Utils

Utils provides shared helpers used across the framework, notably a similarity metric registry and consistent logging for LLM invocations.

Similarity Metric Registry
`PyDI.utils.similarity_registry.SimilarityRegistry` centralizes access to similarity functions with name/category lookup and recommended sets for common use cases.

Available Metrics
- Edit‑based: hamming, levenshtein, damerau_levenshtein, jaro_winkler, jaro, strcmp95, needleman_wunsch, gotoh, smith_waterman, mlipns, editex
- Token‑based: jaccard, sorensen_dice, tversky, overlap, tanimoto, cosine, monge_elkan, bag
- Sequence‑based: lcsseq, lcsstr, ratcliff_obershelp
- Simple: prefix, postfix, length, identity
- Phonetic: mra

LLM Invocation Logging
The LLM logging helpers (`PyDI.utils.llm`) standardize how prompts, responses, token usage, and model/provider details are captured. They enable comparable debugging and usage tracking across LLM‑based extractors and matchers, and integrate with artifact writing so traces can be reviewed alongside other pipeline outputs.
