# Utils

The utils module provides shared helpers used across the framework: a similarity metric registry using `textdistance`, consistent logging for LLM invocations, and data profiling utilities.


## Data Profiling

`DataProfiler` generates dataset profiles and quick summaries for exploratory data analysis. It wraps ydata-profiling and sweetviz to produce HTML reports for individual datasets and comparisons.

```python
from PyDI.utils import DataProfiler

profiler = DataProfiler()

# HTML report via ydata-profiling
profiler.profile(df, out_dir="output/profiles")

# Compare two datasets via sweetviz
profiler.compare(df_a, df_b, out_dir="output/profiles")

# Quick console summary
profiler.summary(df)

# Analyze attribute coverage across multiple datasets
coverage = profiler.analyze_coverage([df1, df2, df3])
```

Methods:
- `profile()` - HTML report for a single dataset (requires `ydata-profiling`)
- `compare()` - HTML comparison of two datasets (requires `sweetviz`)
- `summary()` - Console output with row/column counts, null statistics
- `analyze_coverage()` - DataFrame showing column overlap and missing value rates across datasets


## Similarity Metric Registry

`SimilarityRegistry` centralizes access to similarity functions with name/category lookup and recommended sets for common use cases.

Available metrics:
- Edit‑based: hamming, levenshtein, damerau_levenshtein, jaro_winkler, jaro, strcmp95, needleman_wunsch, gotoh, smith_waterman, mlipns, editex
- Token‑based: jaccard, sorensen_dice, tversky, overlap, tanimoto, cosine, monge_elkan, bag
- Sequence‑based: lcsseq, lcsstr, ratcliff_obershelp
- Simple: prefix, postfix, length, identity
- Phonetic: mra


## LLM Invocation Logging

The LLM logging helpers (`PyDI.utils.llm`) standardize how prompts, responses, token usage, and model/provider details are captured. They enable comparable debugging and usage tracking across LLM‑based extractors and matchers, and integrate with artifact writing so traces can be reviewed alongside other pipeline outputs.