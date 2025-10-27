This module contains methods to automatically find 1-to-1 correspondences between columns across two datasets using label-, instance-, or duplicate-based schema matching strategies. Mapping columns containing corresponding information to a unified schema is important to prepare datasets for a high quality matching and merging during the following entity matching and data fusion steps.

Module: `PyDI.schemamatching`
- `LabelBasedSchemaMatcher`: compares the labels of the columns using similarity metrics to find schema correspondences. Fast and accurate when column labels are meaningful.
- `InstanceBasedSchemaMatcher`: compares the distributions of values per column via TF/TF‑IDF/binary vectors and cosine/Jaccard/containment similarity. Better suited than LabelBasedMatcher if column labels are ambigous.
- `DuplicateBasedSchemaMatcher`: leverage known record correspondences to infer column alignments from co‑occurring values in columns of the corresponding records. Great when a labeled set of matching records between datasets exists.
- `SchemaMappingEvaluator`: offers methods for evaluating a generated schema mapping given a labeled set of schema correspondences.

Example schema matching
```python
from langchain_openai import ChatOpenAI
from PyDI.schemamatching import LLMBasedSchemaMatcher

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
matcher = LLMBasedSchemaMatcher(
    chat_model=chat,
    num_rows=10,
    debug=True
    )
corr = matcher.match(source_df, target_df)
```

Example evaluation
```python
from PyDI.schemamatching import SchemaMappingEvaluator

metrics = SchemaMappingEvaluator.evaluate(corr, test_set)
```

### WDC SMB Evaluation

Use ``run_wdc_smb_benchmark`` to execute ``LLMBasedSchemaMatcher`` on the
official WDC Schema Matching Benchmark (SOTAB-SM and T2D-SM). The helper takes
care of loading the released correspondences, aligning column indices with
their headers, and reporting precision, recall, and F1 via
``SchemaMappingEvaluator``.

Steps:
- Download the desired benchmark zip from the WDC SMB release page and extract
  it locally (e.g. ``SOTAB_SM_V500`` or ``T2D_SM_WH``).
- Instantiate your preferred LangChain ``BaseChatModel`` (for example
  ``ChatOpenAI``) and create ``LLMBasedSchemaMatcher`` with it.
- Call ``run_wdc_smb_benchmark`` with a ``WDCBenchmarkConfig`` pointing to the
  extracted directory and the split you want to evaluate.

Minimal example:
```python
from pathlib import Path
from langchain_openai import ChatOpenAI

from PyDI.schemamatching import (
    LLMBasedSchemaMatcher,
    WDCBenchmarkConfig,
    run_wdc_smb_benchmark,
)

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
matcher = LLMBasedSchemaMatcher(chat_model=chat, num_rows=8)

config = WDCBenchmarkConfig(
    dataset_root=Path("/data/WDC/SOTAB_SM_V500"),
    task="sotab",
    split="test",
)

result = run_wdc_smb_benchmark(matcher, config)
print(result["overall"])  # precision / recall / F1
```

There is also a CLI for OpenAI models:
```
python -m PyDI.schemamatching.wdc_smb \
    --dataset-root /data/WDC/SOTAB_SM_V500 \
    --task sotab \
    --split test \
    --openai-model gpt-4o-mini
```

Benchmark data should never be fed into model training. The utility only reads
the provided evaluation splits for on-demand benchmarking.

Artifacts
- Schema correspondences written to file