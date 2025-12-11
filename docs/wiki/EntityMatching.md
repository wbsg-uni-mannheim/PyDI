# Entity Matching

PyDI's Entity Matching module identifies records describing the same real-world entity across datasets. It provides a three-stage pipeline: blocking to generate candidate pairs, matching to score those candidates and generate correspondences between matching records, and optional post-clustering to refine matching results. The module provides rule-based, machine learning-based, and (large) language model-based matchers as well as evaluation methods and debugging logs.

- Blockers generate a set of candidate pairs
- Matchers score candidates and decide for match or non-match and output a set of correspondences
- Post-clustering algorithms refine correspondences for e.g. transitivity or one-to-one constraints.
- Evaluation using evaluation set with precision, recall, and F1
- Detailed debug logs for inspection of matching results


## Table of Contents

- [Requirements](#requirements)
- [Blocking](#blocking)
- [Comparators](#comparators)
- [Matchers](#matchers)
- [Post-Clustering](#post-clustering)
- [Evaluation](#evaluation)
- [Debug Logs and Tuning](#debug-logs-and-tuning)
- [Example: End-to-End](#example-end-to-end)


## Requirements

- Each input `DataFrame` must set `df.attrs["dataset_name"]`.
- Records need a unique identifier. Provide a column name via `id_column` parameter.
- Evaluation set: `DataFrame` with `id1`, `id2`, `label` (True/False or 1/0).


## Blocking

Blocking reduces the O(n×m) Cartesian product to a manageable candidate set. Blockers generate pairs that share some property (blocking key, token, semantic similarity).

PyDI offers the following blockers
- `NoBlocker` – full Cartesian product
- `StandardBlocker` – equality-based blocking on one or more blocking keys
- `SortedNeighbourhoodBlocker` – sliding window on sorted (by blocking key) sequence
- `TokenBlocker` – token or n-gram overlap
- `EmbeddingBlocker` – nearest neighbors via dense embeddings


### Standard blocking

Block on exact equality of one or more attributes.

```python
from PyDI.entitymatching import StandardBlocker

# Preprocess name to create blocking key
df_left["name_prefix"] = df_left["name"].astype(str).apply(
    lambda x: "".join([word[:2].upper() for word in x.split()[:3]])
)
df_right["name_prefix"] = df_right["name"].astype(str).apply(
    lambda x: "".join([word[:2].upper() for word in x.split()[:3]])
)

blocker = StandardBlocker(
    df_left, df_right,
    on=["name_prefix"],
    id_column="id",
    output_dir="output/blocking"
)
```

Standard blocking writes debug CSV files with blocking key frequencies to help tune key selection.


### Sorted neighbourhood blocking

Sort records by a key and slide a window over the sequence, pairing each record with its neighbors. Effective for handling typos and small variations in strings.

```python
from PyDI.entitymatching import SortedNeighbourhoodBlocker

blocker = SortedNeighbourhoodBlocker(
    df_left, df_right,
    key="name",
    window=20,
    id_column="id",
    output_dir="output/blocking"
)
```

Window size controls recall vs. candidate set size trade-off.


### Token blocking

Tokenize text and generate candidate pairs for records sharing tokens or n-grams.

```python
from PyDI.entitymatching import TokenBlocker

blocker = TokenBlocker(
    df_left, df_right,
    column="name",
    min_token_len=3,
    ngram_size=2,
    ngram_type="word",
    id_column="id",
    output_dir="output/blocking"
)
```

`ngram_type` supports `word`- and `character`-level n-grams.


### Embedding blocking

Use sentence embeddings to find semantically similar pairs. Supports multiple index backends for scalability.

```python
from PyDI.entitymatching import EmbeddingBlocker

blocker = EmbeddingBlocker(
    df_left, df_right,
    text_cols=["name", "description"],
    model="sentence-transformers/all-MiniLM-L6-v2",
    index_backend="sklearn",
    metric="cosine",
    top_k=20,
    id_column="id",
    output_dir="output/blocking"
)
```

Performance notes
- `sklearn` backend (default): good for datasets up to ~100k records
- `faiss` backend: recommended for 100k–1M records, requires `faiss-cpu` or `faiss-gpu`
- `hnsw` backend: recommended for >1M records, requires `hnswlib`

You can pass pre-computed embeddings via `left_embeddings` and `right_embeddings` to avoid re-embedding or to use custom embeddings.


### Streaming and materialization

Blockers are iterators. Iterate in batches to keep memory usage low:

```python
for batch in blocker:
    # batch is a DataFrame with columns: id1, id2, (optional) block_key
    process_batch(batch)
```

Or materialize the full candidate set:

```python
candidates = blocker.materialize()
```

Debug output includes blocking key frequencies and pair counts to help diagnose blocking effectiveness.


## Comparators

Comparators are building blocks for similarity computation. Each comparator measures similarity between two values and returns a score in [0, 1]. Comparators are used as feature generators for the RuleBasedMatcher and MLBasedMatcher

Built-in comparators
- `StringComparator` – string similarity using textdistance metrics (jaro_winkler, jaccard, levenshtein, etc.)
- `NumericComparator` – numeric similarity (absolute or relative difference)
- `DateComparator` – temporal similarity

All comparators support preprocessing (e.g., lowercase, strip) and handle list-valued attributes (concatenate, best match, set operations).


### String comparator

```python
from PyDI.entitymatching import StringComparator

# Single-valued attributes
name_comp = StringComparator(
    "name",
    similarity_function="jaro_winkler",
    preprocess=lambda x: str(x).lower()
)

# Tokenized comparison
address_comp = StringComparator(
    "address",
    similarity_function="jaccard",
    preprocess=lambda x: str(x).lower()
)

# List-valued attributes (e.g., tags, categories)
tags_comp = StringComparator(
    "tags",
    similarity_function="jaccard",
    list_strategy="set_jaccard"
)
```

List strategies: `"concatenate"`, `"best_match"`, `"set_jaccard"`, `"set_overlap"`.


### Numeric and date comparators

```python
from PyDI.entitymatching import NumericComparator, DateComparator

# Numeric comparison
price_comp = NumericComparator(
    "price",
    method="absolute_difference"
)

# Date comparison
date_comp = DateComparator(
    "created_date",
    max_days_difference=365
)
```


### Custom comparators

Write a callable that takes two records and returns a similarity score:

```python
def year_only_match(rec1, rec2):
    val1 = rec1.get("created_date")
    val2 = rec2.get("created_date")
    if pd.isna(val1) or pd.isna(val2):
        return 0.0
    year1 = pd.to_datetime(val1, errors="coerce").year if val1 else None
    year2 = pd.to_datetime(val2, errors="coerce").year if val2 else None
    return 1.0 if year1 == year2 else 0.0

# Store for later use in matcher
comparators = [name_comp, address_comp, year_only_match] # you can pass custom functions together with comparators to the matchers
```


## Matchers

Matchers score candidate pairs and return a `CorrespondenceSet` (DataFrame with `id1`, `id2`, `score`, `notes`). The `CorrespondenceSet` contains all candidate pairs that were scores as matches.


### Rule-based matcher

Combines multiple comparators with manually assigned `weights`. The final similarity score is the weighted average of comparator outputs. Given a manually set `threshold`, candidate pairs are scored either as matches if above the threshold or non-matches if below.

```python
from PyDI.entitymatching import RuleBasedMatcher, StringComparator, DateComparator

comparators = [
    StringComparator("name", similarity_function="jaro_winkler"),
    DateComparator("created_date", max_days_difference=365)
]

weights = [0.5, 0.5]

matcher = RuleBasedMatcher()
correspondences = matcher.match(
    df_left=df_left,
    df_right=df_right,
    candidates=blocker, # pass blocker or materalized candidates set
    id_column="id",
    comparators=comparators,
    threshold=0.7,
    weights=weights
    debug=True, # debug mode writes per-pair comparator scores to a CSV for tuning weights and thresholds.
)
```




### ML-based matcher

For use with e.g. scikit-learn classifiers. Comparators are the features. Train on labeled pairs to learn optimal weights.

Feature extraction converts record pairs into feature vectors using the set of comparators. The `FeatureExtractor` class handles this transformation.

```python
from PyDI.entitymatching import MLBasedMatcher, FeatureExtractor, StringComparator
from PyDI.io import load_csv
from sklearn.ensemble import RandomForestClassifier

# Load training data
training_pairs = load_csv(
    "path/to/training_pairs.csv",
    header=None,
    names=["id1", "id2", "label"]
)

# Define feature extractors (same comparators as rule-based)
comparators = [
    StringComparator("name", similarity_function="jaro_winkler"),
    StringComparator("address", similarity_function="jaccard"),
    DateComparator("created_date", max_days_difference=365)
]

extractor = FeatureExtractor(comparators)

# Extract features for training pairs
train = extractor.create_features(
    df_left, df_right, pairs=training_pairs[['id1','id2']], labels=training_pairs['label'] id_column="id"
)

feature_columns = [col for col in train.columns if col not in ['id1', 'id2', 'label']]

X_train = train[feature_columns]
y_train = train['label']

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Match on candidate set
matcher = MLBasedMatcher(extractor)
correspondences = matcher.match(
    df_left, df_right, blocker, id_column="id", trained_classifier=clf
)
```

If the classifier supports `feature_importances_` or `coef_`, the matcher logs feature importance for interpretability.

Alternatively, you can use the VectorFeatureExtractor to create features from an embedding model.

```python
from PyDI.entitymatching import VectorFeatureExtractor


# Define vector-based feature extractor using embeddings
extractor = VectorFeatureExtractor(
    embedding_model='sentence-transformers/all-MiniLM-L6-v2',
    columns=['name', 'address'],
    distance_metrics=['cosine', 'euclidean'],
    pooling_strategy='concatenate',
    list_strategies={'address': 'concatenate'}  # if address contains lists
)

# Extract features for training pairs
train = extractor.create_features(
    df_left, df_right,
    pairs=training_pairs[['id1','id2']],
    labels=training_pairs['label'],
    id_column="id"
)
```


### PLM-based matcher

PyDI expects a pre-trained or fine-tuned transformer model from the HuggingFace ecosystem. Refer to the [HuggingFace documentation on text classification](https://huggingface.co/docs/transformers/tasks/sequence_classification) for detailed guidance on fine-tuning models for binary classification (match vs. non-match).

**Using a fine-tuned model with PyDI**

Once you have a fine-tuned model, you can use it with PyDI's `PLMBasedMatcher`. The matcher requires a `TextFormatter` that defines how entity pairs should be formatted as text input for the model.

```python
from PyDI.entitymatching import PLMBasedMatcher
from PyDI.entitymatching.text_formatting import TextFormatter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load your fine-tuned model and tokenizer
model_path = "path/to/your/fine-tuned-model"  # or HuggingFace model ID
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define how entity pairs should be formatted as text
text_formatter = TextFormatter(
    text_fields=['name', 'address', 'created_date'],  # Attributes to include
    template="{left} [SEP] {right}",  # Template for the PAIR - uses {left} and {right}
    single_template="Name: {name}, Address: {address}, Date: {created_date}",  # Template for individual entities
    max_length=128  # Maximum sequence length
)

# Initialize PLM-based matcher
plm_matcher = PLMBasedMatcher(text_formatter=text_formatter)

# Perform matching
correspondences = plm_matcher.match(
    df_left=df_left,
    df_right=df_right,
    candidates=blocker,  # Use blocking to reduce search space
    id_column='id',
    trained_model=model,
    tokenizer=tokenizer,
    model_type='classification',  # 'classification' for binary classifiers
    batch_size=16,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

PLM-based matchers integrate seamlessly with PyDI's blocking strategies, evaluation framework, and post-clustering algorithms.


### LLM-based matcher

Leverages large language models for semantic matching. Supports few-shot examples and custom prompts.

```python
from PyDI.entitymatching import LLMBasedMatcher
from langchain_openai import ChatOpenAI

# Initialize chat model
chat_model = ChatOpenAI(model="gpt-4", temperature=0)

# Few-shot examples (optional)
few_shots = [
    (
        {"name": "Acme Corp", "address": "123 Main St"},
        {"name": "ACME Corporation", "address": "123 Main Street"},
        {"match": True, "confidence": 0.95, "reason": "Same company, minor address abbreviation"}
    ),
    (
        {"name": "Alpha Inc", "address": "456 Oak Ave"},
        {"name": "Beta LLC", "address": "789 Pine Rd"},
        {"match": False, "confidence": 0.0, "reason": "Different companies and addresses"}
    )
]

matcher = LLMBasedMatcher()

correspondences = matcher.match(
    df_left=df_left,
    df_right=df_right,
    candidates=blocker,
    chat_model=chat_model,
    fields=['name', 'address'],
    id_column='id',
    out_dir= "output/llm",
    few_shots=few_shots
)
```

The matcher writes artifacts to `out_dir`: prompts, responses, errors, and statistics. These can be used for debugging and cost tracking.


## Post-Clustering

Post-clustering algorithms refine correspondences by enforcing constraints (e.g., transitivity, one-to-one) or optimizing global objectives. Apply post-clustering after matching to improve precision or satisfy problem requirements.

Built-in algorithms
- `ConnectedComponentClusterer` – transitive closure (if A↔B and B↔C, then A↔C)
- `CentreClusterer` – star-shaped clusters with max diameter 2
- `HierarchicalClusterer` – agglomerative clustering with configurable linkage
- `GreedyOneToOneMatchingAlgorithm` – fast one-to-one greedy selection
- `MaximumBipartiteMatching` – optimal one-to-one via Hungarian algorithm
- `StableMatching` – preference-based one-to-one stable matching


### Connected component clustering

Groups all transitively connected entities. If record A matches B and B matches C, all three are clustered even if A and C were not directly compared.

```python
from PyDI.entitymatching import ConnectedComponentClusterer

clusterer = ConnectedComponentClusterer(
    threshold=0.6,
    min_cluster_size=2,
    preserve_scores=True
)
refined = clusterer.cluster(correspondences)
```

Output is all-pairs within each cluster (fully connected graph). Use when transitivity is desired.


### Centre clustering

Creates star-shaped clusters: one center connected to multiple leaves. Max cluster diameter is 2.

```python
from PyDI.entitymatching import CentreClusterer

clusterer = CentreClusterer(
    threshold=0.7,
    min_cluster_size=2
)
refined = clusterer.cluster(correspondences)
```

Use for hub-based grouping (e.g., citations to a canonical paper, product variants to a master record).


### Hierarchical clustering

Bottom-up agglomerative clustering with linkage modes: MIN (single-linkage), MAX (complete-linkage), AVG (average-linkage).

```python
from PyDI.entitymatching import HierarchicalClusterer, LinkageMode

clusterer = HierarchicalClusterer(
    linkage_mode=LinkageMode.AVG,
    num_clusters=10,
    threshold=0.5
)
refined = clusterer.cluster(correspondences)
```

Stop merging when `num_clusters` is reached or when similarity falls below `threshold`.


### One-to-one matching

Enforce one-to-one constraint: each entity matches at most one other entity.

Greedy matching:

```python
from PyDI.entitymatching import GreedyOneToOneMatchingAlgorithm

greedy = GreedyOneToOneMatchingAlgorithm()
greedy_matches = greedy.cluster(correspondences)
```

Maximum weighted bipartite matching:

```python
from PyDI.entitymatching import MaximumBipartiteMatching

mbp = MaximumBipartiteMatching()
mbp_matches = mbp.cluster(correspondences)
```

Stable matching:

Finds stable pairs based on preference (similarity) lists. A matching is stable if no two entities prefer each other over their assigned matches.

```python
from PyDI.entitymatching import StableMatching

stable = StableMatching()
stable_matches = stable.cluster(correspondences)
```

## Evaluation

The evaluator supports blocking evaluation: pass candidate pairs and gold standard to measure blocking recall and reduction ratio.

```python
from PyDI.entitymatching import EntityMatchingEvaluator
from PyDI.io import load_csv

evaluator = EntityMatchingEvaluator()

# Load gold standard
evaluation_set = load_csv(
    "path/to/evaluation_set.csv",
    header=None,
    names=["id1", "id2", "label"]
)

blocking_metrics = evaluator.evaluate_blocking(
    candidates, blocker, evaluation_set
)
```

The evaluator also supports evaluating a matching against an evaluation set. Returns precision, recall, F1.

```python
metrics = evaluator.evaluate_matching(
    correspondences=correspondences,
    test_pairs=evaluation_set,
    debug_info=debug_info, # add debug info (optional)
    matcher_instance=matcher # add matcher instance for context for debug files (optional)
    )
```

## Debug Logs and Tuning

Enable debug mode in matchers to write per-pair details. Debug logs show comparator scores, final weighted score and true label ("isMatch") if a pair is part of the evaluation set.

Example debug output (CSV format):

```
"MatchingRule","Record1Identifier","Record2Identifier","TotalSimilarity","IsMatch","[0] StringComparator(name, jaro_winkler) similarity","[1] StringComparator(address, levenshtein) similarity","[2] DateComparator(date) similarity"
"RuleBasedMatcher","rec_123","rec_456","0.91","1","0.95","0.82","1.0"
"RuleBasedMatcher","rec_124","rec_457","0.68","0","0.88","0.75","0.0"
```

Use debug logs to:
- Diagnose blocking failures (pair not in candidate set)
- Identify which comparators contribute most to mismatches
- Adjust comparator weights to balance precision and recall
- Tune threshold to optimize F1 score

LLM-based matcher writes JSON artifacts:
- `prompts.jsonl`: All prompts sent to the model
- `responses.jsonl`: Model responses with parsed scores
- `errors.jsonl`: Parse errors and retry attempts
- `stats.json`: Token usage, latency, cost estimates


### Cluster size distribution

After matching (and optional post-clustering), analyzing the size distribution of entity clusters allows understanding how records are grouped which is useful for additional debugging.

```python
from PyDI.entitymatching import EntityMatchingEvaluator
from pathlib import Path

OUTPUT_DIR = Path("output/entity_matching")

# Create cluster size distribution from our matches
cluster_distribution = EntityMatchingEvaluator.create_cluster_size_distribution(
    correspondences=correspondences,
    out_dir=OUTPUT_DIR / "cluster_analysis"
)
```

Use this to validate that matching parameters produce sensible groupings for the use-case.


## Example: End-to-End

```python
from PyDI.io import load_csv
from PyDI.entitymatching import (
    StandardBlocker, RuleBasedMatcher, StringComparator, DateComparator,
    ConnectedComponentClusterer, EntityMatchingEvaluator
)

# 1. Load data
df_left = load_csv("path/to/dataset_a.csv", name="dataset_a")
df_right = load_csv("path/to/dataset_b.csv", name="dataset_b")

# 2. Blocking
df_left["name_prefix"] = df_left["name"].astype(str).apply(
    lambda x: "".join([word[:2].upper() for word in x.split()[:3]])
)
df_right["name_prefix"] = df_right["name"].astype(str).apply(
    lambda x: "".join([word[:2].upper() for word in x.split()[:3]])
)

blocker = StandardBlocker(
    df_left, df_right,
    on=["name_prefix"],
    id_column="id",
    output_dir="output/blocking"
)
candidate_set = blocker.materialize()

# 3. Matching
comparators = [
    StringComparator("name", similarity_function="jaro_winkler"),
    StringComparator("address", similarity_function="jaro_winkler"),
    DateComparator("created_date", max_days_difference=365)
]

weights = [0.5, 0.3, 0.2]

matcher = RuleBasedMatcher()
correspondences = matcher.match(
    df_left, df_right,
    candidates=blocker,
    id_column="id",
    comparators=comparators,
    weights=weights,
    threshold=0.7
)

# 4. Post-clustering (optional)
clusterer = ConnectedComponentClusterer()
refined = clusterer.cluster(correspondences)

# 5. Evaluate
evaluation_set = load_csv(
    "path/to/evaluation_set.csv",
    header=None,
    names=["id1", "id2", "label"]
)

evaluator = EntityMatchingEvaluator()

metrics_blocker = evaluator.evaluate_blocking(candidate_set, evaluation_set)
metrics_matcher = evaluator.evaluate_matching(refined, evaluation_set)
```

## Tutorials

- [Data Integration Tutorial](../tutorial/entity_matching_and_fusion/data_integration_tutorial.ipynb) - Complete pipeline: blocking, matching (rule-based and ML), post-clustering, and evaluation with movie datasets
