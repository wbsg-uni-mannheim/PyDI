# Data Fusion

Fuse matched records into a consolidated dataset using attribute‑level conflict resolution functions, with provenance tracking and rich reports. Fusion closes the loop by reconciling conflicting values and summarizing lineage.

Modules
- `PyDI.fusion.base` — `AttributeValueFuser`, `FusionContext`, `RecordGroup`
- `PyDI.fusion.engine` — fusion process over groups + rules
- `PyDI.fusion.conflict_resolution` — numeric/list/date strategies (average, median, most_recent, union, ...)
- `PyDI.fusion.strategy.DataFusionStrategy` — register per‑attribute fusers and evaluation functions
- `PyDI.fusion.provenance` — provenance tracking and export
- `PyDI.fusion.reporting.FusionReport` — HTML/JSON reports
- `PyDI.fusion.evaluation` — density/coverage/consistency metrics

Conflict Resolution Strategies
- Numeric: `average`, `median`, `min`, `max`, `sum` — robust to noisy numeric disagreement.
- Date: `most_recent`, `earliest` — pick timeline extremes.
- Lists/Text: `union`, `intersection`, `intersection_k_sources` — consolidate multi‑valued fields.
- Custom: implement a callable returning `FusionResult` with value + metadata.

Provenance and Trust
- Track source datasets, per‑record lineage, and optional trust scores to bias resolution.
- Export provenance for audit or explainability.

Record Grouping and 1:1 Enforcement
- Group using correspondences (possibly post‑processed via bipartite matching or clustering).
- Each fused record contains the union of attributes from its group members.

Reporting and Evaluation
- `FusionReport` computes density, conflict stats, and attribute coverage; renders HTML/JSON.
- Evaluate against a gold fused set if available for accuracy.

Example
```python
from PyDI.fusion.strategy import DataFusionStrategy
from PyDI.fusion.engine import DataFuser
from PyDI.fusion.conflict_resolution.numeric import average
from PyDI.fusion.conflict_resolution.date import most_recent
from PyDI.fusion.conflict_resolution.list import union

strategy = DataFusionStrategy()
strategy.add_attribute_fuser("title", union)
strategy.add_attribute_fuser("release_date", most_recent)
strategy.add_attribute_fuser("rating", average)

fuser = DataFuser(strategy=strategy, out_dir="output/fusion")
fused_df = fuser.fuse(datasets=[df_a, df_b], correspondences=matches)
```

Artifacts
- Fused dataset (CSV/Parquet), JSON and HTML fusion reports, provenance export under `out_dir`.
