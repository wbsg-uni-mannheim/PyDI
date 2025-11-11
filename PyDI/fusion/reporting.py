"""
Reporting and diagnostics for data fusion in PyDI.

This module provides comprehensive reporting capabilities for fusion results,
including consistency analysis, conflict reporting, and debug information.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import pandas as pd
import logging
from datetime import datetime
try:
    from jinja2 import Template
except ImportError:
    Template = None

from .base import _is_valid_value

from .evaluation import calculate_consistency_metrics, calculate_coverage_metrics
from .analysis import AttributeCoverageAnalyzer


def _is_missing(value) -> bool:
    """Check if a value is missing (None, NA, or empty list).
    
    This is the inverse of _is_valid_value from the base module.
    """
    return not _is_valid_value(value)


class FusionReport:
    """Comprehensive report for data fusion results.
    
    Parameters
    ----------
    fused_df : pd.DataFrame
        The fused dataset.
    input_datasets : List[pd.DataFrame]
        The original input datasets.
    strategy_name : str
        Name of the fusion strategy used.
    correspondences : pd.DataFrame
        The correspondences used for fusion.
    evaluation_results : Dict[str, float], optional
        Results from fusion evaluation.
    """
    
    def __init__(
        self,
        fused_df: pd.DataFrame,
        input_datasets: List[pd.DataFrame],
        strategy_name: str,
        correspondences: pd.DataFrame,
        evaluation_results: Optional[Dict[str, float]] = None,
        include_coverage_analysis: bool = True,
    ):
        self.fused_df = fused_df
        self.input_datasets = input_datasets
        self.strategy_name = strategy_name
        self.correspondences = correspondences
        self.evaluation_results = evaluation_results or {}
        self.generated_at = datetime.now()
        self._logger = logging.getLogger(__name__)
        
        # Initialize coverage analyzer if requested
        self.coverage_analyzer = None
        if include_coverage_analysis and self.input_datasets:
            try:
                dataset_names = [df.attrs.get('dataset_name', f'dataset_{i}') 
                               for i, df in enumerate(self.input_datasets)]
                self.coverage_analyzer = AttributeCoverageAnalyzer(
                    self.input_datasets, dataset_names
                )
            except Exception as e:
                self._logger.warning(f"Could not initialize coverage analyzer: {e}")
        
        # Calculate metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate all fusion metrics."""
        self.quality_metrics = calculate_consistency_metrics(self.fused_df)
        self.coverage_metrics = calculate_coverage_metrics(
            self.input_datasets, self.fused_df
        )
        self.group_statistics = self._calculate_group_statistics()
        self.attribute_statistics = self._calculate_attribute_statistics()
        self.conflict_analysis = self._analyze_conflicts()
    
    def _calculate_group_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about record groups."""
        stats = {
            "total_groups": 0,
            "multi_record_groups": 0,
            "singleton_groups": 0,
            "largest_group_size": 0,
            "average_group_size": 0.0,
            "group_size_distribution": {},
        }
        
        if "_fusion_sources" in self.fused_df.columns:
            source_counts = self.fused_df["_fusion_sources"].apply(len)
            stats["total_groups"] = len(self.fused_df)
            stats["multi_record_groups"] = (source_counts > 1).sum()
            stats["singleton_groups"] = (source_counts == 1).sum()
            stats["largest_group_size"] = source_counts.max()
            stats["average_group_size"] = source_counts.mean()
            
            # Group size distribution
            size_dist = source_counts.value_counts().sort_index()
            stats["group_size_distribution"] = size_dist.to_dict()
        
        return stats
    
    def _calculate_attribute_statistics(self) -> Dict[str, Any]:
        """Calculate statistics about attribute fusion."""
        stats = {
            "total_attributes": 0,
            "attributes_with_conflicts": 0,
            "most_conflicted_attribute": None,
            "rule_usage": {},
            "attribute_coverage": {},
        }
        
        # Get data attributes (excluding fusion metadata)
        data_columns = [col for col in self.fused_df.columns if not col.startswith("_fusion_")]
        stats["total_attributes"] = len(data_columns)
        
        # Analyze rule usage from metadata
        if "_fusion_metadata" in self.fused_df.columns:
            rule_counts = {}
            attribute_rule_usage = {}
            
            for _, row in self.fused_df.iterrows():
                metadata = row.get("_fusion_metadata", {})
                if isinstance(metadata, dict):
                    for key, value in metadata.items():
                        if key.endswith("_rule"):
                            attribute = key[:-5]  # Remove '_rule' suffix
                            rule_counts[value] = rule_counts.get(value, 0) + 1
                            if attribute not in attribute_rule_usage:
                                attribute_rule_usage[attribute] = {}
                            attribute_rule_usage[attribute][value] = attribute_rule_usage[attribute].get(value, 0) + 1
            
            stats["rule_usage"] = rule_counts
            stats["attribute_rule_usage"] = attribute_rule_usage
            
            # Find most conflicted attribute (most rule diversity)
            max_rules = 0
            most_conflicted = None
            for attr, rules in attribute_rule_usage.items():
                if len(rules) > max_rules:
                    max_rules = len(rules)
                    most_conflicted = attr
            
            stats["most_conflicted_attribute"] = most_conflicted
            stats["attributes_with_conflicts"] = sum(1 for rules in attribute_rule_usage.values() if len(rules) > 1)
        
        # Calculate attribute coverage
        for dataset in self.input_datasets:
            dataset_name = dataset.attrs.get("dataset_name", "unknown")
            common_attrs = set(dataset.columns).intersection(set(data_columns))
            stats["attribute_coverage"][dataset_name] = len(common_attrs) / len(data_columns) if data_columns else 0.0
        
        return stats
    
    def _analyze_conflicts(self) -> Dict[str, Any]:
        """Analyze conflicts that occurred during fusion."""
        analysis = {
            "total_potential_conflicts": 0,
            "resolved_conflicts": 0,
            "unresolved_conflicts": 0,
            "conflict_patterns": {},
            "confidence_distribution": {},
        }
        
        # Analyze confidence scores
        if "_fusion_confidence" in self.fused_df.columns:
            confidences = self.fused_df["_fusion_confidence"].dropna()
            
            # Create confidence bins
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ["very_low", "low", "medium", "high", "very_high"]
            confidence_bins = pd.cut(confidences, bins=bins, labels=labels, include_lowest=True)
            
            analysis["confidence_distribution"] = confidence_bins.value_counts().to_dict()
            
            # Consider records with low confidence as having conflicts
            analysis["total_potential_conflicts"] = len(confidences)
            analysis["resolved_conflicts"] = (confidences >= 0.5).sum()
            analysis["unresolved_conflicts"] = (confidences < 0.5).sum()
        
        # Analyze conflict patterns from source information
        source_column = "_fusion_source_datasets" if "_fusion_source_datasets" in self.fused_df.columns else "_fusion_sources"
        if source_column in self.fused_df.columns:
            source_combinations = {}
            for sources in self.fused_df[source_column]:
                if isinstance(sources, list) and len(sources) > 1:
                    normalized = sorted(set(sources))
                    key = tuple(normalized)
                    source_combinations[key] = source_combinations.get(key, 0) + 1

            analysis["conflict_patterns"] = {
                str(k): v for k, v in source_combinations.items()
            }

        return analysis
    
    def print_summary(self):
        """Print a summary of the fusion report."""
        print(f"\n=== PyDI Data Fusion Report ===")
        print(f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Strategy: {self.strategy_name}")
        print()
        
        # Input/Output Summary
        print("Data Summary:")
        print(f"  Input datasets: {len(self.input_datasets)}")
        print(f"  Input records: {sum(len(df) for df in self.input_datasets)}")
        print(f"  Output records: {len(self.fused_df)}")
        print(f"  Correspondences: {len(self.correspondences)}")
        print(f"  Record coverage: {self.coverage_metrics['record_coverage']:.2%}")
        print()
        
        # Quality Metrics
        print("Quality Metrics:")
        print(f"  Mean confidence: {self.quality_metrics['mean_confidence']:.3f}")
        print(f"  Multi-source records: {self.quality_metrics['multi_source_records']}")
        print(f"  Single-source records: {self.quality_metrics['single_source_records']}")
        print()
        
        # Group Statistics
        print("Group Statistics:")
        print(f"  Total groups: {self.group_statistics['total_groups']}")
        print(f"  Multi-record groups: {self.group_statistics['multi_record_groups']}")
        print(f"  Average group size: {self.group_statistics['average_group_size']:.2f}")
        print(f"  Largest group: {self.group_statistics['largest_group_size']} records")
        print()
        
        # Attribute Statistics
        print("Attribute Statistics:")
        print(f"  Total attributes: {self.attribute_statistics['total_attributes']}")
        print(f"  Attributes with conflicts: {self.attribute_statistics['attributes_with_conflicts']}")
        if self.attribute_statistics['most_conflicted_attribute']:
            print(f"  Most conflicted: {self.attribute_statistics['most_conflicted_attribute']}")
        print()
        
        # Rule Usage
        if self.quality_metrics.get("rule_usage"):
            print("Rule Usage:")
            for rule, count in sorted(self.quality_metrics["rule_usage"].items()):
                print(f"  {rule}: {count} applications")
            print()
        
        # Evaluation Results
        if self.evaluation_results:
            print("Evaluation Results:")
            if "overall_accuracy" in self.evaluation_results:
                print(f"  Overall accuracy: {self.evaluation_results['overall_accuracy']:.3f}")
            if "macro_accuracy" in self.evaluation_results:
                print(f"  Macro accuracy: {self.evaluation_results['macro_accuracy']:.3f}")
            print(f"  Evaluated records: {self.evaluation_results.get('num_evaluated_records', 0)}")
            print()
    
    def print_evaluation_examples(self, gold_df: pd.DataFrame, fused_id_column: str, gold_id_column: str, max_examples: int = 3):
        """Print examples of correct and incorrect fusion results.
        
        Parameters
        ----------
        gold_df : pd.DataFrame
            The expected/validation dataset.
        fused_id_column : str
            ID column name in the fused dataset.
        gold_id_column : str
            ID column name in the expected dataset.
        max_examples : int, default 3
            Maximum number of examples to show for each category.
        """
        if not self.evaluation_results:
            print("No evaluation results available for examples.")
            return
        
        examples = self.get_evaluation_examples(gold_df, fused_id_column, gold_id_column, max_examples)
        
        if "error" in examples:
            print(f"Error getting examples: {examples['error']}")
            return
        
        print("\n=== Fusion Examples ===")
        
        # Show incorrect examples (most important for debugging)
        if examples["incorrect_examples"]:
            print("\n❌ Incorrect Fusion Examples:")
            for i, ex in enumerate(examples["incorrect_examples"], 1):
                print(f"\n  Example {i}:")
                print(f"    Record ID: {ex['record_id']}")
                print(f"    Attribute: {ex['attribute']}")
                print(f"    Fused:  '{ex['fused_value']}'")
                print(f"    Expected:   '{ex['expected_value']}'")
                print(f"    Confidence: {ex['confidence']}")
                if ex['sources']:
                    print(f"    Sources: {', '.join(ex['sources'])}")
        else:
            print("\n❌ No incorrect examples found in sample.")
        
        # Show correct examples (for validation)
        if examples["correct_examples"]:
            print(f"\n✅ Correct Fusion Examples:")
            for i, ex in enumerate(examples["correct_examples"], 1):
                print(f"\n  Example {i}:")
                print(f"    Record ID: {ex['record_id']}")
                print(f"    Attribute: {ex['attribute']}")
                print(f"    Value: '{ex['fused_value']}' (matches expected)")
                print(f"    Confidence: {ex['confidence']}")
                if ex['sources']:
                    print(f"    Sources: {', '.join(ex['sources'])}")
        else:
            print("\n✅ No correct examples found in sample.")
        
        print(f"\nSample size: {examples['total_compared']} records, {len(examples['attributes_compared'])} attributes compared")
        print()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "metadata": {
                "generated_at": self.generated_at.isoformat(),
                "strategy_name": self.strategy_name,
                "num_input_datasets": len(self.input_datasets),
                "num_correspondences": len(self.correspondences),
            },
            "quality_metrics": self.quality_metrics,
            "coverage_metrics": self.coverage_metrics,
            "group_statistics": self.group_statistics,
            "attribute_statistics": self.attribute_statistics,
            "conflict_analysis": self.conflict_analysis,
            "evaluation_results": self.evaluation_results,
        }
    
    def to_json(self, path: Optional[str] = None) -> str:
        """Export report as JSON.
        
        Parameters
        ----------
        path : str, optional
            File path to save JSON. If None, returns JSON string.
            
        Returns
        -------
        str
            JSON representation of the report.
        """
        report_dict = self.to_dict()
        json_str = json.dumps(report_dict, indent=2, default=str)
        
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(json_str)
            self._logger.info(f"Report saved to {path}")
        
        return json_str
    
    def to_html(self, path: str):
        """Export report as HTML file.
        
        Parameters
        ----------
        path : str
            File path to save HTML report.
        """
        html_content = self._generate_html()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            f.write(html_content)
        
        self._logger.info(f"HTML report saved to {path}")
    
    def _generate_html(self) -> str:
        """Generate HTML report content using Jinja2 template."""
        if Template is None:
            # Fallback to simple string formatting if Jinja2 is not available
            return self._generate_html_fallback()
        
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>PyDI Data Fusion Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #2c5aa0; color: white; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-value { font-weight: bold; color: #2c5aa0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .good { color: #28a745; }
        .warning { color: #ffc107; }
        .error { color: #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <h1>PyDI Data Fusion Report</h1>
        <p>Strategy: {{ strategy_name }}</p>
        <p>Generated: {{ generated_at }}</p>
    </div>
    
    <div class="section">
        <h2>Data Summary</h2>
        <div class="metric">Input Datasets: <span class="metric-value">{{ num_input_datasets }}</span></div>
        <div class="metric">Input Records: <span class="metric-value">{{ total_input_records }}</span></div>
        <div class="metric">Output Records: <span class="metric-value">{{ num_output_records }}</span></div>
        <div class="metric">Correspondences: <span class="metric-value">{{ num_correspondences }}</span></div>
        <div class="metric">Record Coverage: <span class="metric-value">{{ coverage_metrics.record_coverage | round(4) * 100 }}%</span></div>
    </div>
    
    <div class="section">
        <h2>Quality Metrics</h2>
        <div class="metric">Mean Confidence: <span class="metric-value">{{ "%.3f" | format(quality_metrics.mean_confidence) }}</span></div>
        <div class="metric">Multi-source Records: <span class="metric-value">{{ quality_metrics.multi_source_records }}</span></div>
        <div class="metric">Single-source Records: <span class="metric-value">{{ quality_metrics.single_source_records }}</span></div>
    </div>
    
    <div class="section">
        <h2>Group Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Groups</td><td>{{ group_statistics.total_groups }}</td></tr>
            <tr><td>Multi-record Groups</td><td>{{ group_statistics.multi_record_groups }}</td></tr>
            <tr><td>Singleton Groups</td><td>{{ group_statistics.singleton_groups }}</td></tr>
            <tr><td>Average Group Size</td><td>{{ "%.2f" | format(group_statistics.average_group_size) }}</td></tr>
            <tr><td>Largest Group Size</td><td>{{ group_statistics.largest_group_size }}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Attribute Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Attributes</td><td>{{ attribute_statistics.total_attributes }}</td></tr>
            <tr><td>Attributes with Conflicts</td><td>{{ attribute_statistics.attributes_with_conflicts }}</td></tr>
            <tr><td>Most Conflicted Attribute</td><td>{{ attribute_statistics.most_conflicted_attribute or 'None' }}</td></tr>
        </table>
    </div>
    
    {% if rule_usage %}
    <div class="section">
        <h2>Rule Usage</h2>
        <table>
            <tr><th>Rule</th><th>Applications</th></tr>
            {% for rule, count in rule_usage | dictsort %}
            <tr><td>{{ rule }}</td><td>{{ count }}</td></tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    
    {% if evaluation_results %}
    <div class="section">
        <h2>Evaluation Results</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {% for key, value in evaluation_results.items() %}
            <tr><td>{{ key }}</td><td>{% if value is number %}{{ "%.3f" | format(value) }}{% else %}{{ value }}{% endif %}</td></tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}
    
</body>
</html>
        """
        
        template = Template(template_str)
        
        context = {
            'strategy_name': self.strategy_name,
            'generated_at': self.generated_at.strftime('%Y-%m-%d %H:%M:%S'),
            'num_input_datasets': len(self.input_datasets),
            'total_input_records': sum(len(df) for df in self.input_datasets),
            'num_output_records': len(self.fused_df),
            'num_correspondences': len(self.correspondences),
            'coverage_metrics': self.coverage_metrics,
            'quality_metrics': self.quality_metrics,
            'group_statistics': self.group_statistics,
            'attribute_statistics': self.attribute_statistics,
            'rule_usage': self.quality_metrics.get("rule_usage"),
            'evaluation_results': self.evaluation_results if self.evaluation_results else None,
        }
        
        return template.render(context)
    
    def _generate_html_fallback(self) -> str:
        """Fallback HTML generation without Jinja2."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PyDI Data Fusion Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c5aa0; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-weight: bold; color: #2c5aa0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .good {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .error {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PyDI Data Fusion Report</h1>
        <p>Strategy: {self.strategy_name}</p>
        <p>Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Data Summary</h2>
        <div class="metric">Input Datasets: <span class="metric-value">{len(self.input_datasets)}</span></div>
        <div class="metric">Input Records: <span class="metric-value">{sum(len(df) for df in self.input_datasets)}</span></div>
        <div class="metric">Output Records: <span class="metric-value">{len(self.fused_df)}</span></div>
        <div class="metric">Correspondences: <span class="metric-value">{len(self.correspondences)}</span></div>
        <div class="metric">Record Coverage: <span class="metric-value">{self.coverage_metrics['record_coverage']:.2%}</span></div>
    </div>
    
    <div class="section">
        <h2>Quality Metrics</h2>
        <div class="metric">Mean Confidence: <span class="metric-value">{self.quality_metrics['mean_confidence']:.3f}</span></div>
        <div class="metric">Multi-source Records: <span class="metric-value">{self.quality_metrics['multi_source_records']}</span></div>
        <div class="metric">Single-source Records: <span class="metric-value">{self.quality_metrics['single_source_records']}</span></div>
    </div>
    
    <div class="section">
        <h2>Group Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Groups</td><td>{self.group_statistics['total_groups']}</td></tr>
            <tr><td>Multi-record Groups</td><td>{self.group_statistics['multi_record_groups']}</td></tr>
            <tr><td>Singleton Groups</td><td>{self.group_statistics['singleton_groups']}</td></tr>
            <tr><td>Average Group Size</td><td>{self.group_statistics['average_group_size']:.2f}</td></tr>
            <tr><td>Largest Group Size</td><td>{self.group_statistics['largest_group_size']}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Attribute Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Attributes</td><td>{self.attribute_statistics['total_attributes']}</td></tr>
            <tr><td>Attributes with Conflicts</td><td>{self.attribute_statistics['attributes_with_conflicts']}</td></tr>
            <tr><td>Most Conflicted Attribute</td><td>{self.attribute_statistics.get('most_conflicted_attribute', 'None')}</td></tr>
        </table>
    </div>
"""
        
        # Add rule usage table if available
        if self.quality_metrics.get("rule_usage"):
            html += """
    <div class="section">
        <h2>Rule Usage</h2>
        <table>
            <tr><th>Rule</th><th>Applications</th></tr>
"""
            for rule, count in sorted(self.quality_metrics["rule_usage"].items()):
                html += f"            <tr><td>{rule}</td><td>{count}</td></tr>\n"
            html += "        </table>\n    </div>\n"
        
        # Add evaluation results if available
        if self.evaluation_results:
            html += """
    <div class="section">
        <h2>Evaluation Results</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""
            for key, value in self.evaluation_results.items():
                if isinstance(value, float):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                html += f"            <tr><td>{key}</td><td>{formatted_value}</td></tr>\n"
            html += "        </table>\n    </div>\n"
        
        html += """
</body>
</html>
"""
        return html
    
    def print_coverage_analysis(self, max_attributes: int = 15):
        """Print attribute coverage analysis if available."""
        if self.coverage_analyzer:
            print("\nAttribute Coverage Analysis:")
            print("=" * 50)
            self.coverage_analyzer.print_summary(max_attributes)
        else:
            print("\nCoverage analysis not available (disabled or failed to initialize)")
    
    def get_coverage_dataframe(self) -> Optional[pd.DataFrame]:
        """Get the coverage analysis DataFrame."""
        if self.coverage_analyzer:
            return self.coverage_analyzer.coverage_df
        return None
    
    def suggest_fusion_rules(self) -> Optional[Dict[str, str]]:
        """Get fusion rule suggestions based on coverage analysis."""
        if self.coverage_analyzer:
            return self.coverage_analyzer.suggest_fusion_rules()
        return None
    
    def get_evaluation_examples(self, gold_df: pd.DataFrame, fused_id_column: str, gold_id_column: str, max_examples: int = 5) -> Dict[str, Any]:
        """Get examples of correct and incorrect fusion results.
        
        Parameters
        ----------
        gold_df : pd.DataFrame
            The expected/validation dataset.
        fused_id_column : str
            ID column name in the fused dataset.
        gold_id_column : str
            ID column name in the expected dataset.
        max_examples : int, default 5
            Maximum number of examples to return for each category.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing examples of correct and incorrect results.
        """
        if not self.evaluation_results:
            return {"error": "No evaluation results available"}
        
        # Align datasets by ID
        fused_ids = set(self.fused_df[fused_id_column].dropna().astype(str))
        gold_ids = set(gold_df[gold_id_column].dropna().astype(str))
        common_ids = fused_ids.intersection(gold_ids)
        
        if not common_ids:
            return {"error": "No common IDs found between fused and expected datasets"}
        
        # Filter to common IDs
        aligned_fused = self.fused_df[self.fused_df[fused_id_column].astype(str).isin(common_ids)].copy()
        aligned_gold = gold_df[gold_df[gold_id_column].astype(str).isin(common_ids)].copy()
        
        # Sort for consistent comparison
        aligned_fused = aligned_fused.sort_values(fused_id_column).reset_index(drop=True)
        aligned_gold = aligned_gold.sort_values(gold_id_column).reset_index(drop=True)
        
        # Get common attributes (excluding metadata columns)
        fused_attrs = set(col for col in aligned_fused.columns if not col.startswith('_fusion_'))
        gold_attrs = set(aligned_gold.columns)
        common_attrs = fused_attrs.intersection(gold_attrs) - {fused_id_column, gold_id_column}
        
        examples = {
            "correct_examples": [],
            "incorrect_examples": [],
            "total_compared": len(aligned_fused),
            "attributes_compared": list(common_attrs)
        }
        
        # Compare records and collect examples
        for idx in range(min(len(aligned_fused), len(aligned_gold))):
            fused_row = aligned_fused.iloc[idx]
            gold_row = aligned_gold.iloc[idx]
            
            record_id = fused_row[fused_id_column]
            
            # Compare each attribute
            for attr in common_attrs:
                if attr in fused_row and attr in gold_row:
                    fused_val = fused_row[attr]
                    gold_val = gold_row[attr]
                    
                    # Skip if both are missing
                    if _is_missing(fused_val) and _is_missing(gold_val):
                        continue
                    
                    # Simple comparison (could be enhanced with evaluation functions)
                    is_correct = self._simple_attribute_match(fused_val, gold_val)
                    
                    example = {
                        "record_id": record_id,
                        "attribute": attr,
                        "fused_value": fused_val,
                        "expected_value": gold_val,
                        "gold_value": gold_val,
                        "confidence": fused_row.get("_fusion_confidence", "N/A"),
                        "sources": fused_row.get("_fusion_sources", [])
                    }
                    
                    if is_correct and len(examples["correct_examples"]) < max_examples:
                        examples["correct_examples"].append(example)
                    elif not is_correct and len(examples["incorrect_examples"]) < max_examples:
                        examples["incorrect_examples"].append(example)
                    
                    # Stop collecting if we have enough examples
                    if (len(examples["correct_examples"]) >= max_examples and 
                        len(examples["incorrect_examples"]) >= max_examples):
                        break
            
            if (len(examples["correct_examples"]) >= max_examples and 
                len(examples["incorrect_examples"]) >= max_examples):
                break
        
        return examples
    
    def _simple_attribute_match(self, fused_val, gold_val) -> bool:
        """Simple attribute matching logic."""
        # Handle missing values
        if _is_missing(fused_val) and _is_missing(gold_val):
            return True
        if _is_missing(fused_val) or _is_missing(gold_val):
            return False
        
        # Handle string comparison (case insensitive, strip whitespace)
        if isinstance(fused_val, str) and isinstance(gold_val, str):
            return fused_val.strip().lower() == gold_val.strip().lower()
        
        # Handle list comparison (for aggregated values)
        if isinstance(fused_val, str) and isinstance(gold_val, str):
            # Check if one might be a comma-separated list
            if ',' in fused_val or ',' in gold_val:
                fused_items = set(item.strip().lower() for item in str(fused_val).split(','))
                gold_items = set(item.strip().lower() for item in str(gold_val).split(','))
                return fused_items == gold_items
        
        # Default exact comparison
        return fused_val == gold_val
    
    def export_detailed_results(self, output_dir: str):
        """Export detailed fusion results to directory.
        
        Parameters
        ----------
        output_dir : str
            Directory to save detailed results.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main fusion results
        self.fused_df.to_csv(output_path / "fused_data.csv", index=False)
        
        # Save report in multiple formats
        self.to_json(str(output_path / "fusion_report.json"))
        self.to_html(str(output_path / "fusion_report.html"))
        
        # Save correspondences
        self.correspondences.to_csv(output_path / "correspondences.csv", index=False)
        
        # Save input dataset summary
        input_summary = []
        for i, df in enumerate(self.input_datasets):
            summary = {
                "dataset_index": i,
                "dataset_name": df.attrs.get("dataset_name", f"dataset_{i}"),
                "num_records": len(df),
                "num_attributes": len(df.columns),
                "attributes": list(df.columns),
            }
            input_summary.append(summary)
        
        with open(output_path / "input_summary.json", 'w') as f:
            json.dump(input_summary, f, indent=2)
        
        # Export coverage analysis if available
        if self.coverage_analyzer:
            try:
                self.coverage_analyzer.export_analysis(output_path / 'coverage_analysis')
                self._logger.info("Coverage analysis exported successfully")
            except Exception as e:
                self._logger.warning(f"Could not export coverage analysis: {e}")
        
        self._logger.info(f"Detailed results exported to {output_dir}")
