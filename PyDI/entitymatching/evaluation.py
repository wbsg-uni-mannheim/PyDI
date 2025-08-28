"""
Entity matching evaluation and cluster analysis tools.

This module provides comprehensive evaluation capabilities for entity matching
results, including precision/recall/F1 metrics, cluster consistency analysis,
and threshold analysis for parameter tuning.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from .base import CorrespondenceSet


class EntityMatchingEvaluator:
    """Static methods for entity matching evaluation and analysis.
    
    This evaluator provides comprehensive analysis of entity matching results
    including standard classification metrics, entity-specific metrics like
    candidate recall and pair reduction, and advanced cluster consistency
    analysis using graph-based approaches.
    
    All methods follow PyDI principles by returning file paths for downstream
    consumption and supporting structured output directories.
    """
    
    @staticmethod
    def evaluate(
        corr: CorrespondenceSet,
        test_pairs: pd.DataFrame,
        *,
        threshold: Optional[float] = None,
        candidate_pairs: Optional[pd.DataFrame] = None,
        total_possible_pairs: Optional[int] = None,
        out_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate entity matching correspondences against ground truth.
        
        Computes comprehensive evaluation metrics including classification
        metrics (P/R/F1), entity matching specific metrics (candidate recall,
        pair reduction), and exports detailed results.
        
        Parameters
        ----------
        corr : CorrespondenceSet
            DataFrame with columns id1, id2, score, notes containing
            entity correspondences to evaluate.
        test_pairs : pandas.DataFrame
            Ground truth test pairs. Should have columns id1, id2, and
            optionally a label column (1 for positive, 0 for negative).
            If no label column, assumes all pairs are positive matches.
        threshold : float, optional
            Similarity threshold to apply to correspondences. If None,
            uses all correspondences regardless of score.
        candidate_pairs : pandas.DataFrame, optional
            Candidate pairs that were considered during blocking.
            Used to compute candidate recall metrics. Should have
            columns id1, id2.
        total_possible_pairs : int, optional
            Total number of possible pairs in the Cartesian product.
            Used to compute pair reduction metrics.
        out_dir : str, optional
            Directory to write evaluation results. If provided, writes
            detailed evaluation report as CSV and JSON files.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing evaluation metrics:
            - precision: float, precision score
            - recall: float, recall score  
            - f1: float, F1 score
            - accuracy: float, accuracy score (if negatives available)
            - true_positives: int, number of correct matches found
            - false_positives: int, number of incorrect matches found
            - false_negatives: int, number of missed correct matches
            - true_negatives: int, number of correct non-matches (if available)
            - candidate_recall: float, fraction of true positives found in candidates (if candidate_pairs provided)
            - pair_reduction: float, reduction ratio from total possible pairs (if total_possible_pairs provided)
            - threshold_used: float, threshold applied to correspondences
            - total_correspondences: int, total correspondences before threshold
            - filtered_correspondences: int, correspondences after threshold
            - evaluation_timestamp: str, ISO timestamp of evaluation
        
        Raises
        ------
        ValueError
            If required columns are missing or data formats are invalid.
        """
        # Input validation
        if corr.empty:
            logging.warning("Empty correspondence set provided")
            
        if test_pairs.empty:
            raise ValueError("Empty test_pairs DataFrame provided")
            
        # Validate required columns
        corr_required = ["id1", "id2", "score"]
        for col in corr_required:
            if col not in corr.columns:
                raise ValueError(f"Correspondence set missing required column: {col}")
                
        test_required = ["id1", "id2"] 
        for col in test_required:
            if col not in test_pairs.columns:
                raise ValueError(f"Test pairs missing required column: {col}")
        
        # Apply threshold filtering if provided
        original_corr_count = len(corr)
        if threshold is not None:
            corr_filtered = corr[corr["score"] >= threshold].copy()
            filtered_count = len(corr_filtered)
            logging.info(f"Applied threshold {threshold}: {original_corr_count} -> {filtered_count} correspondences")
        else:
            corr_filtered = corr.copy()
            filtered_count = len(corr_filtered)
            threshold = 0.0  # For reporting
            
        # Normalize pairs for consistent comparison (ensure id1 <= id2)
        predicted_pairs = EntityMatchingEvaluator._normalize_pairs(
            corr_filtered[["id1", "id2"]]
        )
        
        # Process test pairs - check for label column
        has_labels = "label" in test_pairs.columns
        if has_labels:
            positive_pairs = EntityMatchingEvaluator._normalize_pairs(
                test_pairs[test_pairs["label"] == 1][["id1", "id2"]]
            )
            negative_pairs = EntityMatchingEvaluator._normalize_pairs(
                test_pairs[test_pairs["label"] == 0][["id1", "id2"]]
            )
        else:
            # Assume all test pairs are positive
            positive_pairs = EntityMatchingEvaluator._normalize_pairs(
                test_pairs[["id1", "id2"]]
            )
            negative_pairs = set()
            
        # Convert to sets for efficient operations
        predicted_set = set(predicted_pairs)
        positive_set = set(positive_pairs)
        negative_set = set(negative_pairs) if has_labels else set()
        
        # Compute classification metrics
        true_positives = len(predicted_set & positive_set)
        false_positives = len(predicted_set - positive_set)
        false_negatives = len(positive_set - predicted_set)
        
        if has_labels:
            true_negatives = len(negative_set - predicted_set)
        else:
            true_negatives = 0
            
        # Calculate metrics with zero division protection
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-10)
        
        if has_labels:
            accuracy = (true_positives + true_negatives) / max(
                true_positives + false_positives + false_negatives + true_negatives, 1
            )
        else:
            accuracy = None
            
        # Build results dictionary
        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "threshold_used": threshold,
            "total_correspondences": original_corr_count,
            "filtered_correspondences": filtered_count,
            "evaluation_timestamp": datetime.now().isoformat(),
        }
        
        # Add entity matching specific metrics if data provided
        if candidate_pairs is not None:
            candidate_pairs_norm = EntityMatchingEvaluator._normalize_pairs(
                candidate_pairs[["id1", "id2"]]
            )
            candidate_set = set(candidate_pairs_norm)
            candidate_recall = len(positive_set & candidate_set) / max(len(positive_set), 1)
            results["candidate_recall"] = candidate_recall
            results["total_candidates"] = len(candidate_set)
            
        if total_possible_pairs is not None:
            if candidate_pairs is not None:
                pair_reduction = 1.0 - (len(candidate_set) / max(total_possible_pairs, 1))
            else:
                pair_reduction = 1.0 - (filtered_count / max(total_possible_pairs, 1))
            results["pair_reduction"] = pair_reduction
            results["total_possible_pairs"] = total_possible_pairs
            
        # Write results to files if output directory provided
        output_files = []
        if out_dir is not None:
            output_files = EntityMatchingEvaluator._write_evaluation_results(
                results, corr_filtered, test_pairs, positive_set, predicted_set, out_dir
            )
            results["output_files"] = output_files
            
        logging.info(f"Evaluation complete: P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
        return results
    
    @staticmethod
    def create_cluster_consistency_report(
        correspondences: CorrespondenceSet,
        *,
        out_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """Analyze cluster consistency using graph-based transitivity analysis.
        
        Creates a detailed report of cluster consistency by analyzing the
        transitivity properties of entity correspondences. Uses NetworkX
        to find connected components and checks if each cluster has complete
        internal connections (transitive closure).
        
        Parameters
        ----------
        correspondences : CorrespondenceSet
            DataFrame with id1, id2, score, notes columns containing
            entity correspondences to analyze.
        out_dir : str, optional
            Directory to write consistency report. If provided, saves
            the report as CSV and detailed JSON analysis.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with cluster-level consistency analysis:
            - cluster_id: int, unique cluster identifier  
            - cluster_size: int, number of entities in cluster
            - total_edges: int, number of correspondences in cluster
            - expected_edges: int, number of edges in complete graph  
            - consistency_ratio: float, total_edges / expected_edges
            - is_consistent: bool, whether cluster is fully transitive
            - avg_similarity: float, average similarity score in cluster
            - min_similarity: float, minimum similarity score in cluster
            - max_similarity: float, maximum similarity score in cluster
            - entities: str, comma-separated list of entity IDs in cluster
        
        Raises
        ------
        ValueError
            If correspondence set is empty or missing required columns.
        """
        if correspondences.empty:
            raise ValueError("Empty correspondence set provided")
            
        required_cols = ["id1", "id2", "score"]
        for col in required_cols:
            if col not in correspondences.columns:
                raise ValueError(f"Correspondences missing required column: {col}")
        
        # Create graph from correspondences
        G = nx.Graph()
        
        # Add edges with similarity scores
        for _, row in correspondences.iterrows():
            G.add_edge(row["id1"], row["id2"], weight=row["score"])
        
        # Find connected components (clusters)
        clusters = list(nx.connected_components(G))
        
        cluster_reports = []
        
        for i, cluster in enumerate(clusters):
            cluster_nodes = list(cluster)
            cluster_size = len(cluster_nodes)
            
            # Get subgraph for this cluster
            subgraph = G.subgraph(cluster_nodes)
            total_edges = subgraph.number_of_edges()
            
            # Calculate expected edges for complete graph
            expected_edges = cluster_size * (cluster_size - 1) // 2
            
            # Calculate consistency ratio
            consistency_ratio = total_edges / max(expected_edges, 1)
            is_consistent = consistency_ratio >= 0.999  # Allow for floating point errors
            
            # Calculate similarity statistics
            if total_edges > 0:
                edge_weights = [data["weight"] for _, _, data in subgraph.edges(data=True)]
                avg_similarity = sum(edge_weights) / len(edge_weights)
                min_similarity = min(edge_weights)
                max_similarity = max(edge_weights)
            else:
                avg_similarity = min_similarity = max_similarity = 0.0
            
            cluster_reports.append({
                "cluster_id": i,
                "cluster_size": cluster_size,
                "total_edges": total_edges,
                "expected_edges": expected_edges,
                "consistency_ratio": consistency_ratio,
                "is_consistent": is_consistent,
                "avg_similarity": avg_similarity,
                "min_similarity": min_similarity,
                "max_similarity": max_similarity,
                "entities": ",".join(sorted(cluster_nodes)),
            })
        
        # Create DataFrame report
        report_df = pd.DataFrame(cluster_reports)
        
        # Add summary statistics
        total_clusters = len(clusters)
        consistent_clusters = sum(1 for r in cluster_reports if r["is_consistent"])
        inconsistent_clusters = total_clusters - consistent_clusters
        
        logging.info(f"Cluster analysis complete: {total_clusters} clusters found")
        logging.info(f"Consistent: {consistent_clusters}, Inconsistent: {inconsistent_clusters}")
        
        # Write to files if output directory provided
        if out_dir is not None:
            EntityMatchingEvaluator._write_cluster_report(
                report_df, correspondences, out_dir
            )
        
        return report_df
    
    @staticmethod
    def write_record_groups_by_consistency(
        out_path: str,
        correspondences: CorrespondenceSet,
    ) -> str:
        """Export entity record groups organized by cluster consistency.
        
        Creates a structured JSON file containing entity records grouped
        by their cluster consistency status. This is useful for manual
        inspection of matching quality and debugging inconsistent clusters.
        
        Parameters
        ----------
        out_path : str
            Full path to output JSON file.
        correspondences : CorrespondenceSet
            DataFrame with id1, id2, score, notes columns.
            
        Returns
        -------
        str
            Path to the written JSON file.
        
        Raises
        ------
        ValueError
            If correspondences are empty or path is invalid.
        """
        if correspondences.empty:
            raise ValueError("Empty correspondence set provided")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Get cluster consistency report
        cluster_report = EntityMatchingEvaluator.create_cluster_consistency_report(
            correspondences
        )
        
        # Organize data by consistency
        consistent_groups = []
        inconsistent_groups = []
        
        for _, cluster_info in cluster_report.iterrows():
            cluster_data = {
                "cluster_id": int(cluster_info["cluster_id"]),
                "cluster_size": int(cluster_info["cluster_size"]),
                "consistency_ratio": float(cluster_info["consistency_ratio"]),
                "avg_similarity": float(cluster_info["avg_similarity"]),
                "entities": cluster_info["entities"].split(","),
                "total_edges": int(cluster_info["total_edges"]),
                "expected_edges": int(cluster_info["expected_edges"]),
            }
            
            if cluster_info["is_consistent"]:
                consistent_groups.append(cluster_data)
            else:
                inconsistent_groups.append(cluster_data)
        
        # Create output structure
        output_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_correspondences": len(correspondences),
                "total_clusters": len(cluster_report),
                "consistent_clusters": len(consistent_groups),
                "inconsistent_clusters": len(inconsistent_groups),
            },
            "consistent_clusters": consistent_groups,
            "inconsistent_clusters": inconsistent_groups,
        }
        
        # Write JSON file
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Record groups written to {out_path}")
        return out_path
    
    @staticmethod
    def threshold_sweep(
        corr: CorrespondenceSet,
        test_pairs: pd.DataFrame,
        thresholds: Optional[list] = None,
        *,
        out_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """Analyze performance across multiple similarity thresholds.
        
        Performs threshold sweep analysis to generate precision-recall
        curves and identify optimal thresholds for entity matching.
        
        Parameters
        ----------
        corr : CorrespondenceSet
            DataFrame with correspondence results.
        test_pairs : pandas.DataFrame
            Ground truth test pairs.
        thresholds : list, optional
            List of thresholds to evaluate. If None, uses default range.
        out_dir : str, optional
            Directory to write threshold analysis results.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with threshold analysis results containing columns:
            threshold, precision, recall, f1, true_positives, false_positives,
            false_negatives, correspondences_count.
        """
        if thresholds is None:
            thresholds = [i * 0.1 for i in range(0, 11)]  # 0.0 to 1.0 in 0.1 steps
        
        results = []
        
        for threshold in thresholds:
            try:
                eval_result = EntityMatchingEvaluator.evaluate(
                    corr, test_pairs, threshold=threshold
                )
                
                results.append({
                    "threshold": threshold,
                    "precision": eval_result["precision"],
                    "recall": eval_result["recall"],
                    "f1": eval_result["f1"],
                    "true_positives": eval_result["true_positives"],
                    "false_positives": eval_result["false_positives"],
                    "false_negatives": eval_result["false_negatives"],
                    "correspondences_count": eval_result["filtered_correspondences"],
                })
            except Exception as e:
                logging.warning(f"Error evaluating threshold {threshold}: {e}")
                continue
        
        sweep_df = pd.DataFrame(results)
        
        if out_dir is not None:
            out_path = os.path.join(out_dir, "threshold_sweep.csv")
            os.makedirs(out_dir, exist_ok=True)
            sweep_df.to_csv(out_path, index=False)
            logging.info(f"Threshold sweep results written to {out_path}")
        
        return sweep_df
    
    @staticmethod
    def _normalize_pairs(pairs_df: pd.DataFrame) -> list:
        """Normalize pairs to ensure consistent comparison (id1 <= id2)."""
        normalized = []
        for _, row in pairs_df.iterrows():
            id1, id2 = row["id1"], row["id2"]
            if id1 <= id2:
                normalized.append((id1, id2))
            else:
                normalized.append((id2, id1))
        return normalized
    
    @staticmethod
    def _write_evaluation_results(
        results: Dict[str, Any],
        correspondences: pd.DataFrame,
        test_pairs: pd.DataFrame,
        positive_set: Set[Tuple[str, str]],
        predicted_set: Set[Tuple[str, str]],
        out_dir: str,
    ) -> list:
        """Write detailed evaluation results to files."""
        os.makedirs(out_dir, exist_ok=True)
        output_files = []
        
        # Write summary JSON
        json_path = os.path.join(out_dir, "evaluation_summary.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        output_files.append(json_path)
        
        # Write detailed correspondence analysis
        detailed_results = []
        for _, row in correspondences.iterrows():
            pair = EntityMatchingEvaluator._normalize_pairs(
                pd.DataFrame([{"id1": row["id1"], "id2": row["id2"]}])
            )[0]
            
            is_correct = pair in positive_set
            detailed_results.append({
                "id1": row["id1"],
                "id2": row["id2"],
                "score": row["score"],
                "is_correct": is_correct,
                "classification": "TP" if is_correct else "FP",
            })
        
        detailed_df = pd.DataFrame(detailed_results)
        csv_path = os.path.join(out_dir, "detailed_results.csv")
        detailed_df.to_csv(csv_path, index=False)
        output_files.append(csv_path)
        
        return output_files
    
    @staticmethod
    def _write_cluster_report(
        report_df: pd.DataFrame,
        correspondences: pd.DataFrame,
        out_dir: str,
    ) -> None:
        """Write cluster consistency report to files."""
        os.makedirs(out_dir, exist_ok=True)
        
        # Write CSV report
        csv_path = os.path.join(out_dir, "cluster_consistency_report.csv")
        report_df.to_csv(csv_path, index=False)
        
        # Write summary JSON
        summary = {
            "total_clusters": len(report_df),
            "consistent_clusters": int(report_df["is_consistent"].sum()),
            "inconsistent_clusters": int((~report_df["is_consistent"]).sum()),
            "avg_cluster_size": float(report_df["cluster_size"].mean()),
            "avg_consistency_ratio": float(report_df["consistency_ratio"].mean()),
            "generated_at": datetime.now().isoformat(),
        }
        
        json_path = os.path.join(out_dir, "cluster_analysis_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Cluster report written to {out_dir}")
    
    @staticmethod
    def create_cluster_size_distribution(
        correspondences: CorrespondenceSet,
        *,
        out_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """Create cluster size distribution from correspondences.
        
        Analyzes the distribution of cluster sizes by creating connected components
        from the correspondences and counting the frequency of each cluster size.
        This follows the Winter framework approach for cluster size analysis.
        
        Parameters
        ----------
        correspondences : CorrespondenceSet
            DataFrame with id1, id2, score, notes columns containing
            entity correspondences to analyze.
        out_dir : str, optional
            Directory to write cluster size distribution. If provided, saves
            the distribution as a CSV file.
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with cluster size distribution containing columns:
            - cluster_size: int, size of the cluster (number of entities)
            - frequency: int, number of clusters with this size
            - percentage: float, percentage of total clusters with this size
            
        Raises
        ------
        ValueError
            If correspondence set is empty or missing required columns.
        """
        if correspondences.empty:
            raise ValueError("Empty correspondence set provided")
            
        required_cols = ["id1", "id2", "score"]
        for col in required_cols:
            if col not in correspondences.columns:
                raise ValueError(f"Correspondences missing required column: {col}")
        
        # Create graph from correspondences to find connected components
        G = nx.Graph()
        
        # Add edges (no weights needed for clustering analysis)
        for _, row in correspondences.iterrows():
            G.add_edge(row["id1"], row["id2"])
        
        # Find connected components (clusters)
        clusters = list(nx.connected_components(G))
        
        # Count cluster sizes
        size_distribution = {}
        for cluster in clusters:
            cluster_size = len(cluster)
            size_distribution[cluster_size] = size_distribution.get(cluster_size, 0) + 1
        
        # Create distribution DataFrame
        distribution_data = []
        total_clusters = len(clusters)
        
        for cluster_size in sorted(size_distribution.keys()):
            frequency = size_distribution[cluster_size]
            percentage = (frequency / total_clusters * 100) if total_clusters > 0 else 0.0
            distribution_data.append({
                "cluster_size": cluster_size,
                "frequency": frequency,
                "percentage": percentage
            })
        
        distribution_df = pd.DataFrame(distribution_data)
        
        # Log distribution to console
        logging.info(f"Cluster Size Distribution of {total_clusters} clusters:")
        logging.info("\tCluster Size\t| Frequency\t| Percentage")
        logging.info("\t" + "─" * 50)
        
        for _, row in distribution_df.iterrows():
            size_str = f"{int(row['cluster_size'])}"
            freq_str = f"{int(row['frequency'])}"
            perc_str = f"{row['percentage']:.2f}%"
            logging.info(f"\t\t{size_str}\t|\t{freq_str}\t|\t{perc_str}")
        
        # Write to file if output directory provided
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, "cluster_size_distribution.csv")
            distribution_df.to_csv(output_path, index=False)
            logging.info(f"Cluster size distribution written to {output_path}")
        
        return distribution_df