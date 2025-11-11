"""Shared helpers for analyzing cluster size distributions."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Optional
import logging

import pandas as pd


def cluster_size_distribution_from_sizes(cluster_sizes: Iterable[int]) -> pd.DataFrame:
    """Build a cluster size distribution DataFrame from an iterable of sizes."""
    sizes = [int(size) for size in cluster_sizes if size is not None]
    if not sizes:
        return pd.DataFrame(columns=["cluster_size", "frequency", "percentage"])

    size_counts = Counter(sizes)
    total_clusters = sum(size_counts.values())
    distribution_data = []

    for cluster_size in sorted(size_counts):
        frequency = size_counts[cluster_size]
        percentage = (frequency / total_clusters * 100) if total_clusters > 0 else 0.0
        distribution_data.append(
            {
                "cluster_size": cluster_size,
                "frequency": frequency,
                "percentage": percentage,
            }
        )

    return pd.DataFrame(distribution_data)


def log_cluster_size_distribution(
    distribution_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
    *,
    header: str = "Cluster Size Distribution",
    total_clusters: Optional[int] = None,
) -> None:
    """Log a formatted cluster size distribution table."""
    logger = logger or logging.getLogger(__name__)

    if distribution_df.empty:
        logger.info("%s: none", header)
        return

    total = (
        int(total_clusters)
        if total_clusters is not None
        else int(distribution_df["frequency"].sum())
    )

    logger.info("%s of %d clusters:", header, total)
    logger.info("\tCluster Size\t| Frequency\t| Percentage")
    logger.info("\t" + "─" * 50)

    for _, row in distribution_df.iterrows():
        if row["cluster_size"] == 1:
            continue
        size_str = f"{int(row['cluster_size'])}"
        freq_str = f"{int(row['frequency'])}"
        perc_str = f"{row['percentage']:.2f}%"
        logger.info("\t\t%s\t|\t%s\t|\t%s", size_str, freq_str, perc_str)
