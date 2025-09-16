"""
Provenance tracking utilities for PyDI data fusion.

This module provides utilities for tracking data lineage and provenance
information throughout the fusion process.
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


@dataclass
class ProvenanceInfo:
    """Provenance information for a piece of data.
    
    Parameters
    ----------
    sources : Set[str]
        Set of source identifiers that contributed to this data.
    timestamp : datetime
        When this data was created or last modified.
    operation : str
        The operation that produced this data.
    confidence : float
        Confidence score for this data (0.0 to 1.0).
    metadata : Dict[str, Any]
        Additional metadata about the provenance.
    """
    
    sources: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.now)
    operation: str = "unknown"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sources": list(self.sources),
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProvenanceInfo:
        """Create from dictionary."""
        return cls(
            sources=set(data.get("sources", [])),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            operation=data.get("operation", "unknown"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


class ProvenanceTracker:
    """Track provenance information throughout data processing.
    
    This class maintains a registry of provenance information for data
    elements and provides utilities for tracking lineage.
    """
    
    def __init__(self):
        self._provenance_registry: Dict[str, ProvenanceInfo] = {}
        self._dataset_sources: Dict[str, str] = {}  # record_id -> dataset_name
        self._trust_scores: Dict[str, float] = {}  # source -> trust_score
    
    def register_dataset_source(self, dataset_name: str, trust_score: float = 1.0):
        """Register a data source with optional trust score.
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset/source.
        trust_score : float
            Trust score for this source (0.0 to 1.0).
        """
        self._trust_scores[dataset_name] = trust_score
    
    def track_input_data(self, df: pd.DataFrame, dataset_name: str):
        """Track provenance for input dataset records.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        dataset_name : str
            Name of the dataset.
        """
        trust_score = self._trust_scores.get(dataset_name, 1.0)
        
        for _, record in df.iterrows():
            record_id = record.get("_id")
            if record_id:
                self._dataset_sources[record_id] = dataset_name
                
                # Extract timestamp if available
                timestamp = datetime.now()
                if "_timestamp" in record:
                    try:
                        timestamp = pd.to_datetime(record["_timestamp"])
                    except:
                        pass
                
                # Create provenance info
                provenance = ProvenanceInfo(
                    sources={dataset_name},
                    timestamp=timestamp,
                    operation="input",
                    confidence=trust_score,
                    metadata={
                        "dataset_name": dataset_name,
                        "record_id": record_id,
                    }
                )
                
                self._provenance_registry[record_id] = provenance
    
    def track_fusion_result(
        self,
        fused_id: str,
        source_ids: List[str],
        operation: str,
        confidence: float,
        metadata: Dict[str, Any] = None,
    ):
        """Track provenance for a fusion result.
        
        Parameters
        ----------
        fused_id : str
            ID of the fused record.
        source_ids : List[str]
            List of source record IDs that contributed.
        operation : str
            The fusion operation performed.
        confidence : float
            Confidence in the fusion result.
        metadata : Dict[str, Any], optional
            Additional metadata.
        """
        # Collect all sources from input records
        all_sources = set()
        input_timestamps = []
        
        for source_id in source_ids:
            if source_id in self._provenance_registry:
                source_prov = self._provenance_registry[source_id]
                all_sources.update(source_prov.sources)
                input_timestamps.append(source_prov.timestamp)
            elif source_id in self._dataset_sources:
                # Fallback to dataset source
                dataset_name = self._dataset_sources[source_id]
                all_sources.add(dataset_name)
        
        # Use most recent timestamp from inputs
        if input_timestamps:
            latest_timestamp = max(input_timestamps)
        else:
            latest_timestamp = datetime.now()
        
        # Create fusion provenance
        fusion_metadata = {
            "source_records": source_ids,
            "fusion_operation": operation,
            **(metadata or {}),
        }
        
        fusion_provenance = ProvenanceInfo(
            sources=all_sources,
            timestamp=latest_timestamp,
            operation=operation,
            confidence=confidence,
            metadata=fusion_metadata,
        )
        
        self._provenance_registry[fused_id] = fusion_provenance
    
    def get_provenance(self, record_id: str) -> Optional[ProvenanceInfo]:
        """Get provenance information for a record.
        
        Parameters
        ----------
        record_id : str
            ID of the record.
            
        Returns
        -------
        Optional[ProvenanceInfo]
            Provenance information, or None if not found.
        """
        return self._provenance_registry.get(record_id)
    
    def get_lineage(self, record_id: str) -> List[str]:
        """Get the lineage chain for a record.
        
        Parameters
        ----------
        record_id : str
            ID of the record.
            
        Returns
        -------
        List[str]
            List of operations in the lineage chain.
        """
        lineage = []
        current_id = record_id
        
        # For now, just return the operation chain
        # In a full implementation, we'd track parent-child relationships
        if current_id in self._provenance_registry:
            prov = self._provenance_registry[current_id]
            lineage.append(prov.operation)
        
        return lineage
    
    def get_trust_score(self, record_id: str) -> float:
        """Get the trust score for a record.
        
        Parameters
        ----------
        record_id : str
            ID of the record.
            
        Returns
        -------
        float
            Trust score (0.0 to 1.0).
        """
        if record_id in self._provenance_registry:
            return self._provenance_registry[record_id].confidence
        elif record_id in self._dataset_sources:
            dataset_name = self._dataset_sources[record_id]
            return self._trust_scores.get(dataset_name, 1.0)
        else:
            return 1.0
    
    def get_source_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics about sources.
        
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Statistics for each source.
        """
        source_stats = {}
        
        # Count records per source
        source_counts = {}
        confidence_sums = {}
        
        for record_id, prov in self._provenance_registry.items():
            for source in prov.sources:
                if source not in source_counts:
                    source_counts[source] = 0
                    confidence_sums[source] = 0.0
                
                source_counts[source] += 1
                confidence_sums[source] += prov.confidence
        
        # Calculate statistics
        for source in source_counts:
            count = source_counts[source]
            avg_confidence = confidence_sums[source] / count if count > 0 else 0.0
            trust_score = self._trust_scores.get(source, 1.0)
            
            source_stats[source] = {
                "record_count": count,
                "average_confidence": avg_confidence,
                "trust_score": trust_score,
                "contribution_ratio": count / len(self._provenance_registry) if self._provenance_registry else 0.0,
            }
        
        return source_stats
    
    def export_provenance(self) -> Dict[str, Any]:
        """Export all provenance information.
        
        Returns
        -------
        Dict[str, Any]
            Complete provenance registry.
        """
        return {
            "provenance_registry": {
                record_id: prov.to_dict() 
                for record_id, prov in self._provenance_registry.items()
            },
            "dataset_sources": self._dataset_sources,
            "trust_scores": self._trust_scores,
            "export_timestamp": datetime.now().isoformat(),
        }
    
    def import_provenance(self, data: Dict[str, Any]):
        """Import provenance information.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Provenance data to import.
        """
        # Import provenance registry
        if "provenance_registry" in data:
            self._provenance_registry = {
                record_id: ProvenanceInfo.from_dict(prov_data)
                for record_id, prov_data in data["provenance_registry"].items()
            }
        
        # Import dataset sources
        if "dataset_sources" in data:
            self._dataset_sources.update(data["dataset_sources"])
        
        # Import trust scores
        if "trust_scores" in data:
            self._trust_scores.update(data["trust_scores"])


def add_provenance_columns(df: pd.DataFrame, tracker: ProvenanceTracker) -> pd.DataFrame:
    """Add provenance information as columns to a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    tracker : ProvenanceTracker
        Provenance tracker with record information.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added provenance columns.
    """
    df_with_prov = df.copy()
    
    # Add provenance columns
    sources_list = []
    timestamps_list = []
    operations_list = []
    confidences_list = []
    
    for _, record in df.iterrows():
        record_id = record.get("_id", "")
        prov = tracker.get_provenance(record_id)
        
        if prov:
            sources_list.append(list(prov.sources))
            timestamps_list.append(prov.timestamp)
            operations_list.append(prov.operation)
            confidences_list.append(prov.confidence)
        else:
            sources_list.append([])
            timestamps_list.append(None)
            operations_list.append("unknown")
            confidences_list.append(1.0)
    
    df_with_prov["_provenance_sources"] = sources_list
    df_with_prov["_provenance_timestamp"] = timestamps_list
    df_with_prov["_provenance_operation"] = operations_list
    df_with_prov["_provenance_confidence"] = confidences_list
    
    return df_with_prov


def extract_source_trust_scores(datasets: List[pd.DataFrame]) -> Dict[str, float]:
    """Extract trust scores from dataset metadata.
    
    Parameters
    ----------
    datasets : List[pd.DataFrame]
        List of datasets with potential trust information.
        
    Returns
    -------
    Dict[str, float]
        Mapping from dataset name to trust score.
    """
    trust_scores: Dict[str, float] = {}

    candidate_keys = ("trust", "trust_score", "trust_level", "_trust")

    for df in datasets:
        dataset_name = df.attrs.get("dataset_name")
        if not dataset_name:
            continue

        trust_score: Optional[float] = None

        # Inspect DataFrame attrs for explicit trust hints first
        for key in ("_trust", "trust_score", "trust", "trust_level"):
            if key in df.attrs:
                trust_score = df.attrs[key]
                break

        # Fall back to provenance metadata if available
        if trust_score is None:
            provenance = df.attrs.get("provenance") or {}
            if isinstance(provenance, dict):
                for key in candidate_keys:
                    if key in provenance:
                        trust_score = provenance[key]
                        break

        # Last resort: derive from a _trust column if present
        if trust_score is None and "_trust" in df.columns:
            trust_values = pd.to_numeric(df["_trust"], errors="coerce").dropna()
            if not trust_values.empty:
                trust_score = trust_values.mean()

        # Default when nothing is provided
        if trust_score is None:
            trust_score = 1.0

        try:
            trust_scores[dataset_name] = float(trust_score)
        except (TypeError, ValueError):
            trust_scores[dataset_name] = 1.0

    return trust_scores
