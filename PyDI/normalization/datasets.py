"""
Dataset-level normalization orchestration for PyDI.

This module provides comprehensive dataset normalization capabilities,
orchestrating value-, column-, and dataset-level transformations to
match Winter's DataSetNormalizer functionality.
"""

from __future__ import annotations

import logging
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, asdict

import pandas as pd

from .values import AdvancedValueNormalizer, NullValueHandler
from .columns import AdvancedTypeDetector, ColumnTypeInference, DataTypeExtended
from .text import TextNormalizer
from .transforms import get_transform as _get_builtin_transform
from .units import UnitRegistry

logger = logging.getLogger(__name__)


def _resolve_transform_callable(spec: Any) -> Optional[Callable[[pd.Series], pd.Series]]:
    """Resolve a transform spec into a callable(series) -> series.

    Supported specs:
    - str: built-in transform name (see map below)
    - callable: function(pd.Series) -> pd.Series
    - list/tuple: sequence of specs to be composed
    """
    if callable(spec):
        return spec  # assume correct signature
    if isinstance(spec, str):
        return _get_builtin_transform(spec)
    if isinstance(spec, (list, tuple)):
        callables: List[Callable[[pd.Series], pd.Series]] = []
        for part in spec:
            fn = _resolve_transform_callable(part)
            if fn is None:
                return None
            callables.append(fn)

        def _chained(series: pd.Series) -> pd.Series:
            out = series
            for fn in callables:
                out = fn(out)
            return out

        return _chained
    return None


def apply_column_transforms(
    df: pd.DataFrame,
    transforms: Dict[Any, Any],
    *,
    missing_policy: str = "warn",
) -> pd.DataFrame:
    """Apply per-column transformations efficiently.

    Usage examples
    --------------
    transforms = {
        'title': ['strip', 'normalize_whitespace'],
        'year': 'to_numeric',
        ('gross', 'budget'): lambda s: pd.to_numeric(s.str.replace(',', ''), errors='coerce'),
    }

    Notes
    -----
    - Keys can be a column name or a tuple/list of column names to apply the same transform.
    - Values can be a string (built-in), a callable, or a list/tuple to chain.
    - Operations are vectorized per Series for performance.
    - Missing columns behaviour is controlled by `missing_policy`:
      'warn' (default) logs a warning; 'error' raises a KeyError; 'ignore' is silent.
    """
    if not transforms:
        return df

    out = df.copy()
    for key, spec in transforms.items():
        # Determine requested and missing columns
        if isinstance(key, (list, tuple)):
            requested: List[str] = list(key)
        else:
            requested = [key]

        missing = [c for c in requested if c not in out.columns]
        present = [c for c in requested if c in out.columns]

        if missing:
            message = f"Transform targets missing column(s): {missing}"
            if missing_policy == "error":
                raise KeyError(message)
            elif missing_policy == "warn":
                logger.warning(message)
            # 'ignore' falls through silently

        # Nothing to do if no requested columns are present
        if not present:
            continue

        fn = _resolve_transform_callable(spec)
        if fn is None:
            continue

        for col in present:
            try:
                out[col] = fn(out[col])
            except Exception:
                # Fail soft per column
                pass
    return out


@dataclass
class NormalizationConfig:
    """Configuration for dataset normalization.

    Parameters
    ----------
    enable_type_detection : bool, default True
        If True, run column type detection before value normalization. This powers
        downstream decisions such as which parser to use (numeric/date/text), and
        enables unit-aware normalization when a numeric type with units is detected.

    type_confidence_threshold : float, default 0.6
        Minimum confidence required for an inferred type to be trusted. Lower values
        make the detector more permissive; higher values prefer leaving ambiguous
        columns as strings.

    sample_size_for_detection : int, default 1000
        Maximum number of non-null values per column to consider during type
        detection for performance on large datasets.

    enable_unit_conversion : bool, default True
        If True, normalize numeric values that carry units (e.g., "5 km", "60 mph")
        into a consistent target unit per category (e.g., metres, m/s). The
        `AdvancedValueNormalizer` uses `UnitNormalizer` and the unit registry, and
        also benefits from header-derived units (e.g., columns named "Speed (km/h)").

    enable_quantity_scaling : bool, default True
        If True, scale numbers containing quantity modifiers such as
        "thousand"/"k", "million"/"m", "billion"/"b". For example, "2.5 million"
        is interpreted as 2_500_000. This applies to both standalone numeric values
        and values with units.

    standardize_nulls : bool, default True
        If True, replace common null-like tokens ("n/a", "-", "—", "(n/a)", etc.)
        with a standard representation. See `NullValueHandler` for the full set.

    null_replacement : str or None, default None
        The value to substitute for detected null-like tokens when
        `standardize_nulls` is enabled. If None, values are replaced with Python
        `None` (which becomes NaN in pandas).

    normalize_text : bool, default True
        If True, apply basic text normalization to column names and values where
        applicable (lower-casing, whitespace cleanup, HTML entity removal). The
        exact operations depend on the context and normalizers in use.

    lowercase_text : bool, default True
        If True, convert text to lowercase during normalization when text
        normalization is enabled.

    remove_extra_whitespace : bool, default True
        If True, collapse multiple whitespace characters and strip leading/trailing
        spaces during text normalization.

    preserve_original_columns : bool, default False
        If True, keep original columns alongside normalized ones when renaming or
        transforming. If False, columns are replaced in-place. (Note: current
        pipeline replaces in place; this flag is reserved for workflows that
        duplicate columns.)

    add_metadata_columns : bool, default True
        If True, include auxiliary metadata columns in outputs when appropriate
        (e.g., identifiers, provenance summaries). Dataset-level provenance is
        always stored in `df.attrs` regardless of this flag.
    """

    # Type detection settings
    enable_type_detection: bool = True
    type_confidence_threshold: float = 0.6
    sample_size_for_detection: int = 1000

    # Value normalization settings
    enable_unit_conversion: bool = True
    enable_quantity_scaling: bool = True
    standardize_nulls: bool = True
    null_replacement: Optional[str] = None

    # Text normalization settings
    normalize_text: bool = True
    lowercase_text: bool = True
    remove_extra_whitespace: bool = True

    # Output settings
    preserve_original_columns: bool = False
    add_metadata_columns: bool = True

    # Per-column transformations (applied before detection/normalization)
    # Mapping of column name (or list of names) to a transform spec:
    # - string: built-in transform name (e.g., 'lower', 'strip', 'to_numeric')
    # - callable: function(pd.Series) -> pd.Series
    # - list/tuple of the above to chain multiple transforms
    column_transformations: Optional[Dict[Any, Any]] = None
    # How to handle transform specs that reference missing columns: 'warn' | 'error' | 'ignore'
    missing_transform_column_policy: str = "warn"
    # Require an explicit plan/summary & selected operations; if True, auto type detection/value normalization is skipped
    explicit_plan_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NormalizationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class ColumnNormalizationResult:
    """Result of normalizing a single column."""

    original_name: str
    normalized_name: str
    detected_type: DataTypeExtended
    confidence: float
    unit_category: Optional[str] = None
    specific_unit: Optional[str] = None
    null_count: int = 0
    total_count: int = 0
    normalization_success_rate: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class DatasetNormalizationResult:
    """Result of normalizing an entire dataset."""

    original_shape: Tuple[int, int]
    normalized_shape: Tuple[int, int]
    column_results: List[ColumnNormalizationResult]
    overall_success_rate: float = 0.0
    processing_time_seconds: float = 0.0
    config_used: Optional[Dict[str, Any]] = None

    def get_column_result(self, column_name: str) -> Optional[ColumnNormalizationResult]:
        """Get normalization result for specific column."""
        for result in self.column_results:
            if result.original_name == column_name:
                return result
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'original_columns': self.original_shape[1],
            'normalized_columns': self.normalized_shape[1],
            'rows_processed': self.normalized_shape[0],
            'overall_success_rate': self.overall_success_rate,
            'processing_time': self.processing_time_seconds,
            'type_distribution': self._get_type_distribution()
        }

    def _get_type_distribution(self) -> Dict[str, int]:
        """Get distribution of detected types."""
        type_counts = {}
        for result in self.column_results:
            type_name = result.detected_type.name
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        return type_counts


class DatasetNormalizer:
    """
    Comprehensive dataset normalizer matching Winter's DataSetNormalizer.

    Orchestrates value-, column-, and dataset-level normalization with
    intelligent type detection, unit conversion, and metadata preservation.

    Notes
    -----
    - Type detection uses `AdvancedTypeDetector` (majority voting + patterns).
    - Unit-aware normalization leverages header-derived unit hints when present
      (e.g., "Distance (km)") and value-level unit detection otherwise.
    - Quantity scaling interprets modifiers like thousand/million/billion.
    - Null standardization maps many web-style null tokens to a single value.
    """

    def __init__(
        self,
        config: Optional[NormalizationConfig] = None,
        unit_registry: Optional[UnitRegistry] = None,
        type_detector: Optional[AdvancedTypeDetector] = None
    ):
        self.config = config or NormalizationConfig()

        # Initialize components
        self.unit_registry = unit_registry or UnitRegistry(comprehensive=True)
        self.type_detector = type_detector or AdvancedTypeDetector()
        self.value_normalizer = AdvancedValueNormalizer(
            unit_registry=self.unit_registry,
            enable_unit_conversion=self.config.enable_unit_conversion,
            enable_quantity_scaling=self.config.enable_quantity_scaling
        )
        self.null_handler = NullValueHandler(self.config.null_replacement)
        self.text_normalizer = TextNormalizer(
            lowercase=self.config.lowercase_text,
            strip_whitespace=self.config.remove_extra_whitespace
        )
        self.column_inference = ColumnTypeInference()

        # Normalization statistics
        self.last_result: Optional[DatasetNormalizationResult] = None

    def normalize_dataset(
        self,
        df: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None,
        preserve_attrs: bool = True,
        column_transforms: Optional[Dict[Any, Any]] = None,
        *,
        summary_plan: Optional[Dict[str, Any]] = None,
        selected_operations: Optional[Dict[str, List[Any]]] = None,
    ) -> Tuple[pd.DataFrame, DatasetNormalizationResult]:
        """
        Normalize entire dataset with comprehensive transformations.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset to normalize.
        output_path : str or Path, optional
            Path to save normalization results and metadata.
        preserve_attrs : bool, default True
            Whether to preserve DataFrame and Series attrs.

        Returns
        -------
        Tuple[pd.DataFrame, DatasetNormalizationResult]
            Normalized dataset and detailed results.
        """
        import time
        start_time = time.time()

        logger.info(f"Starting dataset normalization: {df.shape}")

        # Preserve original metadata
        original_attrs = df.attrs.copy() if preserve_attrs else {}
        original_shape = df.shape

        # Initialize result tracking
        column_results = []
        normalized_df = df.copy()

        # Apply user-specified per-column transformations prior to detection
        effective_transforms = column_transforms or self.config.column_transformations
        if effective_transforms:
            normalized_df = apply_column_transforms(
                normalized_df,
                effective_transforms,
                missing_policy=self.config.missing_transform_column_policy)

        # Explicit plan mode: apply only user-selected operations and skip auto-detection/value normalization
        if self.config.explicit_plan_required:
            if summary_plan is None or selected_operations is None:
                raise ValueError(
                    "explicit_plan_required=True: provide summary_plan and selected_operations"
                )

            # Apply selected per-column operations
            for column_name, ops in selected_operations.items():
                if column_name not in normalized_df.columns:
                    continue
                for op in (ops or []):
                    fn = _resolve_transform_callable(op)
                    if fn is None:
                        continue
                    try:
                        normalized_df[column_name] = fn(
                            normalized_df[column_name])
                    except Exception:
                        pass

            # Build minimal results without auto type detection
            for column_name in normalized_df.columns:
                series = normalized_df[column_name]
                column_results.append(ColumnNormalizationResult(
                    original_name=column_name,
                    normalized_name=self._normalize_column_name(column_name),
                    detected_type=DataTypeExtended.UNKNOWN,
                    confidence=0.0,
                    null_count=series.isnull().sum(),
                    total_count=len(series),
                    normalization_success_rate=1.0
                ))

            # Skip auto-processing below
            goto_finalize = True
        else:
            goto_finalize = False

        # Process each column (auto) if not in explicit plan mode
        if not goto_finalize:
            for column_name in df.columns:
                try:
                    column_result = self._normalize_column(
                        normalized_df, column_name, preserve_attrs
                    )
                    column_results.append(column_result)

                except Exception as e:
                    logger.error(
                        f"Failed to normalize column '{column_name}': {e}")
                    error_result = ColumnNormalizationResult(
                        original_name=column_name,
                        normalized_name=column_name,
                        detected_type=DataTypeExtended.UNKNOWN,
                        confidence=0.0,
                        total_count=len(df),
                        errors=[str(e)]
                    )
                    column_results.append(error_result)

        # Restore dataset-level metadata
        if preserve_attrs:
            normalized_df.attrs.update(original_attrs)
            normalized_df.attrs['normalization_applied'] = True
            normalized_df.attrs['normalization_config'] = self.config.to_dict()

        # Calculate overall statistics
        end_time = time.time()
        processing_time = end_time - start_time

        success_rates = [
            r.normalization_success_rate for r in column_results if r.normalization_success_rate > 0]
        overall_success_rate = sum(success_rates) / \
            len(success_rates) if success_rates else 0.0

        # Create result object
        result = DatasetNormalizationResult(
            original_shape=original_shape,
            normalized_shape=normalized_df.shape,
            column_results=column_results,
            overall_success_rate=overall_success_rate,
            processing_time_seconds=processing_time,
            config_used=self.config.to_dict()
        )

        self.last_result = result

        # Save results if path provided
        if output_path:
            self._save_results(normalized_df, result, output_path)

        logger.info(f"Dataset normalization completed: {normalized_df.shape} "
                    f"(success rate: {overall_success_rate:.1%})")

        return normalized_df, result

    def _normalize_column(
        self,
        df: pd.DataFrame,
        column_name: str,
        preserve_attrs: bool
    ) -> ColumnNormalizationResult:
        """Normalize a single column with type detection and value transformation."""

        series = df[column_name]
        original_name = column_name
        original_attrs = series.attrs.copy() if preserve_attrs else {}

        # Sample for type detection if dataset is large
        sample_size = min(len(series), self.config.sample_size_for_detection)
        sample = series.dropna().head(sample_size) if len(
            series) > sample_size else series

        # Detect column type and characteristics
        type_result = None
        if self.config.enable_type_detection and not sample.empty:
            type_result = self.type_detector.detect_type_for_column(
                sample.tolist(),
                column_name,
                confidence_threshold=self.config.type_confidence_threshold
            )

        if not type_result:
            detected_type = DataTypeExtended.STRING
            confidence = 0.5
            unit_category = None
            specific_unit = None
        else:
            detected_type = type_result.data_type
            confidence = type_result.confidence
            unit_category = type_result.unit_category
            specific_unit = type_result.specific_unit

        # Normalize column name if needed
        normalized_name = self._normalize_column_name(column_name)

        # Apply value-level normalization
        normalized_values = []
        success_count = 0
        error_count = 0

        for value in series:
            try:
                if self.config.standardize_nulls and self.null_handler.is_null_value(value):
                    normalized_value = self.null_handler.normalize_nulls(value)
                else:
                    normalized_value = self.value_normalizer.normalize_value(
                        value,
                        detected_type.name.lower(),
                        unit_category,
                        specific_unit
                    )

                normalized_values.append(normalized_value)
                success_count += 1

            except Exception as e:
                logger.debug(f"Value normalization failed for {value}: {e}")
                normalized_values.append(value)  # Keep original on failure
                error_count += 1

        # Update the column
        normalized_series = pd.Series(
            normalized_values, index=series.index, name=normalized_name)

        # Preserve and enhance metadata
        if preserve_attrs:
            normalized_series.attrs.update(original_attrs)
            normalized_series.attrs['detected_type'] = detected_type.name
            normalized_series.attrs['type_confidence'] = confidence
            if unit_category:
                normalized_series.attrs['unit_category'] = unit_category
            if specific_unit:
                normalized_series.attrs['specific_unit'] = specific_unit

        # Update DataFrame
        if normalized_name != original_name:
            df.drop(columns=[original_name], inplace=True)
        df[normalized_name] = normalized_series

        # Calculate statistics
        null_count = normalized_series.isnull().sum()
        total_count = len(normalized_series)
        success_rate = success_count / total_count if total_count > 0 else 0.0

        return ColumnNormalizationResult(
            original_name=original_name,
            normalized_name=normalized_name,
            detected_type=detected_type,
            confidence=confidence,
            unit_category=unit_category,
            specific_unit=specific_unit,
            null_count=null_count,
            total_count=total_count,
            normalization_success_rate=success_rate
        )

    def _normalize_column_name(self, column_name: str) -> str:
        """Normalize column name using text normalizer."""
        if not self.config.normalize_text:
            return column_name

        # Use header-specific normalization
        from .text import HeaderNormalizer
        header_normalizer = HeaderNormalizer()
        return header_normalizer.normalize_header(column_name)

    def _save_results(
        self,
        normalized_df: pd.DataFrame,
        result: DatasetNormalizationResult,
        output_path: Union[str, Path]
    ) -> None:
        """Save normalization results to files."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save normalized dataset
        csv_path = output_path / "normalized_dataset.csv"
        normalized_df.to_csv(csv_path, index=False)

        # Save normalization metadata
        metadata_path = output_path / "normalization_metadata.json"
        metadata = {
            'summary': result.get_summary(),
            'column_results': [asdict(cr) for cr in result.column_results],
            'config': result.config_used
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save column-level analysis
        analysis_path = output_path / "column_analysis.csv"
        analysis_data = []
        for cr in result.column_results:
            analysis_data.append({
                'original_name': cr.original_name,
                'normalized_name': cr.normalized_name,
                'detected_type': cr.detected_type.name,
                'confidence': cr.confidence,
                'unit_category': cr.unit_category,
                'specific_unit': cr.specific_unit,
                'null_count': cr.null_count,
                'total_count': cr.total_count,
                'success_rate': cr.normalization_success_rate,
                'error_count': len(cr.errors)
            })

        analysis_df = pd.DataFrame(analysis_data)
        analysis_df.to_csv(analysis_path, index=False)

        logger.info(f"Normalization results saved to {output_path}")

    def get_column_quality_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate column quality assessment report."""
        return self.column_inference.analyze_columns_quality(df)

    def detect_schema_patterns(self, dfs: List[pd.DataFrame]) -> Dict[str, Any]:
        """Detect common schema patterns across multiple datasets."""
        patterns = {
            'common_columns': set(),
            'type_patterns': {},
            'naming_patterns': set(),
            'unit_patterns': set()
        }

        if not dfs:
            return patterns

        # Find common columns
        all_columns = [set(df.columns) for df in dfs]
        patterns['common_columns'] = set.intersection(
            *all_columns) if all_columns else set()

        # Analyze type patterns
        for df in dfs:
            for column in df.columns:
                if column not in patterns['type_patterns']:
                    patterns['type_patterns'][column] = []

                # Quick type detection for pattern analysis
                sample = df[column].dropna().head(100)
                if not sample.empty:
                    type_result = self.type_detector.detect_type_for_column(
                        sample.tolist(), column, confidence_threshold=0.5
                    )
                    if type_result:
                        patterns['type_patterns'][column].append(
                            type_result.data_type.name)

        return patterns


# Convenience functions for dataset normalization
def normalize_dataset(
    df: pd.DataFrame,
    config: Optional[NormalizationConfig] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[pd.DataFrame, DatasetNormalizationResult]:
    """
    Quick dataset normalization with default settings.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to normalize.
    config : NormalizationConfig, optional
        Normalization configuration.
    output_path : str or Path, optional
        Path to save results.

    Returns
    -------
    Tuple[pd.DataFrame, DatasetNormalizationResult]
        Normalized dataset and results.
    """
    normalizer = DatasetNormalizer(config)
    return normalizer.normalize_dataset(df, output_path)


def create_normalization_config(
    enable_unit_conversion: bool = True,
    enable_quantity_scaling: bool = True,
    normalize_text: bool = True,
    standardize_nulls: bool = True,
    **kwargs
) -> NormalizationConfig:
    """Create normalization configuration with common settings.

    Parameters
    ----------
    enable_unit_conversion : bool, default True
        Normalize numeric values containing units to consistent target units
        per category (e.g., metres, seconds, m/s). Uses header-derived unit hints
        when present.

    enable_quantity_scaling : bool, default True
        Interpret common quantity modifiers (k, million, billion, etc.) and scale
        numbers accordingly.

    normalize_text : bool, default True
        Apply basic text normalization (e.g., lowercasing, whitespace cleanup) in
        text-related paths.

    standardize_nulls : bool, default True
        Map a broad set of null-like tokens to a single standard value.

    **kwargs
        Any additional `NormalizationConfig` fields to override.

    Returns
    -------
    NormalizationConfig
        A populated configuration instance.
    """
    return NormalizationConfig(
        enable_unit_conversion=enable_unit_conversion,
        enable_quantity_scaling=enable_quantity_scaling,
        normalize_text=normalize_text,
        standardize_nulls=standardize_nulls,
        **kwargs
    )


def load_normalization_config(config_path: Union[str, Path]) -> NormalizationConfig:
    """Load normalization configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return NormalizationConfig.from_dict(config_dict)


def save_normalization_config(config: NormalizationConfig, config_path: Union[str, Path]) -> None:
    """Save normalization configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
