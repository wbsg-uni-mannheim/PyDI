"""Base classes and utilities for information extraction."""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd


logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Abstract base class for all information extractors.

    All extractors should inherit from this class and implement the extract method.
    This provides a consistent interface for extracting structured information from
    DataFrame columns.

    Parameters
    ----------
    out_dir : str, optional
        Output directory for artifacts, by default "output/informationextraction"
    debug : bool, optional
        Enable debug mode with verbose logging and artifacts, by default False
    """

    def __init__(self, *, out_dir: str = "output/informationextraction", debug: bool = False):
        self.out_dir = Path(out_dir)
        self.debug = debug
        self._setup_output_dir()

    def _setup_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        if self.debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = self.out_dir / timestamp
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_dir = self.out_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract(
        self,
        df: pd.DataFrame,
        *,
        source_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Extract structured information from DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        source_column : str, optional
            Column to extract from. If None, extractor should specify default
        **kwargs
            Additional extractor-specific parameters

        Returns
        -------
        pd.DataFrame
            DataFrame with extracted columns added
        """
        pass

    def _validate_input(self, df: pd.DataFrame, source_column: Optional[str] = None) -> str:
        """Validate input DataFrame and determine source column.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        source_column : str, optional
            Specified source column

        Returns
        -------
        str
            Validated source column name

        Raises
        ------
        ValueError
            If DataFrame is empty or source column is invalid
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        if source_column is None:
            if hasattr(self, 'default_source') and self.default_source:
                source_column = self.default_source
            else:
                raise ValueError(
                    "source_column must be specified or extractor must have default_source")

        if source_column not in df.columns:
            raise ValueError(
                f"Source column '{source_column}' not found in DataFrame")

        return source_column

    def _write_artifact(self, filename: str, data: Any) -> Path:
        """Write artifact to output directory.

        Parameters
        ----------
        filename : str
            Name of the file to write
        data : Any
            Data to write (dict/list for JSON, str for text, DataFrame for CSV)

        Returns
        -------
        Path
            Path to written file
        """
        if not self.debug:
            return self.run_dir / filename

        filepath = self.run_dir / filename
        # Ensure nested directories exist for artifacts like "errors/", "prompts/", etc.
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, (dict, list)):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif isinstance(data, str):
            with open(filepath, 'w') as f:
                f.write(data)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            raise ValueError(
                f"Unsupported data type for artifact: {type(data)}")

        logger.debug(f"Wrote artifact: {filepath}")
        return filepath

    def _write_artifact_always(self, filename: str, data: Any) -> Path:
        """Write artifact to output directory, always writing regardless of debug mode.
        
        Used for critical artifacts like LLM logs that should always be persisted.

        Parameters
        ----------
        filename : str
            Name of the file to write
        data : Any
            Data to write (dict/list for JSON, str for text, DataFrame for CSV)

        Returns
        -------
        Path
            Path to written file
        """
        filepath = self.run_dir / filename
        # Ensure nested directories exist for artifacts like "errors/", "prompts/", etc.
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, (dict, list)):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif isinstance(data, str):
            with open(filepath, 'w') as f:
                f.write(data)
        elif isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        else:
            raise ValueError(
                f"Unsupported data type for artifact: {type(data)}")

        logger.debug(f"Wrote artifact: {filepath}")
        return filepath

    def _log_extraction_stats(self, df_in: pd.DataFrame, df_out: pd.DataFrame,
                              extracted_columns: List[str]) -> None:
        """Log extraction statistics.

        Parameters
        ----------
        df_in : pd.DataFrame
            Input DataFrame
        df_out : pd.DataFrame
            Output DataFrame
        extracted_columns : List[str]
            Names of extracted columns
        """
        stats = {
            "input_rows": len(df_in),
            "output_rows": len(df_out),
            "extracted_columns": extracted_columns,
            "extraction_stats": {}
        }

        for col in extracted_columns:
            if col in df_out.columns:
                non_null = df_out[col].notna().sum()
                stats["extraction_stats"][col] = {
                    "non_null_values": int(non_null),
                    "extraction_rate": float(non_null / len(df_out)) if len(df_out) > 0 else 0.0
                }

        logger.info(f"Extraction complete: {stats}")

        if self.debug:
            self._write_artifact("extraction_stats.json", stats)


class ExtractorPipeline:
    """Pipeline for chaining multiple extractors.

    Parameters
    ----------
    extractors : List[BaseExtractor]
        List of extractors to apply in sequence
    """

    def __init__(self, extractors: List[BaseExtractor]):
        self.extractors = extractors

    def run(
        self,
        df: pd.DataFrame,
        *,
        out_dir: str = "output/informationextraction",
        debug: bool = False
    ) -> pd.DataFrame:
        """Run extraction pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        out_dir : str, optional
            Output directory for artifacts, by default "output/informationextraction"
        debug : bool, optional
            Enable debug mode, by default False

        Returns
        -------
        pd.DataFrame
            DataFrame with all extracted columns added
        """
        result = df.copy()

        for i, extractor in enumerate(self.extractors):
            logger.info(
                f"Running extractor {i+1}/{len(self.extractors)}: {type(extractor).__name__}")

            # Update extractor output directory to include pipeline info
            if debug:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                extractor.out_dir = Path(
                    out_dir) / f"pipeline_{timestamp}" / f"step_{i+1}_{type(extractor).__name__}"
                extractor.debug = debug
                extractor._setup_output_dir()

            result = extractor.extract(result)

        logger.info(
            f"Pipeline complete. Final DataFrame shape: {result.shape}")
        return result
