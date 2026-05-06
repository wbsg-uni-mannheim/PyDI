"""Taxonomy-based normalization for categorical attributes.

This module provides functionality to normalize categorical values (like industry,
category, or classification) to a standardized taxonomy using LLM-based mapping.

The workflow:
1. Load a taxonomy from a CSV file
2. Extract unique values from the source data column
3. Use an LLM to create a mapping from source values to taxonomy values
4. Cache the mapping to JSON for reuse and manual review
5. Apply the mapping during normalization

Example:
    >>> from langchain_openai import ChatOpenAI
    >>> from PyDI.normalization.taxonomy import TaxonomyLoader, TaxonomyMapper
    >>>
    >>> # Load taxonomy values
    >>> loader = TaxonomyLoader()
    >>> taxonomy_values = loader.load("taxonomies/gics.csv", column="Industry Name")
    >>>
    >>> # Create mapping for source values
    >>> chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    >>> mapper = TaxonomyMapper(chat)
    >>> result = mapper.create_mapping(
    ...     source_values=["Tech", "IT Services", "Banking"],
    ...     taxonomy_values=taxonomy_values,
    ...     column_name="industry",
    ... )
    >>> print(result.mapping)
    {"Tech": "Software", "IT Services": "IT Consulting & Other Services", "Banking": "Banks"}
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatModel = None

from ..utils.llm import LLMCallLogger, _get_model_name

logger = logging.getLogger(__name__)


@dataclass
class TaxonomyMappingResult:
    """Result of creating a taxonomy mapping.

    Attributes:
        mapping: Dictionary mapping source values to taxonomy values.
            Values are None for source values that couldn't be mapped.
        unmapped: List of source values that couldn't be mapped to taxonomy.
        taxonomy_values: List of valid taxonomy values that were available.
        taxonomy_column: Name of the taxonomy column used.
        created_at: ISO timestamp when the mapping was created.
        llm_model: Name of the LLM model used to create the mapping.
        source_value_counts_by_dataset: Per-dataset record counts for each source value.
            Structure: {dataset_name: {source_value: count}}
    """

    mapping: Dict[str, Optional[str]]
    unmapped: List[str]
    taxonomy_values: List[str]
    taxonomy_column: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    llm_model: Optional[str] = None
    source_value_counts_by_dataset: Optional[Dict[str, Dict[str, int]]] = None


class TaxonomyLoader:
    """Load taxonomy values from CSV files.

    This class handles loading taxonomy CSV files and extracting unique values
    from a specified column. It supports both absolute and relative paths.

    Example:
        >>> loader = TaxonomyLoader()
        >>> values = loader.load("taxonomies/gics.csv", column="Industry Name")
        >>> print(values[:3])
        ['Software', 'Banks', 'Energy']
    """

    def __init__(self) -> None:
        """Initialize the taxonomy loader."""
        self._cache: Dict[Tuple[str, Optional[str]], List[str]] = {}

    def load(
        self,
        path: str | Path,
        column: Optional[str] = None,
        *,
        base_path: Optional[str | Path] = None,
    ) -> List[str]:
        """Load taxonomy values from a CSV file.

        Args:
            path: Path to the taxonomy CSV file.
            column: Column name to extract values from. If None, uses the first column.
            base_path: Base path for resolving relative paths.

        Returns:
            List of unique taxonomy values (excluding nulls).

        Raises:
            FileNotFoundError: If the taxonomy file doesn't exist.
            ValueError: If the file is empty, the specified column doesn't exist,
                or no valid values are found.
        """
        # Resolve path
        taxonomy_path = Path(path)
        if not taxonomy_path.is_absolute() and base_path:
            taxonomy_path = Path(base_path) / taxonomy_path

        # Check cache
        cache_key = (str(taxonomy_path.resolve()), column)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check file exists
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

        # Load CSV
        try:
            df = pd.read_csv(taxonomy_path)
        except pd.errors.EmptyDataError:
            raise ValueError(f"Taxonomy file is empty: {taxonomy_path}")

        if df.empty:
            raise ValueError(f"Taxonomy file contains no data: {taxonomy_path}")

        # Determine which column to use
        if column is None:
            target_column = df.columns[0]
            logger.info(
                f"No taxonomy column specified, using first column: '{target_column}'"
            )
        else:
            if column not in df.columns:
                available = ", ".join(f"'{c}'" for c in df.columns)
                raise ValueError(
                    f"Column '{column}' not found in taxonomy file. "
                    f"Available columns: {available}"
                )
            target_column = column

        # Extract unique non-null values
        values = df[target_column].dropna().astype(str).unique().tolist()

        if not values:
            raise ValueError(
                f"No valid values found in column '{target_column}' of taxonomy file"
            )

        # Cache and return
        self._cache[cache_key] = values
        logger.info(f"Loaded {len(values)} taxonomy values from '{target_column}'")

        return values

    def load_full_csv(
        self,
        path: str | Path,
        *,
        base_path: Optional[str | Path] = None,
    ) -> str:
        """Load the full taxonomy CSV content as a string for LLM context.

        Args:
            path: Path to the taxonomy CSV file.
            base_path: Base path for resolving relative paths.

        Returns:
            CSV content as a string.
        """
        taxonomy_path = Path(path)
        if not taxonomy_path.is_absolute() and base_path:
            taxonomy_path = Path(base_path) / taxonomy_path

        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

        with open(taxonomy_path, "r", encoding="utf-8") as f:
            return f.read()

    def clear_cache(self) -> None:
        """Clear the taxonomy cache."""
        self._cache.clear()


class TaxonomyMapper:
    """Create taxonomy mappings using LLM.

    This class uses a LangChain chat model to create mappings from source values
    to taxonomy values. It handles batching for large value sets and integrates
    with the unified LLM call logging.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        >>> mapper = TaxonomyMapper(chat)
        >>> result = mapper.create_mapping(
        ...     source_values=["Tech", "Software Industry"],
        ...     taxonomy_values=["Software", "Hardware"],
        ...     column_name="industry",
        ... )
    """

    DEFAULT_SYSTEM_PROMPT = """You are an expert at mapping categorical values to a standardized taxonomy.

Given source values and a taxonomy CSV, map each source value to the most appropriate value from the specified column, or null if no good match exists.

Rules:
- Each source value maps to exactly one taxonomy value (or null)
- Consider synonyms, abbreviations, alternative spellings, and hierarchical relationships
- If a source value contains multiple values separated by a delimiter (e.g. "|", "/", ";"), prioritize the first value for determining the best taxonomy match
- If a source value already matches a taxonomy value exactly (case-insensitive), map to the exact taxonomy value
- If no good match exists, map to null (the original value will be preserved)
- Be conservative: only map when you are confident in the match

Return your response as valid JSON only, with no additional text."""

    DEFAULT_USER_PROMPT_TEMPLATE = """Map these source values to the taxonomy.

Column to map to: {taxonomy_column}

Source values to map:
{source_values_json}

Taxonomy CSV (use values from the "{taxonomy_column}" column):
{taxonomy_csv}

Return JSON in exactly this format:
{{"mappings": {{"source_value_1": "taxonomy_value_1", "source_value_2": null, ...}}}}"""

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        batch_size: int = 50,
        max_retries: int = 2,
        retry_delay: float = 2.0,
        out_dir: str | Path = "output/normalization/taxonomy",
        debug: bool = False,
    ) -> None:
        """Initialize the taxonomy mapper.

        Args:
            chat_model: LangChain chat model instance.
            batch_size: Maximum number of values to process in a single LLM call.
            max_retries: Number of retry attempts on LLM failure.
            retry_delay: Base delay in seconds between retries (exponential backoff).
            out_dir: Directory for debug artifacts.
            debug: Enable debug mode with detailed logging.
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain dependencies not available. Install with: "
                "pip install langchain-core"
            )

        self.chat_model = chat_model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.debug = debug

        self.out_dir = Path(out_dir)
        if debug:
            self.out_dir.mkdir(parents=True, exist_ok=True)

        self._llm_logger = LLMCallLogger()

    def create_mapping(
        self,
        source_values: List[str],
        taxonomy_csv_content: str,
        taxonomy_column: str,
        *,
        column_name: Optional[str] = None,
        column_description: Optional[str] = None,
    ) -> TaxonomyMappingResult:
        """Create a mapping from source values to taxonomy values using LLM.

        Args:
            source_values: List of unique values from source data to map.
            taxonomy_csv_content: Full CSV content of the taxonomy file.
            taxonomy_column: Name of the column in the taxonomy CSV to map to.
            column_name: Name of the source column (for LLM context).
            column_description: Description of what the values represent.

        Returns:
            TaxonomyMappingResult with the mapping and metadata.
        """
        if not source_values:
            return TaxonomyMappingResult(
                mapping={},
                unmapped=[],
                taxonomy_values=[],
                taxonomy_column=taxonomy_column,
                llm_model=_get_model_name(self.chat_model),
            )

        # Parse taxonomy values from CSV for the result
        taxonomy_values = self._extract_column_values(taxonomy_csv_content, taxonomy_column)

        # Process in batches if needed
        all_mappings: Dict[str, Optional[str]] = {}
        all_unmapped: List[str] = []

        batches = [
            source_values[i : i + self.batch_size]
            for i in range(0, len(source_values), self.batch_size)
        ]

        # Use tqdm for progress bar if available
        try:
            from tqdm import tqdm
            batch_iter = tqdm(
                enumerate(batches),
                total=len(batches),
                desc="    LLM mapping batches",
                unit="batch",
                leave=True,
            )
        except ImportError:
            batch_iter = enumerate(batches)

        for batch_idx, batch in batch_iter:
            logger.debug(
                f"Processing batch {batch_idx + 1}/{len(batches)} "
                f"({len(batch)} values)"
            )

            batch_mapping = self._create_mapping_batch(
                batch,
                taxonomy_csv_content,
                taxonomy_column,
                column_name=column_name,
                column_description=column_description,
                batch_idx=batch_idx,
            )

            for value in batch:
                mapped = batch_mapping.get(value)
                all_mappings[value] = mapped
                if mapped is None:
                    all_unmapped.append(value)

        return TaxonomyMappingResult(
            mapping=all_mappings,
            unmapped=all_unmapped,
            taxonomy_values=taxonomy_values,
            taxonomy_column=taxonomy_column,
            llm_model=_get_model_name(self.chat_model),
        )

    def _extract_column_values(self, csv_content: str, column: str) -> List[str]:
        """Extract unique values from a column in CSV content."""
        from io import StringIO

        df = pd.read_csv(StringIO(csv_content))
        if column not in df.columns:
            return []
        return df[column].dropna().astype(str).unique().tolist()

    def _create_mapping_batch(
        self,
        source_values: List[str],
        taxonomy_csv_content: str,
        taxonomy_column: str,
        *,
        column_name: Optional[str] = None,
        column_description: Optional[str] = None,
        batch_idx: int = 0,
    ) -> Dict[str, Optional[str]]:
        """Create mapping for a single batch of values."""
        # Build the prompt
        system_prompt = self.DEFAULT_SYSTEM_PROMPT
        if column_description:
            system_prompt += f"\n\nContext: The values represent {column_description}"

        user_prompt = self.DEFAULT_USER_PROMPT_TEMPLATE.format(
            taxonomy_column=taxonomy_column,
            source_values_json=json.dumps(source_values, indent=2),
            taxonomy_csv=taxonomy_csv_content,
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        # Call LLM with retries
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                response = self.chat_model.invoke(messages)
                duration_ms = (time.time() - start_time) * 1000

                # Log the call
                self._llm_logger.record_call(
                    chat_model=self.chat_model,
                    messages=messages,
                    response=response,
                    row_index=batch_idx,
                    attempt=attempt,
                    duration_ms=duration_ms,
                )

                # Parse response
                response_text = self._normalize_response_content(response)
                mapping = self._parse_llm_response(response_text, source_values)

                if self.debug:
                    self._write_debug_artifact(
                        f"batch_{batch_idx}_response.json",
                        {
                            "source_values": source_values,
                            "response": response_text,
                            "mapping": mapping,
                        },
                    )

                return mapping

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}"
                )
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    time.sleep(delay)

        # All retries failed - return empty mapping (keep original values)
        logger.error(f"All LLM attempts failed: {last_error}")
        return {v: None for v in source_values}

    def _normalize_response_content(self, response: Any) -> str:
        """Normalize LangChain response content to plain text."""
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                try:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(str(block.get("text", "")))
                    else:
                        parts.append(str(block))
                except Exception:
                    continue
            return "\n".join(p.strip() for p in parts if p)
        try:
            return str(content)
        except Exception:
            return ""

    def _parse_llm_response(
        self,
        response_text: str,
        source_values: List[str],
    ) -> Dict[str, Optional[str]]:
        """Parse LLM response to extract mappings.

        Args:
            response_text: Raw response from LLM.
            source_values: List of source values we asked to map.

        Returns:
            Dictionary mapping source values to taxonomy values or None.
        """
        # Remove markdown code fences if present
        fence_match = re.search(
            r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE
        )
        if fence_match:
            response_text = fence_match.group(1).strip()

        try:
            parsed = json.loads(response_text.strip())

            # Expected format: {"mappings": {"source": "target", ...}}
            if isinstance(parsed, dict) and "mappings" in parsed:
                raw_mappings = parsed["mappings"]
                if isinstance(raw_mappings, dict):
                    # Normalize the mapping
                    result: Dict[str, Optional[str]] = {}
                    for key, value in raw_mappings.items():
                        # Handle string "null" as None
                        if value is None or value == "null" or value == "None":
                            result[key] = None
                        else:
                            result[key] = str(value)
                    return result

            logger.warning(f"Unexpected JSON structure in LLM response: {type(parsed)}")
            return {v: None for v in source_values}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return {v: None for v in source_values}

    def _write_debug_artifact(self, filename: str, data: Any) -> None:
        """Write debug artifact to disk."""
        if not self.debug:
            return

        filepath = self.out_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def flush_logs(self, write_artifact: Any = None) -> None:
        """Flush LLM call logs.

        Args:
            write_artifact: Callable that accepts (filename, data) to persist artifacts.
                If None, writes to out_dir.
        """
        if write_artifact is None:

            def write_artifact(filename: str, data: Any) -> None:
                filepath = self.out_dir / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, default=str)

        self._llm_logger.flush(write_artifact)


# === Cache Functions ===


def load_mapping_cache(path: str | Path) -> Optional[Dict[str, Optional[str]]]:
    """Load an existing taxonomy mapping from a JSON cache file.

    Args:
        path: Path to the cache JSON file.

    Returns:
        Dictionary mapping source values to taxonomy values, or None if
        the cache file doesn't exist or is invalid.
    """
    cache_path = Path(path)

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "mappings" not in data:
            logger.warning(f"Invalid cache file structure: {cache_path}")
            return None

        return data["mappings"]

    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load cache file {cache_path}: {e}")
        return None


def save_mapping_cache(
    path: str | Path,
    result: TaxonomyMappingResult,
) -> None:
    """Save a taxonomy mapping to a JSON cache file.

    Also creates a reverse mapping file showing which source values
    mapped to each taxonomy value.

    Args:
        path: Path where the cache file should be saved.
        result: TaxonomyMappingResult to save.
    """
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Build reverse mapping: taxonomy_value -> [source_values]
    reverse_mapping: Dict[str, List[str]] = {}
    for source_value, taxonomy_value in result.mapping.items():
        if taxonomy_value is not None:
            if taxonomy_value not in reverse_mapping:
                reverse_mapping[taxonomy_value] = []
            reverse_mapping[taxonomy_value].append(source_value)

    # Sort the lists for readability
    for key in reverse_mapping:
        reverse_mapping[key] = sorted(reverse_mapping[key])

    data = {
        "version": "1.2",
        "created_at": result.created_at,
        "llm_model": result.llm_model,
        "taxonomy_column": result.taxonomy_column,
        "taxonomy_values": result.taxonomy_values,
        "mappings": result.mapping,
        "unmapped": result.unmapped,
        "reverse_mapping": reverse_mapping,
        "source_value_counts_by_dataset": result.source_value_counts_by_dataset,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Also save a separate reverse mapping file for easy viewing
    reverse_path = cache_path.with_suffix(".reverse.json")
    with open(reverse_path, "w", encoding="utf-8") as f:
        json.dump(reverse_mapping, f, indent=2)

    logger.info(f"Saved taxonomy mapping cache to {cache_path}")
    logger.info(f"Saved reverse mapping to {reverse_path}")


def apply_taxonomy_mapping(
    series: pd.Series,
    mapping: Dict[str, Optional[str]],
) -> pd.Series:
    """Apply a taxonomy mapping to a pandas Series.

    Values that are not in the mapping or are mapped to None keep their
    original value.

    Args:
        series: Pandas Series to transform.
        mapping: Dictionary mapping source values to taxonomy values.

    Returns:
        Transformed Series with taxonomy values applied.
    """
    def transform_value(value: Any) -> Any:
        if pd.isna(value):
            return value

        text = str(value).strip()
        if not text:
            return value

        mapped = mapping.get(text)
        if mapped is not None:
            return mapped

        # No mapping found or mapped to None - keep original
        return value

    return series.apply(transform_value)


__all__ = [
    "TaxonomyMappingResult",
    "TaxonomyLoader",
    "TaxonomyMapper",
    "load_mapping_cache",
    "save_mapping_cache",
    "apply_taxonomy_mapping",
]
