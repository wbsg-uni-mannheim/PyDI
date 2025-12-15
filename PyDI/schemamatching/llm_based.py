"""LLM-based schema matching using Large Language Models."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

try:
    import pydantic
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatModel = None
    pydantic = None
    ChatPromptTemplate = None

from .base import BaseSchemaMatcher, SchemaMapping, get_schema_columns
from ..utils.llm import LLMCallLogger


logger = logging.getLogger(__name__)


class LLMBasedSchemaMatcher(BaseSchemaMatcher):
    """LLM-based schema matcher using Large Language Models.

    This matcher uses LangChain's BaseChatModel interface to match schemas
    by showing the LLM both datasets in markdown format and asking it to
    identify column correspondences based on column names, data types, and
    sample values.

    Parameters
    ----------
    chat_model : langchain_core.language_models.chat_models.BaseChatModel
        LangChain chat model instance (e.g., ChatOpenAI, ChatAnthropic)
    num_rows : int, optional
        Number of sample rows to show to the LLM, by default 5
    system_prompt : str, optional
        Custom system prompt for the LLM. If None, uses default prompt.
    user_prompt_template : str, optional
        Template for the user prompt. If None, uses default template.
    temperature : float, optional
        Model temperature for generation, by default 0.0
    max_retries : int, optional
        Number of retry attempts on failure, by default 1
    out_dir : str, optional
        Output directory for artifacts, by default "output/schemamatching"
    debug : bool, optional
        Enable debug mode with detailed artifacts, by default False
    target_schema : dict, optional
        JSON Schema dict describing target fields. When provided, the LLM
        sees field names, types, and descriptions for better matching.

    Examples
    --------
    >>> from langchain_openai import ChatOpenAI
    >>> from PyDI.schemamatching import LLMBasedSchemaMatcher
    >>>
    >>> chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    >>> matcher = LLMBasedSchemaMatcher(
    ...     chat_model=chat,
    ...     num_rows=10,
    ...     target_schema={"properties": {"name": {"type": "string", "description": "Full name"}}}
    ... )
    >>> mappings = matcher.match(source_df, target_df)
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        *,
        num_rows: int = 5,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 1,
        out_dir: str = "output/schemamatching",
        debug: bool = False,
        target_schema: Optional[Union[str, Dict[str, Any]]] = None,
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain dependencies not available. Install with: "
                "pip install langchain-core pydantic"
            )

        self.chat_model = chat_model
        self.num_rows = num_rows
        self.temperature = temperature
        self.max_retries = max_retries
        self.debug = debug
        self.target_schema = target_schema

        # Set up output directory for artifacts
        if debug:
            from datetime import datetime
            from pathlib import Path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.out_dir = Path(out_dir) / f"llm_matching_{timestamp}"
            self.out_dir.mkdir(parents=True, exist_ok=True)
        else:
            from pathlib import Path
            self.out_dir = Path(out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)

        # Unified LLM logger for calls
        self._llm_logger = LLMCallLogger()

        # Default prompts
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.user_prompt_template = user_prompt_template or self._get_default_user_prompt_template()

        logger.info(f"Initialized LLMBasedSchemaMatcher with {num_rows} sample rows")

    def _format_target_schema(self, schema: Optional[Dict[str, Any]] = None) -> str:
        """Format target schema as readable text for LLM prompt.

        Parameters
        ----------
        schema : dict, optional
            JSON Schema to format. If not provided, uses self.target_schema.

        Returns
        -------
        str
            Formatted schema description or empty string
        """
        effective_schema = schema if schema is not None else self.target_schema
        if not effective_schema:
            return ""

        lines = []
        title = effective_schema.get("title", "Target Schema")
        description = effective_schema.get("description", "")

        lines.append(f"**{title}**")
        if description:
            lines.append(f"{description}")
        lines.append("")
        lines.append("**Target Fields:**")

        properties = effective_schema.get("properties", {})
        for field_name, field_spec in properties.items():
            field_type = field_spec.get("type", "any")
            field_desc = field_spec.get("description", "")
            if field_desc:
                lines.append(f"- `{field_name}` ({field_type}): {field_desc}")
            else:
                lines.append(f"- `{field_name}` ({field_type})")

        return "\n".join(lines)

    def _format_source_metadata(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Format Schema.org dataset metadata for LLM prompt.

        Supports Schema.org Dataset format with variableMeasured for column descriptions.

        Parameters
        ----------
        metadata : dict, optional
            Schema.org metadata dict with @type "Dataset"

        Returns
        -------
        str
            Formatted metadata description or empty string
        """
        if not metadata:
            return ""

        lines = []

        # Dataset-level info
        name = metadata.get("name", "")
        description = metadata.get("description", "")

        if name:
            lines.append(f"**{name}**")
        if description:
            lines.append(f"{description}")

        # Variable/column descriptions from variableMeasured
        variables = metadata.get("variableMeasured", [])
        if variables:
            lines.append("")
            lines.append("**Source Fields:**")
            for var in variables:
                if isinstance(var, dict):
                    var_name = var.get("name", "")
                    var_desc = var.get("description", "")
                    var_unit = var.get("unitText", "")

                    if var_name:
                        parts = [f"- `{var_name}`"]
                        if var_unit:
                            parts.append(f"({var_unit})")
                        if var_desc:
                            parts.append(f": {var_desc}")
                        lines.append(" ".join(parts) if len(parts) > 1 else parts[0])

        return "\n".join(lines) if lines else ""

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for schema matching."""
        return """You are an expert at aligning table schemas.

Given Table A and Table B, identify the matching columns between them. For every column in Table A, specify the corresponding column in Table B. If a column in Table A has no match in Table B, map it to null. Represent each mapping as a two-item list like ["Table A column", "Table B column" or null].

Return the final answer strictly as JSON in the format {{"column_mappings": [["Table A column", "Table B column"], ["Table A column", null], ...]}} with no additional commentary."""

    def _get_default_user_prompt_template(self) -> str:
        """Get the default user prompt template."""
        return """Description: Please identify the matching columns between Table A and Table B. For each column in Table A, specify the corresponding column in Table B. If a column in Table A has no corresponding column in Table B, map it to null. Represent every mapping using a pair [Table A column, Table B column or null]. Provide a mapping for each column in Table A.

Question:
## Table A (source: {source_name})

### Dataset Description
{source_schema_info}

### Provenance
{source_metadata}

### Sample Data
{source_table}

## Table B (source: {target_name})

### Dataset Description
{target_schema_info}

### Provenance
{target_metadata}

### Sample Data
{target_table}

Return the final result as JSON in the format {{"column_mappings": [["Table A column", "Table B column"], ["Table A column", null], ...]}}."""

    def _extract_dataset_metadata(self, df: pd.DataFrame) -> str:
        """Extract and format metadata from DataFrame attrs for LLM context.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to extract metadata from

        Returns
        -------
        str
            Formatted metadata string
        """
        attrs = getattr(df, 'attrs', {}) or {}
        provenance = attrs.get('provenance', {})

        metadata_lines = []

        # Basic info
        metadata_lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")

        # Source information
        if 'source_path' in provenance:
            source_path = provenance['source_path']
            metadata_lines.append(f"Source: {source_path}")

        # File metadata
        if 'file_size_bytes' in provenance:
            size_mb = provenance['file_size_bytes'] / (1024 * 1024)
            metadata_lines.append(f"File size: {size_mb:.2f} MB")

        if 'modified_time_utc_iso' in provenance:
            metadata_lines.append(f"Modified: {provenance['modified_time_utc_iso']}")

        # Reader information
        if 'reader' in provenance:
            metadata_lines.append(f"Loaded via: {provenance['reader']}")

        # Additional provenance info
        for key in ['sha256_prefix', 'loaded_time_utc_iso']:
            if key in provenance:
                value = provenance[key]
                if key == 'sha256_prefix':
                    metadata_lines.append(f"SHA256: {value[:16]}...")
                elif key == 'loaded_time_utc_iso':
                    metadata_lines.append(f"Loaded: {value}")

        return "**Metadata:** " + " | ".join(metadata_lines) if metadata_lines else ""

    def _dataframe_to_markdown(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        *,
        num_rows: Optional[int] = None,
    ) -> str:
        """Convert DataFrame to markdown table format for LLM consumption.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to convert
        dataset_name : str
            Name of the dataset for headers

        Returns
        -------
        str
            Markdown table representation
        """
        # Get schema columns (excluding PyDI ID columns)
        schema_columns = get_schema_columns(df)

        if not schema_columns:
            return f"Dataset '{dataset_name}' has no analyzable columns."

        # Map string column names back to original column types for DataFrame access
        # get_schema_columns returns strings, but df may have int columns (headerless CSV)
        str_to_orig = {str(col): col for col in df.columns}
        original_columns = [str_to_orig[col] for col in schema_columns if col in str_to_orig]

        # Determine how many rows to include
        row_limit = num_rows if num_rows is not None else self.num_rows
        try:
            row_limit_int = int(row_limit)
        except (TypeError, ValueError):
            row_limit_int = self.num_rows
        if row_limit_int is None or row_limit_int <= 0:
            row_limit_int = max(1, min(len(df), self.num_rows or 1))

        # Sample rows prioritizing well-populated records
        sample_df = self._select_sample_rows(df, original_columns, row_limit_int)

        # Build markdown table
        return sample_df.to_markdown()

    def _select_sample_rows(
        self,
        df: pd.DataFrame,
        schema_columns: List[str],
        limit: int,
    ) -> pd.DataFrame:
        """Select representative rows prioritizing records without missing values."""

        if limit <= 0:
            return df[schema_columns].iloc[0:0]

        data = df[schema_columns]
        if data.empty:
            return data

        limit = min(limit, len(data))
        cleaned = data.replace("", pd.NA)
        missing_counts = cleaned.isna().sum(axis=1)

        full_rows = missing_counts[missing_counts == 0]
        if len(full_rows) >= limit:
            indices = full_rows.index[:limit]
        else:
            ordered_indices = missing_counts.sort_values(kind="mergesort").index[:limit]
            indices = ordered_indices

        sampled = data.loc[indices]
        sampled = sampled.sort_index()
        return sampled.head(limit)

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Build the prompt template with system and user messages."""
        # Don't escape default prompts - they're already properly formatted
        # Template variables use single braces {}, JSON examples use double braces {{}}
        messages = [
            ("system", self.system_prompt),
            ("human", self.user_prompt_template)
        ]
        return ChatPromptTemplate.from_messages(messages)

    def _parse_llm_response(self, response_text: str) -> List[Tuple[str, Optional[str]]]:
        """Parse LLM response to extract column mappings.

        Parameters
        ----------
        response_text : str
            Raw response from LLM

        Returns
        -------
        List[Tuple[str, Optional[str]]]
            List of (source_column, target_column) pairs
        """
        # Remove code fences if present
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            response_text = fence_match.group(1).strip()

        # Try to parse JSON
        try:
            parsed = json.loads(response_text.strip())

            # Handle new format: {"column_mappings": [["col1", "col2"], ...]}
            if isinstance(parsed, dict) and "column_mappings" in parsed:
                mappings = parsed["column_mappings"]
                if isinstance(mappings, list):
                    result = []
                    for mapping in mappings:
                        if isinstance(mapping, list) and len(mapping) == 2:
                            source_col = mapping[0]
                            target_col = mapping[1] if mapping[1] != "None" else None
                            result.append((source_col, target_col))
                    return result

            # Legacy format: [{"source_column": ..., "target_column": ...}, ...]
            elif isinstance(parsed, list):
                result = []
                for mapping in parsed:
                    if isinstance(mapping, dict):
                        source_col = mapping.get("source_column")
                        target_col = mapping.get("target_column")
                        if target_col == "None":
                            target_col = None
                        if source_col:
                            result.append((source_col, target_col))
                return result

            logger.warning(f"Unexpected JSON structure in LLM response: {type(parsed)}")
            return []
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return []

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

    def _write_artifact(self, filename: str, data: Any, force: bool = False) -> None:
        """Persist an artifact if debug mode is enabled or force is True."""
        if not self.debug and not force:
            return

        filepath = self.out_dir / filename
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
            with open(filepath, 'w') as f:
                f.write(str(data))

        logger.debug(f"Wrote artifact: {filepath}")

    def match(
        self,
        source_dataset: pd.DataFrame,
        target_dataset: pd.DataFrame,
        preprocess: Optional[Callable[[str], str]] = None,
        *,
        num_rows: Optional[int] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
        target_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SchemaMapping:
        """Find schema correspondences using LLM analysis.

        Parameters
        ----------
        source_dataset : pandas.DataFrame
            The source dataset
        target_dataset : pandas.DataFrame
            The target dataset
        preprocess : callable, optional
            Preprocessing function (not used in LLM matching)
        num_rows : int, optional
            Override for number of sample rows to show
        source_metadata : dict, optional
            Schema.org metadata for the source dataset. When provided, the LLM
            sees dataset description and column definitions from variableMeasured.
            Example: {"name": "Companies", "variableMeasured": [{"name": "revenue", "description": "Annual revenue"}]}
        target_schema : dict, optional
            JSON Schema for the target dataset. Overrides the target_schema set in
            the constructor. When provided, the LLM sees field names, types, and
            descriptions for better matching.
            Example: {"properties": {"name": {"type": "string", "description": "Full name"}}}
        **kwargs
            Additional keyword arguments (ignored)

        Returns
        -------
        SchemaMapping
            DataFrame with schema correspondences
        """
        # Get dataset names
        source_name = source_dataset.attrs.get("dataset_name", "source")
        target_name = target_dataset.attrs.get("dataset_name", "target")

        logger.info(f"LLM-based schema matching: {source_name} -> {target_name}")

        # Determine how many rows to include in the prompt
        if num_rows is not None:
            try:
                effective_num_rows = int(num_rows)
            except (TypeError, ValueError):
                logger.warning("Invalid num_rows override %r; falling back to matcher default.", num_rows)
                effective_num_rows = self.num_rows
        else:
            effective_num_rows = self.num_rows

        if effective_num_rows is None or effective_num_rows <= 0:
            effective_num_rows = max(1, self.num_rows or 1)

        # Extract provenance metadata and convert DataFrames to markdown
        source_provenance = self._extract_dataset_metadata(source_dataset)
        target_provenance = self._extract_dataset_metadata(target_dataset)
        source_markdown = self._dataframe_to_markdown(
            source_dataset,
            source_name,
            num_rows=effective_num_rows,
        )
        target_markdown = self._dataframe_to_markdown(
            target_dataset,
            target_name,
            num_rows=effective_num_rows,
        )

        # Build prompt template
        prompt_template = self._build_prompt_template()

        results = []

        for attempt in range(self.max_retries + 1):
            try:
                # Format source schema info from Schema.org metadata if available
                source_schema_info = self._format_source_metadata(source_metadata)
                if source_schema_info:
                    source_schema_info = source_schema_info + "\n\n"
                else:
                    source_schema_info = "(No dataset description available)\n\n"

                # Format target schema info - use parameter override if provided
                target_schema_info = self._format_target_schema(target_schema)
                if target_schema_info:
                    target_schema_info = target_schema_info + "\n\n"
                else:
                    target_schema_info = "(No dataset description available)\n\n"

                # Format prompt
                messages = prompt_template.format_messages(
                    source_name=source_name,
                    source_schema_info=source_schema_info,
                    source_metadata=source_provenance,
                    source_table=source_markdown,
                    target_name=target_name,
                    target_schema_info=target_schema_info,
                    target_metadata=target_provenance,
                    target_table=target_markdown
                )

                # Save prompt artifact if debug mode
                if self.debug:
                    prompt_data = {
                        "source_name": source_name,
                        "target_name": target_name,
                        "source_schema_info": source_schema_info,
                        "source_provenance": source_provenance,
                        "target_schema_info": target_schema_info,
                        "target_provenance": target_provenance,
                        "source_table": source_markdown,
                        "target_table": target_markdown,
                        "attempt": attempt,
                        "effective_num_rows": effective_num_rows
                    }
                    self._write_artifact(f"prompt_attempt_{attempt}.json", prompt_data)

                    # Also save as readable text
                    prompt_items = []
                    for msg in messages:
                        msg_type = getattr(msg, "type", type(msg).__name__)
                        msg_content = getattr(msg, "content", str(msg))
                        prompt_items.append(f"=== {msg_type.upper()} ===\n{msg_content}")
                    prompt_text = "\n\n".join(prompt_items)
                    self._write_artifact(f"prompt_attempt_{attempt}.txt", prompt_text)

                # Call LLM with timing
                start_time = time.time()
                response = self.chat_model.invoke(messages)
                duration_ms = (time.time() - start_time) * 1000.0

                # Extract response content
                response_text = self._normalize_response_content(response)

                # Save response artifact if debug mode
                if self.debug:
                    response_data = {
                        "response_text": response_text,
                        "metadata": {
                            "model": getattr(self.chat_model, 'model_name', 'unknown'),
                            "temperature": self.temperature,
                            "attempt": attempt,
                            "duration_ms": duration_ms
                        }
                    }
                    self._write_artifact(f"response_attempt_{attempt}.json", response_data)
                    self._write_artifact(f"response_attempt_{attempt}.txt", response_text)

                # Log LLM call
                self._llm_logger.record_call(
                    chat_model=self.chat_model,
                    messages=messages,
                    response=response,
                    row_index=0,  # Schema matching is a single "row"
                    attempt=attempt,
                    duration_ms=duration_ms,
                    temperature=self.temperature,
                    max_tokens=None
                )

                # Parse response
                mappings = self._parse_llm_response(response_text)

                if mappings:
                    # Process mappings returned by the LLM
                    source_columns = get_schema_columns(source_dataset)
                    target_columns = get_schema_columns(target_dataset)

                    for source_col, target_col in mappings:
                        try:
                            # Validate columns exist
                            source_valid = source_col in source_columns
                            target_valid = target_col is None or target_col in target_columns

                            if source_valid and target_valid:
                                if target_col is None:
                                    # Skip non-matches; LLM will surface these implicitly
                                    continue

                                # Assign a high confidence score for LLM matches that pass validation
                                confidence = 0.95

                                results.append({
                                    "source_dataset": source_name,
                                    "source_column": source_col,
                                    "target_dataset": target_name,
                                    "target_column": target_col,
                                    "score": confidence,
                                    "notes": "llm_based_matching"
                                })

                                logger.debug(f"LLM match: {source_col} -> {target_col} ({confidence:.3f})")
                            else:
                                logger.warning(f"Invalid column mapping: {source_col} -> {target_col}")

                        except Exception as e:
                            logger.warning(f"Error processing mapping {source_col} -> {target_col}: {e}")

                    # Success - break retry loop
                    break
                else:
                    logger.debug(f"No valid mappings found on attempt {attempt + 1}")
                    if attempt < self.max_retries:
                        time.sleep(0.1 * (2 ** attempt))

            except Exception as e:
                logger.debug(f"LLM call failed on attempt {attempt + 1}: {e}")

                if self.debug:
                    error_details = f"Exception type: {type(e).__name__}\nError message: {str(e)}\nRepr: {repr(e)}"
                    self._write_artifact(f"error_attempt_{attempt}.txt", error_details)

                if attempt < self.max_retries:
                    time.sleep(0.1 * (2 ** attempt))

        # Write final artifacts
        if self.debug:
            config = {
                "model": getattr(self.chat_model, 'model_name', 'unknown'),
                "num_rows": self.num_rows,
                "effective_num_rows": effective_num_rows,
                "temperature": self.temperature,
                "max_retries": self.max_retries,
                "system_prompt": self.system_prompt,
                "user_prompt_template": self.user_prompt_template
            }
            self._write_artifact("matching_config.json", config)

            stats = {
                "total_attempts": self.max_retries + 1,
                "matches_found": len(results),
                "source_columns": len(get_schema_columns(source_dataset)),
                "target_columns": len(get_schema_columns(target_dataset))
            }
            self._write_artifact("matching_stats.json", stats)

        # Always flush logs even when debug mode is disabled
        self._llm_logger.flush(lambda name, payload: self._write_artifact(name, payload, force=True))

        logger.info(f"LLM schema matching completed. Found {len(results)} correspondences")

        return pd.DataFrame(results)
