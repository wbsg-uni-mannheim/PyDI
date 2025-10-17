"""
LLM-based entity matching using conversational AI models.

This module provides an LLMBasedMatcher that uses large language models to
determine if pairs of records refer to the same real-world entity.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from ..utils.llm import LLMCallLogger
from .base import BaseMatcher


class LLMBasedMatcher(BaseMatcher):
    """
    LLM-based entity matcher using conversational AI models.

    This matcher uses large language models to determine if pairs of records
    refer to the same real-world entity. It supports both zero-shot and few-shot
    prompting modes with configurable output formats and unified LLM logging.

    Examples
    --------
    Zero-shot matching:

    >>> from langchain_openai import ChatOpenAI
    >>> from PyDI.entitymatching import LLMBasedMatcher
    >>> import pandas as pd
    >>>
    >>> # candidates is a DataFrame with id1, id2 columns
    >>> candidates = pd.DataFrame({"id1": [1, 2], "id2": [101, 102]})
    >>>
    >>> matcher = LLMBasedMatcher()
    >>> chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    >>> matches = matcher.match(
    ...     df_left, df_right, candidates, "_id",
    ...     chat_model=chat,
    ...     fields=["name", "address", "city"],
    ...     threshold=0.7
    ... )

    Few-shot matching with examples:

    >>> few_shots = [
    ...     (
    ...         {"name": "Acme Corp", "address": "12 Main St"},
    ...         {"name": "Acme Corporation", "address": "12 Main Street"},
    ...         '{"match": true, "score": 0.95, "explanation": "name+address match"}'
    ...     )
    ... ]
    >>> matches = matcher.match(
    ...     df_left, df_right, candidates, "_id",
    ...     chat_model=chat,
    ...     few_shots=few_shots,
    ...     threshold=0.8
    ... )
    """

    def __init__(self):
        """Initialize the LLM-based matcher."""
        super().__init__()
        self._llm_logger = LLMCallLogger()
        self._current_run_out_dir: Optional[Path] = None

    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        candidates: Union[pd.DataFrame, Iterable[pd.DataFrame]],
        id_column: str,
        chat_model: BaseChatModel,
        *,
        fields: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        few_shots: Optional[List[Tuple[Dict[str, Any],
                                       Dict[str, Any], str]]] = None,
        threshold: float = 0.5,
        retries: int = 1,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_parser: Optional[Callable[[str],
                                           Optional[Dict[str, Any]]]] = None,
        out_dir: str = "output/entitymatching/llm",
        debug: bool = False,
        max_concurrency: int = 1,
        rate_limit_per_sec: Optional[float] = None,
        parse_strictness: str = "skip",
        notes_detail: bool = True
    ) -> pd.DataFrame:
        """
        Match entity pairs using an LLM.

        Parameters
        ----------
        df_left : pd.DataFrame
            Left dataset with records to match.
        df_right : pd.DataFrame
            Right dataset with records to match.
        candidates : pd.DataFrame or Iterable[pd.DataFrame]
            Single DataFrame or iterable of candidate pair batches with id1, id2 columns
            representing candidate pairs to evaluate.
        id_column : str
            Name of the column containing record identifiers.
        chat_model : BaseChatModel
            LangChain chat model instance (e.g., ChatOpenAI, ChatAnthropic).
        fields : Optional[List[str]], default=None
            List of column names to include in LLM prompts. If None, auto-selects
            string-like columns up to a reasonable limit.
        system_prompt : Optional[str], default=None
            Custom system prompt. If None, uses a default entity matching prompt.
        few_shots : Optional[List[Tuple[Dict, Dict, str]]], default=None
            Few-shot examples as (left_record, right_record, expected_json) tuples.
        threshold : float, default=0.5
            Minimum score threshold for including matches in results.
        retries : int, default=1
            Number of retry attempts on API failures.
        temperature : Optional[float], default=None
            Deprecated: temperature is no longer passed to the chat model. Configure
            temperature on the chat model instance itself.
        max_tokens : Optional[int], default=None
            Maximum tokens in model response.
        response_parser : Optional[Callable[[str], Optional[Dict[str, Any]]]], default=None
            Optional custom parser that takes the raw response text and returns a
            dict with keys {'score': float, 'notes': str} (and optionally 'match').
            Use this when your prompt returns a different format than the default JSON.
        out_dir : str, default="output/entitymatching/llm"
            Directory for writing debug artifacts and logs.
        debug : bool, default=False
            Whether to write debug artifacts (prompts, responses, stats).
        max_concurrency : int, default=1
            Maximum concurrent API calls (not implemented yet).
        rate_limit_per_sec : Optional[float], default=None
            Rate limit for API calls per second (not implemented yet).
        parse_strictness : str, default="skip"
            How to handle JSON parsing errors: "skip" or "zero_score".
        notes_detail : bool, default=True
            Whether to include explanation details in the notes column.

        Returns
        -------
        pd.DataFrame
            CorrespondenceSet with columns: id1, id2, score, notes.
        """
        # Validate inputs using base class
        self._validate_inputs(df_left, df_right, id_column)

        # Initialize output directory and remember for artifact writes
        self._current_run_out_dir = Path(out_dir)
        if debug:
            self._current_run_out_dir.mkdir(parents=True, exist_ok=True)

        # Auto-select fields if not provided
        if fields is None:
            fields = self._auto_select_fields(df_left, df_right, id_column)

        # Build prompt template
        prompt_template = self._build_prompt_template(system_prompt, few_shots)

        # Normalize candidates to iterable of DataFrames
        if isinstance(candidates, pd.DataFrame):
            candidate_batches = [candidates]
        else:
            candidate_batches = list(candidates)

        # Log matching info
        self._log_matching_info(df_left, df_right, candidate_batches)

        matches = []
        pair_index = 0

        # Process candidate batches
        for batch in candidate_batches:
            if batch.empty:
                continue

            # Validate batch has required columns
            if "id1" not in batch.columns or "id2" not in batch.columns:
                raise ValueError("Candidate DataFrame must have 'id1' and 'id2' columns")

            # Process each pair in the batch
            for _, row in batch.iterrows():
                left_id, right_id = row["id1"], row["id2"]

                # Get records
                left_record = df_left[df_left[id_column] == left_id].iloc[0]
                right_record = df_right[df_right[id_column] == right_id].iloc[0]

                # Serialize records for prompt
                left_data = self._serialize_record(left_record, fields, id_column)
                right_data = self._serialize_record(right_record, fields, id_column)

                # Try matching with retries
                match_result = self._match_pair_with_retry(
                    prompt_template, left_data, right_data, chat_model,
                    retries, temperature, max_tokens, pair_index, out_dir, debug,
                    parse_strictness, notes_detail, response_parser)

                if match_result and match_result["score"] >= threshold:
                    matches.append({
                        "id1": left_id,
                        "id2": right_id,
                        "score": match_result["score"],
                        "notes": match_result["notes"]
                    })

                pair_index += 1

        # Flush LLM logs and write artifacts
        if debug:
            self._write_debug_artifacts(out_dir, matches, pair_index)

            # Write configuration for this run
            config = {
                "model": getattr(chat_model, 'model_name', 'unknown'),
                "fields": fields,
                "few_shots_count": len(few_shots) if few_shots else 0,
                "threshold": threshold,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "retries": retries,
                "system_prompt_provided": bool(system_prompt),
                "parse_strictness": parse_strictness,
                "notes_detail": notes_detail,
                "total_candidates": pair_index,
            }
            self._write_artifact("llm_config.json", config)

        # Always flush LLM logs (writes llm_calls.json and llm_usage_summary.json)
        self._llm_logger.flush(self._write_artifact)

        return pd.DataFrame(matches) if matches else pd.DataFrame(columns=["id1", "id2", "score", "notes"])

    def _auto_select_fields(self, df_left: pd.DataFrame, df_right: pd.DataFrame, id_column: str, max_fields: int = 10) -> List[str]:
        """Auto-select string-like fields for LLM prompts."""
        # Get common columns (excluding id_column)
        common_cols = set(df_left.columns) & set(df_right.columns) - {id_column}

        # Prefer string-like columns
        string_cols = []
        for col in common_cols:
            if df_left[col].dtype == 'object' or df_right[col].dtype == 'object':
                string_cols.append(col)

        # If we have too many, limit to first few
        if len(string_cols) > max_fields:
            string_cols = string_cols[:max_fields]

        # If no string columns, take any common columns
        if not string_cols:
            string_cols = list(common_cols)[:max_fields]

        return string_cols

    def _build_prompt_template(
        self,
        system_prompt: Optional[str],
        few_shots: Optional[List[Tuple[Dict[str, Any], Dict[str, Any], str]]]
    ) -> ChatPromptTemplate:
        """Build the chat prompt template with system message and optional few-shot examples."""
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        messages = [SystemMessagePromptTemplate.from_template(system_prompt)]

        # Add few-shot examples if provided
        if few_shots:
            few_shot_examples = []
            for left_ex, right_ex, expected_json in few_shots:
                few_shot_examples.append({
                    "left_record": json.dumps(left_ex, ensure_ascii=False),
                    "right_record": json.dumps(right_ex, ensure_ascii=False),
                    "output": expected_json
                })

            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=ChatPromptTemplate.from_messages([
                    ("human",
                     "Left record: {left_record}\nRight record: {right_record}"),
                    ("assistant", "{output}")
                ]),
                examples=few_shot_examples
            )
            messages.append(few_shot_prompt)

        # Add human message template for the actual comparison
        messages.append(HumanMessagePromptTemplate.from_template(
            "Left record: {left_record}\nRight record: {right_record}\n\n"
            "Return JSON matching the schema described above."
        ))

        return ChatPromptTemplate.from_messages(messages)

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for entity matching."""
        return """You are an expert entity resolver. Your task is to decide if two records refer to the same real-world entity.

Analyze the provided records carefully and return your decision as strict JSON in this format:
{{"match": true|false, "score": <float between 0.0 and 1.0>, "explanation": "<brief explanation>"}}

Guidelines:
- score should reflect your confidence (1.0 = definitely same entity, 0.0 = definitely different)
- match should be true if score >= 0.5, false otherwise  
- explanation should be concise (1-2 sentences)
- Consider variations in naming, formatting, abbreviations, and data quality
- Respond with ONLY the JSON object and nothing else."""

    def _serialize_record(self, record: pd.Series, fields: List[str], id_column: str, max_length: int = 200) -> str:
        """Serialize a record for the LLM prompt, including only specified fields."""
        data = {}

        for field in fields:
            if field in record and pd.notna(record[field]):
                value = str(record[field])
                # Truncate long strings
                if len(value) > max_length:
                    value = value[:max_length] + "..."
                data[field] = value

        return json.dumps(data, ensure_ascii=False)

    def _match_pair_with_retry(
        self,
        prompt_template: ChatPromptTemplate,
        left_data: str,
        right_data: str,
        chat_model: BaseChatModel,
        retries: int,
        temperature: Optional[float],
        max_tokens: Optional[int],
        pair_index: int,
        out_dir: str,
        debug: bool,
        parse_strictness: str,
        notes_detail: bool,
        response_parser: Optional[Callable[[str],
                                           Optional[Dict[str, Any]]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Match a single pair with retry logic."""
        for attempt in range(retries + 1):
            try:
                # Format the prompt
                messages = prompt_template.format_messages(
                    left_record=left_data,
                    right_record=right_data
                )

                # Write debug artifacts
                if debug:
                    self._write_prompt_artifacts(
                        out_dir, pair_index, attempt, messages)

                # Call the model
                start_time = time.time()
                # Build kwargs without overriding model-level configuration
                invoke_kwargs: Dict[str, Any] = {}
                if max_tokens is not None:
                    invoke_kwargs["max_tokens"] = max_tokens
                # Do NOT pass temperature; rely on chat_model's own configuration
                response = chat_model.invoke(messages, **invoke_kwargs)
                duration = time.time() - start_time

                # Log the call using unified logger signature
                self._llm_logger.record_call(
                    chat_model=chat_model,
                    messages=messages,
                    response=response,
                    row_index=pair_index,
                    attempt=attempt,
                    duration_ms=duration * 1000.0,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                response_text = response.content

                # Write debug artifacts
                if debug:
                    self._write_response_artifacts(
                        out_dir, pair_index, attempt, response_text)

                # Parse the response (custom parser first, then default)
                if response_parser is not None:
                    try:
                        custom = response_parser(response_text)
                        if isinstance(custom, dict):
                            # Normalize custom output
                            score: float
                            if "score" in custom:
                                score = float(custom.get("score", 0.0))
                            elif "match" in custom:
                                score = 1.0 if bool(
                                    custom.get("match")) else 0.0
                            else:
                                raise ValueError(
                                    "Custom parser must return 'score' or 'match'.")

                            notes = str(custom.get(
                                "notes", custom.get("explanation", "")))
                            return {"score": max(0.0, min(1.0, score)), "notes": notes}
                    except Exception as e:
                        if parse_strictness == "zero_score":
                            return {"score": 0.0, "notes": f"Custom parse error: {str(e)}"}
                        # else fall through to default parser

                return self._parse_response(response_text, parse_strictness, notes_detail)

            except Exception as e:
                if debug:
                    self._write_error_artifacts(
                        out_dir, pair_index, attempt, str(e))

                if attempt < retries:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                    continue
                else:
                    # Final attempt failed
                    if parse_strictness == "zero_score":
                        return {
                            "score": 0.0,
                            "notes": f"LLM call failed after {retries + 1} attempts: {str(e)}"
                        }
                    else:
                        return None

        return None

    def _parse_response(self, response_text: str, parse_strictness: str, notes_detail: bool) -> Optional[Dict[str, Any]]:
        """Parse the LLM response and extract match information."""
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json_from_response(response_text)
            if not json_str:
                raise ValueError("No JSON found in response")

            data = json.loads(json_str)

            # Extract score
            score = float(
                data.get("score", 1.0 if data.get("match", False) else 0.0))

            # Extract notes
            notes = ""
            if notes_detail and "explanation" in data:
                notes = data["explanation"]

            return {
                "score": max(0.0, min(1.0, score)),  # Clamp to [0,1]
                "notes": notes
            }

        except Exception as e:
            if parse_strictness == "zero_score":
                return {
                    "score": 0.0,
                    "notes": f"Parse error: {str(e)}"
                }
            else:
                return None

    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """Extract JSON object from response text, handling extra text."""
        # Look for JSON object boundaries
        start = response_text.find('{')
        if start == -1:
            return None

        # Find matching closing brace
        brace_count = 0
        for i, char in enumerate(response_text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return response_text[start:i+1]

        return None

    def _write_prompt_artifacts(self, out_dir: str, pair_index: int, attempt: int, messages: List[BaseMessage]):
        """Write prompt artifacts for debugging."""
        prompts_dir = Path(out_dir) / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)

        # Write as text
        with open(prompts_dir / f"row_{pair_index}_attempt_{attempt}.txt", "w") as f:
            for msg in messages:
                f.write(f"{msg.__class__.__name__}: {msg.content}\n\n")

        # Write as JSON
        with open(prompts_dir / f"row_{pair_index}_attempt_{attempt}.json", "w") as f:
            json.dump([{"type": msg.__class__.__name__, "content": msg.content}
                      for msg in messages], f, indent=2)

    def _write_response_artifacts(self, out_dir: str, pair_index: int, attempt: int, response_text: str):
        """Write response artifacts for debugging."""
        responses_dir = Path(out_dir) / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)

        with open(responses_dir / f"row_{pair_index}_attempt_{attempt}.txt", "w") as f:
            f.write(response_text)

    def _write_error_artifacts(self, out_dir: str, pair_index: int, attempt: int, error_text: str):
        """Write error artifacts for debugging."""
        errors_dir = Path(out_dir) / "errors"
        errors_dir.mkdir(parents=True, exist_ok=True)

        with open(errors_dir / f"row_{pair_index}_attempt_{attempt}.txt", "w") as f:
            f.write(error_text)

    def _write_debug_artifacts(self, out_dir: str, matches: List[Dict], total_candidates: int):
        """Write final debug artifacts."""
        # Write stats
        stats = {
            "total_candidates": total_candidates,
            "total_matches": len(matches),
            "match_rate": len(matches) / total_candidates if total_candidates > 0 else 0.0
        }

        with open(Path(out_dir) / "llm_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Write sample matches
        if matches:
            sample_df = pd.DataFrame(matches[:10])  # First 10 matches
            sample_df.to_csv(Path(out_dir) / "sample_matches.csv", index=False)

    def _write_artifact(self, artifact_name: str, content: Any):
        """Write artifact to current run directory.

        Supports JSON (dict/list), text (str), and DataFrame to CSV.
        """
        base_dir = self._current_run_out_dir or Path(
            "output/entitymatching/llm")
        filepath = base_dir / artifact_name
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, (dict, list)):
            with open(filepath, 'w') as f:
                json.dump(content, f, indent=2, default=str)
        elif isinstance(content, str):
            with open(filepath, 'w') as f:
                f.write(content)
        elif isinstance(content, pd.DataFrame):
            content.to_csv(filepath, index=False)
        else:
            # Fallback to string serialization
            with open(filepath, 'w') as f:
                f.write(str(content))
