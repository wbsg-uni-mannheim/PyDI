"""LLM-based information extraction using LangChain."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

try:
    import pydantic
    from pydantic import create_model
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseChatModel = None
    pydantic = None
    ChatPromptTemplate = None
    PydanticOutputParser = None
    FewShotChatMessagePromptTemplate = None

from .base import BaseExtractor
from ..utils.llm import LLMCallLogger


logger = logging.getLogger(__name__)


class LLMExtractor(BaseExtractor):
    """Extract information using Large Language Models via LangChain.

    This extractor uses LangChain's BaseChatModel interface to extract structured
    information from text using natural language prompts and schema definitions.
    Supports Pydantic schemas for structured output and validation.

    Parameters
    ----------
    chat_model : langchain_core.language_models.chat_models.BaseChatModel
        LangChain chat model instance (e.g., ChatOpenAI, ChatAnthropic)
    schema : Type[pydantic.BaseModel] or dict
        Schema defining the fields to extract. Pydantic BaseModel preferred
        for automatic validation and type coercion
    source_column : str
        Default column name to extract from
    system_prompt : str
        System prompt providing extraction instructions
    few_shots : List[Tuple[str, str]], optional
        List of (input_text, expected_json) example pairs for few-shot learning
    temperature : float, optional
        Model temperature for generation, by default 0.0
    max_tokens : int, optional
        Maximum tokens to generate, by default None
    retries : int, optional
        Number of retry attempts on failure, by default 1
    out_dir : str, optional
        Output directory for artifacts, by default "output/informationextraction"
    debug : bool, optional
        Enable debug mode with detailed artifacts, by default False

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> from langchain_openai import ChatOpenAI
    >>> from PyDI.informationextraction import LLMExtractor
    >>> 
    >>> class Product(BaseModel):
    ...     brand: Optional[str] = None
    ...     model: Optional[str] = None
    ...     price: Optional[float] = None
    >>> 
    >>> chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    >>> extractor = LLMExtractor(
    ...     chat_model=chat,
    ...     schema=Product,
    ...     source_column="description",
    ...     system_prompt="Extract product information as JSON matching the schema."
    ... )
    >>> result_df = extractor.extract(df)
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        source_column: str,
        system_prompt: str,
        *,
        schema: Optional[Union[Type[pydantic.BaseModel],
                               Dict[str, Any]]] = None,
        few_shots: Optional[List[Tuple[str, str]]] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        retries: int = 1,
        out_dir: str = "output/informationextraction",
        debug: bool = False
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain dependencies not available. Install with: "
                "pip install langchain-core pydantic"
            )

        super().__init__(out_dir=out_dir, debug=debug)

        self.chat_model = chat_model
        self.default_source = source_column
        self.system_prompt = system_prompt
        self.schema = schema
        self._json_schema: Optional[Dict[str, Any]] = None
        self._pyd_model: Optional[Type[pydantic.BaseModel]] = None
        self.few_shots = few_shots or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries

        # Unified LLM logger for blocking calls
        self._llm_logger = LLMCallLogger()

        # Determine schema mode
        self.open_schema = schema is None
        # Determine if Pydantic model was provided directly
        self.is_pydantic = (
            not self.open_schema and
            hasattr(schema, '__bases__') and
            pydantic is not None and
            pydantic.BaseModel in getattr(schema, '__bases__', [])
        )

        # Setup output parser and fields
        if self.open_schema:
            self.output_parser = None
            self.schema_fields = ['extracted']
        elif self.is_pydantic:
            # Pydantic model provided directly
            self._pyd_model = schema  # type: ignore[assignment]
            self.output_parser = PydanticOutputParser(pydantic_object=schema)
            # Prefer __annotations__ when available; fallback to model_fields for Pydantic v2
            try:
                self.schema_fields = list(schema.__annotations__.keys())  # type: ignore[union-attr]
            except Exception:
                self.schema_fields = list(getattr(schema, 'model_fields', {}).keys())  # type: ignore[arg-type]
        else:
            # Dict schema provided; try to detect JSON Schema and convert to Pydantic
            self.output_parser = None
            self.schema_fields: List[str] = []
            if isinstance(schema, dict) and schema:
                # JSON Schema detection (simple heuristic)
                if 'properties' in schema or schema.get('$schema'):
                    self._json_schema = schema
                    if pydantic is not None:
                        try:
                            self._pyd_model = self._build_pydantic_from_json_schema(schema)
                            if self._pyd_model is not None:
                                self.output_parser = PydanticOutputParser(pydantic_object=self._pyd_model)
                                # Collect fields from model
                                try:
                                    self.schema_fields = list(self._pyd_model.__annotations__.keys())  # type: ignore[union-attr]
                                except Exception:
                                    self.schema_fields = list(getattr(self._pyd_model, 'model_fields', {}).keys())  # type: ignore[arg-type]
                                # Treat as pydantic downstream
                                self.is_pydantic = True
                        except Exception as conv_err:
                            logger.warning(f"Failed to convert JSON Schema to Pydantic model: {conv_err}")
                    # If conversion failed, fall back to property names only
                    if not self.schema_fields:
                        props = schema.get('properties', {})
                        if isinstance(props, dict):
                            self.schema_fields = list(props.keys())
                else:
                    # Plain dict of fields (fallback)
                    self.schema_fields = list(schema.keys())

        logger.info(
            f"Initialized LLMExtractor with {len(self.schema_fields)} schema fields")

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Build the prompt template with system message and few-shot examples."""
        # Escape curly braces in system prompt to prevent unintended template variables
        # (e.g., JSON examples like {"brand": ...} being parsed as a variable).
        safe_system_prompt = self.system_prompt.replace(
            "{", "{{").replace("}", "}}")
        messages = [("system", safe_system_prompt)]

        # Add format instructions if a structured output parser is available
        if self.output_parser is not None:
            try:
                fmt_instructions = self.output_parser.get_format_instructions()
                # Escape braces so ChatPromptTemplate doesn't treat them as variables
                safe_fmt = fmt_instructions.replace("{", "{{").replace("}", "}}")
                messages.append(("system", f"Follow these output format instructions strictly:\n{safe_fmt}"))
            except Exception:
                pass

        # Add few-shot examples if provided
        if self.few_shots:
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=ChatPromptTemplate.from_messages([
                    ("human", "{input}"),
                    ("ai", "{output}")
                ]),
                examples=[
                    {"input": inp, "output": out}
                    for inp, out in self.few_shots
                ]
            )
            messages.append(few_shot_prompt)

        # Add the main input template
        messages.append(("human", "{text}"))

        return ChatPromptTemplate.from_messages(messages)

    def _normalize_response_content(self, response: Any) -> str:
        """Normalize a LangChain response content to plain text.

        Handles string content, and list-of-blocks content such as
        [{"type":"text", "text":"..."}, ...].
        """
        # If it's a standard message with 'content'
        content = getattr(response, "content", response)
        # If content is already a string
        if isinstance(content, str):
            return content
        # If content is a list of blocks (e.g., langchain 0.2 content blocks)
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                try:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(str(block.get("text", "")))
                    else:
                        # Fallback: string representation
                        parts.append(str(block))
                except Exception:
                    continue
            return "\n".join(p.strip() for p in parts if p)
        # Fallback to string cast
        try:
            return str(content)
        except Exception:
            return ""

    def _extract_json_from_response(self, response_content: Any) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response with robust fallback parsing."""
        text = self._normalize_response_content(response_content)

        # Remove common code fences
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

        # First try direct JSON parsing
        try:
            return json.loads(text.strip())
        except Exception:
            pass

        # Try to find JSON within the response (trim to outermost braces)
        json_match = re.search(r"\{[\s\S]*\}", text, re.DOTALL)
        if json_match:
            candidate = json_match.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                pass

        logger.warning("Failed to parse JSON from LLM response")
        return None

    def _validate_and_coerce(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and coerce data using Pydantic schema if available."""
        if not data:
            return {}

        # Prefer Pydantic validation when model is available
        if self.is_pydantic and self._pyd_model is not None:
            try:
                instance = self._pyd_model(**data)
                return instance.model_dump()
            except Exception as e:
                logger.warning(f"Pydantic validation failed: {e}")
                return {field: None for field in self.schema_fields}

        # Optional: validate against JSON Schema if provided
        if self._json_schema is not None:
            try:
                import jsonschema  # optional dependency
                jsonschema.validate(instance=data, schema=self._json_schema)
            except Exception as e:
                logger.warning(f"JSON Schema validation failed: {e}")
                return {field: None for field in self.schema_fields}

        # No validation applied; return data constrained to known fields
        return {k: data.get(k) for k in self.schema_fields}

    def _build_pydantic_from_json_schema(self, schema: Dict[str, Any]) -> Optional[Type[pydantic.BaseModel]]:
        """Create a Pydantic model from a simple JSON Schema object.

        Supports basic 'type' mappings for properties and 'required' fields.
        """
        if pydantic is None:
            return None

        props = schema.get('properties', {})
        if not isinstance(props, dict):
            return None

        required = set(schema.get('required', []) or [])

        type_map = {
            'string': (str, ...),
            'number': (float, ...),
            'integer': (int, ...),
            'boolean': (bool, ...),
            'array': (list, ...),
            'object': (dict, ...),
        }

        fields: Dict[str, tuple] = {}
        for name, spec in props.items():
            if not isinstance(spec, dict):
                continue
            t = spec.get('type')
            # Handle enum as string
            if 'enum' in spec and not t:
                t = 'string'

            base = type_map.get(t, (str, ...))

            # Optional if not required
            default = None if name not in required else ...
            annotated_type = base[0]
            if default is None:
                from typing import Optional as TypingOptional
                annotated_type = TypingOptional[annotated_type]  # type: ignore[assignment]

            fields[name] = (annotated_type, default)

        model = create_model('DynamicExtractionSchema', **fields)  # type: ignore[arg-type]
        return model  # type: ignore[return-value]

    def _extract_from_text_with_retry(
        self,
        text: str,
        row_idx: int
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Extract information from text with retry logic."""

        prompt_template = self._build_prompt_template()

        artifacts = []

        for attempt in range(self.retries + 1):
            try:
                # Format prompt
                messages = prompt_template.format_messages(text=text)

                # Save prompt artifacts if debug mode (both JSON and TXT)
                if self.debug:
                    try:
                        prompt_items = []
                        for msg in messages:
                            msg_type = getattr(msg, "type", type(msg).__name__)
                            msg_content = getattr(msg, "content", str(msg))
                            prompt_items.append(
                                {"type": msg_type, "content": msg_content})

                        prompt_json_path = f"prompts/row_{row_idx}_attempt_{attempt}.json"
                        prompt_txt_path = f"prompts/row_{row_idx}_attempt_{attempt}.txt"

                        prompt_text = "\n\n".join([
                            f"=== {item['type'].upper()} ===\n{item['content']}" for item in prompt_items
                        ])

                        # Queue artifacts for end-of-run write
                        artifacts.append((prompt_json_path, prompt_items))
                        artifacts.append((prompt_txt_path, prompt_text))

                        # Also write immediately to ensure visibility even if later steps fail
                        self._write_artifact(prompt_json_path, prompt_items)
                        self._write_artifact(prompt_txt_path, prompt_text)
                    except Exception as prompt_err:
                        # Make sure prompt serialization errors don't block extraction
                        error_path = f"errors/row_{row_idx}_attempt_{attempt}_prompt_serialize.txt"
                        artifacts.append((error_path, str(prompt_err)))
                        self._write_artifact(error_path, str(prompt_err))

                # Call LLM with timing
                start_time = time.time()
                response = self.chat_model.invoke(messages)
                duration_ms = (time.time() - start_time) * 1000.0

                # Extract response content
                response_text = self._normalize_response_content(response)

                # Save response artifact if debug mode
                if self.debug:
                    response_path = f"responses/row_{row_idx}_attempt_{attempt}.json"
                    response_data = {
                        "response_text": response_text,
                        "metadata": {
                            "model": getattr(self.chat_model, 'model_name', 'unknown'),
                            "temperature": self.temperature,
                            "attempt": attempt
                        }
                    }
                    artifacts.append((response_path, response_data))
                    # Also save raw response as text for easy viewing
                    response_txt_path = f"responses/row_{row_idx}_attempt_{attempt}.txt"
                    artifacts.append((response_txt_path, response_text))
                    # Write immediately as well
                    self._write_artifact(response_path, response_data)
                    self._write_artifact(response_txt_path, response_text)

                # Unified per-call log record (always emit via logger; artifacts written on flush)
                self._llm_logger.record_call(
                    chat_model=self.chat_model,
                    messages=messages,
                    response=response,
                    row_index=row_idx,
                    attempt=attempt,
                    duration_ms=duration_ms,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                # Parse JSON
                parsed_data = self._extract_json_from_response(response_text)
                if parsed_data is not None:
                    if self.open_schema:
                        # In open-schema mode, put raw parsed JSON under 'extracted'
                        return {"extracted": parsed_data}, artifacts
                    # Validate and coerce if Pydantic schema
                    validated_data = self._validate_and_coerce(parsed_data)
                    return validated_data, artifacts

                logger.debug(
                    f"Failed to parse JSON on attempt {attempt + 1} for row {row_idx}")

                # Exponential backoff for retries
                if attempt < self.retries:
                    time.sleep(0.1 * (2 ** attempt))

            except Exception as e:
                logger.debug(
                    f"LLM call failed on attempt {attempt + 1} for row {row_idx}: {e}")

                if self.debug:
                    error_path = f"errors/row_{row_idx}_attempt_{attempt}.txt"
                    artifacts.append((error_path, str(e)))
                    # Write immediately so the error is visible without waiting for the end
                    self._write_artifact(error_path, str(e))

                if attempt < self.retries:
                    time.sleep(0.1 * (2 ** attempt))

        # All attempts failed - return empty values
        logger.warning(f"All extraction attempts failed for row {row_idx}")
        if self.open_schema:
            return {"extracted": None}, artifacts
        return {field: None for field in self.schema_fields}, artifacts

    def extract(
        self,
        df: pd.DataFrame,
        *,
        source_column: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Extract structured information using LLM.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame
        source_column : str, optional
            Column to extract from. If None, uses default source_column

        Returns
        -------
        pd.DataFrame
            DataFrame with extracted schema fields added as columns
        """
        # Validate input DataFrame
        try:
            validated_source = self._validate_input(df, source_column)
        except ValueError as e:
            logger.error(f"Input validation failed: {e}")
            return df

        # Determine source column
        extraction_source = source_column or self.default_source

        if extraction_source not in df.columns:
            logger.error(
                f"Source column '{extraction_source}' not found in DataFrame")
            return df

        logger.info(f"Starting LLM extraction from column '{extraction_source}' "
                    f"for {len(df)} rows with {len(self.schema_fields)} fields")

        result_df = df.copy()
        all_artifacts = []
        validation_errors = []
        extraction_stats = {}

        # Process each row
        for idx, text in enumerate(df[extraction_source]):
            if pd.isna(text) or not str(text).strip():
                # Empty text - set all fields to None
                for field in self.schema_fields:
                    if field not in result_df.columns:
                        result_df.loc[idx, field] = None
                continue

            # Extract from text
            extracted_data, row_artifacts = self._extract_from_text_with_retry(
                str(text), idx
            )

            # Add extracted fields to result
            if self.open_schema:
                result_df.loc[idx, 'extracted'] = json.dumps(extracted_data.get('extracted')) \
                    if extracted_data.get('extracted') is not None else None
            else:
                for field in self.schema_fields:
                    value = extracted_data.get(field)
                    result_df.loc[idx, field] = value

            # Collect artifacts
            all_artifacts.extend(row_artifacts)

            # Track validation errors
            if all(v is None for v in extracted_data.values()):
                validation_errors.append({
                    "row_index": idx,
                    "input_text": str(text)[:200] + "..." if len(str(text)) > 200 else str(text),
                    "error": "All fields returned None"
                })

        # Calculate extraction stats
        for field in (['extracted'] if self.open_schema else self.schema_fields):
            if field in result_df.columns:
                non_null_count = result_df[field].notna().sum()
                success_rate = non_null_count / \
                    len(result_df) if len(result_df) > 0 else 0.0
                extraction_stats[field] = {
                    'total_rows': len(result_df),
                    'successful_extractions': int(non_null_count),
                    'success_rate': success_rate
                }

                logger.info(f"Field '{field}': {non_null_count}/{len(result_df)} successful extractions "
                            f"({success_rate:.2%})")

        # Write debug artifacts
        if self.debug:
            # Write all prompt/response artifacts
            for artifact_path, artifact_content in all_artifacts:
                self._write_artifact(artifact_path, artifact_content)

            # Write validation errors
            if validation_errors:
                self._write_artifact(
                    "validation_errors.json", validation_errors)

            # Write extraction stats
            self._write_artifact("llm_stats.json", extraction_stats)

            # Write sample data
            sample_df = result_df.head(100)
            self._write_artifact("samples.csv", sample_df)

            # Write configuration
            config = {
                "model": getattr(self.chat_model, 'model_name', 'unknown'),
                "schema_fields": self.schema_fields,
                "is_pydantic": self.is_pydantic,
                "open_schema": self.open_schema,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "retries": self.retries,
                "system_prompt": self.system_prompt,
                "few_shots_count": len(self.few_shots)
            }
            self._write_artifact("llm_config.json", config)

        # Log overall stats
        extracted_columns = (
            ['extracted'] if self.open_schema else [
                f for f in self.schema_fields if f in result_df.columns]
        )
        self._log_extraction_stats(df, result_df, extracted_columns)

        # Always flush LLM logs to artifacts (writes llm_calls.json and llm_usage_summary.json)
        self._llm_logger.flush(self._write_artifact_always)

        logger.info(f"LLM extraction completed. Added {len(extracted_columns)} columns. "
                    f"Validation errors: {len(validation_errors)}")

        return result_df
