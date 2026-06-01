"""LLM-based canonicalizer for the Normalization committee (prompt v2).

Sixth-and-only LLM member of the Normalization roster. Prompt v2 (C12,
2026-05-25) lifts the v1 closed-vocab verbatim constraint and asks the
LLM to choose its operation per call from a fixed set
(``vocab_canonicalize``, ``date_normalize``, ``numeric_normalize``,
``categorical_map``, ``synthesize``, ``abstain``). Synthesis is allowed:
partial-date → ISO completion, unit conversion, abbreviation expansion,
and similar.

Determinism + cost discipline:

- ``temperature=0.0`` and a pinned ``model_id`` (default
  ``gpt-5.4-mini``) per the §"LLM model defaults + per-run override"
  policy.
- Per-call result cached at
  ``usecases_synthetic/cache/llm_normalizer/`` via
  :class:`usecases_synthetic.lib.llm_cache.LLMCache`. Cache key embeds
  ``(prompt_version, model_id)``; bumping ``PROMPT_VERSION_V2``
  invalidates the existing cache by construction.
- Operation log: when ``op_log_path`` is wired by the runner, every
  call appends a row recording the operation tag + canonical form. The
  runner sets the path per-(member, domain, level) so the diagnostics
  CSV in ``output/norm_diagnostics/<domain>/<level>/llm_only_operations.csv``
  surfaces per-attribute operation distributions.
- A handful of canonical examples are embedded in the user prompt
  per attribute; examples are sampled deterministically so the cache
  key remains stable across runs.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .llm_cache import LLMCache

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = REPO_ROOT / "usecases_synthetic" / "cache" / "llm_normalizer"


PROMPT_VERSION_V2 = "v2"


VALID_NORM_OPERATIONS_V2: frozenset[str] = frozenset(
    {
        "vocab_canonicalize",
        "date_normalize",
        "numeric_normalize",
        "categorical_map",
        "synthesize",
        "abstain",
    }
)


_SYSTEM_PROMPT_V2 = """You are a careful data normalizer. Given a value from a noisy data source and a small set of canonical examples, return its canonical form.

Pick the operation that best describes what you did:
- vocab_canonicalize: map the value to one of the listed canonical examples (closed vocabulary).
- date_normalize: produce an ISO 8601 date / datetime (e.g. "Aug 5, 2007" -> "2007-08-05").
- numeric_normalize: produce a canonical numeric form (unit-converted / scale-normalized if needed).
- categorical_map: map to a canonical category label (when the canonical form is a string label, not a closed-vocab pick).
- synthesize: produce a canonical form not literally listed but confidently derivable (e.g. unit conversion, abbreviation expansion, partial-date completion).
- abstain: when the value is unparseable, ambiguous, or you cannot confidently normalize.

Synthesis is permitted. The canonical form need not be one of the listed examples.

Respond with strict JSON of the form:
{"value": <canonical form (string or number); null when abstaining>, "operation": "<one of: vocab_canonicalize, date_normalize, numeric_normalize, categorical_map, synthesize, abstain>", "confidence": <float in 0..1>, "reasoning": "<short text, <= 1 sentence>"}
No prose. No markdown. No code fences."""


_USER_PROMPT_TEMPLATE_V2 = """Attribute: {attribute}
Kind: {kind}
Canonical examples:
{examples}

Value: {value}

Respond with strict JSON."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_examples(examples: list[str], max_items: int) -> str:
    """Render a deterministic, capped example list for the prompt."""
    if not examples:
        return "(none provided)"
    seen: set[str] = set()
    rendered: list[str] = []
    for ex in examples:
        s = str(ex).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        rendered.append(f"- {s}")
        if len(rendered) >= max_items:
            break
    return "\n".join(rendered) if rendered else "(none provided)"


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    return s


# ---------------------------------------------------------------------------
# LLMCanonicalizer
# ---------------------------------------------------------------------------


@dataclass
class _ExampleSpec:
    """Per-(domain, attribute) canonical-example reservoir."""

    examples: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Operation log (CSV append; thread-safe)
# ---------------------------------------------------------------------------


_OP_LOG_HEADER: list[str] = [
    "timestamp",
    "domain",
    "attribute",
    "kind",
    "source_value",
    "canonical_value",
    "operation",
    "confidence",
    "reasoning",
    "cache_hit",
    "model_id",
]

_op_log_lock = threading.Lock()


def _append_op_log(
    path: Path,
    *,
    domain: str,
    attribute: str,
    kind: str,
    source_value: str,
    canonical_value: Any,
    operation: str,
    confidence: float,
    reasoning: str,
    cache_hit: bool,
    model_id: str,
) -> None:
    """Append one row to the per-(member, level) llm_only operation log."""
    with _op_log_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_OP_LOG_HEADER)
            writer.writerow(
                [
                    _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    domain,
                    attribute,
                    kind,
                    source_value,
                    "" if canonical_value is None else str(canonical_value),
                    operation,
                    f"{confidence:.4f}",
                    reasoning,
                    int(bool(cache_hit)),
                    model_id,
                ]
            )


class LLMCanonicalizer:
    """LLM normalizer member: maps source value → canonical form (prompt v2).

    Constructor parameters mirror the YAML ``params`` block:
    ``model_name``, ``num_examples``, ``temperature``, ``max_tokens``,
    ``cache_dir``, ``prompt_version``, ``op_log_path``. Reference examples
    are wired in via :meth:`set_examples` by the runner before the first
    call.
    """

    name: str

    def __init__(
        self,
        name: str = "llm_canonicalize",
        *,
        model_name: str = "gpt-5.4-mini",
        num_examples: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        cache_dir: Path | str | None = None,
        prompt_version: str = PROMPT_VERSION_V2,
        op_log_path: Path | str | None = None,
    ) -> None:
        self.name = name
        self.model_name = model_name
        self.num_examples = int(num_examples)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.prompt_version = prompt_version
        self.op_log_path = Path(op_log_path) if op_log_path is not None else None

        cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        # Embed num_examples in the cache prompt_version so different
        # in-context-example counts get distinct cache slots. Without
        # this, the first sweep cell to populate the cache poisons
        # every subsequent cell with its outputs (caught during the
        # 2026-05-10 R5 Norm tuning sweep).
        cache_prompt_version = f"{prompt_version}_n{int(num_examples)}"
        self._cache = LLMCache(
            cache_dir=cache_path,
            prompt_version=cache_prompt_version,
            model_id=model_name,
        )
        self._examples: dict[tuple[str, str], _ExampleSpec] = {}
        self._llm_callable: Callable[[str, str], str] | None = None

    # ---- example wiring ---------------------------------------------------

    def set_examples(
        self,
        per_domain: dict[str, dict[str, list[str]]],
    ) -> None:
        """Wire fusion-reference canonical examples for each (domain, attribute).

        The runner builds *per_domain* from ``protection.load_fusion_target_values``
        before calling :meth:`normalize`. Examples are deduplicated and
        truncated to :pyattr:`num_examples` deterministically (sorted
        alphabetically) so the cache key is stable.

        Parameters
        ----------
        per_domain : dict
            ``{domain: {attribute: [example, ...]}}``.
        """
        new_examples: dict[tuple[str, str], _ExampleSpec] = {}
        for domain, attrs in per_domain.items():
            for attribute, values in attrs.items():
                deduped = sorted({str(v).strip() for v in values if str(v).strip()})
                new_examples[(domain, attribute)] = _ExampleSpec(
                    examples=deduped[: self.num_examples]
                )
        self._examples = new_examples

    # ---- LLM client wiring ------------------------------------------------

    def _ensure_llm_callable(self) -> Callable[[str, str], str]:
        """Construct (lazily) the LangChain chat callable.

        Returns a function ``(system_prompt, user_prompt) -> response_text``.
        """
        if self._llm_callable is not None:
            return self._llm_callable

        from langchain_core.messages import HumanMessage, SystemMessage

        from .llm_client import build_chat_openai

        chat = build_chat_openai(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        def _call(system_prompt: str, user_prompt: str) -> str:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = chat.invoke(messages)
            content = getattr(response, "content", response)
            return str(content)

        self._llm_callable = _call
        return _call

    # ---- public surface ---------------------------------------------------

    def normalize(
        self,
        value: Any,
        *,
        attribute: str,
        kind: str,
        domain: str,
    ) -> str | None:
        """Return the canonical form for *value*, or ``None`` to abstain."""
        s = _stringify(value)
        if s is None:
            return None

        spec = self._examples.get((domain, attribute))
        examples = spec.examples if spec else []
        examples_text = _format_examples(examples, self.num_examples)

        # Detect cache hit by checking exists() pre-call; LLMCache's
        # call_or_cache short-circuits returning the stored payload when
        # the cell hash is present, so this flag reflects whether the
        # api_fn was invoked.
        cell_hash = self._cache.make_cell_hash(
            source=domain, attribute=attribute, value=s
        )
        was_cached = self._cache.exists(cell_hash)

        payload = self._cache.call_or_cache(
            source=domain,
            attribute=attribute,
            value=s,
            api_fn=lambda: self._call_llm(
                value=s,
                attribute=attribute,
                kind=kind,
                examples_text=examples_text,
            ),
            strict=False,
        )
        result = payload.get("result")
        if not isinstance(result, dict):
            canonical: str | None = None
            operation = "abstain"
            confidence = 0.0
            reasoning = ""
        else:
            canonical_raw = result.get("canonical")
            if canonical_raw is None:
                canonical = None
            else:
                canonical_str = str(canonical_raw).strip()
                canonical = (
                    None
                    if not canonical_str or canonical_str.upper() == "NULL"
                    else canonical_str
                )
            operation = str(result.get("operation", "abstain")) or "abstain"
            try:
                confidence = float(result.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            reasoning = str(result.get("reasoning", "") or "")

        if self.op_log_path is not None:
            try:
                _append_op_log(
                    self.op_log_path,
                    domain=domain,
                    attribute=attribute,
                    kind=kind,
                    source_value=s,
                    canonical_value=canonical,
                    operation=operation,
                    confidence=confidence,
                    reasoning=reasoning,
                    cache_hit=was_cached,
                    model_id=self.model_name,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("LLMCanonicalizer op-log append failed: %s", exc)

        return canonical

    # ---- internals --------------------------------------------------------

    def _call_llm(
        self,
        *,
        value: str,
        attribute: str,
        kind: str,
        examples_text: str,
    ) -> dict[str, Any]:
        client = self._ensure_llm_callable()
        user_prompt = _USER_PROMPT_TEMPLATE_V2.format(
            attribute=attribute,
            kind=kind,
            examples=examples_text,
            value=value,
        )
        try:
            response = client(_SYSTEM_PROMPT_V2, user_prompt)
        except Exception as exc:  # pragma: no cover - network/transport error
            logger.warning(
                "LLMCanonicalizer call failed for %s/%s: %s",
                attribute,
                value[:40],
                exc,
            )
            return {
                "canonical": None,
                "operation": "abstain",
                "confidence": 0.0,
                "reasoning": "",
                "raw": "",
                "error": str(exc),
            }

        parsed = _parse_response(response)
        if parsed is None:
            return {
                "canonical": None,
                "operation": "abstain",
                "confidence": 0.0,
                "reasoning": "",
                "raw": response,
                "parse_error": True,
            }
        return {**parsed, "raw": response}


# ---------------------------------------------------------------------------
# Response parsing (v2)
# ---------------------------------------------------------------------------


def _parse_response(text: str) -> dict[str, Any] | None:
    """Parse a raw LLM response into v2 JSON shape.

    Returns
    -------
    dict with keys ``{"canonical", "operation", "confidence", "reasoning"}``
    on success; ``None`` if the response is unparseable or carries an
    unknown ``operation``. The ``canonical`` field is ``None`` when the
    model abstains or returns null.
    """
    if not text:
        return None
    body = text.strip()
    if body.startswith("```"):
        lines = [l for l in body.splitlines() if not l.startswith("```")]
        body = "\n".join(lines).strip()
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    operation = parsed.get("operation")
    if not isinstance(operation, str) or operation not in VALID_NORM_OPERATIONS_V2:
        return None
    if "value" not in parsed:
        return None
    raw_value = parsed["value"]
    canonical: str | None
    if operation == "abstain" or raw_value is None:
        canonical = None
    else:
        canonical_str = str(raw_value).strip()
        canonical = (
            None
            if not canonical_str or canonical_str.upper() == "NULL"
            else canonical_str
        )
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    reasoning_raw = parsed.get("reasoning")
    reasoning = "" if reasoning_raw is None else str(reasoning_raw).strip()
    return {
        "canonical": canonical,
        "operation": operation,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": reasoning,
    }


__all__ = ["LLMCanonicalizer", "PROMPT_VERSION_V2", "VALID_NORM_OPERATIONS_V2"]
