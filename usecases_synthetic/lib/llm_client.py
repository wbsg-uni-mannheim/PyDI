"""Shared ChatOpenAI builder with reasoning-model guard.

Reasoning models (gpt-5*, gpt-5.4*, o1*, o3*) charge their internal
reasoning tokens against the same budget as visible output. When the
caller sets a small ``max_tokens`` (e.g. 8 — the conventional cap for a
yes/no LLM matcher) the model exhausts that budget on reasoning before
emitting a visible token; the API surfaces this as

    openai.BadRequestError: Error code: 400 — "Could not finish the
    message because max_tokens or model output limit was reached."

The committee runners swallow per-matcher exceptions (see
``committee_em._run_pair``), so the matcher silently registers F1=0
on the affected pair and the failure does not surface in the per-domain
sign-off tables. The result is invisible regression.

This module centralises every ``ChatOpenAI`` construction in the
synthetic pipeline and enforces a floor on ``max_tokens`` whenever the
model is detected as a reasoning model. Any future caller that requests
a budget too small for reasoning-token overhead raises ``ValueError``
at construction time rather than silently truncating.

Discovery: 2026-05-13 music-small sanity check (S.3 of plan_s1_scale.md).
Affected sites at discovery time: matchgpt_em_matcher, comem_em_matcher,
llm_normalizer, committee_fusion.
"""

from __future__ import annotations

import re
from typing import Any

# Models that consume internal reasoning tokens against ``max_tokens``.
# The patterns cover the gpt-5 / gpt-5.x families and OpenAI's o1 / o3
# reasoning series. Match is anchored to the start of the model id (after
# any "openai/" prefix) and case-insensitive.
_REASONING_MODEL_PATTERN = re.compile(r"^(gpt-5|o1|o3)", re.IGNORECASE)

# Minimum max_tokens for reasoning models. Empirically 100-3000 reasoning
# tokens are typical per call (EM yes/no, normalization, fusion judge);
# 1024 covers the lower end with headroom but can still truncate hard
# cases. Pipeline default (committee YAMLs + module defaults) is 2048.
_REASONING_MIN_MAX_TOKENS = 1024


def is_reasoning_model(model_id: str) -> bool:
    """Return True if ``model_id`` uses internal reasoning tokens.

    Strips an optional ``"openai/"`` prefix before matching.
    """
    stripped = model_id.split("/", 1)[-1]
    return bool(_REASONING_MODEL_PATTERN.match(stripped))


def build_chat_openai(
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    **extra_model_kwargs: Any,
) -> Any:
    """Construct a ``langchain_openai.ChatOpenAI`` with safety guards.

    Parameters
    ----------
    model : str
        The OpenAI model id. May carry an ``"openai/"`` prefix (stripped
        before being passed to ``ChatOpenAI``).
    temperature : float
        Sampling temperature (default 0.0 for cache determinism).
    max_tokens : int or None
        Hard cap on tokens (reasoning + visible). When the model is a
        reasoning model and ``max_tokens`` is below
        :data:`_REASONING_MIN_MAX_TOKENS`, raises ``ValueError`` rather
        than silently truncating. Pass ``None`` to omit the cap (OpenAI
        applies its model-level default).
    **extra_model_kwargs
        Additional kwargs forwarded into the ``ChatOpenAI``
        ``model_kwargs`` block.

    Raises
    ------
    ValueError
        If ``max_tokens`` is set below the reasoning floor for a
        reasoning model.

    Returns
    -------
    langchain_openai.ChatOpenAI
        Instantiated chat client.
    """
    from langchain_openai import ChatOpenAI

    model_id = model.split("/", 1)[-1]
    if (
        max_tokens is not None
        and is_reasoning_model(model_id)
        and max_tokens < _REASONING_MIN_MAX_TOKENS
    ):
        raise ValueError(
            f"max_tokens={max_tokens} is too small for reasoning model "
            f"{model_id!r}. Reasoning models consume internal reasoning "
            f"tokens against the max_tokens budget; budgets below "
            f"{_REASONING_MIN_MAX_TOKENS} routinely truncate to empty "
            f"visible output. Bump to >= 2048 (pipeline default), or "
            f"switch the YAML to a non-reasoning model (e.g. "
            f"gpt-4o-mini) if the prior 8 / 64 budgets were load-bearing."
        )

    model_kwargs: dict[str, Any] = dict(extra_model_kwargs)
    if max_tokens is not None:
        model_kwargs["max_tokens"] = int(max_tokens)

    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        model_kwargs=model_kwargs,
    )


__all__ = ["build_chat_openai", "is_reasoning_model"]
