"""Surface-rewrite operators for Knob 01 — Surface Augmentation Intensity.

Implements the deterministic operator pool used at easy / medium plus the
cached LLM paraphrase operator used at hard. Each operator is a pure
function of ``(value, rng?, **params)`` and returns
``(new_value, params_dict)`` for provenance, or ``None`` when the
operator cannot apply (value too short, collision with stopword /
key-token skiplist, round-trip check failed, ...).

Operators
---------
normalize_to_canonical
    Replace a cell with a canonical sibling-source form (easy-only
    "normalize-down" path for baseline-above-target attributes).
abbreviate
    Bidirectional abbreviation / expansion from an authored table.
eda_random_swap
    Swap two non-stopword non-key tokens in-place.
eda_random_delete
    Delete one non-stopword non-key token.
llm_paraphrase
    Cache-first LLM paraphrase with contamination guardrails. Used at
    hard level only.

Single-cell export
------------------
``paraphrase_value_for_knob_04`` — the C3 contract consumer for Knob 04's
easy fabrication fallback. Pinned to the medium deterministic pool (no
LLM), deterministic given the RNG.

See ``knobs/knob_01_surface_augmentation.md`` §"Algorithm selection" and
§"Implementation handoff" for the full specification.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Literal

import numpy as np

from .llm_cache import LLMCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def _tokenize(value: str) -> list[str]:
    """Whitespace tokenization preserving original token order.

    We deliberately avoid NLTK / spaCy — operators must run with the
    stdlib only. For the entity-matching surface-rewrite use case,
    whitespace tokenization is sufficient and deterministic.
    """
    return value.split()


def _detokenize(tokens: list[str]) -> str:
    """Join tokens back into a whitespace-separated string."""
    return " ".join(tokens)


def _is_skippable(
    token: str,
    stopwords: set[str],
    key_tokens: set[str],
) -> bool:
    """True if *token* is in the stopword or key-token skiplist."""
    return token.lower() in stopwords or token in key_tokens


# ---------------------------------------------------------------------------
# Easy: canonical form selection
# ---------------------------------------------------------------------------


def normalize_to_canonical(
    current_value: str,
    sibling_values: Iterable[str],
    *,
    strategy: Literal["shortest", "most_frequent"] = "shortest",
) -> tuple[str, dict[str, Any]] | None:
    """Return a canonical form drawn from sibling-source values.

    Used by the easy-level "normalize-down" path: when a cell's
    baseline sits above the easy target (e.g. Forbes' long-form
    country), replace it with the shortest (or most-frequent) sibling
    value for the same entity.

    Parameters
    ----------
    current_value : str
        The cell's current value.
    sibling_values : iterable of str
        Sibling source values for the same entity / target attribute.
        Null / empty values are ignored.
    strategy : {"shortest", "most_frequent"}, default "shortest"
        Canonical-selection strategy. ``"shortest"`` picks the shortest
        non-empty sibling value (ties broken alphabetically for
        determinism). ``"most_frequent"`` picks the most-common value.

    Returns
    -------
    (new_value, params) or None
        ``None`` when the canonical form equals ``current_value`` or no
        sibling value is available.
    """
    candidates = [
        s
        for s in sibling_values
        if isinstance(s, str) and s and s.strip().lower() not in ("null", "nan", "none")
    ]
    if not candidates:
        return None

    if strategy == "shortest":
        canonical = sorted(candidates, key=lambda s: (len(s), s))[0]
    elif strategy == "most_frequent":
        from collections import Counter

        counts = Counter(candidates)
        # Deterministic tie-break: sort by (-count, length, lexicographic).
        canonical = sorted(counts.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[
            0
        ][0]
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    if canonical == current_value:
        return None

    return canonical, {
        "template_source": "sibling",
        "canonical_form_origin": "sibling_source",
        "strategy": strategy,
    }


# ---------------------------------------------------------------------------
# Medium: table-driven operators
# ---------------------------------------------------------------------------


def abbreviate(
    value: str,
    abbrev_table: dict[str, str],
    rng: np.random.Generator | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Apply one bidirectional abbreviation substitution.

    The table is bidirectional: ``{"Incorporated": "Inc."}`` matches
    both ``"Acme Incorporated"`` → ``"Acme Inc."`` (expand→contract)
    and ``"Acme Inc."`` → ``"Acme Incorporated"`` (contract→expand).
    Ordering between direction candidates is deterministic (sorted
    alphabetically); if *rng* is provided, we randomise the choice.

    Parameters
    ----------
    value : str
        Input string.
    abbrev_table : dict[str, str]
        Mapping from long form to short form. Order in the dict is not
        load-bearing.
    rng : Generator or None
        Optional RNG for operator-pick randomisation. When None we pick
        the first alphabetically-sorted match.

    Returns
    -------
    (new_value, params) or None
        ``None`` if no long or short form matches *value*.
    """
    candidates: list[tuple[str, str, str]] = []  # (direction, key, replacement)

    # Sorted iteration for determinism.
    for long_form, short_form in sorted(abbrev_table.items()):
        if long_form in value:
            candidates.append(("expand_to_contract", long_form, short_form))
        if short_form and short_form in value and short_form != long_form:
            # Exact whole-token containment check to avoid flipping "Inc."
            # inside "Inc.orporated" style words.
            if _contains_token(value, short_form):
                candidates.append(("contract_to_expand", short_form, long_form))

    if not candidates:
        return None

    if rng is not None:
        idx = int(rng.integers(0, len(candidates)))
    else:
        idx = 0

    direction, key, replacement = candidates[idx]
    new_value = value.replace(key, replacement, 1)
    if new_value == value:
        return None

    return new_value, {
        "table": "abbreviation",
        "table_key": key,
        "direction": direction,
    }


def _contains_token(value: str, token: str) -> bool:
    """Check if *token* appears as a whole space-delimited token in *value*."""
    if token not in value:
        return False
    # Whole-word check with whitespace / boundary sentinels.
    padded = f" {value} "
    return (
        f" {token} " in padded
        or padded.startswith(f"{token} ")
        or padded.endswith(f" {token}")
    )


# ---------------------------------------------------------------------------
# Medium: EDA-style token-level operators
# ---------------------------------------------------------------------------


def eda_random_swap(
    value: str,
    rng: np.random.Generator,
    stopwords: set[str] | None = None,
    key_tokens: set[str] | None = None,
    n_swaps: int = 1,
) -> tuple[str, dict[str, Any]] | None:
    """Swap two non-stopword non-key tokens in *value*.

    Stopwords and key tokens are untouchable — the swap picks two
    positions from the remaining pool. If the pool has fewer than two
    positions, returns ``None``.

    Parameters
    ----------
    value : str
        Input string.
    rng : Generator
        Seeded RNG.
    stopwords : set of str or None
        Lowercase stopword set (e.g. {"the", "of", "and"}).
    key_tokens : set of str or None
        Case-sensitive key-token skiplist (e.g. artist names for music).
    n_swaps : int, default 1
        Number of independent swap operations to perform. The same
        position may be chosen twice across swaps.

    Returns
    -------
    (new_value, params) or None
    """
    stopwords = stopwords or set()
    key_tokens = key_tokens or set()

    tokens = _tokenize(value)
    positions = [
        i for i, t in enumerate(tokens) if not _is_skippable(t, stopwords, key_tokens)
    ]
    if len(positions) < 2:
        return None

    applied_pairs: list[tuple[int, int]] = []
    tokens_before_snapshot = list(tokens)

    for _ in range(n_swaps):
        chosen = rng.choice(positions, size=2, replace=False)
        i, j = int(chosen[0]), int(chosen[1])
        tokens[i], tokens[j] = tokens[j], tokens[i]
        applied_pairs.append((i, j))

    new_value = _detokenize(tokens)
    if new_value == value:
        return None

    first_i, first_j = applied_pairs[0]
    return new_value, {
        "positions": [first_i, first_j],
        "tokens_before": [
            tokens_before_snapshot[first_i],
            tokens_before_snapshot[first_j],
        ],
        "tokens_after": [tokens[first_i], tokens[first_j]],
        "n_swaps": n_swaps,
    }


def eda_random_delete(
    value: str,
    rng: np.random.Generator,
    stopwords: set[str] | None = None,
    key_tokens: set[str] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Delete one non-stopword non-key token from *value*.

    Parameters
    ----------
    value : str
        Input string.
    rng : Generator
        Seeded RNG.
    stopwords : set of str or None
        Lowercase stopword set.
    key_tokens : set of str or None
        Case-sensitive key-token skiplist.

    Returns
    -------
    (new_value, params) or None
    """
    stopwords = stopwords or set()
    key_tokens = key_tokens or set()

    tokens = _tokenize(value)
    positions = [
        i for i, t in enumerate(tokens) if not _is_skippable(t, stopwords, key_tokens)
    ]
    if not positions:
        return None
    # Preserve at least one token.
    if len(tokens) <= 1:
        return None

    pos = int(rng.choice(positions))
    token_removed = tokens[pos]
    del tokens[pos]

    if not tokens:
        return None

    new_value = _detokenize(tokens)
    if new_value == value:
        return None

    return new_value, {
        "position": pos,
        "token_removed": token_removed,
    }


# ---------------------------------------------------------------------------
# Hard: LLM paraphrase with cache and contamination guardrails
# ---------------------------------------------------------------------------


def ngram_overlap_tokens(a: str, b: str, n: int) -> bool:
    """Return True if *a* and *b* share an n-or-more-token contiguous overlap.

    Used by the contamination spot-check (8-gram overlap). Token
    comparison is case-insensitive.
    """
    a_toks = [t.lower() for t in a.split()]
    b_toks = [t.lower() for t in b.split()]
    if len(a_toks) < n or len(b_toks) < n:
        return False

    b_ngrams: set[tuple[str, ...]] = set()
    for i in range(len(b_toks) - n + 1):
        b_ngrams.add(tuple(b_toks[i : i + n]))
    for i in range(len(a_toks) - n + 1):
        if tuple(a_toks[i : i + n]) in b_ngrams:
            return True
    return False


def contamination_ngram_passed(
    paraphrase: str,
    original: str,
    n: int = 8,
) -> bool:
    """Contamination check: the paraphrase must NOT contain an n-gram overlap.

    Returns True when the check passes (safe to admit).
    """
    return not ngram_overlap_tokens(paraphrase, original, n=n)


def build_first_token_index(
    records: Iterable[tuple[str, str]],
    n_tokens: int = 3,
) -> dict[tuple[str, ...], str]:
    """Build a normalised first-n-token lookup index.

    Parameters
    ----------
    records : iterable of (entity_key, value)
        Entity identifier and canonical primary value. Collisions are
        broken by first-seen (deterministic given a sorted iterable).
    n_tokens : int, default 3
        Number of leading tokens to index.

    Returns
    -------
    dict
        Mapping from the lowercase n-token prefix tuple to an entity key.
    """
    index: dict[tuple[str, ...], str] = {}
    for key, value in records:
        if not value:
            continue
        toks = value.split()
        if len(toks) < n_tokens:
            continue
        probe = tuple(t.lower() for t in toks[:n_tokens])
        index.setdefault(probe, key)
    return index


def contamination_first_token_probe_passed(
    paraphrase: str,
    self_key: str,
    first_token_index: dict[tuple[str, ...], str],
    n_tokens: int = 3,
) -> bool:
    """Check that a paraphrase does not silently alias another entity.

    Passes iff the paraphrase's first *n_tokens* do not resolve to a
    different entity key in *first_token_index*.
    """
    toks = paraphrase.split()
    if len(toks) < n_tokens:
        return True
    probe = tuple(t.lower() for t in toks[:n_tokens])
    other_key = first_token_index.get(probe)
    if other_key is None:
        return True
    return other_key == self_key


UNCHANGED_SENTINEL = "<UNCHANGED>"


def _is_near_identity(paraphrase: str, value: str) -> bool:
    """Return True when *paraphrase* shares the same token set as *value*.

    "Near-identity" means the LLM output has the same whitespace-split
    lowercased token set as the input -- only casing, punctuation, or
    whitespace differs. Used by the R10-D post-filter to reject lazy
    LLM responses that satisfy ``paraphrase != value`` (the legacy
    verbatim check) but carry no substantive token change.
    """
    a = {t for t in value.lower().split() if t}
    b = {t for t in paraphrase.lower().split() if t}
    if not a and not b:
        return True
    if not a or not b:
        return False
    return a == b


def llm_paraphrase(
    value: str,
    *,
    source: str,
    attribute: str,
    attribute_class: Literal["primary", "key", "secondary", "categorical"],
    cache: LLMCache,
    prompt_template: str,
    api_client: Callable[[str, str], str] | None = None,
    entity_key: str = "",
    first_token_index: dict[tuple[str, ...], str] | None = None,
    committee_fn: Callable[[str, str, str, str], bool] | None = None,
    strict_cache: bool = False,
    ngram_n: int = 8,
) -> tuple[str, dict[str, Any]] | None:
    """Apply a cached LLM paraphrase to a single cell value.

    Cache-first: if a cached payload exists for the cell, it is used
    unconditionally. Otherwise, *api_client* is invoked (unless
    *strict_cache* is True, in which case a cache miss raises
    ``LLMCacheMiss``).

    Every fresh paraphrase is gated by:
    1. **N-gram overlap contamination check** — the paraphrase must not
       contain an *ngram_n*-or-more-token contiguous overlap with the
       original value.
    2. **First-token memorization probe** — the paraphrase's first 3
       tokens must not resolve to a different entity key in
       *first_token_index*.
    3. **Committee validation** — *committee_fn* is called with
       ``(source, attribute, original_value, paraphrase)`` and must
       return True.

    Any failing check routes back to the caller (this function returns
    ``None``), which should fall back to a medium-level operator.

    R10-D additions
    ---------------
    * If the LLM returns the ``<UNCHANGED>`` sentinel, the function
      returns ``(value, {"transform_fn": "llm_paraphrase_unchanged",
      ...})``. The caller detects the sentinel transform_fn and skips
      the write; the row lands in the skipped audit under reason
      ``llm_unchanged_sentinel`` so unchanged rate is observable.
    * If the LLM output passes the legacy ``paraphrase != value`` check
      but its lowercased token set equals the input's (casing /
      punctuation / whitespace only -- "near identity"), the function
      returns ``(paraphrase, {"transform_fn": "llm_paraphrase_near_identity",
      ...})``. The caller skips the write and logs under reason
      ``llm_near_identity``.

    Parameters
    ----------
    value : str
        Original cell value.
    source : str
        Source dataset name (hashed into cache key).
    attribute : str
        Attribute name (hashed into cache key).
    attribute_class : {"primary", "key", "secondary", "categorical"}
        Attribute class — informs the prompt template.
    cache : LLMCache
        Shared LLM cache.
    prompt_template : str
        The raw prompt text template — unused here (stored in the cache
        payload for audit). The caller is responsible for formatting
        prompts before invoking the API function.
    api_client : callable or None
        Callable ``(prompt_template, value) -> paraphrase_str``. Only
        called on cache miss when ``strict_cache`` is False.
    entity_key : str, default ""
        The entity identifier used by the first-token probe (defaults to
        an empty string which means "always pass self-alias check").
    first_token_index : dict or None
        Optional lookup for the first-token probe.
    committee_fn : callable or None
        Optional committee validator. Defaults to accept-all.
    strict_cache : bool, default False
        When True, a cache miss raises ``LLMCacheMiss``.
    ngram_n : int, default 8
        Token-count threshold for the contamination overlap check.

    Returns
    -------
    (new_value, params) or None
        Returns ``None`` when the paraphrase fails any guardrail. The
        caller is expected to fall back to a medium-level operator.
        Under R10-D, also returns sentinel tuples for ``<UNCHANGED>``
        and near-identity cases; the caller detects these via the
        ``transform_fn`` key (see the R10-D additions block above).
    """

    def _api_call() -> dict[str, Any]:
        assert api_client is not None
        out = api_client(prompt_template, value)
        return {"paraphrase": out}

    # No api_client and no strict flag would hit the cache's "api_fn
    # required" RuntimeError on miss. Treat as strict so the caller's
    # LLMCacheMiss handler degrades gracefully (skip, log) instead.
    effective_strict = strict_cache or api_client is None
    payload = cache.call_or_cache(
        source=source,
        attribute=attribute,
        value=value,
        api_fn=_api_call if api_client is not None else None,
        strict=effective_strict,
    )

    result = payload.get("result") or {}
    paraphrase = str(result.get("paraphrase", "")).strip()
    if not paraphrase or paraphrase == value:
        return None

    cell_hash = cache.make_cell_hash(source, attribute, value)

    # R10-D: <UNCHANGED> sentinel — LLM judged the cell unparaphrasable.
    # Return a sentinel tuple so the caller can count rates per (domain,
    # level, attribute) distinctly from LLM laziness.
    if paraphrase == UNCHANGED_SENTINEL:
        return value, {
            "transform_fn": "llm_paraphrase_unchanged",
            "prompt_version": cache.prompt_version,
            "model_id": cache.model_id,
            "cache_path": f"{cell_hash}.json",
        }

    # R10-D: near-identity post-filter. Same lowercased token set as the
    # input means the LLM only changed casing / punctuation /
    # whitespace; reject as a shallow paraphrase. The caller skips the
    # write and logs the cell separately from contamination failures.
    if _is_near_identity(paraphrase, value):
        return paraphrase, {
            "transform_fn": "llm_paraphrase_near_identity",
            "prompt_version": cache.prompt_version,
            "model_id": cache.model_id,
            "cache_path": f"{cell_hash}.json",
        }

    ngram_ok = contamination_ngram_passed(paraphrase, value, n=ngram_n)
    first_token_ok = True
    if first_token_index is not None:
        first_token_ok = contamination_first_token_probe_passed(
            paraphrase, entity_key, first_token_index
        )

    committee_ok = True
    if committee_fn is not None:
        committee_ok = bool(committee_fn(source, attribute, value, paraphrase))

    if not (ngram_ok and first_token_ok and committee_ok):
        return None

    if attribute_class == "categorical":
        transform_fn = "llm_paraphrase_categorical"
    elif attribute_class == "secondary":
        transform_fn = "llm_paraphrase_secondary"
    else:
        transform_fn = "llm_paraphrase_short"

    return paraphrase, {
        "transform_fn": transform_fn,
        "prompt_version": cache.prompt_version,
        "model_id": cache.model_id,
        "cache_path": f"{cell_hash}.json",
        "committee_passed": committee_ok,
        "contamination_check": {
            "ngram_overlap_passed": ngram_ok,
            "first_token_probe_passed": first_token_ok,
        },
    }


# ---------------------------------------------------------------------------
# OpenAI-backed K1 paraphrase client
# ---------------------------------------------------------------------------


def build_openai_paraphrase_client(
    *,
    model_id: str,
    temperature: float = 0.0,
    max_tokens: int | None = 2048,
) -> Callable[[str, str], str]:
    """Build an OpenAI-backed paraphrase callable for K1 surface augmentation.

    Returns a callable matching the ``Callable[[str, str], str]`` contract
    that :func:`llm_paraphrase` invokes on cache miss
    (``(prompt_template, value) -> paraphrase``):

    1. Substitutes the ``{value}`` placeholder in the prompt template (see
       ``config/knob_01_surface/_prompts/prompt_{short,secondary,categorical}_*.txt``).
    2. Calls ``langchain_openai.ChatOpenAI`` via :func:`build_chat_openai`
       (which enforces the reasoning-token floor for gpt-5* models).
    3. Returns the response text stripped of whitespace and any surrounding
       quotes the model added despite the prompt's instructions. The v2
       ``<UNCHANGED>`` sentinel is returned verbatim so the caller's R10-D
       sentinel handling fires.

    On any failure (missing placeholder, network error, empty response)
    returns the empty string; :func:`llm_paraphrase` then treats it as a
    failed paraphrase (returns ``None``) and the caller falls back to a
    deterministic operator.

    Parameters
    ----------
    model_id : str
        OpenAI chat model id (e.g. ``"gpt-5.4-mini"``).
    temperature : float, default 0.0
        Sampling temperature -- pinned to 0 for cache determinism.
    max_tokens : int or None, default 2048
        Per-call token cap. Reasoning models (gpt-5*) need >= 1024;
        :func:`build_chat_openai` raises ``ValueError`` otherwise.

    Returns
    -------
    Callable[[str, str], str]
        A paraphrase function suitable for ``apply_knob_01(llm_client=...)``.

    Notes
    -----
    Invoked **only on cache miss**. Committed paraphrases are persisted to
    ``cache/knob_01_paraphrases/<domain>/<level>/`` and reused for free on
    subsequent runs; strict-cache runs short-circuit before reaching it.
    """
    from .llm_client import build_chat_openai

    chat = build_chat_openai(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    def _call(template: str, value: str) -> str:
        try:
            prompt = template.format(value=value)
        except KeyError as exc:
            logger.warning(
                "K1 OpenAI client: prompt template missing placeholder %s; "
                "cannot paraphrase",
                exc,
            )
            return ""

        try:
            response = chat.invoke(prompt)
        except Exception as exc:  # pragma: no cover - network/runtime errors
            logger.warning("K1 OpenAI client: invoke failed: %s", exc)
            return ""

        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "".join(
                str(p) if not isinstance(p, dict) else str(p.get("text", ""))
                for p in content
            )
        text = str(content).strip()

        # Strip a single pair of matching surrounding quotes the model may
        # add despite the "no quotes" instruction.
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
            text = text[1:-1].strip()

        return text

    return _call


# ---------------------------------------------------------------------------
# C3 export: single-cell paraphrase for Knob 04's easy fabrication fallback
# ---------------------------------------------------------------------------


def paraphrase_value_for_knob_04(
    domain: str,
    attribute_class: Literal["primary", "key", "secondary", "categorical"],
    original_value: str,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[str, dict[str, Any]]:
    """Apply one medium-level Knob 01 operator to a single value.

    Pinned to the medium deterministic pool (abbreviation /
    ``eda_random_swap`` / ``eda_random_delete``). Never invokes the LLM.
    Never reads or writes the joint provenance index — Knob 04 owns the
    provenance row for the fabricated cell and should nest the returned
    ``transform_params_dict`` under ``knob_01_paraphrase_params``.

    Deterministic given ``(rng, config)``.

    Parameters
    ----------
    domain : str
        Domain name (currently unused — reserved for domain-specific
        operator routing when categorical vocab maps are consulted).
    attribute_class : {"primary", "key", "secondary", "categorical"}
        Attribute class of the cell being paraphrased.
    original_value : str
        The cell's current value.
    config : dict
        Loaded Knob 01 per-domain config. Reads ``abbreviation_table``,
        ``stopword_list``, ``key_token_skiplist``.
    rng : Generator
        Seeded RNG from Knob 04.

    Returns
    -------
    (new_value, params)
        Always returns a tuple. ``transform_fn`` in the params dict
        indicates which operator fired (or ``"passthrough"`` if none
        applied cleanly).
    """
    del domain  # Reserved.

    abbrev_table: dict[str, str] = config.get("abbreviation_table", {}) or {}
    stopwords = set(t.lower() for t in config.get("stopword_list", []) or [])
    # Key tokens are per-column; K4's consumer does not know the column,
    # so we fall back to the global key-token set if one is declared.
    key_tokens = set(config.get("key_token_skiplist_global", []) or [])

    # Deterministic order: permutation of {abbreviate, swap, delete}.
    ops_order = list(rng.permutation(3))

    for op_idx in ops_order:
        op_idx_int = int(op_idx)
        result: tuple[str, dict[str, Any]] | None = None
        op_name = ""

        if op_idx_int == 0:
            op_name = "abbreviate"
            if abbrev_table:
                result = abbreviate(original_value, abbrev_table, rng=rng)
        elif op_idx_int == 1:
            op_name = "eda_random_swap"
            result = eda_random_swap(
                original_value,
                rng,
                stopwords=stopwords,
                key_tokens=key_tokens,
                n_swaps=1,
            )
        else:
            op_name = "eda_random_delete"
            result = eda_random_delete(
                original_value,
                rng,
                stopwords=stopwords,
                key_tokens=key_tokens,
            )

        if result is not None:
            new_value, params = result
            if new_value != original_value:
                return new_value, {"transform_fn": op_name, **params}

    # Nothing applicable — passthrough.
    return original_value, {"transform_fn": "passthrough"}


# ---------------------------------------------------------------------------
# Operator registry (for dispatcher iteration)
# ---------------------------------------------------------------------------


VALID_TRANSFORM_FNS: frozenset[str] = frozenset(
    {
        "normalize_to_canonical",
        "abbreviate",
        "eda_random_swap",
        "eda_random_delete",
        "llm_paraphrase_short",
        "llm_paraphrase_categorical",
        "llm_paraphrase_secondary",
        "llm_paraphrase_unchanged",
        "llm_paraphrase_near_identity",
        "paraphrase_short",
        "paraphrase_categorical",
        "passthrough",
        "gold_extend_for_committee",
        "soften_for_committee",
    }
)
