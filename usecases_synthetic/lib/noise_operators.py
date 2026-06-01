"""Cell-level corruption operators for Knob 6 — Value Noise Injection.

Implements the FEBRL / Christen-Vatsalan operator taxonomy:

- ``typo_substitute`` — keyboard-adjacency weighted character substitution
- ``ocr_confuse`` — OCR confusion table (single-char + char-pair)
- ``truncate`` — right-truncation
- ``whitespace_corrupt`` — extra/missing spaces, punctuation collapse
- ``case_corrupt`` — random case changes
- ``taxonomy_walk`` — single-level walk up/down a configured taxonomy

Each operator is a pure function of ``(value, rng, **params)`` and returns
``(new_value, params_dict)`` for provenance, or ``None`` if the operator
cannot apply (e.g. value too short).

Static lookup tables (QWERTY adjacency, OCR confusions) are loaded once
from ``usecases_synthetic/config/knob_06_noise/_tables/``.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Table loading
# ---------------------------------------------------------------------------

_TABLES_DIR = (
    Path(__file__).resolve().parents[1] / "config" / "knob_06_noise" / "_tables"
)

_qwerty_cache: dict[str, list[str]] | None = None
_ocr_single_cache: dict[str, str] | None = None
_ocr_pair_cache: dict[str, str] | None = None


def _load_qwerty() -> dict[str, list[str]]:
    """Load the QWERTY adjacency map (cached)."""
    global _qwerty_cache
    if _qwerty_cache is None:
        path = _TABLES_DIR / "qwerty.yaml"
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _qwerty_cache = {str(k): v for k, v in raw["adjacency"].items()}
    return _qwerty_cache


def _load_ocr() -> tuple[dict[str, str], dict[str, str]]:
    """Load OCR confusion tables (cached).

    Returns ``(single_map, pair_map)``.
    """
    global _ocr_single_cache, _ocr_pair_cache
    if _ocr_single_cache is None or _ocr_pair_cache is None:
        path = _TABLES_DIR / "ocr.yaml"
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _ocr_single_cache = {str(k): str(v) for k, v in raw["single"].items()}
        _ocr_pair_cache = {str(k): str(v) for k, v in raw["pair"].items()}
    return _ocr_single_cache, _ocr_pair_cache


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def typo_substitute(
    value: str,
    rng: np.random.Generator,
    n_edits: int = 1,
    use_adjacency: bool = True,
) -> tuple[str, dict[str, Any]] | None:
    """Keyboard-adjacency weighted character substitution.

    Parameters
    ----------
    value : str
        Input string.
    rng : Generator
        Seeded RNG.
    n_edits : int
        Number of character substitutions to apply.
    use_adjacency : bool
        If True, substitute with a QWERTY-adjacent key when available.

    Returns
    -------
    tuple[str, dict] or None
        ``(new_value, params_dict)`` or ``None`` if the value is too short.
    """
    alpha_positions = [i for i, c in enumerate(value) if c.isalnum()]
    if not alpha_positions:
        return None

    qwerty = _load_qwerty() if use_adjacency else {}

    chars = list(value)
    edits: list[dict[str, Any]] = []
    actual_edits = min(n_edits, len(alpha_positions))

    chosen = rng.choice(alpha_positions, size=actual_edits, replace=False)
    for pos in sorted(chosen):
        pos = int(pos)
        original_char = chars[pos]
        lower = original_char.lower()

        # Try adjacency first.
        neighbours = qwerty.get(lower, [])
        if neighbours:
            replacement = str(rng.choice(neighbours))
        else:
            # Fallback: random alphanumeric character.
            pool = "abcdefghijklmnopqrstuvwxyz0123456789"
            pool = pool.replace(lower, "")
            replacement = str(rng.choice(list(pool)))

        # Preserve case.
        if original_char.isupper():
            replacement = replacement.upper()

        chars[pos] = replacement
        edits.append(
            {"position": pos, "original": original_char, "replacement": replacement}
        )

    new_value = "".join(chars)
    if new_value == value:
        return None

    return new_value, {
        "edit_type": "substitute",
        "positions": [e["position"] for e in edits],
        "chars": [e["replacement"] for e in edits],
        "originals": [e["original"] for e in edits],
        "use_adjacency": use_adjacency,
    }


def ocr_confuse(
    value: str,
    rng: np.random.Generator,
    n_chars: int = 1,
) -> tuple[str, dict[str, Any]] | None:
    """OCR confusion table substitution.

    Scans left-to-right, preferring pair matches over single-char matches.

    Parameters
    ----------
    value : str
        Input string.
    rng : Generator
        Seeded RNG.
    n_chars : int
        Maximum number of confusion substitutions.

    Returns
    -------
    tuple[str, dict] or None
        ``(new_value, params_dict)`` or ``None`` if no confusable chars found.
    """
    single, pair = _load_ocr()

    # Find all confusable positions.
    candidates: list[tuple[int, str, str, str]] = []  # (pos, src, dst, kind)
    i = 0
    while i < len(value):
        # Prefer pair matches (longest match first).
        if i + 1 < len(value):
            bigram = value[i : i + 2]
            if bigram in pair:
                candidates.append((i, bigram, pair[bigram], "pair"))
                i += 2
                continue
        # Single char.
        ch = value[i]
        if ch in single:
            candidates.append((i, ch, single[ch], "single"))
        i += 1

    if not candidates:
        return None

    # Select up to n_chars candidates to apply.
    n_apply = min(n_chars, len(candidates))
    chosen_indices = sorted(rng.choice(len(candidates), size=n_apply, replace=False))

    # Apply in reverse order so positions stay valid.
    chars = list(value)
    applied: list[dict[str, Any]] = []
    for ci in reversed(chosen_indices):
        pos, src, dst, kind = candidates[ci]
        # Replace in-place.
        if kind == "pair":
            chars[pos : pos + len(src)] = list(dst)
        else:
            chars[pos] = dst
        applied.append(
            {"position": pos, "original": src, "replacement": dst, "kind": kind}
        )

    new_value = "".join(chars)
    if new_value == value:
        return None

    applied.reverse()  # Restore forward order for provenance.
    return new_value, {
        "positions": [a["position"] for a in applied],
        "chars": [a["replacement"] for a in applied],
        "originals": [a["original"] for a in applied],
        "ocr_kind": [a["kind"] for a in applied],
    }


def truncate(
    value: str,
    rng: np.random.Generator,
    max_truncate_chars: int = 3,
) -> tuple[str, dict[str, Any]] | None:
    """Right-truncate a string value.

    Parameters
    ----------
    value : str
        Input string.
    rng : Generator
        Seeded RNG.
    max_truncate_chars : int
        Maximum number of characters to remove from the right.

    Returns
    -------
    tuple[str, dict] or None
        ``(new_value, params_dict)`` or ``None`` if value too short.
    """
    if len(value) <= 2:
        return None

    # Truncation length: [1, min(max_truncate_chars, len-1)].
    max_cut = min(max_truncate_chars, len(value) - 1)
    cut_len = int(rng.integers(1, max_cut + 1))

    removed = value[-cut_len:]
    new_value = value[:-cut_len]

    return new_value, {
        "cut_length": cut_len,
        "removed_chars": removed,
    }


def whitespace_corrupt(
    value: str,
    rng: np.random.Generator,
) -> tuple[str, dict[str, Any]] | None:
    """Extra/missing spaces, punctuation collapse.

    Applies one of: insert space, delete space, collapse punctuation
    (delete comma/period/hyphen).

    Parameters
    ----------
    value : str
        Input string.
    rng : Generator
        Seeded RNG.

    Returns
    -------
    tuple[str, dict] or None
        ``(new_value, params_dict)`` or ``None`` if no applicable positions.
    """
    ops: list[str] = []

    # Check which sub-operations are applicable.
    has_spaces = " " in value
    has_punct = bool(re.search(r"[,.\-]", value))
    has_alpha = bool(re.search(r"[a-zA-Z0-9]", value))

    if has_alpha and len(value) >= 2:
        ops.append("space_insert")
    if has_spaces:
        ops.append("space_delete")
    if has_punct:
        ops.append("punct_collapse")

    if not ops:
        return None

    op = str(rng.choice(ops))
    chars = list(value)

    if op == "space_insert":
        # Insert a space at a random position between characters.
        insert_positions = list(range(1, len(chars)))
        if not insert_positions:
            return None
        pos = int(rng.choice(insert_positions))
        chars.insert(pos, " ")
        new_value = "".join(chars)
        return new_value, {
            "sub_op": "space_insert",
            "position": pos,
        }

    elif op == "space_delete":
        space_positions = [i for i, c in enumerate(chars) if c == " "]
        if not space_positions:
            return None
        pos = int(rng.choice(space_positions))
        del chars[pos]
        new_value = "".join(chars)
        return new_value, {
            "sub_op": "space_delete",
            "position": pos,
        }

    else:  # punct_collapse
        punct_positions = [i for i, c in enumerate(chars) if c in (",", ".", "-")]
        if not punct_positions:
            return None
        pos = int(rng.choice(punct_positions))
        removed = chars[pos]
        del chars[pos]
        new_value = "".join(chars)
        return new_value, {
            "sub_op": "punct_collapse",
            "position": pos,
            "removed": removed,
        }


def case_corrupt(
    value: str,
    rng: np.random.Generator,
) -> tuple[str, dict[str, Any]] | None:
    """Random case change on one or more alphabetic characters.

    Parameters
    ----------
    value : str
        Input string.
    rng : Generator
        Seeded RNG.

    Returns
    -------
    tuple[str, dict] or None
        ``(new_value, params_dict)`` or ``None`` if no alpha characters.
    """
    alpha_positions = [i for i, c in enumerate(value) if c.isalpha()]
    if not alpha_positions:
        return None

    pos = int(rng.choice(alpha_positions))
    chars = list(value)
    original_char = chars[pos]
    chars[pos] = original_char.swapcase()
    new_value = "".join(chars)

    if new_value == value:
        return None

    return new_value, {
        "position": pos,
        "original": original_char,
        "replacement": chars[pos],
    }


# ---------------------------------------------------------------------------
# Taxonomy walk
# ---------------------------------------------------------------------------


_TAXONOMY_CACHE: dict[str, "Taxonomy"] = {}


class Taxonomy:
    """A hierarchical label taxonomy loaded from CSV.

    Each row encodes one terminal label plus its ancestor labels along an
    ordered chain of *level* columns (most abstract first, most specific
    last).  Used by :func:`taxonomy_walk` to map a cell value to a node in
    the hierarchy and walk one level up (more abstract) or one level down
    (a sibling of the value's children).
    """

    def __init__(self, name: str, levels: list[str], rows: pd.DataFrame) -> None:
        self.name = name
        self.levels = levels
        self.rows = rows.reset_index(drop=True)
        # Lower-cased label-to-(level_index, canonical_label) lookup.
        self._lookup: dict[str, tuple[int, str]] = {}
        # Children of (level_idx, canonical_label) -> set of child labels.
        self._children: dict[tuple[int, str], set[str]] = {}
        for _, row in self.rows.iterrows():
            for li, lvl in enumerate(levels):
                val = row[lvl]
                if pd.isna(val):
                    continue
                vstr = str(val).strip()
                if not vstr:
                    continue
                key = vstr.lower()
                if key not in self._lookup:
                    self._lookup[key] = (li, vstr)
                if li > 0:
                    parent_val = row[levels[li - 1]]
                    if pd.notna(parent_val) and str(parent_val).strip():
                        pkey = (li - 1, str(parent_val).strip())
                        self._children.setdefault(pkey, set()).add(vstr)

    def find(self, value: str) -> tuple[int, str] | None:
        """Return (level_index, canonical_label) for *value*, or None."""
        if value is None:
            return None
        key = str(value).strip().lower()
        if not key:
            return None
        return self._lookup.get(key)

    def parent_of(self, level_idx: int, canonical: str) -> str | None:
        """Return the parent label (one level up) of *canonical*, or None."""
        if level_idx <= 0:
            return None
        # Find the row that contains *canonical* at *level_idx* and read the
        # value at level_idx - 1.
        col = self.levels[level_idx]
        parent_col = self.levels[level_idx - 1]
        mask = self.rows[col].astype(str).str.strip().str.lower() == canonical.lower()
        if not mask.any():
            return None
        for parent in self.rows.loc[mask, parent_col].dropna():
            pstr = str(parent).strip()
            if pstr:
                return pstr
        return None

    def children_of(self, level_idx: int, canonical: str) -> list[str]:
        """Return the children labels (one level down) of *canonical*."""
        if level_idx >= len(self.levels) - 1:
            return []
        kids = self._children.get((level_idx, canonical), set())
        return sorted(kids)


def load_taxonomy(name: str, csv_path: Path, levels: list[str]) -> Taxonomy:
    """Load *and cache* a taxonomy from *csv_path*.

    Parameters
    ----------
    name : str
        Cache key (e.g. ``"music_genres"``).
    csv_path : Path
        Absolute path to the taxonomy CSV.
    levels : list of str
        Column names in the CSV ordered most-abstract first.

    Returns
    -------
    Taxonomy
        Cached instance.
    """
    if name in _TAXONOMY_CACHE:
        return _TAXONOMY_CACHE[name]
    df = pd.read_csv(csv_path)
    missing = [lvl for lvl in levels if lvl not in df.columns]
    if missing:
        raise ValueError(f"Taxonomy {name!r} at {csv_path} missing levels: {missing}")
    tax = Taxonomy(name=name, levels=levels, rows=df[levels])
    _TAXONOMY_CACHE[name] = tax
    return tax


def taxonomy_walk(
    value: str,
    rng: np.random.Generator,
    taxonomy: Taxonomy,
    direction: str = "either",
) -> tuple[str, dict[str, Any]] | None:
    """Walk one level up or down a configured taxonomy.

    Parameters
    ----------
    value : str
        Input cell value.
    rng : Generator
        Seeded RNG.
    taxonomy : Taxonomy
        The configured hierarchy.
    direction : {"up", "down", "either"}
        ``up`` swaps in the parent label (more abstract); ``down`` picks a
        random sibling at the next-deeper level (more specific). ``either``
        draws 50/50 between the two then falls through.

    Returns
    -------
    tuple[str, dict] or None
        ``(new_value, params)`` or ``None`` if the value is not in the
        taxonomy or if the requested walk has no destination (e.g. ``up``
        from a root, ``down`` from a leaf).
    """
    found = taxonomy.find(value)
    if found is None:
        return None
    level_idx, canonical = found

    if direction == "either":
        choice = "up" if rng.random() < 0.5 else "down"
    else:
        choice = direction

    if choice == "up":
        parent = taxonomy.parent_of(level_idx, canonical)
        if parent is None or parent.lower() == canonical.lower():
            return None
        return parent, {
            "taxonomy": taxonomy.name,
            "direction": "up",
            "from_level": level_idx,
            "to_level": level_idx - 1,
            "from_label": canonical,
        }

    # down: pick a sibling at the next-deeper level.
    children = taxonomy.children_of(level_idx, canonical)
    if not children:
        return None
    child = str(rng.choice(children))
    if child.lower() == canonical.lower():
        return None
    return child, {
        "taxonomy": taxonomy.name,
        "direction": "down",
        "from_level": level_idx,
        "to_level": level_idx + 1,
        "from_label": canonical,
    }


# ---------------------------------------------------------------------------
# Numeric jitter cap (Pending #6)
# ---------------------------------------------------------------------------


_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d.%m.%Y",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y%m%d",
)


def _try_parse_float(text: str) -> float | None:
    """Best-effort parse to float. Strips currency/locale punctuation."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    # Strip leading currency symbols and trailing magnitude tokens.
    s_clean = re.sub(r"[^\d.\-,eE+]", "", s)
    if not s_clean or s_clean in ("-", ".", ","):
        return None
    # Both 1,234.56 (en) and 1.234,56 (de): if both separators present, the
    # last one is the decimal mark. With only one, treat as decimal mark
    # only when it would otherwise produce an unparseable string.
    if "," in s_clean and "." in s_clean:
        if s_clean.rfind(",") > s_clean.rfind("."):
            s_clean = s_clean.replace(".", "").replace(",", ".")
        else:
            s_clean = s_clean.replace(",", "")
    elif "," in s_clean:
        # Heuristic: 4+ digit groups → thousands; else decimal.
        comma_parts = s_clean.split(",")
        if len(comma_parts) > 2 or (len(comma_parts) == 2 and len(comma_parts[1]) == 3):
            s_clean = s_clean.replace(",", "")
        else:
            s_clean = s_clean.replace(",", ".")
    try:
        return float(s_clean)
    except ValueError:
        return None


def _try_parse_year(text: str) -> int | None:
    """Best-effort parse to a 4-digit year."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    m = re.search(r"(?<!\d)(\d{4})(?!\d)", s)
    if not m:
        return None
    try:
        y = int(m.group(1))
    except ValueError:
        return None
    if 1500 <= y <= 2200:
        return y
    return None


def _try_parse_date(text: str) -> date | None:
    """Best-effort parse to a calendar date."""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def numeric_jitter_within_cap(
    original_value: str,
    new_value: str,
    column_type: str,
    *,
    max_relative: float = 0.02,
) -> bool:
    """Check whether *new_value* is an acceptable jitter of *original_value*.

    Returns True (allow) when:

    - ``column_type == "continuous"``: both parse as floats and the relative
      perturbation ``abs(new - orig) / max(abs(orig), 1.0)`` is at or below
      ``max_relative``; OR the new value is unparseable as a float (string
      corruption — not a numeric jitter).
    - ``column_type == "year"``: both parse as 4-digit years and the year is
      identical; OR the new value is unparseable as a year (corruption).
    - ``column_type == "date"``: both parse as calendar dates with delta = 0;
      OR the new value is unparseable as a date (corruption).
    - For unknown ``column_type``: always True (cap not applicable).

    Returns False (reject) when both values parse but the perturbation
    exceeds the cap. The dispatcher then retries with a different operator.
    """
    if column_type == "continuous":
        orig_f = _try_parse_float(original_value)
        new_f = _try_parse_float(new_value)
        if orig_f is None or new_f is None:
            return True  # Unparseable: corruption, not jitter.
        denom = max(abs(orig_f), 1.0)
        rel = abs(new_f - orig_f) / denom
        return rel <= max_relative
    if column_type == "year":
        orig_y = _try_parse_year(original_value)
        new_y = _try_parse_year(new_value)
        if orig_y is None or new_y is None:
            return True
        return orig_y == new_y
    if column_type == "date":
        orig_d = _try_parse_date(original_value)
        new_d = _try_parse_date(new_value)
        if orig_d is None or new_d is None:
            return True
        return orig_d == new_d
    return True


# ---------------------------------------------------------------------------
# Operator registry
# ---------------------------------------------------------------------------

OPERATOR_REGISTRY: dict[str, Any] = {
    "typo_substitute": typo_substitute,
    "ocr_confuse": ocr_confuse,
    "truncate": truncate,
    "whitespace_corrupt": whitespace_corrupt,
    "case_corrupt": case_corrupt,
    "taxonomy_walk": taxonomy_walk,
}

VALID_TRANSFORM_FNS = frozenset(
    {
        "typo_substitute",
        "ocr_confuse",
        "truncate",
        "whitespace_corrupt",
        "case_corrupt",
        "taxonomy_walk",
        "cleanup",
        "rollback_for_committee",
    }
)
