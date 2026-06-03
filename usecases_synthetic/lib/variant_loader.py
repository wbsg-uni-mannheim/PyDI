"""Variant loader for committee validation.

``load_variant(domain, level)`` returns a :class:`VariantBundle` that
packages everything a committee needs to score a single variant:

- source DataFrames with ``attrs["dataset_name"]`` preserved
- schema-matching target schema plus the K8 mapping (when present)
- per-source-pair EM gold splits
- fusion gold and fusion validation DataFrames
- pooled positives (when the domain has a pool)

``level == "baseline"`` loads the *original* ``usecases/<domain>/``
directory; every other level loads
``usecases/<domain>-augmented/<level>/``. The baseline case lets
``measure_baseline.py`` reuse the same committee runners as
``validate_variant.py`` — one code path, two inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from PyDI.io import load_csv, load_json, load_xml

from .domain_config import (
    POOLS_DIR,
    USECASES_DIR,
    VALID_LEVELS,
    DomainConfig,
    data_root_for_domain,
    load_domain_config,
)
from .loaders import (
    em_gold_candidates,
    load_source,
    read_em_gold_pair,
)

VALID_BUNDLE_LEVELS: list[str] = ["baseline", *VALID_LEVELS]

EM_SPLITS: tuple[str, ...] = ("train", "val", "test", "all")


@dataclass
class VariantBundle:
    """Everything a committee needs to score a single variant.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    level : str
        One of ``"baseline"``, ``"easy"``, ``"medium"``, ``"hard"``.
    sources : dict[str, DataFrame]
        Source DataFrames keyed by source name, with
        ``attrs["dataset_name"]`` set.
    target_schema : dict
        Parsed ``target_schema.json``.
    sm_mapping : DataFrame or None
        Knob-8 generated schema mapping. ``None`` for ``level ==
        "baseline"`` (original use cases have no SM mapping artefact).
    em_gold : dict[tuple[str, str], DataFrame]
        Per-source-pair test gold, keyed by ``(src1, src2)`` ordered as
        declared in the domain config. Each frame has columns ``id1``,
        ``id2``, ``label``.
    em_gold_regenerated : dict[tuple[str, str], dict[str, dict[str, DataFrame]]]
        Per-source-pair regenerated EM gold, split by train / val /
        test and then by version (``baseline_pruned`` /
        ``corner_filled``). Loaded from
        ``input/entitymatching/<pair>_{train,val,test}_{baseline_pruned,corner_filled}.csv``
        (Knob 2 output, plan_revision.md C11). Outer key follows the
        same ``(src1, src2)`` convention as ``em_gold``; inner key is
        the split name; innermost key is the version. Empty dict when
        the files are missing (baseline variants; K2 not yet applied).
        Frames have columns ``id1``, ``id2``, ``label`` — source
        columns are dropped after loading. The **val** split is the
        primary variant-specific F1 surface; the **test** split is
        scored under both versions by the EM committee (4d);
        **train** is emitted for downstream public benchmark consumers.
    em_splits : dict[tuple[str, str], dict[str, DataFrame]]
        Per-source-pair split map. The inner dict has keys ``"train"``,
        ``"val"``, ``"test"``, ``"all"``; missing splits are omitted.
    fusion_gold : DataFrame
        Parsed ``test_set.xml`` used as the fusion gold standard.
    fusion_validation : DataFrame or None
        Parsed ``validation_set.xml`` if present.
    pooled_positives : DataFrame or None
        Pooled positives from ``pools/<domain>/pooled_positives.csv``
        when present. Used as the protection set / pool diagnostic.
    variant_root : Path
        Root directory this bundle was loaded from.
    """

    domain: str
    level: str
    sources: dict[str, pd.DataFrame]
    target_schema: dict[str, Any]
    sm_mapping: pd.DataFrame | None
    em_gold: dict[tuple[str, str], pd.DataFrame]
    em_splits: dict[tuple[str, str], dict[str, pd.DataFrame]]
    fusion_gold: pd.DataFrame
    fusion_validation: pd.DataFrame | None
    pooled_positives: pd.DataFrame | None
    variant_root: Path
    em_gold_regenerated: dict[tuple[str, str], dict[str, dict[str, pd.DataFrame]]] = (
        field(default_factory=dict)
    )
    knob_08_renames: dict[str, dict[str, str]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def source_pairs(self) -> list[tuple[str, str]]:
        """Return the ordered source pairs this bundle contains EM gold for."""
        return list(self.em_gold.keys())

    def resolve_column_mapping(
        self,
        static_mapping: dict[str, dict[str, str]],
    ) -> dict[str, dict[str, str]]:
        """Translate a committee's static column_mapping through K8 renames.

        The static column_mapping in ``em_blocking_committee.yaml`` /
        ``em_matching_committee.yaml`` / ``fusion_committee.yaml`` is keyed
        on *original* source column
        names (pre-K8). Two translations are applied so the resulting
        mapping renames the columns that actually exist in the variant:

        1. For every static entry ``{orig: canonical}``, the key is
           rewritten to the post-K8 column name (when K8 renamed it).
        2. For every K8 rename ``orig -> k8_col`` on a column *not* in
           the static mapping, we add ``{k8_col: orig}`` so the column
           is restored to its pre-K8 (baseline, canonical) name.

        Together these guarantee every column the variant's DataFrame
        carries is either left alone or renamed back to the name the
        downstream committee code expects.

        Columns not touched by K8 pass through unchanged. Sources
        absent from ``knob_08_renames`` pass through identity.

        Parameters
        ----------
        static_mapping : dict
            ``{source: {orig_col: canonical_col}}`` from the committee
            roster YAML.

        Returns
        -------
        dict
            ``{source: {post_k8_col: canonical_col}}`` suitable for
            renaming the variant's source DataFrames.
        """
        if not self.knob_08_renames:
            return {src: dict(m) for src, m in static_mapping.items()}

        resolved: dict[str, dict[str, str]] = {}
        all_sources = set(static_mapping) | set(self.knob_08_renames)
        for src in all_sources:
            static = static_mapping.get(src, {})
            k8 = self.knob_08_renames.get(src, {})

            result: dict[str, str] = {}
            # K8-only renames: restore post-K8 name to its pre-K8 name.
            for orig, k8_col in k8.items():
                if orig not in static:
                    result[k8_col] = orig
            # Static entries: rewrite key through K8 (identity if unrenamed).
            for orig, canonical in static.items():
                result[k8.get(orig, orig)] = canonical

            resolved[src] = result
        return resolved


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def variant_root(domain: str, level: str) -> Path:
    """Return the root directory for ``(domain, level)``.

    Parameters
    ----------
    domain : str
        Domain name.
    level : str
        ``"baseline"`` maps to ``usecases/<domain>/``; any other level
        maps to ``usecases/<domain>-augmented/<level>/``.

    Returns
    -------
    Path
        Variant root directory.
    """
    if level == "baseline":
        # Original-input bundle honors the ``data_root`` override for
        # the input domain (e.g. products → usecases_synthetic/usecases).
        root = data_root_for_domain(domain) or USECASES_DIR
        return root / domain
    # Augmented variants always live under the top-level USECASES_DIR
    # (``usecases/<domain>-augmented/<level>/``) for cross-domain
    # consistency; the per-domain ``data_root`` does NOT apply here.
    return USECASES_DIR / f"{domain}-augmented" / level


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------


def _load_baseline_sources(config: DomainConfig) -> dict[str, pd.DataFrame]:
    """Load original-format sources for the baseline variant."""
    sources: dict[str, pd.DataFrame] = {}
    for spec in config.sources:
        df = load_source(
            domain=config.domain,
            source_name=spec.name,
            source_file=spec.file,
            source_format=spec.format,
            reader_kwargs=spec.reader_kwargs,
            inject_id=spec.inject_id,
            id_column=spec.id_column,
        )
        df.attrs["dataset_name"] = spec.name
        sources[spec.name] = df
    return sources


def _load_augmented_sources(
    config: DomainConfig,
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load CSV-serialised sources from a packaged variant directory.

    The packager writes every source as ``<source>.csv`` regardless of
    original format (XML / JSON / TSV all collapse to CSV). See
    :func:`usecases_synthetic.scripts.package_variant.write_sources_as_csv`.

    Applies the same post-load normalization as
    :func:`usecases_synthetic.lib.loaders.load_source` (music ``tracks``
    list-literal parse, games ``genres`` comma-split, discogs
    ``duration`` 0-sentinel coalesce) so list-aware fusion strategies
    + Jaccard eval consume identical in-memory shapes on baseline and
    variant runs. ``DataFrame.to_csv`` collapses Python lists to their
    string repr, so without this step the variant's tracks/genres
    columns would come back as strings and break list-shape eval.
    """
    from .loaders import normalize_loaded_source

    sources: dict[str, pd.DataFrame] = {}
    for spec in config.sources:
        csv_path = data_dir / f"{spec.name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Variant source missing: {csv_path}")
        df = load_csv(csv_path, name=spec.name)
        df = normalize_loaded_source(df, domain=config.domain, source_name=spec.name)
        df.attrs["dataset_name"] = spec.name
        sources[spec.name] = df
    return sources


# ---------------------------------------------------------------------------
# Schema matching
# ---------------------------------------------------------------------------


def _load_target_schema(sm_dir: Path) -> dict[str, Any]:
    """Parse ``target_schema.json`` from a schema-matching directory."""
    path = sm_dir / "target_schema.json"
    if not path.exists():
        raise FileNotFoundError(f"Target schema missing: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_sm_mapping(sm_dir: Path, *, baseline: bool = False) -> pd.DataFrame | None:
    """Load schema-matching gold mapping if present.

    Parameters
    ----------
    sm_dir : Path
        Schema-matching directory.
    baseline : bool
        When ``True``, look for ``sm_mapping_gold.csv`` (the hand-authored
        baseline mapping). When ``False``, look for ``sm_mapping.csv``
        (knob-8 generated variant mapping).

    Returns
    -------
    DataFrame or None
        Gold mapping, or ``None`` if the file does not exist.
    """
    filename = "sm_mapping_gold.csv" if baseline else "sm_mapping.csv"
    path = sm_dir / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Entity matching
# ---------------------------------------------------------------------------


def _em_file(em_dir: Path, pair: tuple[str, str], split: str) -> Path:
    """Return the EM CSV path for ``<src1>_2_<src2>_<split>.csv``."""
    src1, src2 = pair
    return em_dir / f"{src1}_2_{src2}_{split}.csv"


def _load_em_gold(
    em_dir: Path,
    source_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], pd.DataFrame]:
    """Load EM gold for each declared source pair.

    Prefers the ``_all`` file (full gold standard) when available;
    falls back to ``_test`` when it doesn't exist.

    Pair direction is matched modulo orientation
    (``<src1>_2_<src2>_<split>.csv`` OR
    ``<src2>_2_<src1>_<split>.csv``). When loaded from the reverse
    direction, ``id1`` / ``id2`` are swapped so that the returned frame
    always carries ``id1`` belonging to ``src1`` and ``id2`` belonging
    to ``src2``. Downstream consumers (``committee_em.py``, the
    matcher inputs) look up ``id1`` in ``df_left = sources[src1]`` and
    ``id2`` in ``df_right = sources[src2]``; without the swap, every
    lookup fails on reverse-direction gold (regression observed 2026-05-26
    on games ``metacritic_dbpedia`` after the initial direction-tolerance
    fix landed without the swap — magellan crashed, Ditto returned 0
    predictions).

    Parameters
    ----------
    em_dir : Path
        Entity-matching directory.
    source_pairs : list of tuple
        Source pairs to load. Pairs without a gold CSV are skipped.

    Returns
    -------
    dict
        ``{(src1, src2): DataFrame}`` for every pair with a gold CSV.
        Each frame has columns ``id1`` (src1's ids), ``id2`` (src2's
        ids), ``label``.
    """
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for pair in source_pairs:
        for split in ("all", "test"):
            match = next(
                (
                    (path, swap)
                    for path, swap in em_gold_candidates(em_dir, pair, split)
                    if path.exists()
                ),
                None,
            )
            if match is not None:
                # read_em_gold_pair swaps id1<->id2 for reverse-direction
                # files so id1 always belongs to the pair's src1.
                out[pair] = read_em_gold_pair(*match)
                break
    return out


_REGEN_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
_REGEN_VERSION_NAMES: tuple[str, ...] = ("baseline_pruned", "corner_filled")


def _load_em_gold_regenerated(
    em_dir: Path,
    source_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, dict[str, pd.DataFrame]]]:
    """Load Knob-2 regenerated EM gold per source pair per split per version.

    Looks for files at
    ``input/entitymatching/<src1>_2_<src2>_<split>_<version>.csv`` for
    each authored pair × each split in ``train / val / test`` × each
    version in ``baseline_pruned / corner_filled`` (plan_revision.md
    C11). Each file carries ``id1, id2, source_1, source_2, label``;
    source columns are dropped after loading.

    Pair direction is matched modulo orientation (``src1_2_src2`` OR
    ``src2_2_src1``) so the loader tolerates either ordering on disk.

    Pre-C11 ``*_regenerated.csv`` files are no longer recognised — the
    next variant regen overwrites with the new naming, and per the
    2026-05-25 sign-off the legacy artifacts are not preserved.

    Parameters
    ----------
    em_dir : Path
        Entity-matching directory.
    source_pairs : list of tuple
        Source pairs declared by the domain config (canonical ordering).

    Returns
    -------
    dict
        ``{(src1, src2): {split: {version: DataFrame}}}`` with columns
        ``id1``, ``id2``, ``label``. Empty outer dict when no regen
        files exist (baseline variants; K2 not yet applied). Inner
        dicts omit splits / versions whose file is missing.
    """
    out: dict[tuple[str, str], dict[str, dict[str, pd.DataFrame]]] = {}
    for pair in source_pairs:
        src1, src2 = pair
        per_split: dict[str, dict[str, pd.DataFrame]] = {}
        for split in _REGEN_SPLIT_NAMES:
            per_version: dict[str, pd.DataFrame] = {}
            for version in _REGEN_VERSION_NAMES:
                candidates = [
                    em_dir / f"{src1}_2_{src2}_{split}_{version}.csv",
                    em_dir / f"{src2}_2_{src1}_{split}_{version}.csv",
                ]
                path = next((p for p in candidates if p.exists()), None)
                if path is None:
                    continue
                df = pd.read_csv(path)
                if df.empty:
                    continue
                required = {"id1", "id2", "label"}
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(
                        f"{path} missing required columns {sorted(missing)}; "
                        "regenerate with the current apply_knob_02_niche.py."
                    )
                per_version[version] = (
                    df[["id1", "id2", "label"]].reset_index(drop=True).copy()
                )
            if per_version:
                per_split[split] = per_version
        if per_split:
            out[pair] = per_split
    return out


def _load_em_splits(
    em_dir: Path,
    source_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, pd.DataFrame]]:
    """Load all EM splits for each source pair.

    Parameters
    ----------
    em_dir : Path
        Entity-matching directory.
    source_pairs : list of tuple
        Source pairs to load.

    Returns
    -------
    dict
        ``{pair: {split: DataFrame}}``. Missing split files are omitted.
    """
    out: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for pair in source_pairs:
        per_split: dict[str, pd.DataFrame] = {}
        for split in EM_SPLITS:
            match = next(
                (
                    (path, swap)
                    for path, swap in em_gold_candidates(em_dir, pair, split)
                    if path.exists()
                ),
                None,
            )
            if match is not None:
                per_split[split] = read_em_gold_pair(*match)
        if per_split:
            out[pair] = per_split
    return out


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def _load_fusion_xml(path: Path, name: str) -> pd.DataFrame:
    """Load a fusion XML file via PyDI with the standard flags."""
    df = load_xml(path, name=name, nested_handling="aggregate")
    df.attrs["dataset_name"] = name
    return df


def _load_fusion_file(path: Path, name: str) -> pd.DataFrame:
    """Load a fusion gold file, dispatching on file extension.

    ``.xml`` (every pre-2026 domain) loads through PyDI's XML reader with
    the aggregate flag, exactly as before. The 2026 papers domain ships
    its fusion gold as JSON-lines (``fusion_test.jsonl`` /
    ``fusion_val.jsonl``: one flat fused record per line, joined to fused
    output on ``doi``), read with ``lines=True``; a plain ``.json`` array
    is also accepted. Non-XML gold carries no per-attribute
    ``provenance`` (the papers notebook reconstructs cluster membership
    from the EM correspondences at eval time), so source-attribution
    fusion metrics are unavailable for such domains — value-correctness
    by the configured ``gold_id_column`` join still works.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        df = load_json(path, name=name, lines=True)
    elif suffix == ".json":
        df = load_json(path, name=name)
    else:
        return _load_fusion_xml(path, name)
    df.attrs["dataset_name"] = name
    return df


def _load_fusion(
    fusion_dir: Path,
    test_filename: str = "test_set.xml",
    validation_filename: str = "validation_set.xml",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load the fusion test (gold) + validation (optional) files.

    ``test_filename`` / ``validation_filename`` default to the canonical
    variant-directory names. ``load_variant`` overrides them with the
    domain config's ``fusion_files`` block when ``level == "baseline"``
    so the source-side ``*_set_final.xml`` for games + music (or the
    papers ``fusion_{test,val}.jsonl``) is picked up; packaged variants
    always carry the canonical names per ``package_variant.copy_fusion``.
    Format is chosen per file extension (see :func:`_load_fusion_file`).
    """
    test_path = fusion_dir / test_filename
    if not test_path.exists():
        raise FileNotFoundError(f"Fusion gold missing: {test_path}")
    gold = _load_fusion_file(test_path, "fusion_test_set")

    val_path = fusion_dir / validation_filename
    val: pd.DataFrame | None = None
    if val_path.exists():
        val = _load_fusion_file(val_path, "fusion_validation_set")
    return gold, val


# ---------------------------------------------------------------------------
# Pooled positives
# ---------------------------------------------------------------------------


def _load_pooled_positives(domain: str) -> pd.DataFrame | None:
    """Load ``pools/<domain>/pooled_positives.csv`` if present."""
    path = POOLS_DIR / domain / "pooled_positives.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Knob-08 renames
# ---------------------------------------------------------------------------


def _load_knob_08_renames(root: Path) -> dict[str, dict[str, str]]:
    """Load per-source column renames from the K8 provenance CSV.

    The K8 apply script writes ``output/provenance/knob_08_naming.csv``
    with one row per non-identity column rename, using columns
    ``source``, ``attribute`` (pre-K8 column name) and ``new_value``
    (post-K8 column name).

    Parameters
    ----------
    root : Path
        Variant root directory.

    Returns
    -------
    dict
        ``{source: {pre_k8_col: post_k8_col}}``. Empty dict when the
        provenance CSV is absent (baseline variants or identity rung).
    """
    path = root / "output" / "provenance" / "knob_08_naming.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}

    renames: dict[str, dict[str, str]] = {}
    for row in df.itertuples(index=False):
        src = str(row.source)
        orig = str(row.attribute)
        new = str(row.new_value)
        renames.setdefault(src, {})[orig] = new
    return renames


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_variant(
    domain: str,
    level: str = "baseline",
    *,
    root_override: Path | None = None,
) -> VariantBundle:
    """Load a variant bundle for ``(domain, level)``.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    level : str, optional
        One of ``"baseline"``, ``"easy"``, ``"medium"``, ``"hard"``.
        Default ``"baseline"`` (original use case directory).
    root_override : Path, optional
        Override the variant root directory entirely. Used by tests to
        point at a fixture directory. When set, ``level`` still controls
        which source-loading strategy is used (original formats for
        baseline, CSV for augmented).

    Returns
    -------
    VariantBundle
        Everything the committees need.
    """
    if level not in VALID_BUNDLE_LEVELS:
        raise ValueError(f"Invalid level: {level!r}. Valid: {VALID_BUNDLE_LEVELS}")

    config = load_domain_config(domain)
    root = root_override if root_override is not None else variant_root(domain, level)

    input_dir = root / "input"
    data_dir = input_dir / "data"
    sm_dir = input_dir / "schemamatching"
    em_dir = input_dir / "entitymatching"
    fusion_dir = input_dir / "fusion"

    if level == "baseline" and root_override is None:
        sources = _load_baseline_sources(config)
    else:
        sources = _load_augmented_sources(config, data_dir)

    target_schema = _load_target_schema(sm_dir)
    sm_mapping = _load_sm_mapping(sm_dir, baseline=(level == "baseline"))

    em_gold = _load_em_gold(em_dir, config.source_pairs)
    em_gold_regenerated = _load_em_gold_regenerated(em_dir, config.source_pairs)
    em_splits = _load_em_splits(em_dir, config.source_pairs)

    if level == "baseline" and root_override is None:
        fusion_gold, fusion_validation = _load_fusion(
            fusion_dir,
            test_filename=config.fusion_files["test"],
            validation_filename=config.fusion_files["validation"],
        )
    else:
        fusion_gold, fusion_validation = _load_fusion(fusion_dir)

    pooled = _load_pooled_positives(domain)

    knob_08_renames = _load_knob_08_renames(root)

    return VariantBundle(
        domain=domain,
        level=level,
        sources=sources,
        target_schema=target_schema,
        sm_mapping=sm_mapping,
        em_gold=em_gold,
        em_gold_regenerated=em_gold_regenerated,
        em_splits=em_splits,
        fusion_gold=fusion_gold,
        fusion_validation=fusion_validation,
        pooled_positives=pooled,
        variant_root=root,
        knob_08_renames=knob_08_renames,
    )
