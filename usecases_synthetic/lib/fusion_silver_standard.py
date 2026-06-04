"""Fusion silver standard builder (plan_revision.md C9 / step 4b).

Applies each domain's human-baseline fusion stack to every cluster in
the pool, producing one fused value per (cluster, attribute). The
output replaces the prior "one value close to gold" variant-generation
protection target with a stricter "post-knob fusion close to silver"
check (wired in a later sub-step).

Per-domain fusion stacks mirror the workflow notebooks at
``usecases/<domain>/<domain>_workflow.ipynb``:

* ``music`` — extracted from cell 38 of ``music_workflow.ipynb``.
  Trust: ``musicbrainz=3 / lastfm=2 / discogs=1``.
* ``games`` — extracted from cell ~26 of ``games_workflow.ipynb``.
  Trust: ``metacritic=3 / sales=2 / dbpedia=1``. **Not yet wired**
  (lands after the music user-gate review per plan §4b).
* ``companies`` — extracted from cell ~26 of
  ``companies_workflow.ipynb``. Trust:
  ``dbpedia=3 / fullcontact=2 / forbes=1``. **Not yet wired**
  (lands after the music user-gate review per plan §4b).
* ``products`` — deferred until R1 lands the
  ``data_cleaned_final`` schema. The notebook's
  ``hardware_fusion_strategy`` references columns
  (``vram_gb`` / ``chipset_name`` / ``storage_gb`` / ...) that do
  not exist in the current synthetic source files.

Cluster IDs are picked as the highest-trust source's record id within
the cluster (lexicographically smallest as tiebreaker).

Output is a long-format DataFrame keyed by ``(cluster_id, attribute)``
and persisted as both CSV (analysis surface) and JSON (per-cluster
nested view).
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from PyDI.fusion.engine import DataFusionEngine
from PyDI.fusion.strategy import DataFusionStrategy
from PyDI.normalization import load_normalization_spec
from PyDI.schemamatching import SchemaTranslator

from .domain_config import (
    POOLS_DIR,
    SYNTHETIC_DIR,
    USECASES_DIR,
    data_root_for_domain,
    load_domain_config,
)
from .fusion_perfect_clusters import _partner_graph, _transitive_closure
from .loaders import load_domain_sources

logger = logging.getLogger(__name__)

SILVER_DIR: Path = SYNTHETIC_DIR / "baselines"

# ---------------------------------------------------------------------------
# Per-domain wiring (mirrors the workflow notebooks)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DomainStack:
    """Per-domain fusion stack spec, extracted from the workflow notebook."""

    domain: str
    trust_scores: dict[str, int]
    attributes: tuple[str, ...]
    # Source-id prefix → source name. Used to pick the canonical cluster id.
    id_prefix_to_source: dict[str, str]


_MUSIC_STACK = _DomainStack(
    domain="music",
    # music_workflow.ipynb cell 36: mbrainz=3, lastfm=2, discogs=1.
    trust_scores={"musicbrainz": 3, "lastfm": 2, "discogs": 1},
    attributes=(
        "name",
        "artist",
        "release-date",
        "release-country",
        "duration",
        "tracks",
        "label",
    ),
    id_prefix_to_source={
        "mbrainz_": "musicbrainz",
        "discogs_": "discogs",
        "lastFM_": "lastfm",
    },
)


_GAMES_STACK = _DomainStack(
    domain="games",
    # games_workflow.ipynb cells 3049-3051: metacritic=3, sales=2, dbpedia=1.
    trust_scores={"metacritic": 3, "sales": 2, "dbpedia": 1},
    attributes=(
        "name",
        "platform",
        "developer",
        "releaseYear",
        "ESRB",
        "criticScore",
        "userScore",
        "genres",
    ),
    id_prefix_to_source={
        "dbpedia_": "dbpedia",
        "metacritic_": "metacritic",
        "sales_": "sales",
    },
)


_COMPANIES_STACK = _DomainStack(
    domain="companies",
    # companies_workflow.ipynb cells 2921-2923: dbpedia=3, fullcontact=2, forbes=1.
    trust_scores={"dbpedia": 3, "fullcontact": 2, "forbes": 1},
    # Notebook calls the multi-truth attribute ``founders`` (cell 2960-2966);
    # the synthetic pipeline / protection.py call it ``keypeople``. The
    # silver uses the synthetic-pipeline canonical name so it is directly
    # usable by the post-knob recoverability check.
    attributes=(
        "name",
        "assets",
        "revenue",
        "keypeople",
        "founded",
        "country",
        "city",
    ),
    id_prefix_to_source={
        # Companies sources expose URL-shaped ids for dbpedia/forbes; the
        # synthetic loader keeps the raw form (no rename to fullcontact_*).
        "http://dbpedia.org/": "dbpedia",
        "http://www.forbes.com/": "forbes",
        "fullcontact_": "fullcontact",
    },
)


_PRODUCTS_STACK = _DomainStack(
    domain="products",
    # products_workflow_minimal.ipynb cell f1b27fcf:
    # products_1=3, products_2=2, products_3=1, products_4=2.
    trust_scores={
        "products_1": 3,
        "products_2": 2,
        "products_3": 1,
        "products_4": 2,
    },
    # The 17 attributes the notebook's fusion strategy emits (cell
    # 8ff11ed2). title / description / title_description / price /
    # priceCurrency / color / url / cluster_id are intentionally omitted
    # — the notebook does not author per-attribute fusers for them, so
    # silver does not protect those cells. K1/K6 noise on title /
    # description therefore stays unconstrained (preserves the K1/K6
    # difficulty signal on products' free-text columns).
    attributes=(
        "brand",
        "product_type",
        "model_number",
        "vram_gb",
        "storage_gb",
        "read_speed_mb_s",
        "write_speed_mb_s",
        "chipset_name",
        "bus_type",
        "interface_type",
        "storage_connection_type",
        "memory_type",
        "form_factor",
        "width_mm",
        "length_mm",
        "height_mm",
        "weight_g",
    ),
    id_prefix_to_source={
        "products_1_": "products_1",
        "products_2_": "products_2",
        "products_3_": "products_3",
        "products_4_": "products_4",
    },
)


_PAPERS_STACK = _DomainStack(
    domain="papers",
    # papers_workflow_minimal.ipynb "Version_5" global trust: dblp=1,
    # crossref=3, open_alex=1. crossref is therefore the canonical
    # cluster representative (highest trust); per-attribute trust maps
    # (e.g. keywords prefers open_alex) are applied in the fusion
    # strategy, not here.
    trust_scores={"dblp": 1, "crossref": 3, "open_alex": 1},
    # The 11 attributes the notebook's Version_5 strategy fuses. doi /
    # id are the join key + record id (never fused/perturbed); publisher
    # and cited_by_count are intentionally omitted -- the notebook drops
    # them before fusion (no per-attribute fuser), so silver does not
    # protect those cells and K1/K6 noise on them stays unconstrained.
    attributes=(
        "type",
        "title",
        "authors",
        "publication_year",
        "journal",
        "keywords",
        "volume",
        "issue",
        "first_page",
        "last_page",
        "referenced_works_count",
    ),
    # Source ids are minted as "<source>-NNNNN" (dash); the prefix carries
    # the trailing dash so startswith() is unambiguous across sources.
    id_prefix_to_source={
        "dblp-": "dblp",
        "crossref-": "crossref",
        "open_alex-": "open_alex",
    },
)


_STACKS: dict[str, _DomainStack] = {
    "music": _MUSIC_STACK,
    "games": _GAMES_STACK,
    "companies": _COMPANIES_STACK,
    "products": _PRODUCTS_STACK,
    "papers": _PAPERS_STACK,
}


def supported_domains() -> list[str]:
    """Return the list of domains the silver-standard builder supports today."""
    return sorted(_STACKS.keys())


def _resolve_stack(domain: str) -> _DomainStack:
    if domain not in _STACKS:
        raise NotImplementedError(
            f"Fusion silver standard not yet wired for domain {domain!r}. "
            f"Supported: {supported_domains()}. "
            "Products is deferred until R1 (data_cleaned_final schema); "
            "products is deferred until R1 (data_cleaned_final schema)."
        )
    return _STACKS[domain]


# ---------------------------------------------------------------------------
# Music: pre/post normalization helpers extracted from the notebook
# ---------------------------------------------------------------------------


def _parse_track_list(value: Any) -> list[str]:
    """Parse music's ``tracks`` column to a deduplicated list of titles.

    Verbatim port of ``parse_track_list`` in
    ``usecases/music/music_workflow.ipynb`` cell 20.
    """
    if isinstance(value, list):
        items = value
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            items = parsed if isinstance(parsed, list) else [parsed]
        except (SyntaxError, ValueError):
            items = [part.strip() for part in text.split("|")]
    else:
        items = [value]

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None or (isinstance(item, float) and pd.isna(item)):
            continue
        title = str(item).strip()
        if not title:
            continue
        key = re.sub(r"\s+", " ", title.casefold())
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(title)
    return cleaned


def _sum_duration(val: Any) -> Any:
    """Reduce a list-valued duration to a single int (per notebook cell 31)."""
    if isinstance(val, list):
        digits = [int(x) for x in val if str(x).isdigit()]
        if not digits:
            return np.nan
        return int(np.nansum(digits))
    try:
        return int(val)
    except (TypeError, ValueError):
        return np.nan


def _fix_discogs_zero_date(value: Any) -> Any:
    """Map ``-00`` day/month placeholders to ``-01`` (per notebook cell 17)."""
    if isinstance(value, str):
        return re.sub(r"-00", "-01", value)
    return value


def prefer_track_list_by_source(
    values: list[Any],
    *,
    sources: list[str] | None = None,
    source_datasets: dict[str, str] | None = None,
    **_kwargs: Any,
) -> tuple[list[str] | None, float, dict[str, Any]]:
    """Pick one coherent track list by source priority (notebook cell 38)."""
    source_priority = ["musicbrainz", "discogs", "lastfm"]
    candidates: list[tuple[int, int, str, str, list[str]]] = []
    for value, record_id in zip(values, sources or [], strict=False):
        tracks = _parse_track_list(value)
        if not tracks:
            continue
        dataset = (source_datasets or {}).get(record_id, "unknown")
        priority = (
            source_priority.index(dataset)
            if dataset in source_priority
            else len(source_priority)
        )
        candidates.append((priority, -len(tracks), dataset, record_id, tracks))

    if not candidates:
        return (
            None,
            0.0,
            {"rule": "prefer_track_list_by_source", "reason": "no_valid_track_lists"},
        )

    priority, _, dataset, record_id, tracks = sorted(candidates)[0]
    confidence = 1.0 if priority == 0 else max(0.5, 1.0 - 0.2 * priority)
    return (
        tracks,
        confidence,
        {
            "rule": "prefer_track_list_by_source",
            "selected_dataset": dataset,
            "selected_record_id": record_id,
            "num_tracks": len(tracks),
        },
    )


# ---------------------------------------------------------------------------
# Spec-based normalization (per-domain)
# ---------------------------------------------------------------------------


def _schema_spec_path(domain: str) -> Path:
    """Return the per-domain ``target_schema.json`` path used by the notebook."""
    root = data_root_for_domain(domain) or USECASES_DIR
    return root / domain / "input" / "schemamatching" / "target_schema.json"


def _build_identity_mapping(
    source_name: str, source_columns: Iterable[str]
) -> pd.DataFrame:
    """Build an identity column mapping for SchemaTranslator.

    Synthetic sources are already loaded with canonical column names, so
    schema matching is a no-op rename. SchemaTranslator still expects a
    mapping DataFrame to drive value normalization.
    """
    cols = list(source_columns)
    return pd.DataFrame(
        {
            "source_dataset": [source_name] * len(cols),
            "source_column": cols,
            "target_dataset": ["target_schema"] * len(cols),
            "target_column": cols,
            "score": [1.0] * len(cols),
            "notes": ["identity"] * len(cols),
        }
    )


def _normalize_music_sources(
    sources: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Replicate music_workflow.ipynb's normalization stack.

    Order (per cells 17, 18, 20, 31):
    1. Fix discogs ``-00`` placeholder dates.
    2. Load the schema spec; set ``release-country`` (country name),
       ``release-date`` (datetime), and ``tracks`` (list) overrides.
    3. Translate + normalize each source via ``SchemaTranslator``.
    4. Apply ``parse_track_list`` to every source's ``tracks`` column.
    5. Reduce mbrainz ``duration`` from list (multi-track aggregation)
       to a single int via ``sum_duration``.
    """
    out = {name: df.copy() for name, df in sources.items()}

    # 1. Discogs date repair.
    if "discogs" in out and "release-date" in out["discogs"].columns:
        out["discogs"]["release-date"] = out["discogs"]["release-date"].apply(
            _fix_discogs_zero_date
        )

    # 2. Build the spec (notebook cells 6 + 18).
    spec = load_normalization_spec(_schema_spec_path("music"))
    spec.set_column("tracks", output_type="list")
    spec.set_column("release-country", country_format="name")
    spec.set_column("release-date", output_type="datetime")

    # 3. Translate + normalize per source. ``schema_base_path`` points
    # at the notebook's working directory (``usecases/<domain>/``) so
    # spec-relative paths like ``input/schemamatching/Music_Genres_Taxonomy.csv``
    # resolve the same way the notebook resolves them with cwd == that dir.
    schema_base = str(_schema_spec_path("music").parents[2])
    translator = SchemaTranslator()
    for name, df in list(out.items()):
        df.attrs.setdefault("dataset_name", name)
        mapping = _build_identity_mapping(name, df.columns)
        translated = translator.translate(
            df,
            mapping,
            normalize=spec,
            on_failure="keep",
            schema_base_path=schema_base,
        )
        # ``translate`` returns just the DataFrame when ``return_result=False``.
        out[name] = translated

    # 4. parse_track_list (notebook cell 20).
    for name, df in out.items():
        if "tracks" in df.columns:
            df["tracks"] = df["tracks"].apply(_parse_track_list)

    # 5. sum_duration on mbrainz (notebook cell 31).
    if "musicbrainz" in out and "duration" in out["musicbrainz"].columns:
        out["musicbrainz"]["duration"] = out["musicbrainz"]["duration"].apply(
            _sum_duration
        )

    return out


def _load_sm_mapping(domain: str, source_name: str) -> pd.DataFrame:
    """Load the per-source slice of the SM gold (JSON preferred, CSV fallback).

    The gold mapping is the canonical column-rename spec used across the
    synthetic pipeline (committee_fusion, protection.py). The committed
    ``sm_mapping_gold.json`` (``kind: pydi_schema_mapping_gold``) is preferred
    over the legacy ``sm_mapping_gold.csv``; because this builder runs the
    :class:`SchemaTranslator` against the *loaded* (renamed) frames from
    :func:`loaders.load_domain_sources`, the gold's raw source-column names are
    reconciled onto the loaded names via
    :func:`variant_loader._reconcile_sm_gold_source_columns` (id-col -> ``id``;
    papers raw -> canonical). Music's gold is identity so this is equivalent to
    :func:`_build_identity_mapping` on that domain.
    """
    from .domain_config import load_domain_config
    from .variant_loader import (
        _load_sm_mapping as _load_sm_gold,
        _reconcile_sm_gold_source_columns,
    )

    root = data_root_for_domain(domain) or USECASES_DIR
    sm_dir = root / domain / "input" / "schemamatching"
    df = _load_sm_gold(sm_dir, baseline=True)
    if df is None:
        raise FileNotFoundError(
            f"sm_mapping_gold.(json|csv) not found for domain {domain!r} at {sm_dir}"
        )
    df = _reconcile_sm_gold_source_columns(df, load_domain_config(domain), domain)
    df = df[df["source_dataset"] == source_name].copy()
    if df.empty:
        raise ValueError(f"No mapping rows for source {source_name!r} in {sm_dir}")
    if "notes" not in df.columns:
        df["notes"] = "gold"
    return df


def _normalize_games_sources(
    sources: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Replicate games_workflow.ipynb's normalization stack.

    Order (per cells 218 (spec), 1222-1224 (franchise clean),
    1252-1276 (per-source date format)):

    1. Clean dbpedia ``franchise`` (strip ``" (video game)"`` suffix) —
       outside the silver's attribute scope but applied for parity.
    2. Build the spec with ``genres → list``.
    3. Translate per source, setting ``releaseYear`` date format per
       source before each translate call:

       - dbpedia: auto-detect (``output_type="datetime"``).
       - metacritic: ``date_format="%b %d, %Y"`` (e.g. ``Oct 26, 2018``).
       - sales: ``date_format="%Y"`` (year-only).
    """
    out = {name: df.copy() for name, df in sources.items()}

    # 1. dbpedia franchise cleanup (notebook cell 1224). Cheap and harmless.
    if "dbpedia" in out and "franchise" in out["dbpedia"].columns:
        out["dbpedia"]["franchise"] = (
            out["dbpedia"]["franchise"]
            .astype("string")
            .str.replace(r" \(video game\)$", "", regex=True)
        )

    spec = load_normalization_spec(_schema_spec_path("games"))
    spec.set_column("genres", output_type="list")
    schema_base = str(_schema_spec_path("games").parents[2])

    translator = SchemaTranslator()
    per_source_date_formats: dict[str, dict[str, Any]] = {
        # dbpedia: auto-detect (no date_format).
        "dbpedia": {"output_type": "datetime"},
        # metacritic: "Oct 26, 2018".
        "metacritic": {"output_type": "datetime", "date_format": "%b %d, %Y"},
        # sales: "2010".
        "sales": {"output_type": "datetime", "date_format": "%Y"},
    }
    for name, df in list(out.items()):
        # Per notebook: spec is mutated between translates so this source's
        # releaseYear format is in effect.
        per_source = per_source_date_formats.get(name)
        if per_source:
            spec.set_column("releaseYear", **per_source)
        df.attrs.setdefault("dataset_name", name)
        mapping = _load_sm_mapping("games", name)
        translated = translator.translate(
            df,
            mapping,
            normalize=spec,
            on_failure="keep",
            schema_base_path=schema_base,
        )
        out[name] = translated

    return out


def _strip_thousands_separators(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    text = str(value).strip()
    if not text:
        return value
    return text.replace(",", "")


def _empty_to_nan(value: Any) -> Any:
    if value is None:
        return np.nan
    if isinstance(value, float) and pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    return value


def _normalize_companies_sources(
    sources: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Replicate companies_workflow.ipynb's normalization stack.

    Order (per cells 184 (spec), 415 + 1205 (dbpedia clean), 1211
    (country spec)):

    1. dbpedia: coerce empty ``keypeople_name`` to NaN; strip commas
       from ``total_assets_val`` (notebook cells 415, 1205).
    2. Build the spec with ``keypeople → list<string>`` (notebook's
       ``founders → list<string>``; renamed because the synthetic
       gold maps ``keypeople_name → keypeople``) and ``country →
       name`` country-format normalization.
    3. Translate per source via the gold mapping.

    Note on the ``founders`` vs ``keypeople`` divergence: the notebook
    runs an LLM-based schema match that retargets ``keypeople_name``
    to the schema's ``founders`` column; ``sm_mapping_gold.csv``
    instead targets ``keypeople``. As of R10-N (2026-06-02) the SM gold
    correctly maps fullcontact ``Attribute_5`` to ``keypeople`` (the
    column holds people-names, not industry), so the silver's
    ``keypeople`` is a multi-source union of dbpedia ``keypeople_name``
    + fullcontact ``Attribute_5``, matching the notebook. The silver
    uses the synthetic-pipeline gold mapping so the per-attribute names
    line up with :data:`protection._DEFAULT_KIND_BY_DOMAIN_ATTR`.
    """
    out = {name: df.copy() for name, df in sources.items()}

    # 1. dbpedia pre-cleanup (notebook cells 415 + 1205).
    if "dbpedia" in out:
        db = out["dbpedia"]
        if "keypeople_name" in db.columns:
            db["keypeople_name"] = db["keypeople_name"].apply(_empty_to_nan)
        if "total_assets_val" in db.columns:
            db["total_assets_val"] = db["total_assets_val"].apply(
                _strip_thousands_separators
            )

    spec = load_normalization_spec(_schema_spec_path("companies"))
    # Notebook uses ``founders``; we apply the equivalent list<string>
    # rule under the gold's ``keypeople`` target name.
    spec.set_column("keypeople", output_type="list<string>")
    spec.set_column("country", country_format="name")
    schema_base = str(_schema_spec_path("companies").parents[2])

    translator = SchemaTranslator()
    for name, df in list(out.items()):
        df.attrs.setdefault("dataset_name", name)
        mapping = _load_sm_mapping("companies", name)
        translated = translator.translate(
            df,
            mapping,
            normalize=spec,
            on_failure="keep",
            schema_base_path=schema_base,
        )
        out[name] = translated

    return out


def _normalize_products_sources(
    sources: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Canonicalize the per-source-native products schemas via sm_mapping.

    Pre-2026-06-02 the synthetic products sources shipped already-canonical
    column names (the notebook loads pre-normalized
    ``dataset_<n>_normalized.json``), so this was a pass-through. After the
    source-native schema adoption (2026-06-02) the four sources each carry
    a distinct native vocabulary (products_1 ``manufacturer/product_name/
    list_price``; products_2 ``brandName/name/priceAmount``; etc.), so the
    "already canonical" assumption no longer holds. Mirror the games/
    companies pattern: rename each source's native columns to the canonical
    schema via the per-source ``sm_mapping_gold.csv`` slice.

    Per C9 the silver must replicate exactly the notebook's pre-fusion
    normalization. The products notebook applies NONE (it loads canonical
    data directly), and the native data is a pure rename of that same data
    (identical values), so the faithful replication is a column rename with
    no further value normalization.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        mapping = _load_sm_mapping("products", name)
        rename = dict(zip(mapping["source_column"], mapping["target_column"]))
        copy = df.rename(columns=rename)
        copy.attrs = dict(df.attrs)
        copy.attrs["dataset_name"] = name
        out[name] = copy
    return out


def _normalize_papers_sources(
    sources: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Pass-through normalization for papers (data already target-shaped).

    Unlike the pre-2026 domains, the papers sources ship in the canonical
    target schema with native Python types already in place after load:
    ``authors`` is a list, ``publication_year`` is an integer, and the
    count/page columns are numeric. The Version_5 notebook applies no
    pre-fusion value normalization on top of that, so the faithful C9
    replication is an identity copy that only re-stamps ``dataset_name``.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        copy = df.copy()
        copy.attrs = dict(df.attrs)
        copy.attrs["dataset_name"] = name
        out[name] = copy
    return out


_NORMALIZERS: dict[
    str, Callable[[dict[str, pd.DataFrame]], dict[str, pd.DataFrame]]
] = {
    "music": _normalize_music_sources,
    "games": _normalize_games_sources,
    "companies": _normalize_companies_sources,
    "products": _normalize_products_sources,
    "papers": _normalize_papers_sources,
}


def _normalize_sources(
    domain: str, sources: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    normalizer = _NORMALIZERS.get(domain)
    if normalizer is None:
        raise NotImplementedError(
            f"No normalization stack wired for domain {domain!r}."
        )
    return normalizer(sources)


# ---------------------------------------------------------------------------
# Per-domain fusion strategy (mirrors the notebook)
# ---------------------------------------------------------------------------


def _build_music_strategy() -> DataFusionStrategy:
    """Mirror ``DataFusionStrategy('music_fusion_strategy')`` from cell 38."""
    from PyDI.fusion import (
        DataFusionStrategy as _Strategy,
        longest_string,
        maximum,
        prefer_higher_trust,
        shortest_string,
    )

    strategy = _Strategy("music_silver_strategy")
    strategy.add_attribute_fuser("name", shortest_string)
    strategy.add_attribute_fuser("artist", longest_string)
    strategy.add_attribute_fuser("release-date", prefer_higher_trust)
    strategy.add_attribute_fuser("release-country", prefer_higher_trust)
    strategy.add_attribute_fuser("duration", maximum)
    strategy.add_attribute_fuser("tracks", prefer_track_list_by_source)
    strategy.add_attribute_fuser("label", longest_string)
    return strategy


def _build_games_strategy() -> DataFusionStrategy:
    """Mirror ``DataFusionStrategy('game_fusion_strategy')`` from cell 3290."""
    from PyDI.fusion import (
        DataFusionStrategy as _Strategy,
        average,
        prefer_higher_trust,
        union,
        voting,
    )

    strategy = _Strategy("games_silver_strategy")
    strategy.add_attribute_fuser("name", voting)
    strategy.add_attribute_fuser("platform", voting)
    strategy.add_attribute_fuser("developer", voting)
    # Notebook passes ``trust_key="trust_score"`` even on voting; voting
    # ignores it but we keep the kwarg for parity with the notebook.
    strategy.add_attribute_fuser("releaseYear", voting, trust_key="trust_score")
    strategy.add_attribute_fuser("ESRB", prefer_higher_trust, trust_key="trust_score")
    strategy.add_attribute_fuser(
        "criticScore", prefer_higher_trust, trust_key="trust_score"
    )
    strategy.add_attribute_fuser("userScore", average)
    strategy.add_attribute_fuser("genres", union)
    return strategy


def _build_companies_strategy() -> DataFusionStrategy:
    """Mirror ``DataFusionStrategy('company_fusion_strategy')`` from cell 2956.

    The notebook attribute name ``founders`` is mapped to ``keypeople`` here
    (see :func:`_normalize_companies_sources` for the rename rationale).
    """
    from PyDI.fusion import (
        DataFusionStrategy as _Strategy,
        prefer_higher_trust,
        shortest_string,
        union,
        voting,
    )

    strategy = _Strategy("companies_silver_strategy")
    strategy.add_attribute_fuser("name", voting)
    strategy.add_attribute_fuser("assets", prefer_higher_trust)
    strategy.add_attribute_fuser("revenue", prefer_higher_trust)
    strategy.add_attribute_fuser("keypeople", union)
    strategy.add_attribute_fuser("founded", prefer_higher_trust)
    strategy.add_attribute_fuser("country", voting)
    strategy.add_attribute_fuser("city", shortest_string)
    return strategy


def _build_products_strategy() -> DataFusionStrategy:
    """Mirror ``DataFusionStrategy('hardware_fusion_strategy')`` from
    ``products_workflow_minimal.ipynb`` cell 8ff11ed2.

    Three attribute groups (notebook comments):

    - **Identity** (``voting``): brand, product_type, model_number.
    - **Performance specs** (``minimum``): vram_gb, storage_gb,
      read_speed_mb_s, write_speed_mb_s. The notebook author chose
      ``minimum`` over the more common ``maximum`` ("i think min could
      be better. lets see"); silver mirrors that choice so the silver
      target matches the notebook's authoritative fused value.
    - **Technical + dimensions** (``prefer_higher_trust`` with
      ``trust_key="trust_score"``): chipset_name, bus_type,
      interface_type, storage_connection_type, memory_type, form_factor,
      width_mm, length_mm, height_mm, weight_g. The trust order is
      products_1=3, products_2=2, products_3=1, products_4=2 (cell
      f1b27fcf).
    """
    from PyDI.fusion import (
        DataFusionStrategy as _Strategy,
        minimum,
        prefer_higher_trust,
        voting,
    )

    strategy = _Strategy("products_silver_strategy")

    # 1. Identity
    for attr in ("brand", "product_type", "model_number"):
        strategy.add_attribute_fuser(attr, voting)

    # 2. Performance specs
    for attr in (
        "vram_gb",
        "storage_gb",
        "read_speed_mb_s",
        "write_speed_mb_s",
    ):
        strategy.add_attribute_fuser(attr, minimum)

    # 3. Technical + dimensions
    for attr in (
        "chipset_name",
        "bus_type",
        "interface_type",
        "storage_connection_type",
        "memory_type",
        "form_factor",
        "width_mm",
        "length_mm",
        "height_mm",
        "weight_g",
    ):
        strategy.add_attribute_fuser(attr, prefer_higher_trust, trust_key="trust_score")

    return strategy


def _build_papers_strategy() -> DataFusionStrategy:
    """Mirror the papers ``Version_5`` fusion strategy from
    ``papers_workflow_minimal.ipynb``.

    Per-attribute fusers (notebook authoritative):

    - **Identity / categorical** (``voting``): type, journal, volume,
      issue, first_page.
    - **Title** (``longest_string``): the disambiguating free-text field.
    - **publication_year** (``maximum``): the notebook uses
      ``most_recent``; on a bare 4-digit integer year that is numerically
      identical to ``maximum`` and avoids date-parsing fragility, so the
      silver target matches the notebook's fused value.
    - **Trust-routed** (``prefer_higher_trust`` with an explicit
      per-attribute ``trust_map``): authors / last_page /
      referenced_works_count prefer crossref (global trust); keywords
      prefer open_alex (the notebook's keyword authority).

    ``publisher`` and ``cited_by_count`` are intentionally absent -- the
    notebook drops them before fusion.
    """
    from PyDI.fusion import (
        DataFusionStrategy as _Strategy,
        longest_string,
        maximum,
        prefer_higher_trust,
        voting,
    )

    # Global trust (Version_5): crossref is the authority; keywords are
    # the one attribute where open_alex wins.
    global_trust = {"dblp": 1, "crossref": 3, "open_alex": 1}
    keywords_trust = {"dblp": 1, "crossref": 1, "open_alex": 3}

    strategy = _Strategy("papers_silver_strategy")
    strategy.add_attribute_fuser("type", voting)
    strategy.add_attribute_fuser("title", longest_string)
    strategy.add_attribute_fuser("authors", prefer_higher_trust, trust_map=global_trust)
    strategy.add_attribute_fuser("publication_year", maximum)
    strategy.add_attribute_fuser("journal", voting)
    strategy.add_attribute_fuser(
        "keywords", prefer_higher_trust, trust_map=keywords_trust
    )
    strategy.add_attribute_fuser("volume", voting)
    strategy.add_attribute_fuser("issue", voting)
    strategy.add_attribute_fuser("first_page", voting)
    strategy.add_attribute_fuser(
        "last_page", prefer_higher_trust, trust_map=global_trust
    )
    strategy.add_attribute_fuser(
        "referenced_works_count", prefer_higher_trust, trust_map=global_trust
    )
    return strategy


_STRATEGY_BUILDERS: dict[str, Callable[[], DataFusionStrategy]] = {
    "music": _build_music_strategy,
    "games": _build_games_strategy,
    "companies": _build_companies_strategy,
    "products": _build_products_strategy,
    "papers": _build_papers_strategy,
}


def _build_strategy(domain: str) -> DataFusionStrategy:
    builder = _STRATEGY_BUILDERS.get(domain)
    if builder is None:
        raise NotImplementedError(f"No fusion strategy wired for domain {domain!r}.")
    return builder()


# ---------------------------------------------------------------------------
# Cluster construction from the pool
# ---------------------------------------------------------------------------


def build_pool_clusters(domain: str) -> dict[str, set[str]]:
    """Build all clusters from the pool's partner graph.

    Returns a ``{representative_id: cluster_members}`` map. The
    representative is the lexicographically smallest member id; the
    canonical cluster id (per the trust-priority rule) is picked
    separately at fusion time.
    """
    partners = _partner_graph(domain)
    seen: set[str] = set()
    clusters: dict[str, set[str]] = {}
    for seed in sorted(partners.keys()):
        if seed in seen:
            continue
        cluster = _transitive_closure(seed, partners)
        seen.update(cluster)
        rep = min(cluster)
        clusters[rep] = cluster
    return clusters


def _build_correspondences(clusters: dict[str, set[str]]) -> pd.DataFrame:
    """Emit hub-and-spoke correspondences for the fusion engine.

    Mirrors :func:`fusion_perfect_clusters.build_perfect_clusters_correspondences`
    but operates on every pool cluster (not just fusion-gold entities).
    """
    rows: list[tuple[str, str, float]] = []
    for members in clusters.values():
        sorted_members = sorted(members)
        if len(sorted_members) == 1:
            sole = sorted_members[0]
            rows.append((sole, sole, 1.0))
            continue
        hub = sorted_members[0]
        for other in sorted_members[1:]:
            rows.append((hub, other, 1.0))
    if not rows:
        return pd.DataFrame(columns=["id1", "id2", "score"])
    return pd.DataFrame(rows, columns=["id1", "id2", "score"]).drop_duplicates(
        ["id1", "id2"], ignore_index=True
    )


# ---------------------------------------------------------------------------
# Cluster-id picker
# ---------------------------------------------------------------------------


def _source_for_id(record_id: str, prefix_map: dict[str, str]) -> str | None:
    for prefix, source in prefix_map.items():
        if record_id.startswith(prefix):
            return source
    return None


def canonical_cluster_id(member_ids: Iterable[str], stack: _DomainStack) -> str:
    """Pick the canonical cluster id: highest-trust source's id, lex-min tiebreaker."""
    ids = list(member_ids)
    if not ids:
        raise ValueError("canonical_cluster_id called on empty member list")
    candidates: list[tuple[int, str]] = []
    for sid in ids:
        src = _source_for_id(sid, stack.id_prefix_to_source)
        if src is None:
            continue
        trust = stack.trust_scores.get(src, 0)
        # negate so that higher trust sorts earlier.
        candidates.append((-trust, sid))
    if not candidates:
        return sorted(ids)[0]
    candidates.sort()
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Silver-standard build pipeline
# ---------------------------------------------------------------------------


def _stamp_trust(
    sources: dict[str, pd.DataFrame], trust_scores: dict[str, int]
) -> list[pd.DataFrame]:
    """Copy and stamp ``attrs['trust_score']`` for the fusion engine."""
    out: list[pd.DataFrame] = []
    for name, df in sources.items():
        copy = df.copy()
        copy.attrs = dict(df.attrs)
        copy.attrs["dataset_name"] = name
        copy.attrs["trust_score"] = float(trust_scores.get(name, 1.0))
        out.append(copy)
    return out


def build_silver_standard(
    domain: str,
    *,
    sources: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build the per-cluster silver standard for *domain*.

    Parameters
    ----------
    domain : str
        Domain name (``music`` for the v1 sign-off scope).
    sources : dict[str, DataFrame], optional
        Pre-loaded raw source DataFrames keyed by source name. When
        ``None``, sources are loaded via :func:`loaders.load_domain_sources`.
        The notebook-equivalent normalization stack runs on top of
        these in either case.

    Returns
    -------
    DataFrame
        Long-format DataFrame with columns:

        ``cluster_id`` (str) — canonical cluster id (highest-trust
        source's id; lex-min tiebreaker).
        ``attribute`` (str) — canonical attribute name.
        ``fused_value`` (object) — fused value; list-typed for
        list attributes (e.g. music ``tracks``), otherwise the
        scalar emitted by the fusion engine.
        ``fused_value_repr`` (str) — display-friendly stringified
        form (``json.dumps`` for lists, ``str()`` otherwise; empty
        string for ``None``/``NaN``).
        ``confidence`` (float) — fusion engine's per-cell confidence.
        ``source_ids`` (str) — comma-joined sorted member ids in the
        cluster (audit trail).
        ``num_sources`` (int) — distinct source datasets contributing.
    """
    stack = _resolve_stack(domain)
    logger.info("Building fusion silver standard for domain=%s", domain)

    if sources is None:
        sources = load_domain_sources(domain)
        logger.info("Loaded %d raw sources for %s", len(sources), domain)

    sources = _normalize_sources(domain, sources)
    logger.info("Applied notebook-equivalent normalization stack")

    clusters = build_pool_clusters(domain)
    logger.info("Built %d pool clusters from partner graph", len(clusters))

    correspondences = _build_correspondences(clusters)
    logger.info(
        "Emitted %d hub-and-spoke correspondences across pool clusters",
        len(correspondences),
    )

    datasets = _stamp_trust(sources, stack.trust_scores)
    strategy = _build_strategy(domain)
    engine = DataFusionEngine(strategy, debug=False)
    fused = engine.run(
        datasets=datasets,
        correspondences=correspondences,
        id_column="id",
        # We only want pool clusters; records not in any correspondence
        # are not silver-standard candidates.
        include_singletons=False,
    )
    logger.info("Fusion engine produced %d fused records", len(fused))

    return _to_long_silver(fused, stack)


def _to_long_silver(fused: pd.DataFrame, stack: _DomainStack) -> pd.DataFrame:
    """Reshape the fusion engine's wide output to long silver format.

    The engine surfaces per-attribute ``{attr}_rule`` keys in
    ``_fusion_metadata`` plus a single per-record ``_fusion_confidence``
    (the mean of all attribute-level confidences). We use that
    per-record confidence as the silver's confidence column — the
    engine does not expose per-attribute confidence in the metadata.
    """
    if fused.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "attribute",
                "fused_value",
                "fused_value_repr",
                "fusion_rule",
                "confidence",
                "source_ids",
                "num_sources",
            ]
        )

    rows: list[dict[str, Any]] = []

    for _, record in fused.iterrows():
        sources_in_cluster = record.get("_fusion_sources", [])
        if not isinstance(sources_in_cluster, (list, tuple)):
            sources_in_cluster = [sources_in_cluster]
        sorted_members = sorted(str(s) for s in sources_in_cluster if s)
        if not sorted_members:
            continue
        cluster_id = canonical_cluster_id(sorted_members, stack)

        member_sources: set[str] = set()
        for sid in sorted_members:
            src = _source_for_id(sid, stack.id_prefix_to_source)
            if src:
                member_sources.add(src)

        member_join = ",".join(sorted_members)
        metadata = record.get("_fusion_metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        confidence_raw = record.get("_fusion_confidence")
        try:
            confidence = (
                float(confidence_raw) if confidence_raw is not None else float("nan")
            )
        except (TypeError, ValueError):
            confidence = float("nan")

        for attribute in stack.attributes:
            if attribute not in record.index:
                continue
            value = record[attribute]
            rule = metadata.get(f"{attribute}_rule", "")
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "attribute": attribute,
                    "fused_value": value,
                    "fused_value_repr": _value_repr(value),
                    "fusion_rule": rule,
                    "confidence": confidence,
                    "source_ids": member_join,
                    "num_sources": len(member_sources),
                }
            )

    return pd.DataFrame(rows)


def _value_repr(value: Any) -> str:
    """Display-friendly stringification of a fused value (for the CSV)."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return ""
        iso: str = value.isoformat()
        return iso
    return str(value)


# ---------------------------------------------------------------------------
# Persistence (CSV + JSON)
# ---------------------------------------------------------------------------


def silver_path(domain: str, ext: str = "csv") -> Path:
    """Return the canonical silver-standard artifact path."""
    if ext not in {"csv", "json"}:
        raise ValueError(f"Unsupported silver standard extension: {ext!r}")
    return SILVER_DIR / domain / f"fusion_silver_standard.{ext}"


def write_silver_standard(
    domain: str,
    silver: pd.DataFrame,
    *,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Persist the silver-standard DataFrame as CSV + JSON.

    Parameters
    ----------
    domain : str
        Domain name.
    silver : DataFrame
        Output of :func:`build_silver_standard`.
    out_dir : Path, optional
        Override directory. Defaults to
        ``usecases_synthetic/baselines/<domain>/``.

    Returns
    -------
    dict
        ``{"csv": Path, "json": Path}``.
    """
    base = out_dir if out_dir is not None else SILVER_DIR / domain
    base.mkdir(parents=True, exist_ok=True)

    csv_path = base / "fusion_silver_standard.csv"
    json_path = base / "fusion_silver_standard.json"

    # CSV: persist the display-safe repr column for the value (lists +
    # timestamps get a stable string form). Keep numeric columns numeric.
    csv_view = silver.drop(columns=["fused_value"], errors="ignore").rename(
        columns={"fused_value_repr": "fused_value"}
    )
    csv_view.to_csv(csv_path, index=False)
    logger.info("Wrote %d silver rows -> %s", len(csv_view), csv_path)

    # JSON: nested {cluster_id: {attribute: {value, fusion_rule}}} +
    # per-cluster source_ids / num_sources / mean confidence.
    nested: dict[str, dict[str, Any]] = {}
    for _, row in silver.iterrows():
        cid = str(row["cluster_id"])
        attr = str(row["attribute"])
        bucket = nested.setdefault(
            cid,
            {
                "source_ids": (
                    str(row["source_ids"]).split(",") if row["source_ids"] else []
                ),
                "num_sources": int(row["num_sources"]),
                "confidence": (
                    float(row["confidence"]) if not pd.isna(row["confidence"]) else None
                ),
                "attributes": {},
            },
        )
        value = row["fused_value"]
        if isinstance(value, pd.Timestamp):
            value_json: Any = value.isoformat() if not pd.isna(value) else None
        elif isinstance(value, (list, tuple)):
            value_json = [
                v.isoformat() if isinstance(v, pd.Timestamp) else v for v in value
            ]
        elif isinstance(value, float) and pd.isna(value):
            value_json = None
        elif value is None:
            value_json = None
        else:
            value_json = value
        bucket["attributes"][attr] = {
            "value": value_json,
            "fusion_rule": row.get("fusion_rule", "") or "",
        }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(nested, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Wrote nested silver JSON -> %s", json_path)

    return {"csv": csv_path, "json": json_path}


def load_silver_standard(domain: str) -> pd.DataFrame:
    """Load the on-disk silver-standard CSV.

    Returns the same long-format shape :func:`build_silver_standard`
    produces. ``fused_value`` carries the string repr; list-typed
    attributes can be re-parsed via :func:`json.loads` when needed.
    """
    path = silver_path(domain, "csv")
    if not path.exists():
        raise FileNotFoundError(
            f"No silver standard at {path}. "
            f"Run scripts/build_fusion_silver_standard.py --domain {domain}."
        )
    return pd.read_csv(path, keep_default_na=False, na_values=[""])
