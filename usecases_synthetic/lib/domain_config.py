"""Domain configuration loading and validation.

Loads per-knob, per-domain YAML configs from ``usecases_synthetic/config/``
and validates monotonicity across easy/medium/hard levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
USECASES_DIR: Path = REPO_ROOT / "usecases"
SYNTHETIC_DIR: Path = REPO_ROOT / "usecases_synthetic"
CONFIG_DIR: Path = SYNTHETIC_DIR / "config"
POOLS_DIR: Path = SYNTHETIC_DIR / "pools"

VALID_LEVELS: list[str] = ["easy", "medium", "hard"]
VALID_DOMAINS: list[str] = [
    "companies",
    "companies-small",
    "games",
    "games-small",
    "music",
    "music-small",
    "movies",
    "products",
    "products-small",
]


@dataclass
class SourceSpec:
    """Specification for a single data source within a domain.

    Parameters
    ----------
    name : str
        Source label (e.g. ``"dbpedia"``).
    file : str
        Filename relative to ``usecases/<domain>/input/data/``.
    format : str
        File format: ``"xml"``, ``"csv"``, or ``"json"``.
    id_prefix : str
        Prefix used in entity IDs (e.g. ``"http://dbpedia.org/"``).
    reader_kwargs : dict[str, Any]
        Extra keyword arguments forwarded to the underlying
        ``PyDI.io`` loader (e.g. ``{"delimiter": "\\t"}`` for TSV files
        with a ``.csv`` extension).
    inject_id : bool
        When ``True``, ensure the loaded DataFrame has an ``id`` column.
        Existing ``id`` columns are preserved; if absent, the synthetic
        loader injects ``id`` as ``f"{name}_{1-based-row-index}"``. Used
        for source files that ship without an explicit ID column (e.g.
        the games corpus). Defaults to ``False`` to preserve the
        companies behavior of using its native ``identifier`` /
        ``Identifier`` / ``id`` columns.
    id_column : str or None
        Name of the on-disk column that carries the entity id, when it is
        not already named ``id``. The loader renames this column to
        ``id`` immediately after read so downstream code can keep its
        ``id_column="id"`` assumption. Used by the 2026-05-04 refreshed
        CSV sources whose native id columns ship under semantic names
        (e.g. companies/dbpedia ``entity_uri``, games/metacritic
        ``mc_id``). When set, the rename happens before ``inject_id`` is
        evaluated, and the on-disk values must already be prefix-bearing
        ids that match the EM gold (e.g. ``mbrainz_1`` not ``1``).
    """

    name: str
    file: str
    format: str
    id_prefix: str
    reader_kwargs: dict[str, Any] = field(default_factory=dict)
    inject_id: bool = False
    id_column: str | None = None


@dataclass
class DomainConfig:
    """Per-domain configuration for the synthetic generation pipeline.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).
    sources : list of SourceSpec
        Ordered list of data sources.
    attribute_classes : dict
        Mapping from attribute name to class
        (``"primary"``, ``"key"``, ``"secondary"``).
    master_seed : int
        Master RNG seed for this domain.
    source_pairs : list of tuple[str, str]
        Ordered list of source pairs with EM correspondences.
    """

    domain: str
    sources: list[SourceSpec]
    attribute_classes: dict[str, str]
    master_seed: int = 42
    source_pairs: list[tuple[str, str]] = field(default_factory=list)
    knob_config_alias: str | None = None
    knob_config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    fusion_files: dict[str, str] = field(
        default_factory=lambda: {
            "validation": "validation_set.xml",
            "test": "test_set.xml",
        }
    )
    # Optional override for the directory containing ``<domain>/`` and
    # ``<domain>-augmented/``. Defaults to ``USECASES_DIR`` (the top-
    # level ``usecases/`` folder), but products points here at
    # ``usecases_synthetic/usecases/`` so the synthetic pipeline reads
    # synthetic-side data without touching the original ``usecases/products/``
    # notebook workflows.
    data_root: Path | None = None

    @property
    def source_names(self) -> list[str]:
        """Return ordered list of source names."""
        return [s.name for s in self.sources]

    def _root(self) -> Path:
        """Resolve the directory containing ``<domain>/`` and
        ``<domain>-augmented/``. Honors ``data_root`` when set, falls back
        to :data:`USECASES_DIR`. Use :func:`data_root_for_domain` outside
        this class to honor test monkeypatching of the module-level
        ``USECASES_DIR``."""
        return self.data_root if self.data_root is not None else USECASES_DIR

    def domain_dir(self) -> Path:
        """Return the per-domain root, i.e. ``<data_root>/<domain>``."""
        return self._root() / self.domain

    def data_dir(self) -> Path:
        """Return the input data directory for this domain."""
        return self.domain_dir() / "input" / "data"

    def em_dir(self) -> Path:
        """Return the entity matching directory for this domain."""
        return self.domain_dir() / "input" / "entitymatching"

    def fusion_dir(self) -> Path:
        """Return the fusion directory for this domain."""
        return self.domain_dir() / "input" / "fusion"

    def fusion_validation_path(self) -> Path:
        """Return the configured fusion validation-set XML path."""
        return self.fusion_dir() / self.fusion_files["validation"]

    def fusion_test_path(self) -> Path:
        """Return the configured fusion test-set XML path."""
        return self.fusion_dir() / self.fusion_files["test"]

    def fusion_paths(self) -> list[Path]:
        """Return ``[validation_path, test_path]`` in the canonical iteration order."""
        return [self.fusion_validation_path(), self.fusion_test_path()]

    def pool_dir(self) -> Path:
        """Return the pooled positives directory for this domain."""
        return POOLS_DIR / self.domain


def load_domain_config(domain: str) -> DomainConfig:
    """Load the domain configuration from YAML.

    Parameters
    ----------
    domain : str
        Domain name (e.g. ``"companies"``).

    Returns
    -------
    DomainConfig
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the domain config YAML does not exist.
    ValueError
        If the domain name is not recognized.
    """
    if domain not in VALID_DOMAINS:
        raise ValueError(f"Unknown domain: {domain!r}. Valid: {VALID_DOMAINS}")

    path = CONFIG_DIR / "domains" / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Domain config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    sources = [
        SourceSpec(
            name=s["name"],
            file=s["file"],
            format=s["format"],
            id_prefix=s.get("id_prefix", ""),
            reader_kwargs=dict(s.get("reader_kwargs", {}) or {}),
            inject_id=bool(s.get("inject_id", False)),
            id_column=s.get("id_column"),
        )
        for s in raw["sources"]
    ]

    source_pairs = [(p[0], p[1]) for p in raw.get("source_pairs", [])]

    raw_fusion = raw.get("fusion_files") or {}
    fusion_files = {
        "validation": str(raw_fusion.get("validation", "validation_set.xml")),
        "test": str(raw_fusion.get("test", "test_set.xml")),
    }

    raw_data_root = raw.get("data_root")
    data_root = (REPO_ROOT / raw_data_root).resolve() if raw_data_root else None

    return DomainConfig(
        domain=raw["domain"],
        sources=sources,
        attribute_classes=raw.get("attribute_classes", {}),
        master_seed=raw.get("master_seed", 42),
        source_pairs=source_pairs,
        knob_config_alias=raw.get("knob_config_alias"),
        knob_config_overrides=raw.get("knob_config_overrides") or {},
        fusion_files=fusion_files,
        data_root=data_root,
    )


def data_root_for_domain(domain: str) -> Path | None:
    """Return the per-domain ``data_root`` override, or ``None`` if unset.

    Honors the optional ``data_root`` field in
    ``config/domains/<domain>.yaml`` (resolved relative to
    :data:`REPO_ROOT`). Returns ``None`` when the YAML does not declare
    an override, when the YAML is missing, or when it cannot be read.

    Callers should fall back to their module's :data:`USECASES_DIR`
    (which test fixtures may monkeypatch). Typical pattern::

        root = data_root_for_domain(domain) or USECASES_DIR
        data_dir = root / domain / "input" / "data"
    """
    path = CONFIG_DIR / "domains" / f"{domain}.yaml"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return None
    raw_root = raw.get("data_root")
    if raw_root:
        return (REPO_ROOT / raw_root).resolve()
    return None


def load_knob_config(knob: int, domain: str) -> dict[str, Any]:
    """Load a per-knob, per-domain YAML config.

    Parameters
    ----------
    knob : int
        Knob number (1-10).
    domain : str
        Domain name.

    Returns
    -------
    dict
        Raw parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    knob_dir_name = _knob_dir_name(knob)
    path = CONFIG_DIR / knob_dir_name / f"{domain}.yaml"
    if not path.exists():
        alias = _resolve_knob_config_alias(domain)
        if alias is not None:
            alias_path = CONFIG_DIR / knob_dir_name / f"{alias}.yaml"
            if alias_path.exists():
                path = alias_path
            else:
                raise FileNotFoundError(
                    f"Knob config not found: {path} (alias {alias_path} also missing)"
                )
        else:
            raise FileNotFoundError(f"Knob config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    overrides = _resolve_knob_config_overrides(domain, knob_dir_name)
    if overrides:
        raw = _deep_merge_dict(raw, overrides)
    return raw


def _resolve_knob_config_overrides(
    domain: str,
    knob_dir_name: str,
) -> dict[str, Any]:
    """Return the ``knob_config_overrides[knob_dir_name]`` block for *domain*.

    Empty dict when the domain declares no overrides (or none for this
    knob). Reads the domain YAML directly so this function does not
    depend on a prior ``load_domain_config`` call.
    """
    domain_yaml = CONFIG_DIR / "domains" / f"{domain}.yaml"
    if not domain_yaml.exists():
        return {}
    with open(domain_yaml, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    all_overrides = raw.get("knob_config_overrides") or {}
    return dict(all_overrides.get(knob_dir_name) or {})


def _deep_merge_dict(
    base: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    """Return a deep-merged copy of *base* with *overlay* applied.

    Values in *overlay* win; nested dicts are recursed, lists and
    scalars are replaced wholesale.
    """
    out: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_knob_config_alias(domain: str) -> str | None:
    """Return ``knob_config_alias`` for ``domain`` if declared, else ``None``."""
    domain_yaml = CONFIG_DIR / "domains" / f"{domain}.yaml"
    if not domain_yaml.exists():
        return None
    with open(domain_yaml, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    alias = raw.get("knob_config_alias")
    return str(alias) if alias else None


def resolve_cache_domain(domain: str) -> str:
    """Return the domain whose LLM caches ``domain`` should read/write.

    When ``domain`` declares a ``knob_config_alias`` in its domain YAML,
    returns the alias so aliased domains (e.g. ``companies-small``)
    share the source domain's content-hashed LLM caches rather than
    rebuilding them. Returns ``domain`` itself otherwise.
    """
    alias = _resolve_knob_config_alias(domain)
    return alias if alias else domain


def validate_monotonicity(
    values: dict[str, float],
    direction: str = "increasing",
) -> None:
    """Validate that easy/medium/hard values are monotone.

    Parameters
    ----------
    values : dict
        Mapping from level name to numeric value. Must contain
        ``"easy"``, ``"medium"``, ``"hard"``.
    direction : str
        ``"increasing"`` (easy <= medium <= hard) or
        ``"decreasing"`` (easy >= medium >= hard).

    Raises
    ------
    ValueError
        If the monotonicity constraint is violated.
    """
    e, m, h = values["easy"], values["medium"], values["hard"]
    if direction == "increasing":
        if not (e <= m <= h):
            raise ValueError(
                f"Non-monotone (increasing): easy={e}, medium={m}, hard={h}"
            )
    elif direction == "decreasing":
        if not (e >= m >= h):
            raise ValueError(
                f"Non-monotone (decreasing): easy={e}, medium={m}, hard={h}"
            )
    else:
        raise ValueError(f"Unknown direction: {direction!r}")


def validate_knob_config_monotonicity(
    config: dict[str, Any],
    rate_keys: list[str],
    direction: str = "increasing",
) -> list[str]:
    """Validate monotonicity for all rate keys in a knob config.

    Looks for keys that map to ``{easy: float, medium: float, hard: float}``
    dicts and validates each one.

    Parameters
    ----------
    config : dict
        Parsed knob config YAML.
    rate_keys : list of str
        Config keys to check (e.g. ``["noise_rate_primary", ...]``).
    direction : str
        ``"increasing"`` or ``"decreasing"``.

    Returns
    -------
    list of str
        List of error messages (empty if all valid).
    """
    errors: list[str] = []
    for key in rate_keys:
        if key not in config:
            continue
        val = config[key]
        if not isinstance(val, dict):
            continue
        if not all(level in val for level in VALID_LEVELS):
            continue
        try:
            validate_monotonicity(val, direction=direction)
        except ValueError as exc:
            errors.append(f"{key}: {exc}")
    return errors


def _knob_dir_name(knob: int) -> str:
    """Return the config directory name for a knob number."""
    knob_names = {
        1: "knob_01_surface",
        2: "knob_02_niche",
        3: "knob_03_drop",
        4: "knob_04_coverage",
        5: "knob_05_format",
        6: "knob_06_noise",
        7: "knob_07_ambiguity",
        8: "knob_08_naming",
        9: "knob_09_completeness",
        10: "knob_10_reliability",
    }
    if knob not in knob_names:
        raise ValueError(f"Unknown knob number: {knob}")
    return knob_names[knob]
