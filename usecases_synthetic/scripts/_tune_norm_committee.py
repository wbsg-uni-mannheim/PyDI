"""Tune Normalization committee hyperparameters per member.

One-off sweep harness for the R5 Normalization-stage hyperparameter
optimisation (2026-05-10). Loads each domain's baseline bundle,
instantiates each member under a parameter grid, scores per-attribute
F1 against the fusion val/test reference values via the Pending #5
closeness contract, and reports the best param combo per member by
mean F1 across companies + games + music.

Mirrors :mod:`_tune_sm_committee` in shape; per-member SPECS define
``init_param_grid`` (cartesian over constructor kwargs).

C12 note: the SPECS "members" (text_clean / date_iso / number_locale /
country_iso / taxonomy_lookup / llm_canonicalize) are NOT the C12 norm
committee's three wrapper members (rule_per_attribute_optimal / llm_only
/ passthrough). They are the per-rule candidates declared under
``rule_normalizers`` in the committee YAML, which ``rule_per_attribute_
optimal`` selects among per attribute (and ``llm_canonicalize`` ==
``llm_only``'s LLMCanonicalizer). So this tuner IS C12-compatible: it
tunes the candidate-rule params; apply winners to the matching
``rule_normalizers[*].params`` / ``llm_normalizer.params`` blocks. (The
llm_canonicalize grid pins max_tokens=2048 + prompt v2 to match the
committee's live llm_only member and clear the reasoning-model floor.)

Usage::

    python usecases_synthetic/scripts/_tune_norm_committee.py \\
        --members text_clean,date_iso,number_locale,country_iso,taxonomy_lookup,llm_canonicalize \\
        --domains companies,games,music

Output is printed and written to ``cache/norm_tuning/sweep.json``.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load OPENAI_API_KEY for LLM members.
try:
    from dotenv import load_dotenv  # noqa: E402

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from usecases_synthetic.lib.committee_norm import (  # noqa: E402
    _build_entity_linkage,
    _build_source_attribute_index,
)
from usecases_synthetic.lib.committee_norm_scoring import (  # noqa: E402
    MemberPerAttributeScores,
)
from usecases_synthetic.lib.protection import (  # noqa: E402
    fusion_cell_tolerance,
    kind_map_for_domain,
    load_fusion_target_values,
)
from usecases_synthetic.lib.variant_loader import load_variant  # noqa: E402

logger = logging.getLogger("tune_norm")


CACHE_DIR = REPO_ROOT / "cache" / "norm_tuning"


# ---------------------------------------------------------------------------
# Per-domain context (shared across sweep cells)
# ---------------------------------------------------------------------------


def _domain_context(domain: str) -> dict[str, Any]:
    """Build the per-domain shared context once per sweep run."""
    bundle = load_variant(domain, level="baseline")
    if bundle.sm_mapping is None or bundle.sm_mapping.empty:
        raise RuntimeError(f"No SM mapping for {domain}")
    fusion_targets = load_fusion_target_values(domain)
    if not fusion_targets:
        raise RuntimeError(f"No fusion targets for {domain}")
    attr_index = _build_source_attribute_index(
        bundle.sm_mapping, bundle.knob_08_renames
    )
    sm_attrs = {ca for (_, ca) in attr_index.keys()}
    gold_attrs: set[str] = set()
    for ents in fusion_targets.values():
        gold_attrs.update(ents.keys())
    kind_map = kind_map_for_domain(domain)
    eligible = sorted(sm_attrs & gold_attrs & set(kind_map.keys()))

    source_id_index: dict[str, dict[str, int]] = {}
    for source_name, source_df in bundle.sources.items():
        if "id" not in source_df.columns:
            continue
        source_id_index[source_name] = {
            str(eid): int(idx) for idx, eid in enumerate(source_df["id"].tolist())
        }
    linkage = _build_entity_linkage(bundle)
    return {
        "domain": domain,
        "bundle": bundle,
        "fusion_targets": fusion_targets,
        "attr_index": attr_index,
        "eligible": eligible,
        "kind_map": kind_map,
        "source_id_index": source_id_index,
        "linkage": linkage,
    }


def _score_member(
    member,
    ctx: dict[str, Any],
    *,
    member_name: str | None = None,
) -> MemberPerAttributeScores:
    """Run a single member across (entity, attr, source) cells *the member applies to*.

    Per-member kind-applicability filter (avoid scoring abstentions on
    inapplicable attributes — e.g. ``date_iso`` would always abstain on
    ``name`` cells, which would artificially deflate its macro F1):

    - ``text_clean`` / ``llm_canonicalize`` apply to string-like kinds
      (``long_string``, ``nominal``, ``free_text``, ``list``).
    - ``date_iso`` applies to ``date``, ``year``.
    - ``number_locale`` applies to ``continuous``.
    - ``country_iso`` applies to ``nominal`` *and* the attribute name
      must be one of the canonical country attrs (``country`` /
      ``release-country``).
    - ``taxonomy_lookup`` applies to ``nominal`` / ``list`` *and* the
      domain has a taxonomy binding for that attribute.
    """
    scores = MemberPerAttributeScores(member=getattr(member, "name", "unnamed"))
    domain = ctx["domain"]
    bundle = ctx["bundle"]
    fusion_targets = ctx["fusion_targets"]
    attr_index = ctx["attr_index"]
    eligible = ctx["eligible"]
    kind_map = ctx["kind_map"]
    source_id_index = ctx["source_id_index"]
    # Wire LLM examples once if applicable.
    if hasattr(member, "set_examples"):
        per_attribute: dict[str, list[str]] = {}
        for ents in fusion_targets.values():
            for attr, vals in ents.items():
                per_attribute.setdefault(attr, []).extend(str(v) for v in vals)
        member.set_examples({domain: per_attribute})

    name = member_name or getattr(member, "name", "")

    string_like = {"long_string", "nominal", "free_text", "list"}
    if name == "text_clean":
        kinds_filter = string_like
    elif name == "llm_canonicalize":
        kinds_filter = string_like
    elif name == "date_iso":
        kinds_filter = {"date", "year"}
    elif name == "number_locale":
        kinds_filter = {"continuous"}
    elif name == "country_iso":
        kinds_filter = {"nominal"}
    elif name == "taxonomy_lookup":
        kinds_filter = {"nominal", "list"}
    else:
        kinds_filter = None

    country_attrs = {"country", "release-country"}
    taxonomy_attrs = set((TAXONOMY_BINDINGS.get(domain, {}) or {}).keys())

    def _attr_applies(attr: str, kind: str) -> bool:
        if kinds_filter is not None and kind not in kinds_filter:
            return False
        if name == "country_iso" and attr not in country_attrs:
            return False
        if name == "taxonomy_lookup" and attr not in taxonomy_attrs:
            return False
        return True

    for attribute in eligible:
        kind = kind_map.get(attribute, "long_string")
        if not _attr_applies(attribute, kind):
            continue
        tolerance = fusion_cell_tolerance(domain, attribute)
        for entity_id, ent_attrs in fusion_targets.items():
            target_values = ent_attrs.get(attribute)
            if not target_values:
                continue
            entity_linkage = ctx.get("linkage", {}).get(str(entity_id), {})
            for source_name in bundle.sources:
                cols = attr_index.get((source_name, attribute), [])
                if not cols:
                    continue
                src_df = bundle.sources[source_name]
                id_lookup = source_id_index.get(source_name)
                if id_lookup is None:
                    continue
                source_record_id = entity_linkage.get(source_name, str(entity_id))
                row_idx = id_lookup.get(source_record_id)
                if row_idx is None and source_record_id != str(entity_id):
                    row_idx = id_lookup.get(str(entity_id))
                if row_idx is None:
                    continue
                for col in cols:
                    if col not in src_df.columns:
                        continue
                    raw = src_df.iat[row_idx, src_df.columns.get_loc(col)]
                    try:
                        out = member.normalize(
                            raw, attribute=attribute, kind=kind, domain=domain
                        )
                    except Exception:
                        logger.exception(
                            "Member %s failed on (%s, %s, %s, %r)",
                            member.name,
                            domain,
                            source_name,
                            attribute,
                            raw,
                        )
                        out = None
                    scores.record(attribute, out, target_values, tolerance)
    return scores


# ---------------------------------------------------------------------------
# Per-member sweep specs
# ---------------------------------------------------------------------------
# Each spec: ((module, cls), init_param_grid).


SPECS: dict[str, tuple[tuple[str, str], dict[str, list[Any]]]] = {
    "text_clean": (
        ("usecases_synthetic.lib.normalizer_members", "TextCleanNormalizer"),
        {
            "lowercase": [True, False],
            "strip_whitespace": [True],
            "normalize_unicode": [True, False],
            "remove_punctuation": [True, False],
            "remove_html": [True],
        },
    ),
    "date_iso": (
        ("usecases_synthetic.lib.normalizer_members", "DateIsoNormalizer"),
        {
            "date_format": ["%Y-%m-%d"],
            "year_only_format": ["%Y"],
            "handle_timezone": [True, False],
        },
    ),
    "number_locale": (
        ("usecases_synthetic.lib.normalizer_members", "NumberLocaleNormalizer"),
        {
            "babel_candidate_locales": [
                ["en_US"],
                ["en_US", "de_DE"],
                ["en_US", "de_DE", "fr_FR"],
            ],
            "handle_currency": [True, False],
            "handle_percentages": [True],
        },
    ),
    "country_iso": (
        ("usecases_synthetic.lib.normalizer_members", "CountryIsoNormalizer"),
        {
            "output_format": ["alpha_2", "alpha_3", "name", "official_name"],
        },
    ),
    "taxonomy_lookup": (
        ("usecases_synthetic.lib.normalizer_members", "TaxonomyLookupNormalizer"),
        {
            "case_insensitive": [True, False],
        },
    ),
    "llm_canonicalize": (
        ("usecases_synthetic.lib.llm_normalizer", "LLMCanonicalizer"),
        {
            "model_name": ["gpt-5.4-mini"],
            # num_examples (few-shot count) is the real tunable knob.
            "num_examples": [0, 3, 5, 10],
            "temperature": [0.0],
            # 2048 = pipeline default + the committee llm_only member's
            # budget. gpt-5.4-mini is a reasoning model; build_chat_openai
            # rejects max_tokens < _REASONING_MIN_MAX_TOKENS (2048), so the
            # prior [64] made every llm_canonicalize cell fail -> 0.0.
            "max_tokens": [2048],
            # Match the committee's llm_normalizer prompt (v2) so the swept
            # num_examples winner transfers to the live committee member;
            # the committee config pins prompt_version: v2.
            "prompt_version": ["v2"],
        },
    ),
}


# Per-domain taxonomy bindings for the taxonomy_lookup sweep.
TAXONOMY_BINDINGS: dict[str, dict[str, dict[str, Any]]] = {
    "companies": {
        "industry": {
            "path": "companies/input/schemamatching/GICS_Industry_Taxonomy.csv",
            "columns": [
                "Sector Name",
                "Industry Group Name",
                "Industry Name",
                "Sub-Industry Name",
            ],
        },
    },
    "games": {
        "platform": {
            "path": "games/input/schemamatching/Gaming_Platforms_Taxonomy.csv",
            "columns": ["Platform Name"],
        },
        "genres": {
            "path": "games/input/schemamatching/Video_Game_Genres_Taxonomy.csv",
            "columns": ["Genre Name", "Subgenre Name", "Sub-Subgenre Name"],
        },
    },
    "music": {
        "genre": {
            "path": "music/input/schemamatching/Music_Genres_Taxonomy.csv",
            "columns": ["Genre Name", "Subgenre Name", "Sub-Subgenre Name"],
        },
    },
}


def _instantiate(
    module_path: str,
    cls_name: str,
    params: dict[str, Any],
    *,
    member_name: str,
    domain: str | None = None,
):
    """Build a member instance, injecting per-domain taxonomies when needed."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    inst_kwargs = dict(params)
    if cls_name == "TaxonomyLookupNormalizer" and domain is not None:
        inst_kwargs["taxonomies"] = {domain: TAXONOMY_BINDINGS.get(domain, {})}
    return cls(name=member_name, **inst_kwargs)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    keys = list(grid)
    out: list[dict[str, Any]] = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        out.append(dict(zip(keys, vals, strict=True)))
    return out


def _sweep_member(
    member_name: str,
    domains: list[str],
    contexts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if member_name not in SPECS:
        raise KeyError(f"Unknown member {member_name!r}; known: {sorted(SPECS)}")
    (module_path, cls_name), init_grid = SPECS[member_name]
    init_combos = _expand_grid(init_grid)
    rows: list[dict[str, Any]] = []
    for init in init_combos:
        per_domain: dict[str, dict[str, float]] = {}
        per_domain_per_attr: dict[str, dict[str, float]] = {}
        for domain in domains:
            ctx = contexts[domain]
            try:
                member = _instantiate(
                    module_path,
                    cls_name,
                    init,
                    member_name=member_name,
                    domain=domain,
                )
            except Exception as e:
                logger.warning(
                    "Failed to instantiate %s with %s on %s: %s",
                    member_name,
                    init,
                    domain,
                    e,
                )
                continue
            t0 = time.monotonic()
            scores = _score_member(member, ctx, member_name=member_name)
            elapsed = time.monotonic() - t0
            metrics = scores.macro_metrics()
            per_domain[domain] = metrics
            per_domain[domain]["runtime_s"] = round(elapsed, 2)
            per_domain_per_attr[domain] = {
                attr: s.f1 for attr, s in scores.by_attribute.items()
            }
        # Aggregate.
        f1_vals = [m["macro_f1"] for m in per_domain.values()]
        if f1_vals:
            mean_f1 = sum(f1_vals) / len(f1_vals)
            min_f1 = min(f1_vals)
        else:
            mean_f1 = 0.0
            min_f1 = 0.0
        rows.append(
            {
                "member": member_name,
                "init": init,
                "mean_f1": round(mean_f1, 4),
                "min_f1": round(min_f1, 4),
                "per_domain": per_domain,
                "per_attr": per_domain_per_attr,
            }
        )
        logger.info(
            "%s init=%s → mean_f1=%.4f (per-domain: %s)",
            member_name,
            init,
            mean_f1,
            {d: round(m["macro_f1"], 3) for d, m in per_domain.items()},
        )
    rows.sort(key=lambda r: (-r["mean_f1"], -r["min_f1"]))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--members",
        default=",".join(SPECS),
        help="Comma-separated member names (default: all).",
    )
    parser.add_argument(
        "--domains",
        default="companies,games,music",
        help="Comma-separated domains.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CACHE_DIR / "sweep.json",
        help="Write rolled-up results here.",
    )
    args = parser.parse_args()

    members = [m.strip() for m in args.members.split(",") if m.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]

    contexts = {d: _domain_context(d) for d in domains}
    for d, ctx in contexts.items():
        logger.info(
            "Domain %s: %d eligible attrs, %d fusion entities",
            d,
            len(ctx["eligible"]),
            len(ctx["fusion_targets"]),
        )

    all_results: dict[str, list[dict[str, Any]]] = {}
    for m in members:
        logger.info("Sweeping member: %s", m)
        rows = _sweep_member(m, domains, contexts)
        all_results[m] = rows
        if rows:
            top = rows[0]
            logger.info(
                "  TOP %s: init=%s mean_f1=%.4f min_f1=%.4f",
                m,
                top["init"],
                top["mean_f1"],
                top["min_f1"],
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Wrote %s", args.out)

    # Print a concise winner table.
    print()
    print(f"{'Member':<22} {'mean_f1':>8} {'min_f1':>8} init")
    print("-" * 80)
    for member, rows in all_results.items():
        if not rows:
            continue
        top = rows[0]
        print(
            f"{member:<22} {top['mean_f1']:>8.4f} {top['min_f1']:>8.4f} "
            f"{json.dumps(top['init'], default=str)}"
        )


if __name__ == "__main__":
    main()
