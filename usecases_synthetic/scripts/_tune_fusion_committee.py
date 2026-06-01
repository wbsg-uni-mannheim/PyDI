"""Tune the fusion committee per-domain.

R5 Fusion sign-off (plans/plan_s1_scale.md, 2026-05-12).

Each fusion sweep cell uses **perfect-cluster correspondences** as input
(see ``lib/fusion_perfect_clusters.py``) — the cluster ground truth
declared in the fusion validation + test XMLs. This isolates the
"how good is this fusion strategy" signal from "how good is the
upstream EM committee", per the R5 design directive.

Sub-sweeps (each opt-in via ``--sub-sweeps``):

Each sweep cell runs the committee with ``reselect=True`` so val-selection
is recomputed against the cell's mutated candidate params and never reads
or writes the persisted ``fusion_committee_selection.json`` (otherwise the
``(domain, member, attr)`` cache key ignores swept params and freezes
selection — which previously made the ``trim`` sub-sweep inert).

Sub-sweeps (each opt-in via ``--sub-sweeps``):

- **A. trust**: every 3-source permutation of ``trust_scores``.
- **B. tolerance**: per-numeric-attr ``evaluation_params.<attr>.tolerance
  ∈ {0.05, 0.10, 0.15}``. NOTE: numeric tolerances are normally HARD-SET
  from the human-baseline notebook (not swept) — keep this sub-sweep out
  of ``--sub-sweeps`` unless deliberately re-tuning tolerance.
- **C. trim**: ``trimmed_mean.trim ∈ {0.05, 0.10, 0.20, 0.30}``.
- **D. list_threshold**: ``evaluation_params.<list_attr>.threshold ∈
  {0.1, 0.3, 0.5, 0.75, 1.0}`` for keypeople / genres / tracks.
- **E. truthfinder**: ``gamma × init_trust`` (9 cells).
- **F. accusim**: ``accuracy_prior × sim_threshold`` (9 cells).
- **G. casefusion**: ``alpha × lr`` (9 cells).
- **H. fusionquery**: ``temperature × threshold`` (9 cells).
- **I. ltm**: ``alpha_0 × alpha_1`` (6 cells).
- **J. llm_judge**: enabled vs disabled (2 cells). The 'enabled' cell is
  skipped when ``OPENAI_API_KEY`` is unset (it would score ``llm_only``
  0.0 and corrupt the comparison).

Output: ``cache/fusion_tuning/sweep.json`` — each run records ``aggregated``,
``per_attribute`` AND ``per_member``. Winner-pick for a single-member
sub-sweep MUST use that member's ``per_member[<member>].macro_accuracy``,
NOT ``aggregated.overall_accuracy`` (which is the best-member macro and is
invariant to the swept member's params when another member dominates).

Usage::

    python usecases_synthetic/scripts/_tune_fusion_committee.py \\
        --domains companies,games,music \\
        --sub-sweeps trust,tolerance,trim,list_threshold,truthfinder,accusim,casefusion,fusionquery,ltm,llm_judge
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from usecases_synthetic.lib.committee_fusion import FusionCommitteeRunner  # noqa: E402
from usecases_synthetic.lib.committee_paths import resolve_committee_path  # noqa: E402
from usecases_synthetic.lib.fusion_perfect_clusters import (  # noqa: E402
    build_perfect_clusters_correspondences,
)
from usecases_synthetic.lib.variant_loader import load_variant  # noqa: E402

logger = logging.getLogger("tune_fusion")

CACHE_DIR = REPO_ROOT / "cache" / "fusion_tuning"
COMMITTEE_DIR = REPO_ROOT / "usecases_synthetic" / "config" / "committees"

NUMERIC_ATTRS_BY_DOMAIN: dict[str, list[str]] = {
    "companies": ["assets", "revenue"],
    "games": ["criticScore", "userScore"],
    "music": ["duration"],
    # Products' numeric attributes per the products fusion strategy
    # (vram_gb, storage_gb in the minimum-fuser group; price under
    # numeric_tolerance_match; read_speed_mb_s / write_speed_mb_s
    # also numeric per the canonical_schema).
    "products": [
        "price",
        "vram_gb",
        "storage_gb",
        "read_speed_mb_s",
        "write_speed_mb_s",
    ],
}

LIST_ATTRS_BY_DOMAIN: dict[str, list[str]] = {
    "companies": ["keypeople"],
    "games": ["genres"],
    "music": ["tracks"],
    # Products has no list-typed attributes in the canonical schema
    # (all attributes are scalar — categorical or numeric).
    "products": [],
}


# ---------------------------------------------------------------------------
# Generic sweep helpers
# ---------------------------------------------------------------------------


def _load_roster_yaml(domain: str) -> tuple[Path, dict[str, Any]]:
    path = resolve_committee_path(
        "fusion_committee", domain, committee_dir=COMMITTEE_DIR
    )
    with open(path, encoding="utf-8") as f:
        return path, yaml.safe_load(f)


def _write_temp_yaml(base_path: Path, mutated: dict[str, Any], tag: str) -> Path:
    out_dir = CACHE_DIR / "temp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{base_path.stem}__{tag}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(mutated, f, sort_keys=False)
    return out_path


def _score_run(
    yaml_path: Path,
    domain: str,
    bundle: Any,
    correspondences: Any,
) -> dict[str, Any]:
    runner = FusionCommitteeRunner(yaml_path)
    t0 = time.monotonic()
    # reselect=True forces fresh val-selection per sweep cell (the persisted
    # (domain,member,attr) cache otherwise freezes selection and makes
    # candidate-param sweeps like trim inert) and never writes the on-disk
    # cache. See C12FusionCommitteeRunner.run.
    result = runner.run(bundle, correspondences=correspondences, reselect=True)
    elapsed = time.monotonic() - t0
    return {
        "aggregated": dict(result.aggregated),
        "per_attribute": {
            attr: dict(scores) for attr, scores in result.per_attribute.items()
        },
        # per_member macro_accuracy is the correct winner-pick signal for a
        # single-member sub-sweep: aggregated.overall_accuracy is the
        # best-member macro (committee_fusion_c12.py), which is invariant to
        # the swept member's params when another member dominates.
        "per_member": {
            name: dict(mr.metrics) for name, mr in result.per_member.items()
        },
        "runtime_s": elapsed,
    }


# Mapping from per-sub-sweep "method name" (kept from the pre-C12 script
# for backward-compat of the sub-sweep names) to the C12 member name
# in ``members:``. Under C12, each pre-C12 TD strategy is a coherent
# end-to-end member with the ``_only`` suffix (per plan_revision.md §C12).
_METHOD_TO_MEMBER: dict[str, str] = {
    "truthfinder": "truthfinder_only",
    "accusim": "accusim_only",
    "casefusion": "casefusion_only",
    "fusionquery": "fusionquery_only",
    "ltm": "ltm_only",
    "llm_judge": "llm_only",
}


def _mutate_member_params(
    base: dict[str, Any], method_name: str, new_params: dict[str, Any]
) -> dict[str, Any]:
    """Deep-copy ``base`` and override the C12 member's params block.

    Method-name → member-name lookup uses ``_METHOD_TO_MEMBER`` (e.g.
    ``"truthfinder"`` → ``"truthfinder_only"``). Pre-C12 the sweep
    helper targeted per-(attribute, strategy) blocks; the C12 restructure
    (plan_revision.md §C12) collapsed those into per-member coherent
    approaches, so a single sweep cell now overrides one member's
    ``params:`` block.
    """
    member_name = _METHOD_TO_MEMBER.get(method_name, method_name)
    mutated = copy.deepcopy(base)
    members = mutated.get("members") or []
    found = False
    for member in members:
        if member.get("name") == member_name:
            params = member.setdefault("params", {})
            params.update(new_params)
            found = True
    if not found:
        raise ValueError(
            f"_mutate_member_params: member {member_name!r} (from method "
            f"{method_name!r}) not found in YAML — the domain's fusion "
            f"committee may not include this coherent member, or the C12 "
            f"member name mapping in _METHOD_TO_MEMBER is wrong."
        )
    return mutated


def _disable_llm_judge(base: dict[str, Any]) -> dict[str, Any]:
    """Deep-copied YAML with the C12 ``llm_only`` member's
    ``llm_callable`` set to None (effectively disables the LLM judge,
    forcing the deterministic fallback path)."""
    mutated = copy.deepcopy(base)
    for member in mutated.get("members") or []:
        if member.get("name") == "llm_only":
            params = member.setdefault("params", {})
            params["llm_callable"] = None
    return mutated


# ---------------------------------------------------------------------------
# Sub-sweeps
# ---------------------------------------------------------------------------


def _sub_trust(
    domain: str,
    base_yaml: dict[str, Any],
    base_path: Path,
    bundle: Any,
    corres: Any,
) -> list[dict[str, Any]]:
    """Sweep ``trust_scores`` permutations across the domain's sources.

    Permutes the descending integer sequence ``[N, N-1, ..., 1]`` across
    the N sources declared in ``trust_scores``. Pre-C12 the script
    hardcoded ``[3, 2, 1]`` which fails for products' 4-source layout.
    For N=3 the behaviour is unchanged (6 perms); for N=4 it produces
    24 perms (~10s each ≈ 4 min total at cached runtimes).
    """
    sources = list(base_yaml["trust_scores"].keys())
    n_sources = len(sources)
    if n_sources < 2:
        return []
    base_values = list(range(n_sources, 0, -1))  # [N, N-1, ..., 1]
    perms = list(itertools.permutations(base_values))
    results: list[dict[str, Any]] = []
    for perm in perms:
        mutated = copy.deepcopy(base_yaml)
        for source, score in zip(sources, perm, strict=True):
            mutated["trust_scores"][source] = float(score)
        tag = "trust_" + "_".join(
            f"{s}-{score}" for s, score in zip(sources, perm, strict=True)
        )
        yaml_path = _write_temp_yaml(base_path, mutated, tag)
        run = _score_run(yaml_path, domain, bundle, corres)
        run["trust_scores"] = dict(zip(sources, perm, strict=True))
        results.append(run)
    return results


def _sub_tolerance(
    domain: str,
    base_yaml: dict[str, Any],
    base_path: Path,
    bundle: Any,
    corres: Any,
) -> list[dict[str, Any]]:
    attrs = NUMERIC_ATTRS_BY_DOMAIN.get(domain, [])
    if not attrs:
        return []
    results: list[dict[str, Any]] = []
    for tol in [0.05, 0.10, 0.15]:
        mutated = copy.deepcopy(base_yaml)
        ep = mutated.setdefault("evaluation_params", {})
        for attr in attrs:
            ep.setdefault(attr, {})["tolerance"] = tol
        yaml_path = _write_temp_yaml(base_path, mutated, f"tol_{tol:.2f}")
        run = _score_run(yaml_path, domain, bundle, corres)
        run["tolerance"] = tol
        run["affected_attrs"] = list(attrs)
        results.append(run)
    return results


def _sub_trim(
    domain: str,
    base_yaml: dict[str, Any],
    base_path: Path,
    bundle: Any,
    corres: Any,
) -> list[dict[str, Any]]:
    """Sweep ``trimmed_mean.trim`` across {0.05, 0.10, 0.20, 0.30}.

    Under C12, ``trimmed_mean`` is a registered val-selectable candidate
    inside ``pydi_candidates.numeric`` (and ``pydi_candidates.list`` if
    the domain declares one). The val-selection inside coherent members
    (``pydi_per_attribute_optimal`` and TD members' numeric/list
    fallbacks) picks per-attribute winners from these candidates, so
    mutating the ``trim`` param on the registered candidate flows
    through to whichever attributes select it.
    """
    results: list[dict[str, Any]] = []
    for trim in [0.05, 0.10, 0.20, 0.30]:
        mutated = copy.deepcopy(base_yaml)
        touched: list[str] = []
        for type_key in ("numeric", "list"):
            candidates = (mutated.get("pydi_candidates") or {}).get(type_key, [])
            for cand in candidates:
                if cand.get("name") == "trimmed_mean":
                    cand.setdefault("params", {})["trim"] = trim
                    touched.append(f"pydi_candidates.{type_key}/trimmed_mean")
        if not touched:
            # No trimmed_mean candidate registered for this domain → nothing
            # to sweep. Return no rows rather than score a no-op.
            continue
        yaml_path = _write_temp_yaml(base_path, mutated, f"trim_{trim:.2f}")
        run = _score_run(yaml_path, domain, bundle, corres)
        run["trim"] = trim
        run["touched_candidates"] = touched
        results.append(run)
    return results


def _sub_list_threshold(
    domain: str,
    base_yaml: dict[str, Any],
    base_path: Path,
    bundle: Any,
    corres: Any,
) -> list[dict[str, Any]]:
    attrs = LIST_ATTRS_BY_DOMAIN.get(domain, [])
    if not attrs:
        return []
    results: list[dict[str, Any]] = []
    for thr in [0.1, 0.3, 0.5, 0.75, 1.0]:
        mutated = copy.deepcopy(base_yaml)
        ep = mutated.setdefault("evaluation_params", {})
        for attr in attrs:
            ep.setdefault(attr, {})["threshold"] = thr
        yaml_path = _write_temp_yaml(base_path, mutated, f"listthr_{thr:.2f}")
        run = _score_run(yaml_path, domain, bundle, corres)
        run["threshold"] = thr
        run["affected_attrs"] = list(attrs)
        results.append(run)
    return results


def _sub_truthfinder(
    domain: str,
    base_yaml: dict[str, Any],
    base_path: Path,
    bundle: Any,
    corres: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    grid = list(itertools.product([0.1, 0.3, 0.5], [0.8, 0.9, 0.95]))
    for gamma, init_trust in grid:
        mutated = _mutate_member_params(
            base_yaml, "truthfinder", {"gamma": gamma, "init_trust": init_trust}
        )
        yaml_path = _write_temp_yaml(
            base_path, mutated, f"tf_g{gamma:.2f}_t{init_trust:.2f}"
        )
        run = _score_run(yaml_path, domain, bundle, corres)
        run["params"] = {"gamma": gamma, "init_trust": init_trust}
        results.append(run)
    return results


def _sub_accusim(
    domain: str,
    base_yaml: dict[str, Any],
    base_path: Path,
    bundle: Any,
    corres: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    grid = list(itertools.product([0.7, 0.8, 0.9], [0.5, 0.7, 0.85]))
    for ap, st in grid:
        mutated = _mutate_member_params(
            base_yaml, "accusim", {"accuracy_prior": ap, "sim_threshold": st}
        )
        yaml_path = _write_temp_yaml(base_path, mutated, f"as_a{ap:.2f}_s{st:.2f}")
        run = _score_run(yaml_path, domain, bundle, corres)
        run["params"] = {"accuracy_prior": ap, "sim_threshold": st}
        results.append(run)
    return results


def _sub_casefusion(
    domain: str,
    base_yaml: dict[str, Any],
    base_path: Path,
    bundle: Any,
    corres: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    grid = list(itertools.product([1.05, 1.1, 1.2], [0.01, 0.05, 0.1]))
    for alpha, lr in grid:
        mutated = _mutate_member_params(
            base_yaml, "casefusion", {"alpha": alpha, "lr": lr}
        )
        yaml_path = _write_temp_yaml(base_path, mutated, f"cf_a{alpha:.2f}_lr{lr:.2f}")
        run = _score_run(yaml_path, domain, bundle, corres)
        run["params"] = {"alpha": alpha, "lr": lr}
        results.append(run)
    return results


def _sub_fusionquery(
    domain: str,
    base_yaml: dict[str, Any],
    base_path: Path,
    bundle: Any,
    corres: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    grid = list(itertools.product([0.3, 0.5, 0.7], [0.5, 0.7, 0.9]))
    for temp, thr in grid:
        mutated = _mutate_member_params(
            base_yaml, "fusionquery", {"temperature": temp, "threshold": thr}
        )
        yaml_path = _write_temp_yaml(base_path, mutated, f"fq_t{temp:.2f}_thr{thr:.2f}")
        run = _score_run(yaml_path, domain, bundle, corres)
        run["params"] = {"temperature": temp, "threshold": thr}
        results.append(run)
    return results


def _sub_ltm(
    domain: str,
    base_yaml: dict[str, Any],
    base_path: Path,
    bundle: Any,
    corres: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    alpha_0_grid = [[50.0, 10.0], [20.0, 5.0], [30.0, 15.0]]
    alpha_1_grid = [[10.0, 10.0], [5.0, 5.0]]
    for a0 in alpha_0_grid:
        for a1 in alpha_1_grid:
            mutated = _mutate_member_params(
                base_yaml, "ltm", {"alpha_0": a0, "alpha_1": a1}
            )
            tag = f"ltm_a0_{int(a0[0])}-{int(a0[1])}" f"_a1_{int(a1[0])}-{int(a1[1])}"
            yaml_path = _write_temp_yaml(base_path, mutated, tag)
            run = _score_run(yaml_path, domain, bundle, corres)
            run["params"] = {"alpha_0": a0, "alpha_1": a1}
            results.append(run)
    return results


def _sub_llm_judge(
    domain: str,
    base_yaml: dict[str, Any],
    base_path: Path,
    bundle: Any,
    corres: Any,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    # The 'enabled' cell calls the live LLM judge; without an API key the
    # llm_only member raises, is caught, and silently scores 0.0 — which
    # would masquerade as a real enabled-vs-disabled comparison. Skip the
    # enabled cell when no key is present so the comparison isn't corrupted.
    if os.environ.get("OPENAI_API_KEY"):
        enabled_yaml = _write_temp_yaml(
            base_path, copy.deepcopy(base_yaml), "llm_judge_enabled"
        )
        enabled_run = _score_run(enabled_yaml, domain, bundle, corres)
        enabled_run["enabled"] = True
        results.append(enabled_run)
    else:
        logger.warning(
            "llm_judge sub-sweep: OPENAI_API_KEY not set — skipping the "
            "'enabled' cell (it would score llm_only=0.0 and corrupt the "
            "enabled-vs-disabled comparison); running disabled cell only."
        )
    disabled = _disable_llm_judge(base_yaml)
    disabled_yaml = _write_temp_yaml(base_path, disabled, "llm_judge_disabled")
    disabled_run = _score_run(disabled_yaml, domain, bundle, corres)
    disabled_run["enabled"] = False
    results.append(disabled_run)
    return results


SUB_SWEEPS: dict[str, Any] = {
    "trust": _sub_trust,
    "tolerance": _sub_tolerance,
    "trim": _sub_trim,
    "list_threshold": _sub_list_threshold,
    "truthfinder": _sub_truthfinder,
    "accusim": _sub_accusim,
    "casefusion": _sub_casefusion,
    "fusionquery": _sub_fusionquery,
    "ltm": _sub_ltm,
    "llm_judge": _sub_llm_judge,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="companies,games,music")
    parser.add_argument(
        "--sub-sweeps",
        default=",".join(SUB_SWEEPS.keys()),
        help=f"Comma-separated subset of {sorted(SUB_SWEEPS.keys())}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CACHE_DIR / "sweep.json",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    sub_sweeps = [s.strip() for s in args.sub_sweeps.split(",") if s.strip()]
    unknown = set(sub_sweeps) - set(SUB_SWEEPS)
    if unknown:
        raise SystemExit(f"Unknown sub-sweeps: {unknown}")

    out: dict[str, Any] = {
        "domains": domains,
        "sub_sweeps": sub_sweeps,
        "results": {},
    }

    for domain in domains:
        logger.info("=== Domain: %s ===", domain)
        bundle = load_variant(domain, level="baseline")
        corres = build_perfect_clusters_correspondences(domain, bundle)
        base_path, base_yaml = _load_roster_yaml(domain)
        domain_results: dict[str, Any] = {
            "yaml": str(base_path.relative_to(REPO_ROOT)),
            "n_correspondences": int(len(corres)),
        }
        for sub in sub_sweeps:
            t0 = time.monotonic()
            logger.info("--- Sub %s on %s ---", sub, domain)
            domain_results[sub] = SUB_SWEEPS[sub](
                domain, base_yaml, base_path, bundle, corres
            )
            logger.info(
                "Sub %s on %s: %d cells, %.1fs",
                sub,
                domain,
                len(domain_results[sub]),
                time.monotonic() - t0,
            )
        out["results"][domain] = domain_results
        args.output.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        logger.info("checkpoint -> %s", args.output)

    args.output.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info("Sweep complete: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
