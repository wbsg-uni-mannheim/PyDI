"""Tests for the M9 per-knob ablation runner + analyzer.

Covers:

- :mod:`usecases_synthetic.scripts.run_ablation_validation` — plumbing,
  skip-existing short-circuit, and per-knob metrics placement, all
  against stubbed ``generate_variant`` / ``validate_variant`` runners.
- :mod:`usecases_synthetic.scripts.analyze_ablation` — per-knob signal
  evaluation, direction-match logic, and the four interaction flags
  (cross-stage leakage, primary under-signal, primary over-signal,
  direction mismatch).

The analyzer tests build a minimal ``knob_expected_signals.yaml`` and
canned per-knob ``metrics.json`` payloads under ``tmp_path``. This keeps
the tests independent of the real validation pipeline while still
exercising the full analyzer code path including the markdown / CSV
writers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from usecases_synthetic.scripts import analyze_ablation as aa
from usecases_synthetic.scripts import run_ablation_validation as rav

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` as JSON, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _stage(**aggregated: float) -> dict[str, Any]:
    """Shape one ``per_stage[<stage>]`` block from an aggregated dict."""
    return {"aggregated": dict(aggregated), "per_member": {}}


def _metrics(
    *,
    sm: dict[str, float] | None = None,
    em: dict[str, float] | None = None,
    fusion: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compose a minimal validation metrics payload."""
    per_stage: dict[str, Any] = {}
    if sm is not None:
        per_stage["sm"] = _stage(**sm)
    if em is not None:
        per_stage["em"] = _stage(**em)
    if fusion is not None:
        per_stage["fusion"] = _stage(**fusion)
    return {"per_stage": per_stage, "meta": {"with_llm": False}}


def _expectations_yaml(path: Path) -> None:
    """Write a tiny two-knob expectations YAML for the analyzer tests."""
    yaml_body = """
schema_version: 1
target_domain: companies
knob_08:
  label: Schema naming
  source_card: knobs/knob_08_schema_naming.md
  primary_stage: sm
  signals:
    - id: sm_drop
      stage: sm
      metric: aggregated.macro_f1
      direction: down
      qualitative_only: true
      target_delta_range: null
      pool_check: false
      notes: Primary target — SM macro F1 should drop.
    - id: em_flat
      stage: em
      metric: aggregated.macro_f1
      direction: flat
      qualitative_only: true
      target_delta_range: null
      pool_check: false
      notes: EM unaffected by K8.
    - id: fusion_flat
      stage: fusion
      metric: aggregated.overall_accuracy
      direction: flat
      qualitative_only: true
      target_delta_range: null
      pool_check: false
      notes: Fusion unaffected by K8.
knob_10:
  label: Source reliability
  source_card: knobs/knob_10_source_reliability.md
  primary_stage: fusion
  signals:
    - id: sm_flat
      stage: sm
      metric: aggregated.macro_f1
      direction: flat
      qualitative_only: true
      target_delta_range: null
      pool_check: false
      notes: SM unaffected by K10.
    - id: fusion_drop
      stage: fusion
      metric: aggregated.overall_accuracy
      direction: down
      qualitative_only: true
      target_delta_range: null
      pool_check: false
      notes: Primary target.
"""
    path.write_text(yaml_body, encoding="utf-8")


def _seed_analyzer_world(tmp_path: Path) -> dict[str, Path]:
    """Populate a fake ``usecases_synthetic`` tree for the analyzer.

    Returns
    -------
    dict[str, Path]
        Paths: ``expectations``, ``baseline``, ``hard``,
        ``ablation_dir``, ``out_dir``.
    """
    syn = tmp_path / "usecases_synthetic"
    cfg = syn / "config"
    cfg.mkdir(parents=True)
    expectations = cfg / "knob_expected_signals.yaml"
    _expectations_yaml(expectations)

    baseline = syn / "baselines" / "companies" / "baseline_metrics.json"
    _write_json(
        baseline,
        _metrics(
            sm={"macro_f1": 0.80},
            em={"macro_f1": 0.70},
            fusion={"overall_accuracy": 0.85},
        ),
    )

    hard = syn / "validation" / "companies" / "hard" / "metrics.json"
    _write_json(
        hard,
        _metrics(
            sm={"macro_f1": 0.40},
            em={"macro_f1": 0.40},
            fusion={"overall_accuracy": 0.50},
        ),
    )

    ablation_dir = syn / "validation" / "companies" / "ablation"
    out_dir = ablation_dir
    return {
        "expectations": expectations,
        "baseline": baseline,
        "hard": hard,
        "ablation_dir": ablation_dir,
        "out_dir": out_dir,
    }


def _patch_analyzer_paths(
    monkeypatch: pytest.MonkeyPatch,
    world: dict[str, Path],
) -> None:
    """Redirect analyzer module-level paths to the temp world."""
    monkeypatch.setattr(aa, "VALIDATION_DIR", world["hard"].parents[2])
    # baseline_path is imported into aa from baseline_loader; patch the
    # bound import.
    monkeypatch.setattr(aa, "baseline_path", lambda domain: world["baseline"])


# ---------------------------------------------------------------------------
# Analyzer unit tests
# ---------------------------------------------------------------------------


class TestAnalyzeAblation:
    """Signal evaluation + flag classification tests."""

    def test_clean_ablations_produce_no_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """K8 ablation: SM drops, EM/Fusion flat -> no flags, direction ok."""
        world = _seed_analyzer_world(tmp_path)
        _patch_analyzer_paths(monkeypatch, world)

        # K8 ablation: big SM drop, EM and Fusion steady.
        _write_json(
            world["ablation_dir"] / "knob_08" / "metrics.json",
            _metrics(
                sm={"macro_f1": 0.45},
                em={"macro_f1": 0.69},
                fusion={"overall_accuracy": 0.84},
            ),
        )
        # K10 ablation: big Fusion drop, SM steady.
        _write_json(
            world["ablation_dir"] / "knob_10" / "metrics.json",
            _metrics(
                sm={"macro_f1": 0.79},
                fusion={"overall_accuracy": 0.55},
            ),
        )

        result = aa.analyze_domain(
            "companies",
            expectations_path=world["expectations"],
            out_dir=world["out_dir"],
        )
        by_knob = {r.knob: r for r in result["results"]}
        assert set(by_knob) == {"knob_08", "knob_10"}

        k8 = by_knob["knob_08"]
        k10 = by_knob["knob_10"]

        # Every signal is direction-correct and flag-free.
        for result_obj in (k8, k10):
            for sig in result_obj.signals:
                assert sig.direction_match, (
                    f"{result_obj.knob} {sig.expectation.signal_id} direction failed: "
                    f"{sig.reason}"
                )
                assert sig.flags == [], (
                    f"{result_obj.knob} {sig.expectation.signal_id} unexpected "
                    f"flags: {sig.flags}"
                )

        # Primary deltas are dominant.
        k8_primary = k8.primary_signals()[0]
        assert k8_primary.delta_vs_baseline < -0.3
        k10_primary = k10.primary_signals()[0]
        assert k10_primary.delta_vs_baseline < -0.2

        # Report files materialised.
        assert result["report_md"].exists()
        assert result["report_csv"].exists()
        md = result["report_md"].read_text(encoding="utf-8")
        assert "knob_08" in md and "knob_10" in md

    def test_cross_stage_leakage_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """K8 ablation that moves Fusion > flat tolerance triggers leakage."""
        world = _seed_analyzer_world(tmp_path)
        _patch_analyzer_paths(monkeypatch, world)

        # K8: SM drops correctly, but Fusion drops by 0.3 (leakage).
        _write_json(
            world["ablation_dir"] / "knob_08" / "metrics.json",
            _metrics(
                sm={"macro_f1": 0.45},
                em={"macro_f1": 0.69},
                fusion={"overall_accuracy": 0.55},  # -0.30 vs baseline
            ),
        )

        result = aa.analyze_domain(
            "companies",
            knobs=["knob_08"],
            expectations_path=world["expectations"],
            out_dir=world["out_dir"],
        )
        k8 = result["results"][0]
        assert k8.any_flag("cross_stage_leakage")
        # Leakage is on the fusion signal, not the em signal.
        fusion_sig = [
            s for s in k8.signals if s.expectation.signal_id == "fusion_flat"
        ][0]
        em_sig = [s for s in k8.signals if s.expectation.signal_id == "em_flat"][0]
        assert "cross_stage_leakage" in fusion_sig.flags
        assert "cross_stage_leakage" not in em_sig.flags

    def test_under_signal_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Small primary-stage delta vs full-hard triggers under-signal."""
        world = _seed_analyzer_world(tmp_path)
        _patch_analyzer_paths(monkeypatch, world)

        # K8 ablation: SM drops only 0.05 vs baseline (0.80 -> 0.75), while
        # the full-hard SM dropped 0.40. 0.05 / 0.40 = 0.125 < under_ratio.
        _write_json(
            world["ablation_dir"] / "knob_08" / "metrics.json",
            _metrics(
                sm={"macro_f1": 0.75},
                em={"macro_f1": 0.69},
                fusion={"overall_accuracy": 0.84},
            ),
        )
        result = aa.analyze_domain(
            "companies",
            knobs=["knob_08"],
            expectations_path=world["expectations"],
            out_dir=world["out_dir"],
        )
        k8 = result["results"][0]
        assert k8.any_flag("primary_under_signal")
        # No leakage, no over-signal, no mismatch.
        assert not k8.any_flag("cross_stage_leakage")
        assert not k8.any_flag("primary_over_signal")
        assert not k8.any_flag("direction_mismatch")

    def test_over_signal_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ablation primary delta > full-hard delta triggers over-signal."""
        world = _seed_analyzer_world(tmp_path)
        _patch_analyzer_paths(monkeypatch, world)

        # Full-hard SM drop is 0.40. Ablation SM drop is 0.60 -> over-signal.
        _write_json(
            world["ablation_dir"] / "knob_08" / "metrics.json",
            _metrics(
                sm={"macro_f1": 0.20},
                em={"macro_f1": 0.69},
                fusion={"overall_accuracy": 0.84},
            ),
        )
        result = aa.analyze_domain(
            "companies",
            knobs=["knob_08"],
            expectations_path=world["expectations"],
            out_dir=world["out_dir"],
        )
        k8 = result["results"][0]
        assert k8.any_flag("primary_over_signal")

    def test_direction_mismatch_flagged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Primary SM *rising* when predicted 'down' is a mismatch."""
        world = _seed_analyzer_world(tmp_path)
        _patch_analyzer_paths(monkeypatch, world)

        _write_json(
            world["ablation_dir"] / "knob_08" / "metrics.json",
            _metrics(
                sm={"macro_f1": 0.95},  # went UP vs 0.80 baseline
                em={"macro_f1": 0.69},
                fusion={"overall_accuracy": 0.84},
            ),
        )
        result = aa.analyze_domain(
            "companies",
            knobs=["knob_08"],
            expectations_path=world["expectations"],
            out_dir=world["out_dir"],
        )
        k8 = result["results"][0]
        assert k8.any_flag("direction_mismatch")
        sm_sig = [s for s in k8.signals if s.expectation.signal_id == "sm_drop"][0]
        assert not sm_sig.direction_match

    def test_missing_metrics_skip_knob(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A knob without metrics.json is recorded under ``missing``."""
        world = _seed_analyzer_world(tmp_path)
        _patch_analyzer_paths(monkeypatch, world)

        # Only write K8; K10 metrics absent.
        _write_json(
            world["ablation_dir"] / "knob_08" / "metrics.json",
            _metrics(
                sm={"macro_f1": 0.45},
                em={"macro_f1": 0.69},
                fusion={"overall_accuracy": 0.84},
            ),
        )
        result = aa.analyze_domain(
            "companies",
            expectations_path=world["expectations"],
            out_dir=world["out_dir"],
        )
        assert [r.knob for r in result["results"]] == ["knob_08"]
        assert result["missing"] == ["knob_10"]

    def test_csv_has_one_row_per_signal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CSV layout: one row per (knob, signal) with the expected columns."""
        world = _seed_analyzer_world(tmp_path)
        _patch_analyzer_paths(monkeypatch, world)

        _write_json(
            world["ablation_dir"] / "knob_08" / "metrics.json",
            _metrics(
                sm={"macro_f1": 0.45},
                em={"macro_f1": 0.69},
                fusion={"overall_accuracy": 0.84},
            ),
        )
        _write_json(
            world["ablation_dir"] / "knob_10" / "metrics.json",
            _metrics(
                sm={"macro_f1": 0.79},
                fusion={"overall_accuracy": 0.55},
            ),
        )

        result = aa.analyze_domain(
            "companies",
            expectations_path=world["expectations"],
            out_dir=world["out_dir"],
        )
        rows = list(csv.DictReader(result["report_csv"].open(encoding="utf-8")))
        # K8 has 3 signals, K10 has 2 signals -> 5 rows.
        assert len(rows) == 5
        first = rows[0]
        for field in (
            "knob",
            "signal_id",
            "stage",
            "is_primary",
            "delta_vs_baseline",
            "flags",
        ):
            assert field in first


# ---------------------------------------------------------------------------
# Runner plumbing tests
# ---------------------------------------------------------------------------


class TestRunAblationValidation:
    """Verify run_ablation uses the ablation label, writes metrics, and
    honours skip-existing."""

    def _patch_runners(
        self,
        monkeypatch: pytest.MonkeyPatch,
        generated: list[tuple[str, dict[str, Any]]],
        validated: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Replace the heavy generate/validate calls with light stubs."""

        def _fake_generate(
            *,
            domain: str,
            level: str,
            master_seed: int | None,
            knob_levels: dict[str, str],
            label: str,
        ) -> dict[str, Any]:
            generated.append(
                (
                    domain,
                    {
                        "level": level,
                        "knob_levels": dict(knob_levels),
                        "label": label,
                        "master_seed": master_seed,
                    },
                )
            )
            # Create a fake packaged variant so skip-existing can see it.
            variant_dir = rav.ablation_variant_dir(domain, _target_knob(knob_levels))
            (variant_dir / "input" / "data").mkdir(parents=True, exist_ok=True)
            return {"variant_dir": variant_dir}

        def _fake_validate(
            *,
            domain: str,
            level: str,
            stages: Any,
            with_llm: bool,
            fusion_input_member: Any,
            out_dir: Path,
            variant_root: Path,
        ) -> dict[str, Any]:
            out_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = out_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps({"per_stage": {"sm": {"aggregated": {"macro_f1": 0.5}}}}),
                encoding="utf-8",
            )
            validated.append(
                (
                    domain,
                    {
                        "level": level,
                        "out_dir": str(out_dir),
                        "variant_root": str(variant_root),
                    },
                )
            )
            return {"metrics_path": metrics_path}

        monkeypatch.setattr(rav, "generate_variant", _fake_generate)
        monkeypatch.setattr(rav, "validate_variant", _fake_validate)

    def test_run_ablation_writes_metrics_and_uses_label(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """run_ablation uses ablation_label for the variant dir and writes metrics."""
        generated: list[tuple[str, dict[str, Any]]] = []
        validated: list[tuple[str, dict[str, Any]]] = []
        self._patch_runners(monkeypatch, generated, validated)

        # Redirect outputs into tmp_path.
        monkeypatch.setattr(rav, "ABLATION_VALIDATION_ROOT", tmp_path / "validation")
        monkeypatch.setattr(rav, "USECASES_DIR", tmp_path / "usecases")

        result = rav.run_ablation("companies", "knob_08")
        assert result["status"] if "status" in result else True
        assert result["generated"] is True

        metrics_dir = rav.ablation_metrics_dir("companies", "knob_08")
        assert (metrics_dir / "metrics.json").exists()

        # Generation was invoked with the ablation label and the right
        # per-knob level map.
        assert len(generated) == 1
        _, call = generated[0]
        assert call["label"] == "ablation_knob_08"
        assert call["level"] == "hard"
        assert call["knob_levels"]["knob_08"] == "hard"
        assert call["knob_levels"]["knob_01"] == "easy"

        # Validation was invoked pointing at the ablation variant root.
        assert len(validated) == 1
        _, vcall = validated[0]
        assert vcall["level"] == "hard"
        assert "ablation_knob_08" in vcall["variant_root"]

    def test_skip_existing_short_circuits_generation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """skip-existing=True reuses a packaged variant on rerun."""
        generated: list[tuple[str, dict[str, Any]]] = []
        validated: list[tuple[str, dict[str, Any]]] = []
        self._patch_runners(monkeypatch, generated, validated)
        monkeypatch.setattr(rav, "ABLATION_VALIDATION_ROOT", tmp_path / "validation")
        monkeypatch.setattr(rav, "USECASES_DIR", tmp_path / "usecases")

        # Pre-create the variant directory so skip-existing fires.
        variant_dir = rav.ablation_variant_dir("companies", "knob_10")
        (variant_dir / "input" / "data").mkdir(parents=True, exist_ok=True)

        result = rav.run_ablation("companies", "knob_10", skip_existing_variant=True)
        assert result["generated"] is False
        assert generated == []  # Generator not invoked.
        assert len(validated) == 1  # Validator still runs.


def _target_knob(knob_levels: dict[str, str]) -> str:
    """Return the single knob set to 'hard' in a build_ablation_knob_levels map."""
    hard = [k for k, v in knob_levels.items() if v == "hard"]
    if len(hard) != 1:
        raise AssertionError(f"Expected exactly one hard knob: {hard}")
    return hard[0]
