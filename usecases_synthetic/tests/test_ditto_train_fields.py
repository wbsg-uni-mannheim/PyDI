"""R10-I: ``ditto/train.py`` --domain train-fields wiring.

``_resolve_field_scope`` sources the serialization field list from the
canonical wide committee scope when ``--domain`` is set, overriding any
stale narrow ``fields`` default in ``--config`` and rejecting an explicit
``--fields`` that disagrees — so a baseline (R10-H) retrain can never train
on a narrower surface than wide inference serializes.
"""

from __future__ import annotations

import pytest

from usecases_synthetic.scripts.ditto.prepare_em_training_data import (
    committee_ditto_fields,
)
from usecases_synthetic.scripts.ditto.train import _resolve_field_scope

_NARROW = ["title", "brand", "description", "price", "priceCurrency"]


class TestResolveFieldScope:
    def test_no_domain_passes_fields_through(self) -> None:
        assert _resolve_field_scope(_NARROW, None, cli_fields_given=True) == _NARROW

    def test_domain_overrides_stale_config_fields(self) -> None:
        # --config default carries the narrow legacy list; --domain wins.
        out = _resolve_field_scope(_NARROW, "products", cli_fields_given=False)
        assert out == committee_ditto_fields("products")
        assert "form_factor" in out and len(out) == 19

    def test_domain_drops_reserved_for_music(self) -> None:
        out = _resolve_field_scope(["x"], "music", cli_fields_given=False)
        assert out == committee_ditto_fields("music")
        assert "label" not in out

    def test_explicit_matching_fields_ok(self) -> None:
        canonical = committee_ditto_fields("products")
        assert (
            _resolve_field_scope(list(canonical), "products", cli_fields_given=True)
            == canonical
        )

    def test_explicit_conflicting_fields_raises(self) -> None:
        with pytest.raises(SystemExit, match="does not match the canonical"):
            _resolve_field_scope(_NARROW, "products", cli_fields_given=True)

    def test_alias_domain_resolves(self) -> None:
        assert _resolve_field_scope(
            ["x"], "companies-small", cli_fields_given=False
        ) == committee_ditto_fields("companies")
