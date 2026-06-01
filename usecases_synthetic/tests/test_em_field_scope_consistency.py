"""R10-I: guard the EM PLM-tier field-scope invariant.

The Ditto checkpoint is trained on exactly the fields it serializes at
inference, and the SC-Block encoder likewise. So per domain the three
field lists MUST be byte-identical (same set + order):

    em_matching_committee[_<d>].yaml  ditto_plm.fields
    em_blocking_committee[_<d>].yaml  sc_block.text_cols
    lib/sc_block_train.py             DOMAIN_TEXT_COLS[<d>]

This test catches drift between them (which would silently mis-train or
mis-serialize a checkpoint).
"""

from __future__ import annotations

import yaml

from usecases_synthetic.lib.domain_config import SYNTHETIC_DIR
from usecases_synthetic.lib.sc_block_train import DOMAIN_TEXT_COLS

_DOMAINS = ("products", "music", "games", "companies")


def _committee_path(domain: str, kind: str):
    suffix = "" if domain == "companies" else f"_{domain}"
    return SYNTHETIC_DIR / "config" / "committees" / f"em_{kind}_committee{suffix}.yaml"


def _member_param(domain: str, kind: str, member_name: str, param: str):
    raw = yaml.safe_load(_committee_path(domain, kind).read_text(encoding="utf-8"))
    sub = "matcher" if kind == "matching" else "blocker"
    for member in raw["members"]:
        if member["name"] == member_name:
            return member[sub]["params"][param]
    raise AssertionError(f"{member_name} not found in em_{kind}_committee for {domain}")


class TestEmFieldScopeConsistency:
    def test_ditto_eq_scblock_eq_domain_text_cols(self) -> None:
        for domain in _DOMAINS:
            ditto = _member_param(domain, "matching", "ditto_plm", "fields")
            scblock = _member_param(domain, "blocking", "sc_block", "text_cols")
            dtc = DOMAIN_TEXT_COLS[domain]
            assert ditto == scblock, (domain, "ditto != sc_block", ditto, scblock)
            assert ditto == dtc, (domain, "ditto != DOMAIN_TEXT_COLS", ditto, dtc)

    def test_llm_and_comem_match_ditto_scope(self) -> None:
        """llm_matcher + comem (zero-shot) widen to the same full scope as
        ditto_plm under R10-I (no narrower curated subset remains)."""
        for domain in _DOMAINS:
            ditto = set(_member_param(domain, "matching", "ditto_plm", "fields"))
            for member in ("llm_matcher", "comem"):
                fields = set(_member_param(domain, "matching", member, "fields"))
                assert fields == ditto, (domain, member, fields ^ ditto)


class TestDittoSerializationScope:
    """R10-I: Ditto's *serialization* scope = committee fields minus any
    name that collides with a Ditto WDC reserved metadata key, applied
    identically on the training side (``committee_ditto_fields`` /
    ``wdc_to_pair_examples``) and the inference side (``DittoMatcher``)."""

    def test_committee_ditto_fields_eq_yaml_minus_reserved(self) -> None:
        from usecases_synthetic.scripts.ditto.prepare_em_training_data import (
            committee_ditto_fields,
        )
        from usecases_synthetic.third_party.ditto_modern.data import (
            RESERVED_SERIALIZATION_FIELDS,
        )

        for domain in _DOMAINS:
            yaml_fields = _member_param(domain, "matching", "ditto_plm", "fields")
            fields = committee_ditto_fields(domain)
            expected = [
                f for f in yaml_fields if f not in RESERVED_SERIALIZATION_FIELDS
            ]
            assert fields == expected, (domain, fields, expected)
            assert not any(f in RESERVED_SERIALIZATION_FIELDS for f in fields)

    def test_music_label_is_the_only_dropped_field(self) -> None:
        from usecases_synthetic.scripts.ditto.prepare_em_training_data import (
            committee_ditto_fields,
        )

        assert "label" in DOMAIN_TEXT_COLS["music"]
        assert "label" not in committee_ditto_fields("music")
        for domain in ("products", "games", "companies"):
            assert committee_ditto_fields(domain) == DOMAIN_TEXT_COLS[domain]

    def test_ditto_matcher_drops_reserved_at_construction(self) -> None:
        from usecases_synthetic.lib.ditto_matcher import DittoMatcher

        m = DittoMatcher(
            checkpoint_path="/tmp/nonexistent-ckpt",
            fields=["name", "label", "artist", "tracks"],
            cache_dir=False,
        )
        assert m.fields == ["name", "artist", "tracks"]

    def test_matcher_and_builder_agree_on_music_scope(self) -> None:
        """The inference matcher (fed the verbatim YAML list) and the
        training builder land on the same field set for music."""
        from usecases_synthetic.lib.ditto_matcher import DittoMatcher
        from usecases_synthetic.scripts.ditto.prepare_em_training_data import (
            committee_ditto_fields,
        )

        yaml_fields = _member_param("music", "matching", "ditto_plm", "fields")
        matcher = DittoMatcher(
            checkpoint_path="/tmp/nonexistent-ckpt",
            fields=yaml_fields,
            cache_dir=False,
        )
        assert matcher.fields == committee_ditto_fields("music")
