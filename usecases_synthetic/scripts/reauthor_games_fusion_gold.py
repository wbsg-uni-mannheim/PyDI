"""Re-author the games fusion validation_set.xml + test_set.xml.

The pre-refresh games fusion gold has only 2/25 overlap with EM-gold positives
(per K10 sign-off follow-up #1 + R5 Fusion baseline measurement on 2026-05-12),
so the fusion engine cannot form multi-source record groups for any of the 25
fusion gold IDs and the fusion evaluation collapses to 0.0 across every member.

This script rebuilds the fusion val + test from EM-positive metacritic IDs.
The selection strategy: take metacritic IDs that participate in EM-positive
clusters spanning at least 2 source pairs (so the fusion engine has cross-
source candidates to fuse). Author the gold record from metacritic's
attribute values (it is the trust-2 curated authority on game metadata) +
sales' publisher (sales is publisher-of-record). DBpedia-only attributes
(``series``) are intentionally omitted from the gold — the fusion engine
still emits ``series`` but evaluation skips it (mirrors the K10 eligible-
attribute filter).

Splits: 100 validation, 25 test (mirrors the existing split shape so
downstream consumers don't shift).

Run once after the source refresh. Output:
  usecases/games/input/fusion/validation_set.xml  (overwritten)
  usecases/games/input/fusion/test_set.xml        (overwritten)

The previous XML files are preserved at ``.old.xml`` for diff.
"""

from __future__ import annotations

import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import pandas as pd

# Repo paths.
REPO_ROOT = Path(__file__).resolve().parents[2]
GAMES_DIR = REPO_ROOT / "usecases" / "games"
EM_DIR = GAMES_DIR / "input" / "entitymatching"
FUSION_DIR = GAMES_DIR / "input" / "fusion"
DATA_DIR = GAMES_DIR / "input" / "data"

SEED = 42
N_VAL = 100
N_TEST = 25


def _load_em_positives() -> dict[str, set[str]]:
    """Walk every CSV in EM_DIR (+ train_test/) and return `{metacritic_id: {partner_id, ...}}` for label-True rows."""
    partners: dict[str, set[str]] = {}

    def _walk(base: Path) -> None:
        for p in base.iterdir():
            if p.is_dir():
                continue
            if p.suffix != ".csv":
                continue
            try:
                df = pd.read_csv(p, header=None, names=["id1", "id2", "label"])
            except Exception:
                continue
            pos = df[
                df["label"].astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])
            ]
            for _, row in pos.iterrows():
                a, b = str(row["id1"]), str(row["id2"])
                if a.startswith("metacritic_"):
                    partners.setdefault(a, set()).add(b)
                elif b.startswith("metacritic_"):
                    partners.setdefault(b, set()).add(a)

    _walk(EM_DIR)
    tt = EM_DIR / "train_test"
    if tt.is_dir():
        _walk(tt)

    return partners


def _format_year(launch: str | float | None) -> str | None:
    if launch is None or (isinstance(launch, float) and pd.isna(launch)):
        return None
    s = str(launch).strip()
    if not s:
        return None
    m = re.match(r"(\d{4})", s)
    if not m:
        return None
    return f"{m.group(1)}-01-01"


def _split_genres(raw: str | float | None) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for tok in s.split(","):
        t = tok.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _coalesce_publisher(partners: set[str], sales_by_id: dict[str, dict]) -> str | None:
    for partner in partners:
        if not partner.startswith("sales_"):
            continue
        row = sales_by_id.get(partner)
        if row is None:
            continue
        pub = row.get("dist")
        if pub is None or (isinstance(pub, float) and pd.isna(pub)):
            continue
        s = str(pub).strip()
        if s:
            return s
    return None


def _build_record(
    metacritic_id: str,
    mc_row: dict,
    partners: set[str],
    sales_by_id: dict[str, dict],
) -> ET.Element:
    el = ET.Element("videogame")

    def _add(tag: str, value: object) -> None:
        if value is None:
            return
        if isinstance(value, float) and pd.isna(value):
            return
        text = str(value).strip()
        if not text:
            return
        ET.SubElement(el, tag).text = text

    _add("id", metacritic_id)
    _add("name", mc_row.get("game_title"))
    _add("releaseYear", _format_year(mc_row.get("year_published")))
    _add("developer", mc_row.get("made_by"))

    genres = _split_genres(mc_row.get("genres"))
    if genres:
        genres_el = ET.SubElement(el, "genres")
        for g in genres:
            ET.SubElement(genres_el, "genre").text = g

    publisher = _coalesce_publisher(partners, sales_by_id)
    if publisher:
        _add("publisher", publisher)

    _add("platform", mc_row.get("console"))

    press = mc_row.get("press_rating")
    if press is not None and not (isinstance(press, float) and pd.isna(press)):
        try:
            _add("criticScore", str(int(float(press))))
        except (ValueError, TypeError):
            pass

    user = mc_row.get("player_rating")
    if user is not None and not (isinstance(user, float) and pd.isna(user)):
        try:
            _add("userScore", f"{float(user):.1f}")
        except (ValueError, TypeError):
            pass

    _add("ESRB", mc_row.get("age_rating"))

    return el


def _serialise(root: ET.Element) -> str:
    rough = ET.tostring(root, encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="    ")
    # minidom adds <?xml ... ?> on the first line — drop it to match the
    # legacy on-disk shape, which starts directly with <videogames>.
    lines = [line for line in pretty.splitlines() if line.strip()]
    if lines and lines[0].startswith("<?xml"):
        lines = lines[1:]
    return "\n".join(lines) + "\n"


def main() -> int:
    rng = random.Random(SEED)

    # Load EM positives + filter to multi-pair-participating metacritic ids.
    partners = _load_em_positives()
    eligible = sorted(
        mc_id
        for mc_id, parts in partners.items()
        if any(p.startswith("dbpedia_") for p in parts)
        and any(p.startswith("sales_") for p in parts)
    )
    if len(eligible) < N_VAL + N_TEST:
        # Fallback to any metacritic id with >= 1 partner across either pair.
        eligible = sorted(partners.keys())

    sys.stderr.write(
        f"Eligible metacritic ids (multi-pair-participating preferred): {len(eligible)}\n"
    )

    rng.shuffle(eligible)
    val_ids = eligible[:N_VAL]
    test_ids = eligible[N_VAL : N_VAL + N_TEST]

    mc_df = pd.read_csv(DATA_DIR / "metacritic.csv")
    mc_by_id = {row["mc_id"]: row.to_dict() for _, row in mc_df.iterrows()}

    sales_df = pd.read_csv(DATA_DIR / "sales.csv")
    sales_by_id = {row["rec_id"]: row.to_dict() for _, row in sales_df.iterrows()}

    for split_name, ids, out_name in (
        ("val", val_ids, "validation_set.xml"),
        ("test", test_ids, "test_set.xml"),
    ):
        root = ET.Element("videogames")
        for mc_id in ids:
            row = mc_by_id.get(mc_id)
            if row is None:
                continue
            root.append(
                _build_record(mc_id, row, partners.get(mc_id, set()), sales_by_id)
            )

        out_path = FUSION_DIR / out_name
        backup = FUSION_DIR / out_name.replace(".xml", ".old.xml")
        if out_path.exists() and not backup.exists():
            shutil.copy(out_path, backup)

        out_path.write_text(_serialise(root), encoding="utf-8")
        sys.stderr.write(f"  wrote {len(ids)} ids to {out_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
