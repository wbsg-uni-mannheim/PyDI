#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from usecases_synthetic.third_party.ditto_modern.data import (
    examples_to_ditto_lines,
    load_wdc_json_gz,
    wdc_to_pair_examples,
)

DEFAULT_FIELDS = ["title", "brand", "description", "price", "priceCurrency"]


def parse_fields(raw: str):
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert WDC json.gz to Ditto text format"
    )
    parser.add_argument("--input-json-gz", required=True)
    parser.add_argument("--output-txt", required=True)
    parser.add_argument("--max-field-len", type=int, default=350)
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    args = parser.parse_args()

    fields = parse_fields(args.fields)
    if not fields:
        raise ValueError("At least one field is required")

    df = load_wdc_json_gz(args.input_json_gz)
    examples = wdc_to_pair_examples(df, fields=fields, max_field_len=args.max_field_len)
    lines = examples_to_ditto_lines(examples)

    out = Path(args.output_txt)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Wrote {len(lines)} lines -> {out}")


if __name__ == "__main__":
    main()
