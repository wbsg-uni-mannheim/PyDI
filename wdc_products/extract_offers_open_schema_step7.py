#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from langchain_openai import ChatOpenAI

from PyDI.informationextraction import LLMExtractor


def load_api_key(env_path: Path | None) -> str:
    if env_path and env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "OPENAI_API_KEY":
                return value.strip().strip('"').strip("'")
    return os.environ.get("OPENAI_API_KEY", "")


def build_text(offer: dict[str, Any]) -> str:
    title = offer.get("title") or ""
    description = offer.get("description") or ""
    return f"{title}\n{description}".strip()


def parse_extracted(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offers-dir",
        type=Path,
        default=Path("./"),
        help="Directory containing offers_1..offers_4.json.",
    )
    parser.add_argument(
        "--env",
        dest="env_path",
        type=Path,
        default=Path("../.env"),
        help="Path to .env with OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens per extraction response.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Process only the first N offers per file (0 = all).",
    )
    args = parser.parse_args()

    api_key = load_api_key(args.env_path)
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not found in .env or environment.")

    system_prompt = (
        "Extract product attributes from the offer text. "
        "Return only a JSON object. Use flat, snake_case keys. "
        "Include only fields explicitly present; omit unknowns. "
        "Prefer numeric values when possible (e.g., capacity_gb: 512). "
        "If nothing is extractable, return an empty object {}."
    )

    chat = ChatOpenAI(
        model=args.model,
        temperature=0,
        max_tokens=args.max_tokens,
        api_key=api_key,
    )

    extractor = LLMExtractor(
        chat_model=chat,
        schema=None,
        source_column="text",
        system_prompt=system_prompt,
        retries=1,
        debug=False,
    )

    for idx in range(1, 5):
        in_path = args.offers_dir / f"offers_{idx}.json"
        offers = json.loads(in_path.read_text(encoding="utf-8"))
        if args.limit > 0:
            offers = offers[: args.limit]

        rows = []
        for offer in offers:
            row = dict(offer)
            row["text"] = build_text(offer)
            rows.append(row)

        df = pd.DataFrame(rows)
        result_df = extractor.extract(df, source_column="text")

        structured = []
        key_counts: Counter[str] = Counter()
        for _, row in result_df.iterrows():
            extracted = parse_extracted(row.get("extracted"))
            record = {
                k: row.get(k)
                for k in df.columns
                if k != "text"
            }
            record["extracted"] = extracted
            structured.append(record)
            key_counts.update(extracted.keys())

        out_path = args.offers_dir / f"structured_offers_{idx}.json"
        out_path.write_text(
            json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        schema_path = args.offers_dir / f"schema_suggestions_offers_{idx}.json"
        schema_path.write_text(
            json.dumps(key_counts.most_common(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"offers_{idx}: {len(offers)} -> {out_path}")
        print(f"schema suggestions: {schema_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
