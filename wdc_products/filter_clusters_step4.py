#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


def build_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


GENERAL_NEGATIVE = build_patterns(
    [
        r"\blaptop\b",
        r"\bnotebook\b",
        r"\bultrabook\b",
        r"\bmacbook\b",
        r"\bzenbook\b",
        r"\bchromebook\b",
        r"\btablet\b",
        r"\bsmartphone\b",
        r"\bmobile phone\b",
        r"\bphone\b",
        r"\bmotherboard\b",
        r"\bmainboard\b",
        r"\bcase\b",
        r"\bmid-?tower\b",
        r"\bcabinet\b",
        r"\bpsu\b",
        r"\bpower supply\b",
        r"\bbracket\b",
        r"\bmounting\b",
        r"\badapter\b",
        r"\benclosure\b",
        r"\bdock\b",
        r"\bcd-?key\b",
        r"\blicense\b",
        r"\bsoftware\b",
        r"\bwindows\b",
    ]
)

CATEGORY_RULES: dict[str, dict[str, Any]] = {
    "hdd": {
        "positive": build_patterns(
            [
                r"\bhdd\b",
                r"\bhard drive\b",
                r"\bhard disk\b",
                r"\b3\.5\s*\"\b",
                r"\b2\.5\s*\"\b",
                r"\bsata\b",
                r"\b7200\s*rpm\b",
                r"\b5400\s*rpm\b",
            ]
        ),
        "negative": build_patterns([r"\benclosure\b", r"\bcaddy\b"]),
        "weights": [4, 4, 4, 2, 2, 1, 2, 2],
    },
    "ssd": {
        "positive": build_patterns(
            [
                r"\bssd\b",
                r"\bsolid state\b",
                r"\bnvme\b",
                r"\bm\.2\b",
                r"\bpcie\b",
            ]
        ),
        "negative": build_patterns(
            [r"\bbracket\b", r"\bmounting\b", r"\badapter\b", r"\benclosure\b"]
        ),
        "weights": [4, 4, 3, 3, 1],
    },
    "sticks": {
        "positive": build_patterns(
            [
                r"\bflash drive\b",
                r"\bthumb drive\b",
                r"\bpen drive\b",
                r"\busb drive\b",
                r"\bdata\s*traveler\b",
                r"\bmemory stick\b",
                r"\busb stick\b",
            ]
        ),
        "negative": build_patterns([r"\breader\b", r"\bcard reader\b"]),
        "weights": [4, 4, 4, 3, 3, 2, 2],
    },
    "gpu": {
        "positive": build_patterns(
            [
                r"\bgpu\b",
                r"\bgraphics card\b",
                r"\bvideo card\b",
                r"\bgeforce\b",
                r"\brtx\b",
                r"\bgtx\b",
                r"\bradeon\b",
            ]
        ),
        "negative": build_patterns(
            [r"\bprocessor\b", r"\bcpu\b", r"\bapu\b", r"\bathlon\b", r"\bryzen\b"]
        ),
        "weights": [4, 4, 4, 3, 3, 3, 2],
    },
}


def text_blob(offer: dict[str, Any]) -> str:
    title = offer.get("title") or ""
    description = offer.get("description") or ""
    return f"{title} {description}".strip()


def has_any(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def score_offer(offer: dict[str, Any]) -> tuple[int, str]:
    text = text_blob(offer).lower()
    if not text:
        return 0, ""
    if has_any(GENERAL_NEGATIVE, text):
        return 0, ""

    best_score = 0
    best_cat = ""
    for cat, rules in CATEGORY_RULES.items():
        if has_any(rules["negative"], text):
            continue
        score = 0
        for pattern, weight in zip(rules["positive"], rules["weights"]):
            if pattern.search(text):
                score += weight
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_score, best_cat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=Path("clusters_filtered_1.json"),
        help="Input clusters JSON file.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        default=Path("clusters_filtered_2.json"),
        help="Filtered clusters JSON output.",
    )
    parser.add_argument(
        "--rejected-out",
        dest="rejected_path",
        type=Path,
        default=Path("clusters_filtered_2_rejected.json"),
        help="Rejected clusters JSON output.",
    )
    parser.add_argument(
        "--min-positive",
        type=int,
        default=2,
        help="Minimum number of positive offers required to keep a cluster.",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.5,
        help="Minimum ratio of positive offers in a cluster.",
    )
    parser.add_argument(
        "--max-offers",
        type=int,
        default=4,
        help="Keep at most this many strongest offers per cluster.",
    )
    args = parser.parse_args()

    clusters = json.loads(args.input_path.read_text(encoding="utf-8"))
    kept_clusters = []
    rejected_clusters = []

    for cluster in clusters:
        offers = cluster.get("offers", [])
        scored = []
        for offer in offers:
            score, cat = score_offer(offer)
            if score > 0:
                scored.append((score, cat, offer))

        positive_count = len(scored)
        ratio = positive_count / max(len(offers), 1)

        if positive_count < args.min_positive or ratio < args.min_ratio:
            rejected_clusters.append(cluster)
            continue

        scored.sort(key=lambda item: item[0], reverse=True)
        if args.max_offers > 0:
            scored = scored[: args.max_offers]

        kept_clusters.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "offers": [item[2] for item in scored],
            }
        )

    args.output_path.write_text(
        json.dumps(kept_clusters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.rejected_path.write_text(
        json.dumps(rejected_clusters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Clusters kept: {len(kept_clusters)}")
    print(f"Clusters rejected: {len(rejected_clusters)}")
    print(f"Output: {args.output_path.resolve()}")
    print(f"Rejected: {args.rejected_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
