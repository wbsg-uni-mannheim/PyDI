#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


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


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_prompt(cluster: dict[str, Any], max_chars: int) -> str:
    offers = cluster.get("offers", [])
    lines = [
        "Decide if the cluster is a true product cluster for exactly one of:",
        "HDD, SSD, USB flash drive, GPU.",
        "Laptops containing SSDs are false positives.",
        "Accessories (brackets, mounting kits, cases), software keys, phones, motherboards, PSU are false positives.",
        'Return only "KEEP" or "DROP".',
        "",
        f"Cluster ID: {cluster.get('cluster_id')}",
        "Offers:",
    ]
    for idx, offer in enumerate(offers, 1):
        title = truncate(str(offer.get("title") or ""), max_chars)
        desc = truncate(str(offer.get("description") or ""), max_chars)
        lines.append(f"{idx}. Title: {title}")
        if desc:
            lines.append(f"   Desc: {desc}")
    return "\n".join(lines)


def parse_response(payload: dict[str, Any]) -> str:
    # Supports both Responses API and legacy Chat Completions style.
    if "output" in payload:
        for item in payload.get("output", []):
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    return part.get("text", "").strip()
    if "choices" in payload:
        return payload["choices"][0]["message"]["content"].strip()
    return ""


def call_openai(
    api_key: str,
    model: str,
    prompt: str,
    reasoning: str,
    timeout: int,
    max_output_tokens: int,
) -> str:
    url = "https://api.openai.com/v1/responses"
    req_body = {
        "model": model,
        "input": prompt,
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": reasoning},
    }
    data = json.dumps(req_body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return parse_response(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=Path("clusters_filtered_2.json"),
        help="Input clusters JSON file.",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        type=Path,
        default=Path("clusters_filtered_3.json"),
        help="Filtered clusters JSON output.",
    )
    parser.add_argument(
        "--rejected-out",
        dest="rejected_path",
        type=Path,
        default=Path("clusters_filtered_3_rejected.json"),
        help="Rejected clusters JSON output.",
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
        "--reasoning",
        default="none",
        help="Reasoning effort (use 'none' to disable).",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=360,
        help="Max characters per title/description field sent to the API.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N clusters (0 = all).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between API calls.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts per cluster on failure.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16,
        help="Max output tokens (Responses API requires >= 16).",
    )
    args = parser.parse_args()

    api_key = load_api_key(args.env_path)
    if not api_key:
        print("OPENAI_API_KEY not found in .env or environment.", file=sys.stderr)
        return 2

    clusters = json.loads(args.input_path.read_text(encoding="utf-8"))
    if args.limit > 0:
        clusters = clusters[: args.limit]

    kept_clusters = []
    rejected_clusters = []
    total = len(clusters)

    for idx, cluster in enumerate(clusters, 1):
        print(f"[{idx}/{total}] Processing cluster {cluster.get('cluster_id')}", file=sys.stderr)
        prompt = build_prompt(cluster, args.max_chars)
        decision = ""
        for attempt in range(1, args.retries + 1):
            try:
                decision = call_openai(
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    reasoning=args.reasoning,
                    timeout=args.timeout,
                    max_output_tokens=args.max_output_tokens,
                )
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                print(f"[{idx}] HTTPError {exc.code}: {body}", file=sys.stderr)
            except Exception as exc:
                print(f"[{idx}] Error: {exc}", file=sys.stderr)
            time.sleep(min(2 ** attempt, 8))

        decision = decision.strip().upper()
        if decision == "KEEP":
            kept_clusters.append(cluster)
        else:
            rejected_clusters.append(cluster)

        print(
            f"[{idx}/{total}] Decision: {decision or 'DROP'} for cluster {cluster.get('cluster_id')}",
            file=sys.stderr,
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

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
