#!/usr/bin/env python3
"""Compare paired API token-ID captures with prompt-identity gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True, type=Path)
    parser.add_argument("--extreme", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    stock = json.loads(args.stock.read_text(encoding="utf-8"))["requests"]
    extreme = json.loads(args.extreme.read_text(encoding="utf-8"))["requests"]
    assert len(stock) == len(extreme)
    rows = []
    for base, candidate in zip(stock, extreme):
        assert base["index"] == candidate["index"]
        prompt_match = (
            base.get("error") is None
            and candidate.get("error") is None
            and base["prompt_sha256"] == candidate["prompt_sha256"]
            and base["prompt_length"] == candidate["prompt_length"]
        )
        a = base.get("output_token_ids") or []
        b = candidate.get("output_token_ids") or []
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if first is None and len(a) != len(b):
            first = min(len(a), len(b))
        rows.append({
            "index": base["index"],
            "prompt_match": prompt_match,
            "stock_length": len(a),
            "extreme_length": len(b),
            "first_mismatch": first,
            "exact_output": prompt_match and first is None,
            "stock_error": base.get("error"),
            "extreme_error": candidate.get("error"),
        })
    result = {
        "requests": len(rows),
        "paired_prompts": sum(r["prompt_match"] for r in rows),
        "exact_outputs": sum(r["exact_output"] for r in rows),
        "first_mismatch_by_request": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "requests": result["requests"],
        "paired_prompts": result["paired_prompts"],
        "exact_outputs": result["exact_outputs"],
    }))
    if result["paired_prompts"] != result["requests"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
