#!/usr/bin/env python3
"""Collect exact API prompt/output token IDs for a paired Stock/Extreme cohort."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import struct
from pathlib import Path

import aiohttp


def token_digest(ids: list[int]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(ids)}i", *ids)).hexdigest()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    prompts = [
        json.loads(line)["question"]
        for line in args.dataset.read_text(encoding="utf-8").splitlines()[: args.limit]
    ]
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def one(index: int, prompt: str) -> dict:
            body = {
                "model": "dsv4",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": args.max_tokens,
                "ignore_eos": True,
                "stream": False,
                "return_token_ids": True,
                "request_id": f"token-oracle-{index:03d}",
            }
            try:
                async with session.post(args.url, json=body) as response:
                    payload = await response.json(content_type=None)
                    if response.status != 200:
                        return {"index": index, "error": f"HTTP {response.status}: {str(payload)[:300]}"}
                prompt_ids = payload.get("prompt_token_ids")
                choices = payload.get("choices") or []
                output_ids = choices[0].get("token_ids") if choices else None
                if not isinstance(prompt_ids, list) or not isinstance(output_ids, list):
                    return {"index": index, "error": "API omitted token IDs"}
                return {
                    "index": index,
                    "prompt_length": len(prompt_ids),
                    "prompt_sha256": token_digest(prompt_ids),
                    "output_token_ids": output_ids,
                    "output_length": len(output_ids),
                    "finish_reason": choices[0].get("finish_reason"),
                    "error": None,
                }
            except Exception as exc:
                return {"index": index, "error": repr(exc)}

        rows = await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"requests": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"requests": len(rows), "success": sum(r["error"] is None for r in rows)}))
    if any(r["error"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
