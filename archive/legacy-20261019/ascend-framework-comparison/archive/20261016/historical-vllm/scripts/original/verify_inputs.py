#!/usr/bin/env python3
"""Verify prepared input lengths and identical shared-prefix token IDs."""
import hashlib
import json
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer

root = Path(os.environ.get("ROOT", "/data/tiankuan/aisbench/warm_w2/codex_4p2d_20261008"))
tokenizer_path = os.environ.get("TOKENIZER_PATH", "/data/tiankuan/aisbench/GLM-5.3-w8a8")
tok = AutoTokenizer.from_pretrained(tokenizer_path)
prefix_file = root / "shared/prefix_warmup/gsm8k/test.jsonl"
prefix = json.loads(prefix_file.open(encoding="utf-8").readline())["question"]
prefix_ids = tok.encode(prefix, add_special_tokens=False)
results = []
datasets = sorted((root / "runs").glob("*/data/gsm8k/test.jsonl"))
selected = set(sys.argv[1:])
if selected:
    datasets = [p for p in datasets if p.parents[2].name in selected]
    missing = selected - {p.parents[2].name for p in datasets}
    if missing:
        raise SystemExit(f"Unknown/unprepared run names: {sorted(missing)}")
for dataset in datasets:
    sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    count = min_len = max_len = prefix_ok = bad = 0
    first_mismatch = None
    for line in dataset.open(encoding="utf-8"):
        q = json.loads(line)["question"]
        ids = tok.encode(q, add_special_tokens=False)
        count += 1
        min_len = len(ids) if count == 1 else min(min_len, len(ids))
        max_len = max(max_len, len(ids))
        same = ids[:len(prefix_ids)] == prefix_ids
        prefix_ok += int(same)
        if not same:
            bad += 1
            if first_mismatch is None:
                first_mismatch = next((i for i, (a, b) in enumerate(zip(ids, prefix_ids)) if a != b), None)
    results.append({"run": dataset.parents[2].name, "rows": count,
                    "token_length_min": min_len, "token_length_max": max_len,
                    "common_prefix_rows_ok": prefix_ok, "common_prefix_rows_bad": bad,
                    "first_prefix_mismatch_token": first_mismatch,
                    "dataset_sha256": sha})
print(json.dumps({"prefix_tokens": len(prefix_ids), "runs": results}, indent=2))
if len(prefix_ids) != 74400 or any(r["token_length_min"] != 80000 or
                                    r["token_length_max"] != 80000 or
                                    r["common_prefix_rows_bad"] for r in results):
    raise SystemExit(1)
