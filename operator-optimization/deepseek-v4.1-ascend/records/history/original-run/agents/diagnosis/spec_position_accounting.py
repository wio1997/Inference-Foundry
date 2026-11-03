#!/usr/bin/env python3
"""Recompute per-position counter deltas from saved benchmark text, no NPU."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/results')
PATTERN = re.compile(r'^vllm:spec_decode_num_accepted_tokens_per_pos_total\{engine="([01])",model_name="dsv41",position="([0-4])"\} ([0-9.]+)$')

def load(path):
    data = path.read_bytes()
    metrics = {}
    for line in data.decode().splitlines():
        match = PATTERN.fullmatch(line)
        if match:
            key = (match[1], match[2])
            assert key not in metrics and float(match[3]).is_integer()
            metrics[key] = int(float(match[3]))
    assert set(metrics) == {(str(e), str(p)) for e in range(2) for p in range(5)}
    return metrics, hashlib.sha256(data).hexdigest()

for phase in ('measure1', 'measure2'):
    directory = ROOT / phase
    before, before_hash = load(directory / 'metrics-before.txt')
    after, after_hash = load(directory / 'metrics-after.txt')
    progress_bytes = (directory / 'progress.json').read_bytes()
    counters = json.loads(progress_bytes)['server_counter_deltas']
    totals = [sum(after[(str(e), str(p))] - before[(str(e), str(p))] for e in range(2)) for p in range(5)]
    assert all(n >= 0 for n in totals)
    assert sum(totals) == counters['spec_decode_num_accepted_tokens_total']
    print(json.dumps({'phase': phase, 'accepted_per_position': totals,
                      'accepted': counters['spec_decode_num_accepted_tokens_total'],
                      'draft_tokens': counters['spec_decode_num_draft_tokens_total'],
                      'drafts': counters['spec_decode_num_drafts_total'],
                      'sha256': {'metrics-before.txt': before_hash, 'metrics-after.txt': after_hash,
                                 'progress.json': hashlib.sha256(progress_bytes).hexdigest()}}, ensure_ascii=False))
