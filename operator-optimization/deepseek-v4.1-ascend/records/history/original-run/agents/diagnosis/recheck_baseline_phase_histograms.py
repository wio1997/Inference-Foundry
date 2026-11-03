#!/usr/bin/env python3
"""Recompute anonymous phase histogram deltas from existing benchmark snapshots."""
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json
import re
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/results')
expected = {
    ('measure1', 'before'): 'd9dacde0fbdd5d82a4f37b2f4bfde5501f5a5109b54dc7c70b753f3dd50c7df8',
    ('measure1', 'after'): 'e1bfaffa59f9cdfe989f908c90b0e6e7c678c4a63d69377d78014b8fb8d1ea6e',
    ('measure2', 'before'): 'df4760abb2e74ecaf8f89e3abbe7f4345d77349d5b69b53eaa5ec0426d29e5b0',
    ('measure2', 'after'): '9db394fab194048e381262e1fefa741f92aac2d8c73fa083c3c95f88a10dd1ec',
}
names = ('request_queue_time_seconds', 'request_prefill_time_seconds', 'request_decode_time_seconds',
         'time_to_first_token_seconds', 'e2e_request_latency_seconds', 'request_num_preemptions', 'request_generation_tokens')
pattern = re.compile(r'^vllm:(' + '|'.join(names) + r')_(sum|count)\{([^}]*)\}\s+([^\s]+)$')
bucket_pattern = re.compile(r'^vllm:request_generation_tokens_bucket\{([^}]*)\}\s+([^\s]+)$')

def load(phase, when):
    data = (root / phase / ('metrics-' + when + '.txt')).read_bytes()
    assert sha256(data).hexdigest() == expected[(phase, when)]
    values = {}
    for line in data.decode().splitlines():
        bucket = bucket_pattern.fullmatch(line)
        if bucket:
            engine = re.search(r'(?:^|,)engine="([01])"(?:,|$)', bucket[1])
            limit = re.search(r'(?:^|,)le="([^"]+)"(?:,|$)', bucket[1])
            assert engine is not None and limit is not None
            if limit[1] in ('1.0', '1000.0', '2000.0', '+Inf'):
                key = ('request_generation_tokens_bucket', limit[1], engine[1])
                assert key not in values
                values[key] = Decimal(bucket[2])
            continue
        match = pattern.fullmatch(line)
        if match is None:
            continue
        engine = re.search(r'(?:^|,)engine="([01])"(?:,|$)', match[3])
        assert engine is not None
        key = (match[1], match[2], engine[1])
        assert key not in values
        values[key] = Decimal(match[4])
    assert len(values) == len(names) * 2 * 2 + 8
    return values

snapshots = {(phase, when): load(phase, when) for phase in ('measure1', 'measure2') for when in ('before', 'after')}
result = {}
for phase in ('measure1', 'measure2'):
    before, after = snapshots[(phase, 'before')], snapshots[(phase, 'after')]
    result[phase] = {}
    for name in names:
        counts = [after[(name, 'count', e)] - before[(name, 'count', e)] for e in ('0', '1')]
        sums = [after[(name, 'sum', e)] - before[(name, 'sum', e)] for e in ('0', '1')]
        assert counts == [Decimal(4), Decimal(4)] and min(sums) >= 0
        if name == 'request_num_preemptions':
            assert sum(sums) == 0
            result[phase][name] = {'count_per_engine': [int(x) for x in counts], 'sum': str(sum(sums))}
        elif name == 'request_generation_tokens':
            bucket_delta = {limit: [after[(name + '_bucket', limit, e)] - before[(name + '_bucket', limit, e)] for e in ('0', '1')] for limit in ('1.0', '1000.0', '2000.0', '+Inf')}
            assert bucket_delta == {'1.0': [0, 0], '1000.0': [0, 0], '2000.0': [4, 4], '+Inf': [4, 4]}
            result[phase][name] = {'count_per_engine': [int(x) for x in counts], 'sum': str(sum(sums)),
                                   'bucket_delta_per_engine': {k: [int(x) for x in v] for k, v in bucket_delta.items()}}
        else:
            result[phase][name] = {'count_per_engine': [int(x) for x in counts], 'sum_seconds': str(sum(sums)),
                                   'mean_ms': str(sum(sums) * 1000 / sum(counts))}
    r = result[phase]
    residual = (Decimal(r['time_to_first_token_seconds']['sum_seconds']) - Decimal(r['request_queue_time_seconds']['sum_seconds']) - Decimal(r['request_prefill_time_seconds']['sum_seconds']))
    result[phase]['ttft_less_queue_and_prefill_sums'] = {'seconds': str(residual), 'mean_ms_for_8_observations': str(residual * 1000 / 8),
                                                        'note': 'Arithmetic across separate histograms; same requests, timestamps and clocks unproven.'}
    residual_e2e = (Decimal(r['e2e_request_latency_seconds']['sum_seconds']) - Decimal(r['time_to_first_token_seconds']['sum_seconds']) - Decimal(r['request_decode_time_seconds']['sum_seconds']))
    result[phase]['e2e_less_ttft_and_decode_sums'] = {'seconds': str(residual_e2e), 'mean_ms_for_8_observations': str(residual_e2e * 1000 / 8),
                                                      'note': 'Can be negative: front-end TTFT/E2E and core decode use distinct timestamps and histogram cohorts.'}
gap = {e: int(snapshots[('measure2', 'before')][('e2e_request_latency_seconds', 'count', e)] -
              snapshots[('measure1', 'after')][('e2e_request_latency_seconds', 'count', e)]) for e in ('0', '1')}
print(json.dumps({'rounds': result, 'between_round_e2e_count_increase_per_engine': gap,
                  'limits': 'Aggregated snapshots, not linked to request IDs; per-request times and causes are not established.'}, indent=2))
