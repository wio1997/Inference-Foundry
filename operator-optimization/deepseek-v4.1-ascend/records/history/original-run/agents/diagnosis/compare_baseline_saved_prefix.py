#!/usr/bin/env python3
"""Compare saved input prefixes without printing or saving input contents or IDs.

Counts are Unicode code points in the saved JSON string, not bytes or tokens.
No tokenization, service access, or NPU use. Anonymous positions are derived
from the same data_id sort as the independently computed latency evidence.
"""
import hashlib
import itertools
import json
import os
import pathlib

ROOT = pathlib.Path('/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/results')
HERE = pathlib.Path(__file__).parent
latency = json.loads((HERE / 'baseline-latency-anonymized.json').read_text())
phases = {}
result = {'definition': 'longest common prefix of saved JSON input strings, Unicode code points; not bytes, token counts, cache hits, or proof of transmitted inputs', 'phases': {}}
for phase in ('measure1', 'measure2'):
    candidates = list((ROOT / phase).rglob('*details.jsonl'))
    if len(candidates) != 1:
        raise ValueError('frozen details ambiguous')
    raw = candidates[0].read_bytes()
    if hashlib.sha256(raw).hexdigest() != latency['phases'][phase]['details_sha256']:
        raise ValueError('latency and input evidence are from different details')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != 8 or len({row['data_id'] for row in rows}) != 8:
        raise ValueError('unexpected request set')
    rows.sort(key=lambda r: r['data_id'])
    strings = [r['input'] for r in rows]
    if not all(isinstance(s, str) for s in strings):
        raise ValueError('input is not a string')
    phases[phase] = (rows, strings)
    common = len(os.path.commonprefix(strings))
    pairs = {}
    for (i, a), (j, b) in itertools.combinations(enumerate(strings, 1), 2):
        length = len(os.path.commonprefix((a, b)))
        pairs.setdefault(length, []).append((i, j))
    records = latency['phases'][phase]['records']
    if [r['ordinal_by_data_id'] for r in records] != list(range(1, 9)):
        raise ValueError('anonymous positions not aligned')
    result['phases'][phase] = {
        'details_sha256': hashlib.sha256(raw).hexdigest(),
        'request_count': len(strings),
        'all_request_common_prefix_characters': common,
        'saved_input_characters_range': [min(map(len, strings)), max(map(len, strings))],
        'common_prefix_fraction_range': [common / max(map(len, strings)), common / min(map(len, strings))],
        'pairwise_common_prefix_length_counts': {str(k): len(v) for k, v in sorted(pairs.items())},
        'extended_group_positions': sorted(set(itertools.chain.from_iterable(pairs[max(pairs)]))),
        'extended_group_first_response_ms_range': [min(records[i-1]['ttft_ms'] for i in set(itertools.chain.from_iterable(pairs[max(pairs)]))), max(records[i-1]['ttft_ms'] for i in set(itertools.chain.from_iterable(pairs[max(pairs)])))],
    }
if [r['data_id'] for r in phases['measure1'][0]] != [r['data_id'] for r in phases['measure2'][0]]:
    raise ValueError('incompatible anonymous positions')
cross = [len(os.path.commonprefix((a, b))) for a in phases['measure1'][1] for b in phases['measure2'][1]]
result['cross_round'] = {'pair_count': len(cross), 'common_prefix_characters_range': [min(cross), max(cross)]}
result['limitations'] = ['saved input field is not verified as exact transmitted token sequence', 'character common prefix is not token common prefix or proof of cache hit', 'anonymous position and input prefix do not establish causal effects on first response', 'first response is client SSE observation, not independently verified token generation time', 'only two rounds and eight requests per round']
(HERE / 'baseline-saved-prefix-anonymized.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print('wrote aggregate prefix comparisons; no input contents or request IDs')
