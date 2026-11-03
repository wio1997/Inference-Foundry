#!/usr/bin/env python3
"""Recompute per-engine speculative and decode metrics from frozen snapshots."""
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json
import re
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    '/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/results'
)
expected = {
    ('measure1', 'before'): 'd9dacde0fbdd5d82a4f37b2f4bfde5501f5a5109b54dc7c70b753f3dd50c7df8',
    ('measure1', 'after'): 'e1bfaffa59f9cdfe989f908c90b0e6e7c678c4a63d69377d78014b8fb8d1ea6e',
    ('measure2', 'before'): 'df4760abb2e74ecaf8f89e3abbe7f4345d77349d5b69b53eaa5ec0426d29e5b0',
    ('measure2', 'after'): '9db394fab194048e381262e1fefa741f92aac2d8c73fa083c3c95f88a10dd1ec',
}
names = (
    'spec_decode_num_drafts_total', 'spec_decode_num_draft_tokens_total',
    'spec_decode_num_accepted_tokens_total', 'generation_tokens_total',
    'request_decode_time_seconds_count', 'request_decode_time_seconds_sum',
)
pattern = re.compile(r'^vllm:(' + '|'.join(names) + r')\{([^}]*)\}\s+([^\s]+)$')

def load(phase, when):
    raw = (root / phase / ('metrics-' + when + '.txt')).read_bytes()
    assert sha256(raw).hexdigest() == expected[(phase, when)]
    values = {}
    for line in raw.decode().splitlines():
        match = pattern.fullmatch(line)
        if match is None:
            continue
        engine = re.search(r'(?:^|,)engine="([01])"(?:,|$)', match[2])
        model = re.search(r'(?:^|,)model_name="dsv41"(?:,|$)', match[2])
        assert engine is not None and model is not None
        key = (match[1], engine[1])
        assert key not in values
        values[key] = Decimal(match[3])
    assert len(values) == len(names) * 2
    return values

output = {}
for phase in ('measure1', 'measure2'):
    before, after = load(phase, 'before'), load(phase, 'after')
    output[phase] = {}
    for engine in ('0', '1'):
        d = {name: after[(name, engine)] - before[(name, engine)] for name in names}
        assert min(d.values()) >= 0
        assert d['request_decode_time_seconds_count'] == 4
        assert d['generation_tokens_total'] == 4096
        assert d['spec_decode_num_draft_tokens_total'] > 0
        assert d['spec_decode_num_drafts_total'] > 0
        assert d['spec_decode_num_accepted_tokens_total'] <= d['spec_decode_num_draft_tokens_total']
        output[phase][engine] = {
            'drafts': int(d['spec_decode_num_drafts_total']),
            'proposed_tokens': int(d['spec_decode_num_draft_tokens_total']),
            'accepted_tokens': int(d['spec_decode_num_accepted_tokens_total']),
            'accepted_over_proposed_percent': str(
                100 * d['spec_decode_num_accepted_tokens_total'] / d['spec_decode_num_draft_tokens_total']
            ),
            'generation_tokens': int(d['generation_tokens_total']),
            'decode_observations': int(d['request_decode_time_seconds_count']),
            'decode_sum_seconds': str(d['request_decode_time_seconds_sum']),
            'decode_mean_seconds': str(
                d['request_decode_time_seconds_sum'] / d['request_decode_time_seconds_count']
            ),
        }
print(json.dumps({
    'rounds': output,
    'scope': 'Two fixed scrape windows, engine labels 0 and 1; no request-level pairing.',
    'limits': 'Different metrics can update at different times; snapshots are not atomic across metrics. '
              'Drafts are not hardware steps. Aggregation does not establish causality or request identity.',
}, ensure_ascii=False, indent=2))
