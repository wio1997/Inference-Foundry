"""Keep Prefill-only counters from qualifying Decode cache evidence."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path


PREFILL_COUNTER = 'sglang:prefill_effective_tokens_total'
COUNTER_FIELDS = ('prefix_hit_counter', 'prefix_hit_counter_labels', 'device_hit_count_at_gate')


def scoped_formal_gate(gate, source_sha=None):
    assert gate['engine'] == 'sglang' and gate['warmup_only'] is False
    assert gate['stage'] == 'formal' and gate['completed_warmups'] == 5
    assert set(gate['roles']) == {'prefill', 'decode'}
    scoped = deepcopy(gate)
    for role, definition in scoped['roles'].items():
        assert definition['prefix_hit_counter'] == PREFILL_COUNTER
        labels = definition['prefix_hit_counter_labels']
        assert labels['engine_type'] == role and labels['model_name'] == gate['model_id']
        assert labels['tp_rank'] == '0' and labels['mode'] == 'device_hit'
    decode = scoped['roles']['decode']
    scoped['D_prefill_counter_observation'] = {key:decode.pop(key) for key in COUNTER_FIELDS}
    scoped['D_prefix_counter_scope'] = 'Independent Decode radix hits and connector tokens remain unverified by this gate.'
    scoped['P_prefix_counter_scope'] = 'Effective Prefill device hits for TP rank 0.'
    if source_sha is not None:
        scoped['cache_scope_source_gate_SHA256'] = source_sha
    return scoped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    source = args.gate.resolve(strict=True)
    target = args.output.resolve()
    assert source.parent == target.parent, 'Keep the role receipts in the gate directory'
    raw = source.read_bytes()
    result = scoped_formal_gate(json.loads(raw),sha256(raw).hexdigest())
    with target.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
