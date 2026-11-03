"""Compare existing quant-gap selections with same-scope rank-local MC2 durations."""
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
source = root / 'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert sha256(source.read_bytes()).hexdigest() == '0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
gap_file = root / 'agents/diagnosis/check_second_quant_matmul_gaps.py'
gaps = json.loads(subprocess.check_output([sys.executable, str(gap_file)], text=True))
long_keys = {(x['digest_prefix'], x['layer'], x['ep']) for x in gaps['long_gaps']}
assert len(long_keys) == 15 and all(ep == 3 for _, _, ep in long_keys)
groups = defaultdict(dict)
for line in source.read_text().splitlines():
    rec = json.loads(line)
    if rec['kind'] != 'rank':
        continue
    for scope in rec['data']['layers']:
        key = scope['scheduler_digest'][:8], scope['layer']
        if key[0] not in ('7ddf7ae9', 'db852bff') or key[1] not in (2,8,14,20,24,28,32,36) or scope['ep'] not in range(8):
            continue
        group = [g for g in scope['operator_groups'] if g['operator_name'] == 'DispatchFFNCombine']
        assert len(group) == 1 and len(group[0]['tasks']) == 1
        task = group[0]['tasks'][0]
        signature = (json.dumps(group[0]['physical_shape_dtype_format'], sort_keys=True),
                     task['csv']['Accelerator Core'], task['csv']['Block Num'], task['csv']['Mix Block Num'])
        assert scope['ep'] not in groups[key]
        groups[key][scope['ep']] = (signature, Decimal(task['trace_duration_us']))
assert len(groups) == 16
for key, ranks in sorted(groups.items()):
    assert set(ranks) == set(range(8)) and len({v[0] for v in ranks.values()}) == 1
    assert all(ranks[3][1] < ranks[i][1] for i in range(8) if i != 3)
    print(key[0], 'layer', key[1], 'EP3 gap >10us:', (key[0],key[1],3) in long_keys,
          'EP3 MC2 shortest:', str(ranks[3][1]), 'us')
print('Same saved scope and physical MC2 signature only; no equal-work or causal conclusion.')
