"""Recompute diagnostic statistics from anonymized numeric observations."""
import argparse
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True, help='Operator-supplied private derived observation file')
args = parser.parse_args()
data = gzip.decompress(args.data.read_bytes())
provenance = json.loads((root / 'PROVENANCE.json').read_text())
assert hashlib.sha256(data).hexdigest() == provenance['public_transformation']['anonymous_data_sha256']
rows = [json.loads(line) for line in data.splitlines()]
steps, first = defaultdict(dict), {}
for row in rows:
    assert row['dp'] not in steps[row['step']]
    steps[row['step']][row['dp']] = row
    for request in row['request_ids']:
        first.setdefault(request, row['step'])
assert len(rows) == 11776 and len(steps) == 736 and len(first) == 48
assert all(set(ranks) == set(range(16)) for ranks in steps.values())
triggers = [r for r in rows if r['local_mode'] == 'NONE']
assert len(triggers) == 48
assert all(r['tokens'] == r['max_sched'] == r['reqs'] == 1
           and r['uniform_query_len'] == 6 and r['is_all_decode']
           and not r['uniform'] and len(r['request_ids']) == 1
           and first[r['request_ids'][0]] == r['step'] for r in triggers)
none_steps = {r['step'] for r in triggers}
assert len(none_steps) == 46
for step, ranks in steps.items():
    vector = [ranks[dp]['local_mode'] for dp in range(16)]
    expected = 'NONE' if step in none_steps else 'FULL'
    assert all(r['dp_modes'] == vector and r['final_mode'] == expected for r in ranks.values())
transitions, long_steps = defaultdict(list), set()
for step, ranks in sorted(steps.items()):
    if step + 1 not in steps:
        continue
    following = steps[step + 1]
    delta = statistics.median([(following[dp]['relative_ns'] - ranks[dp]['relative_ns']) / 1e6
                               for dp in range(16)])
    key = ranks[0]['final_mode'] + '->' + following[0]['final_mode']
    transitions[key].append(delta)
    if delta > 150:
        long_steps.add(step)
assert long_steps == none_steps
print(json.dumps({'verified': True, 'records': len(rows), 'global_steps': len(steps),
                  'requests': len(first), 'global_none_steps': len(none_steps),
                  'transitions': {k: {'n': len(v), 'median_ms': statistics.median(v),
                                      'max_ms': max(v)} for k, v in transitions.items()}}, indent=2))
