#!/usr/bin/env python3
"""Offline replay of the evidence. Does not contact or modify any server."""
import hashlib
import json
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path
from analyze_cg_dispatch_diag import read_records

ROOT = Path(__file__).resolve().parent
paths = [ROOT / f'D{n}_repro600_events.log' for n in (170, 171)]
records = read_records(paths)
assert len(records) == 11776
steps = defaultdict(dict)
first = {}
for r in sorted(records, key=lambda r: (r['step'], r['dp'])):
    assert r['dp'] not in steps[r['step']]
    steps[r['step']][r['dp']] = r
    for request in r['req_ids']:
        first.setdefault(request, r['step'])
assert len(steps) == 736 and len(first) == 48
assert all(set(ranks) == set(range(16)) for ranks in steps.values())
local_none = [r for r in records if r['local_mode'] == 'NONE']
assert len(local_none) == 48
assert all(r['tokens'] == r['max_sched'] == r['reqs'] == 1
           and r['uniform_query_len'] == 6 and r['is_all_decode']
           and not r['uniform'] and len(r['req_ids']) == 1
           and first[r['req_ids'][0]] == r['step'] for r in local_none)
uuid_re = re.compile(r'chatcmpl-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
kv_end_lines = {}
for path in paths:
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        match = uuid_re.search(line)
        if match and 'PD_OBSERVE_END' in line:
            kv_end_lines.setdefault((str(path), match.group()), line_number)
for r in local_none:
    request_uuid = uuid_re.search(r['req_ids'][0]).group()
    assert kv_end_lines[(r['source'], request_uuid)] < r['line']
none_steps = {r['step'] for r in local_none}
assert len(none_steps) == 46
for step, ranks in steps.items():
    vector = [ranks[dp]['local_mode'] for dp in range(16)]
    expected = 'NONE' if step in none_steps else 'FULL'
    assert all(r['dp_modes'] == vector and r['final_mode'] == expected
               for r in ranks.values())

transitions = defaultdict(list)
long_steps = []
weighted_excess = []
for step in sorted(steps):
    if step + 1 not in steps:
        continue
    before, after = steps[step], steps[step + 1]
    delta = statistics.median([(after[dp]['ts_ns'] - before[dp]['ts_ns']) / 1e6
                               for dp in range(16)])
    label = before[0]['final_mode'] + '->' + after[0]['final_mode']
    transitions[label].append(delta)
    if delta > 150:
        long_steps.append(step)
    if step in none_steps:
        old_ids = {req for row in before.values() for req in row['req_ids']
                   if first[req] < step}
        weighted_excess.append((len(old_ids), delta))
assert set(long_steps) == none_steps
normal_ms = statistics.median(transitions['FULL->FULL'])

archive_inventory = {}
for archive in sorted((ROOT / 'raw_archive').glob('*.tar.gz')):
    members = []
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            if not member.isfile():
                continue
            data = tf.extractfile(member).read()
            members.append({'name': member.name, 'bytes': len(data),
                            'sha256': hashlib.sha256(data).hexdigest()})
    archive_inventory[archive.name] = members
assert set(archive_inventory) == {'benchmark600.tar.gz', 'D170_diagnostics.tar.gz',
                                  'D171_diagnostics.tar.gz', 'runtime_source.tar.gz'}
for node in (170, 171):
    with tarfile.open(ROOT / 'raw_archive' / f'D{node}_diagnostics.tar.gz') as tf:
        raw = tf.extractfile(f'repro600_20261007/D{node}.log').read().decode()
    filtered = (ROOT / f'D{node}_repro600_events.log').read_text().splitlines()
    raw_lines = set(raw.splitlines())
    assert all(line in raw_lines for line in filtered if line.strip())

summary = {
    'verified': True, 'records': len(records), 'complete_global_steps': len(steps),
    'real_request_first_appearances': len(first), 'q1_local_none_records': len(local_none),
    'kv_end_precedes_first_dispatch_in_merged_node_log': len(local_none),
    'global_none_steps': len(none_steps), 'all_intervals_over_150ms_follow_none': True,
    'transitions': {k: {'n': len(v), 'median_ms': statistics.median(v), 'max_ms': max(v)}
                    for k, v in transitions.items()},
    'old_request_step_occupancies': sum(n for n, _ in weighted_excess),
    'assumed_excess_ms_per_token': sum(n * (ms - normal_ms) for n, ms in weighted_excess)
                                  / (48 * 599),
    'caveat': 'Dispatch intervals include source execution and next-step preparation/synchronization. '
              'The excess estimate is not a causal graph-only A/B result.',
    'raw_log_membership_verified': True,
}
(ROOT / 'evidence_recheck.json').write_text(json.dumps(summary, indent=2) + '\n')
(ROOT / 'archive_inventory.json').write_text(json.dumps(archive_inventory, indent=2) + '\n')
print(json.dumps(summary, indent=2))
