#!/usr/bin/env python3
"""Check whether the extractor's AIC- and AIV-eligible group lists overlap.

Group identity is reconstructed exactly as analyze_recapture_all_kernel_groups.py
builds it (its line 83):

    groupkey = (scheduler_digest, layer_index, operator_name, csv_type,
                canonical_json(physical_shape_dtype_format), accelerator_core)

The eligible entries emitted by --all-groups carry those same six components,
so the key is rebuilt from the emitted fields; nothing else is invented.
CPU only: reads local JSON, performs no device or network work.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALL = HERE / 'all-groups-output.json'
SUMMARY = HERE.parents[2] / 'diagnostic-20261029/all-kernel-groups-summary.json'
DESC_FIELDS = ['Input Shapes', 'Output Shapes', 'Input Data Types',
               'Output Data Types', 'Input Formats', 'Output Formats']


def canon(shape_obj):
    """Same canonicalization the extractor applies to the physical descriptor."""
    return json.dumps(shape_obj, sort_keys=True, separators=(',', ':'))


def key_of(entry):
    return (entry['digest'], entry['layer'], entry['operator_name'],
            entry['csv_type'], canon(entry['physical_shape_dtype_format']),
            entry['core'])


d = json.loads(ALL.read_text())
s = json.loads(SUMMARY.read_text())

# --- counts must match the compact summary produced by the same run family ---
checks = []
def check(name, cond, detail=''):
    checks.append({'check': name, 'ok': bool(cond), 'detail': detail})
    if not cond:
        raise SystemExit('FAILED CHECK: %s %s' % (name, detail))

check('summary physical_group_count == 6862', s['physical_group_count'] == 6862, s['physical_group_count'])
check('summary physically_complete_groups == 4930',
      s['physical_group_status']['full_8_ranks_equal_count'] == 4930)
check('--all-groups aic count == summary aic count',
      len(d['all_aic_eligible_groups']) == s['aic_eligible_group_count'],
      (len(d['all_aic_eligible_groups']), s['aic_eligible_group_count']))
check('--all-groups aiv count == summary aiv count',
      len(d['all_aiv_eligible_groups']) == s['aiv_eligible_group_count'],
      (len(d['all_aiv_eligible_groups']), s['aiv_eligible_group_count']))
check('source hash matches summary', d['source_sha256'] == s['source_sha256'])

# --- descriptor round-trip must be lossless, else keys are ambiguous ---
bad_desc = 0
for engine in ('all_aic_eligible_groups', 'all_aiv_eligible_groups'):
    for e in d[engine]:
        o = e['physical_shape_dtype_format']
        if sorted(o) != sorted(DESC_FIELDS) or any(not isinstance(o[k], str) for k in DESC_FIELDS):
            bad_desc += 1
        elif canon(json.loads(canon(o))) != canon(o):
            bad_desc += 1
check('canonical descriptor round-trip is lossless', bad_desc == 0, bad_desc)

aic_entries, aiv_entries = d['all_aic_eligible_groups'], d['all_aiv_eligible_groups']
k_aic = [key_of(e) for e in aic_entries]
k_aiv = [key_of(e) for e in aiv_entries]

check('AIC keys are unique', len(set(k_aic)) == len(k_aic), (len(k_aic), len(set(k_aic))))
check('AIV keys are unique', len(set(k_aiv)) == len(k_aiv), (len(k_aiv), len(set(k_aiv))))

A, B = set(k_aic), set(k_aiv)
inter, only_a, only_b = A & B, A - B, B - A
union = A | B

check('no duplicate physical group across the two lists is counted twice',
      len(aic_entries) + len(aiv_entries) == len(inter) * 2 + len(only_a) + len(only_b))
check('union is within the 4930 physically complete groups', len(union) <= 4930,
      (len(union), 4930))

# shared groups must agree on record-derived fields across engines
mismatch = []
aiv_by_key = {key_of(e): e for e in aiv_entries}
for e in aic_entries:
    k = key_of(e)
    if k in aiv_by_key:
        o = aiv_by_key[k]
        for f in ('task_count', 'per_rank_task_count', 'eps'):
            if e[f] != o[f]:
                mismatch.append((k[:4], f, e[f], o[f]))
check('intersecting groups agree on task_count/per_rank_task_count/eps', not mismatch, mismatch[:3])

# Is the key over-split by Accelerator Core? (5-tuple collapse test)
coarse = defaultdict(set)
for k in union:
    coarse[k[:5]].add(k[5])
over_split = {canon5: cores for canon5, cores in coarse.items() if len(cores) > 1}

core_dist = {}
for label, keys in (('aic_only', only_a), ('both', inter), ('aiv_only', only_b)):
    core_dist[label] = dict(Counter(k[5] for k in keys).most_common())

out = {
    'inputs': {'all_groups_output': str(ALL), 'compact_summary': str(SUMMARY)},
    'group_key': "(.scheduler_digest, .layer, .operator_name, .csv_type, "
                 "canonical_json(.physical_shape_dtype_format), .core); "
                 "core is the CSV 'Accelerator Core' column",
    'physically_complete_groups': 4930,
    'aic_eligible': len(A),
    'aiv_eligible': len(B),
    'overlap_both': len(inter),
    'aic_only': len(only_a),
    'aiv_only': len(only_b),
    'union_eligible_in_at_least_one_engine': len(union),
    'neither_engine': 4930 - len(union),
    'sum_of_two_lists': len(A) + len(B),
    'accelerator_core_by_bucket': core_dist,
    'distinct_5tuple_keys_ignoring_core': len(coarse),
    'distinct_5tuple_keys_that_split_over_multiple_cores': len(over_split),
    'checks': checks,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
