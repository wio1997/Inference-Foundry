#!/usr/bin/env python3
"""Count AIC/AIV eligibility overlap using the extractor's physical grouping key."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
extractor = root / 'agents/diagnosis/analyze_recapture_all_kernel_groups.py'
x = json.loads(subprocess.check_output(['python3', str(extractor), '--all-groups'], cwd=root))
def key(g):
    return json.dumps([g[k] for k in (
        'digest', 'layer', 'operator_name', 'csv_type',
        'physical_shape_dtype_format', 'core')], sort_keys=True, separators=(',', ':'))
a_rows, v_rows = x['all_aic_eligible_groups'], x['all_aiv_eligible_groups']
a, v = {key(g): g for g in a_rows}, {key(g): g for g in v_rows}
assert len(a) == len(a_rows) == x['aic_eligible_group_count']
assert len(v) == len(v_rows) == x['aiv_eligible_group_count']
physical = x['physical_group_status']['full_8_ranks_equal_count']
both = a.keys() & v.keys()
counts = {'both': len(both), 'aic_only': len(a.keys() - v.keys()),
          'aiv_only': len(v.keys() - a.keys()), 'neither': physical - len(a.keys() | v.keys())}
assert min(counts.values()) >= 0 and sum(counts.values()) == physical
print(json.dumps({'source_sha256': x['source_sha256'],
                  'extractor_sha256': hashlib.sha256(extractor.read_bytes()).hexdigest(),
                  'physically_complete_groups': physical, 'aic_eligible_groups': len(a),
                  'aiv_eligible_groups': len(v), 'mutually_exclusive_counts': counts,
                  'both_core_labels': dict(Counter(a[k]['core'] for k in both)),
                  'limits': 'Selected physical groups, not all tasks or model; AIC and AIV count different fields and denominators, overlap does not prove simultaneous activity or comparable utilization.'}, indent=2))
