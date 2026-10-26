import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--raw', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
files = glob.glob(str(Path(a.raw) / 'PROF_*' / 'mindstudio_profiler_output' / 'op_summary*.csv'))
if len(files) != 1:
    raise RuntimeError(f'expected one op_summary, found {files}')
d = pd.read_csv(files[0], usecols=['Task ID', 'OP Type', 'Task Start Time(us)', 'Task Duration(us)'])
d = d.dropna(subset=['Task ID', 'OP Type', 'Task Start Time(us)', 'Task Duration(us)'])
t0 = d['Task Start Time(us)'].min()
t1 = (d['Task Start Time(us)'] + d['Task Duration(us)']).max()
d['category'] = np.where(d['OP Type'].str.startswith('hcom_'), 'communication', 'compute_or_copy')

def interval_union(frame):
    starts = frame['Task Start Time(us)'].to_numpy(dtype=float)
    ends = starts + frame['Task Duration(us)'].to_numpy(dtype=float)
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    if not len(starts):
        return 0.0, 0.0
    total = 0.0
    longest_gap = 0.0
    right = ends[0]
    for start, end in zip(starts[1:], ends[1:]):
        if start > right:
            longest_gap = max(longest_gap, start - right)
            right = end
        else:
            right = max(right, end)
    total = 0.0
    left, right = starts[0], ends[0]
    for start, end in zip(starts[1:], ends[1:]):
        if start > right:
            total += right - left
            left, right = start, end
        else:
            right = max(right, end)
    total += right - left
    return total, longest_gap

summary = {'source': files[0], 'events': len(d), 'span_s': (t1-t0)/1e6,
           'total_duration_s_sum_overlaps': float(d['Task Duration(us)'].sum()/1e6)}
for category, frame in [('all', d), *list(d.groupby('category'))]:
    busy, gap = interval_union(frame)
    summary[category] = {'events': len(frame), 'union_busy_s': busy/1e6,
                         'busy_fraction_of_span': busy/(t1-t0), 'longest_inter_event_gap_ms': gap/1000,
                         'sum_duration_s': float(frame['Task Duration(us)'].sum()/1e6)}
summary['top_op_duration_sum_s'] = {k: float(v/1e6) for k, v in
    d.groupby('OP Type')['Task Duration(us)'].sum().sort_values(ascending=False).head(20).items()}
summary['top_single_events'] = d.nlargest(20, 'Task Duration(us)')[['OP Type', 'Task Start Time(us)', 'Task Duration(us)']].to_dict('records')
d['second'] = ((d['Task Start Time(us)']-t0)/1e6).astype(int)
summary['per_second'] = {str(int(sec)): {cat: {'count': len(f), 'sum_duration_ms': float(f['Task Duration(us)'].sum()/1000)}
    for cat, f in group.groupby('category')} for sec, group in d.groupby('second')}
Path(a.out).write_text(json.dumps(summary, indent=2))
print(json.dumps({k:v for k,v in summary.items() if k != 'per_second' and k != 'top_single_events'}, indent=2))
