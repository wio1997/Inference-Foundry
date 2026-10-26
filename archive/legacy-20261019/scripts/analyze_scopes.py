import argparse
import glob
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--raw', required=True)
p.add_argument('--phases', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
paths = glob.glob(str(Path(a.raw) / 'dp0_pp0_tp0*' / 'ASCEND_PROFILER_OUTPUT' / 'trace_view.json'))
if len(paths) != 1:
    raise RuntimeError(f'expected one TP0 trace_view.json, found {paths}')
marks = json.loads(Path(a.phases).read_text())
phases = {}
for kind in ('warm', 'cold'):
    start = datetime.fromisoformat(marks[f'{kind}_start_utc']).timestamp() * 1e6
    end = datetime.fromisoformat(marks[f'{kind}_end_utc']).timestamp() * 1e6
    phases[kind] = {'start_us': start, 'end_us': end, 'wall_s': (end-start)/1e6}
scopes = {'prepare input', 'forward', 'post process', 'sample_token', 'draft_token', 'async_state_update'}
records = {phase: defaultdict(list) for phase in phases}
trace = json.loads(Path(paths[0]).read_text())
for event in trace:
    if event.get('ph') != 'X' or event.get('cat') != 'cpu_op' or event.get('name') not in scopes:
        continue
    ts = float(event['ts'])
    dur = float(event['dur'])
    for phase, window in phases.items():
        if window['start_us'] <= ts < window['end_us']:
            records[phase][event['name']].append({'start_us': ts, 'duration_us': dur})

def percentile(values, q):
    values = sorted(values)
    return values[min(len(values)-1, int((len(values)-1)*q))] if values else None

summary = {'source': paths[0], 'event_count': len(trace), 'phases': {}}
for phase, window in phases.items():
    x = {'wall_s': window['wall_s'], 'scopes': {}}
    for name, events in records[phase].items():
        durations = [e['duration_us']/1000 for e in events]
        x['scopes'][name] = {'count': len(events), 'sum_ms': sum(durations),
                             'mean_ms': statistics.mean(durations), 'p50_ms': percentile(durations,.5),
                             'p90_ms': percentile(durations,.9), 'max_ms': max(durations)}
    summary['phases'][phase] = x
Path(a.out).write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary['phases'], indent=2))
