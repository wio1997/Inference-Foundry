#!/usr/bin/env python3
"""Analyze legal eight-rank host stage timestamps from Run137."""
import json,statistics
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'evidence/20260925_loop042_phase/run137'
ranks={r:json.loads((base/f'phase/rank{r}.json').read_text()) for r in range(8)}
assert all(v['cycles']>=256 for v in ranks.values())
rows={r:{x['cycle']:dict(x['marks']) for x in v['phase_rows']} for r,v in ranks.items()}
labels=['begin','prepare_target','derived_target_metadata','target','acceptance','state_advance','proposer','draft_commit']
cycles=[]
for cycle in range(64,256):
    samples={r:rows[r][cycle] for r in range(8)}
    assert all(set(labels)==set(x) for x in samples.values())
    skews={label:(max(x[label] for x in samples.values())-min(x[label] for x in samples.values()))/1e6 for label in labels}
    lasts={label:max(samples,key=lambda r:samples[r][label]) for label in labels}
    duration={r:(samples[r]['draft_commit']-samples[r]['begin'])/1e6 for r in samples}
    stages={label:{r:(samples[r][label]-samples[r][prev])/1e6 for r in samples} for prev,label in zip(labels[:-1],labels[1:])}
    cycles.append({'cycle':cycle,'skew_ms':skews,'last_rank':lasts,'duration_spread_ms':max(duration.values())-min(duration.values()),
                   'cycle_wall_ms':(max(x['draft_commit'] for x in samples.values())-min(x['begin'] for x in samples.values()))/1e6,
                   'stage_duration_ms':stages})
def describe(v):return {'median':statistics.median(v),'p10':sorted(v)[int(.1*(len(v)-1))], 'p90':sorted(v)[int(.9*(len(v)-1))], 'min':min(v),'max':max(v)}
summary={'skew_ms':{label:describe([c['skew_ms'][label] for c in cycles]) for label in labels},
         'stage_duration_ms':{label:describe([c['stage_duration_ms'][label][r] for c in cycles for r in range(8)]) for label in labels[1:]},
         'last_rank_counts':{label:dict(Counter(c['last_rank'][label] for c in cycles)) for label in labels},
         'cycle_wall_ms':describe([c['cycle_wall_ms'] for c in cycles]),
         'duration_spread_ms':describe([c['duration_spread_ms'] for c in cycles]),
         'begin_to_next_begin_ms':describe([(rows[r][c+1]['begin']-rows[r][c]['begin'])/1e6 for c in range(64,255) for r in range(8)])}
bench=json.loads((base/'bench.json').read_text())['summary']
assert bench['n']==12 and bench['success']==12 and bench['fail']==0 and bench['max_tokens']==1024
out={'run':'run137','contract':'legal 8-rank 12x1024 c12 DSpark7', 'bench_summary':bench,
     'clock':'host time.time_ns on one physical host; marks after asynchronous dispatch, no added NPU synchronization',
     'steady_cycles':[64,255],'summary':summary,'cycles':cycles,
     'interpretation_limit':'Host stage timestamps reveal CPU launch and blocking phase only. They cannot be equated directly with NPU collective start/transfer or causal removable time. Diagnostic throughput is not formal E2E.'}
p=base/'phase_analysis.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
