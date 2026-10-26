#!/usr/bin/env python3
"""Analyze Run140 proposer wall/thread-CPU spans without summing nested calls."""
import json,statistics
from collections import Counter
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'evidence/20260925_loop042_phase/run140'
bench=json.loads((base/'bench.json').read_text())['summary']
assert bench['n']==12 and bench['success']==12 and bench['fail']==0 and bench['max_tokens']==1024
rank_rows={r:{x['cycle']:x for x in json.loads((base/f'subphase/rank{r}.json').read_text())['rows']} for r in range(8)}
assert all(all(c in rank_rows[r] for c in range(64,256)) for r in rank_rows)
labels=['begin','host_mirror','refresh_common','prepare_inputs','pack_hidden','model']
rows=[]
for rank in range(8):
    for cycle in range(64,256):
        x=rank_rows[rank][cycle]
        marks={m['name']:m for m in x['marks']}
        assert set(marks)==set(labels)
        stages={}
        for prev,label in zip(labels[:-1],labels[1:]):
            a,b=marks[prev],marks[label]
            stages[label]={'wall_ms':(b['wall_ns']-a['wall_ns'])/1e6,
                           'thread_cpu_ms':(b['thread_cpu_ns']-a['thread_cpu_ns'])/1e6}
        calls={q['name']:{'wall_ms':(q['wall_end_ns']-q['wall_start_ns'])/1e6,
                          'thread_cpu_ms':q['thread_cpu_ns']/1e6} for q in x['calls']}
        assert len(calls)==len(x['calls'])
        rows.append({'rank':rank,'cycle':cycle,'stages':stages,'calls':calls,
                     'begin_ns':marks['begin']['wall_ns'],'end_ns':marks['model']['wall_ns']})
def desc(xs):
    xs=sorted(xs);return {'median':statistics.median(xs),'p10':xs[int(.1*(len(xs)-1))],
                          'p90':xs[int(.9*(len(xs)-1))],'min':xs[0],'max':xs[-1]}
methods=sorted(set(k for x in rows for k in x['calls']))
summary={'stage':{label:{metric:desc([x['stages'][label][metric] for x in rows])
                          for metric in ('wall_ms','thread_cpu_ms')} for label in labels[1:]},
         'calls':{name:{'count':sum(name in x['calls'] for x in rows),
                        **{metric:desc([x['calls'][name][metric] for x in rows if name in x['calls']])
                            for metric in ('wall_ms','thread_cpu_ms')}} for name in methods},
         'method_presence_by_rank':{str(r):dict(Counter(k for c in range(64,256) for k in next(x for x in rows if x['rank']==r and x['cycle']==c)['calls'])) for r in range(8)}}
# These are nested method calls. Runnable includes possible subcalls, so never add them.
latest_end=[max(rank_rows[r][c]['marks'][-1]['wall_ns'] for r in range(8)) for c in range(64,256)]
summary['latest_rank_proposer_end_cadence_ms']=desc([(b-a)/1e6 for a,b in zip(latest_end,latest_end[1:])])
summary['latest_rank_proposer_end_span_mean_ms']=(latest_end[-1]-latest_end[0])/1e6/(len(latest_end)-1)
out={'run':'run140','contract':'8-rank legal 12x1024 c12 DSpark7, 290 cycles/rank; steady cycles64-255',
     'bench_summary':bench,'clock':'perf_counter_ns and thread_time_ns, no added NPU synchronization',
     'summary':summary,
     'interpretation_limit':'Thread CPU near wall supports CPU work in this thread, but may include busy waits in C++ and cannot by itself prove removable overhead. Nested calls cannot be summed; diagnostic benchmark is not formal E2E.'}
p=base/'subphase_analysis.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
