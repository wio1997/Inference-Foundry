#!/usr/bin/env python3
"""Summarize Run142 actual DSpark proposer segments, all eight ranks."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'evidence/20260925_loop043_dspark/run142'
bench=json.loads((base/'bench.json').read_text())['summary']
assert bench['n']==12 and bench['success']==12 and bench['fail']==0 and bench['max_tokens']==1024
names=['begin','post_context_kv','post_model','pre_lmhead','post_lmhead','post_markov']
rows=[]
for rank in range(8):
    trace=json.loads((base/f'segments/rank{rank}.json').read_text())
    assert len(trace['rows'])==64
    assert [x['cycle'] for x in trace['rows']]==list(range(64,128))
    for row in trace['rows']:
        marks=row['marks'];assert [m['name'] for m in marks]==names
        stages={}
        for a,b in zip(marks,marks[1:]):
            stages[b['name']]={'host_wall_ms':(b['wall_ns']-a['wall_ns'])/1e6,
                               'host_thread_cpu_ms':(b['thread_cpu_ns']-a['thread_cpu_ns'])/1e6,
                               'device_event_ms':row['device_stage_ms'][b['name']]}
        rows.append({'rank':rank,'cycle':row['cycle'],'stages':stages})
assert len(rows)==512
def describe(xs):
    xs=sorted(xs);return {'median':statistics.median(xs),'p10':xs[int(.1*(len(xs)-1))],
                          'p90':xs[int(.9*(len(xs)-1))],'min':xs[0],'max':xs[-1]}
summary={name:{metric:describe([x['stages'][name][metric] for x in rows])
               for metric in ('host_wall_ms','host_thread_cpu_ms','device_event_ms')}
         for name in names[1:]}
summary['runnable_total']={metric:describe([sum(x['stages'][n][metric] for n in names[1:]) for x in rows])
                          for metric in ('host_wall_ms','host_thread_cpu_ms','device_event_ms')}
out={'run':'run142','classification':'diagnostic_profile','contract':'8 ranks, legal 12x1024 c12 DSpark7, 64 consecutive steady segment captures/rank',
     'bench_summary':bench,'summary':summary,
     'segment_names':{'post_context_kv':'input preparation through build_model_inputs_first_pass',
                      'post_model':'three-layer DSpark forward plus adjacent setup',
                      'pre_lmhead':'gather and sample-hidden selection',
                      'post_lmhead':'compute_logits/TP path',
                      'post_markov':'seed plus seven sequential Markov corrections'},
     'limits':'Device event intervals are diagnostic stream intervals; sums may include waits and need not equal uninstrumented proposer time. No forced sync inside the cycle; one sync after cohort for event readout. Host/thread timing may include driver busy wait. Diagnostic output_tps is not formal E2E.',
     'decision_hint':'If Markov tail device interval is sub-ms and total proposer event ~6ms, replay has small possible cycle effect; prioritize larger target device path unless measured end-to-end critical-path gain can be shown.'}
p=base/'segment_analysis.json';p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
