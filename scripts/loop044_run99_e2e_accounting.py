#!/usr/bin/env python3
"""Reconcile formal Run99 client envelopes with fixed-runtime timer scopes."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'evidence/20260924_loop036_metadata/run99'
repeats=[]
for repeat in (1,2,3):
 d=json.loads((base/f'bench48_{repeat}.json').read_text())
 summary=d['summary'];req=sorted(d['requests'],key=lambda x:x['i'])
 assert len(req)==48 and summary['success']==48 and summary['fail']==0
 cohort_rows=[]
 runtime_sum=0.0
 for j in range(4):
  subset=req[j*12:(j+1)*12]
  assert len(subset)==12
  runtime_id=4*repeat+j+1
  runtime=json.loads((base/f'runtime/rank0_cohort{runtime_id}.json').read_text())
  envelope=max(x['end'] for x in subset)-min(x['start'] for x in subset)
  wall=float(runtime['wall_seconds']);runtime_sum+=wall
  cohort_rows.append({'client_cohort_index':j,'runtime_cohort_id':runtime_id,
   'request_envelope_s':envelope,'runtime_wall_s':wall,
   'outside_runtime_scope_s':envelope-wall,
   'max_ttft_s':max(x['ttft_ms'] for x in subset)/1000,
   'mean_ttft_s':statistics.mean(x['ttft_ms'] for x in subset)/1000,
   'cycles':runtime['cycles'],
   'acceptance_last_window':list(runtime['acceptance_window_means'].items())[-1]})
 duration=float(summary['duration_s'])
 repeats.append({'repeat':repeat,'client_duration_s':duration,'runtime_sum_s':runtime_sum,
  'difference_s':duration-runtime_sum,'outside_fraction':(duration-runtime_sum)/duration,
  'cohorts':cohort_rows})
source=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py').read_text()
assert source.index('_extreme_runtime = build_extreme_runtime(')<source.index('_extreme_wall_start = time.perf_counter()')
stock_max_ttft=[]
for repeat in (1,2,3):
 req=json.loads((root/f'evidence/20260920_baseline/bench48_{repeat}.json').read_text())['requests']
 for j in range(4):stock_max_ttft.append(max(x['ttft_ms'] for x in req[j*12:(j+1)*12])/1000)
cohorts=[c for r in repeats for c in r['cohorts']]
residual=[c['outside_runtime_scope_s']-c['max_ttft_s'] for c in cohorts]
out={'run':'run153','source':'Run99 three formal bench48 repeats and rank0 cohort5..16 runtime JSON',
 'repeats':repeats,
 'median_client_duration_s':statistics.median(r['client_duration_s'] for r in repeats),
 'median_outside_scope_s':statistics.median(r['difference_s'] for r in repeats),
 'median_outside_fraction':statistics.median(r['outside_fraction'] for r in repeats),
 'median_outside_minus_max_ttft_s':statistics.median(residual),
 'outside_minus_max_ttft_range_s':[min(residual),max(residual)],
 'median_cohort_max_ttft_extreme_s':statistics.median(c['max_ttft_s'] for c in cohorts),
 'median_cohort_max_ttft_stock_s':statistics.median(stock_max_ttft),
 'scope_boundary':'model_runner_v1.py constructs Extreme Runtime before _extreme_wall_start, then times FixedCohortServing.run plus NPU synchronize. Client envelope also includes request admission/prefill/bootstrap/publication. Existing JSON lacks common-clock start/end timestamps for these stages. Run99 outside-minus-max-TTFT correlation supports a first-token-range attribution but not a causal split; Stock baseline was on a different date.',
 'tail_limit':'Run99 acceptance_window_means are aggregate over 12 slots. Per-slot parked cycle counts and count_history were not persisted; tail compaction opportunity cannot be quantified from these files.',
 'decision':'The 11-17s client-vs-runtime difference is real accounting but not established removable overhead. Highest-value next legal capture must timestamp admission, prefill, runtime construction, handoff, FixedCohortServing, and publication on a common host clock, plus per-slot completion cycles. Do not subtract cross-scope medians as a performance gain.'}
p=root/'evidence/20260925_loop044_target/run153/e2e_accounting.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ('repeats',)},indent=2))
for r in repeats:print(json.dumps({'repeat':r['repeat'],'client_s':r['client_duration_s'],'runtime_s':r['runtime_sum_s'],'outside_s':r['difference_s'],'cohorts':r['cohorts']},indent=2))
