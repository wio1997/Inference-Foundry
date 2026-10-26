#!/usr/bin/env python3
"""Reconstruct Run667 cohort wall from the frozen bench and rank0 Runtime rows."""
import json
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry/evidence/20260929_loop081_bound/run667')
BASE=ROOT/'live'
def main():
 out={'scope':'existing Run667 Product wall; group join inferred from serialized c12 admission; no absolute Runtime timestamps','arms':{}}
 for arm in ('off_a','on'):
  a=BASE/arm; repeats=[]
  for rep in (1,2,3):
   bench=json.loads((a/f'bench48_{rep}.json').read_text());q=bench['requests']
   assert len(q)==48 and [x['i'] for x in q]==list(range(48))
   cohorts=[]
   for j in range(4):
    z=q[12*j:12*j+12]; r=json.loads((a/'runtime'/f'rank0_cohort{4*rep+1+j}.json').read_text())
    assert len(r['req_ids'])==12 and r['pass'] and r['generated_output_counts']==[1024]*12
    start=min(x['start'] for x in z);end=max(x['end'] for x in z)
    first=max(x['start']+x['ttft_ms']/1000 for x in z)
    cohorts.append({'cohort':r['cohort'],'product_wall_s':end-start,
      'start_to_last_first_token_s':first-start,'fixed_runtime_s':r['wall_seconds'],
      'after_last_first_minus_runtime_s':end-first-r['wall_seconds'],
      'cycles':r['cycles'],'useful_tokens':12288,'useful_tokens_per_cycle':12288/r['cycles'],
      'overshoot_tokens':r['overshoot_tokens'],'acceptance_window_means':r['acceptance_window_means']})
   repeats.append({'repeat':rep,'client_duration_s':bench['summary']['duration_s'],
      'cohorts':cohorts,'cycles':sum(x['cycles'] for x in cohorts),
      'start_to_last_first_token_s':sum(x['start_to_last_first_token_s'] for x in cohorts),
      'fixed_runtime_s':sum(x['fixed_runtime_s'] for x in cohorts),
      'after_last_first_minus_runtime_s':sum(x['after_last_first_minus_runtime_s'] for x in cohorts)})
  out['arms'][arm]=repeats
 (ROOT/'product_wall_decomposition.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# Run667 Product wall reconstruction','',
  'The c12 client admits four serial groups of 12. The cohort join below follows their order and validates 12 Runtime request IDs and 12×1024 output per group. `bench.py` did not save response IDs, so the join is order-based, not an exact ID join. Runtime rows have duration but no absolute start/end stamp.','',
  '| repeat | arm | client s | sum start→last first s | fixed Runtime s | remaining after last first s | cycles | useful tok/cycle |',
  '|---|---|---:|---:|---:|---:|---:|---:|']
 for rep in range(3):
  for arm in ('off_a','on'):
   x=out['arms'][arm][rep]
   lines.append(f"| {rep+1} | {arm} | {x['client_duration_s']:.3f} | {x['start_to_last_first_token_s']:.3f} | {x['fixed_runtime_s']:.3f} | {x['after_last_first_minus_runtime_s']:.3f} | {x['cycles']} | {49152/x['cycles']:.3f} |")
 lines += ['','The larger ON client−Runtime residual in Run667 is nearly all in the start→last-first-token side of the cohort. The remaining post-first term is 0.48–0.77 s per four cohorts. ON repeat3 has 1256 vs OFF 1166 cycles, lowering useful tokens/cycle from 42.15 to 39.13 and adding ~3.1 s of Runtime wall independently of the ~10 s pre-first residual difference. This decomposition locates the wall region but cannot separate prefill, seed, scheduler handoff and per-cohort Graph capture, or establish causality. Run666 OFF/ON/OFF had ON pre-first shorter than both controls, so a direct Graph-causes-residual claim is unsupported. Run668 collects only missing absolute stage boundaries.']
 (ROOT/'product_wall_decomposition.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':'ok','repeats':6,'cohorts':24}))
if __name__=='__main__':main()
