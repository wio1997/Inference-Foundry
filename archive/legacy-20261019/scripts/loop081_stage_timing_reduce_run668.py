#!/usr/bin/env python3
"""Exact response-ID join and absolute stage-clock reduction for Run668."""
import json,statistics
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry/evidence/20260929_loop081_bound/run668')
BASE=ROOT/'live'
STAGES=('stage_handoff_entry_ns','stage_build_done_ns','stage_graph_begin_ns',
        'stage_graph_end_ns','stage_runtime_begin_ns','stage_runtime_end_ns')
def ms(a,b): return (b-a)/1e6
def main():
 out={'scope':'stage-aligned diagnostic; 48 warmup+48 measured; not formal TPS','arms':{}}
 for arm in ('off_a','on','off_b'):
  p=BASE/arm
  requests=[json.loads(f.read_text())['row'] for f in (p/'measured_client').glob('request_*.json')]
  assert len(requests)==48
  byid={r['response_id']:r for r in requests}
  assert len(byid)==48
  cohorts=[]
  for c in (5,6,7,8):
   rows=[json.loads((p/'runtime'/f'rank{k}_cohort{c}.json').read_text()) for k in range(8)]
   assert all(r['pass'] and r['target_graph_mode']=='FULL' for r in rows)
   assert all(all(isinstance(r[key],int) for key in STAGES) for r in rows)
   req_ids=rows[0]['req_ids'];assert all(r['req_ids']==req_ids for r in rows)
   q=[byid[rid.rsplit('-',1)[0]] for rid in req_ids]
   st=min(r['start_monotonic_ns'] for r in q)
   first=max(r['start_monotonic_ns']+round(r['ttft_ms']*1e6) for r in q)
   end=max(r['end_monotonic_ns'] for r in q)
   # Rank0 is the client-facing TP root; also retain all-rank ranges to expose skew.
   r=rows[0]
   cohort={'cohort':c,'cycles':r['cycles'],'useful_tokens_per_cycle':12288/r['cycles'],
       'product_wall_ms':ms(st,end),'start_to_last_first_ms':ms(st,first),
       'client_start_to_handoff_ms':ms(st,r['stage_handoff_entry_ns']),
       'handoff_to_build_done_ms':ms(r['stage_handoff_entry_ns'],r['stage_build_done_ns']),
       'graph_capture_scope_ms':ms(r['stage_graph_begin_ns'],r['stage_graph_end_ns']),
       'graph_end_to_runtime_begin_ms':ms(r['stage_graph_end_ns'],r['stage_runtime_begin_ns']),
       'runtime_ms':ms(r['stage_runtime_begin_ns'],r['stage_runtime_end_ns']),
       'runtime_end_to_client_end_ms':ms(r['stage_runtime_end_ns'],end),
       'last_first_minus_handoff_ms':ms(r['stage_handoff_entry_ns'],first),
       'capture_rank_ms':[ms(x['stage_graph_begin_ns'],x['stage_graph_end_ns']) for x in rows],
       'runtime_rank_ms':[ms(x['stage_runtime_begin_ns'],x['stage_runtime_end_ns']) for x in rows],
       'first_token_after_capture_all_ranks':first>=max(x['stage_graph_end_ns'] for x in rows),
       'rank0_runtime_vs_report_ms':ms(r['stage_runtime_begin_ns'],r['stage_runtime_end_ns'])-r['wall_seconds']*1000}
   cohorts.append(cohort)
  out['arms'][arm]={'cohorts':cohorts,'totals':{
   key:sum(x[key] for x in cohorts) for key in ('product_wall_ms','start_to_last_first_ms',
    'client_start_to_handoff_ms','handoff_to_build_done_ms','graph_capture_scope_ms',
    'graph_end_to_runtime_begin_ms','runtime_ms','runtime_end_to_client_end_ms')},
   'cycles':sum(x['cycles'] for x in cohorts)}
 out['source_restored']=(BASE/'source_before.sha256').read_bytes()==(BASE/'source_after.sha256').read_bytes()
 out['cleanup']=(BASE/'cleanup_status.txt').read_text().splitlines()
 (ROOT/'stage_timing_summary.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'status':'reduced','arms':3,'cohorts':12,'source_restored':out['source_restored']}))
 for arm,x in out['arms'].items():
  print(arm,'cycles',x['cycles'],'totals_ms',{k:round(v,2) for k,v in x['totals'].items()})
  print('capture per cohort rank0 ms',[round(c['graph_capture_scope_ms'],2) for c in x['cohorts']])
  print('first after all-rank capture',[c['first_token_after_capture_all_ranks'] for c in x['cohorts']])
if __name__=='__main__':main()
