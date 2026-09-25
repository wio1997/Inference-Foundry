#!/usr/bin/env python3
"""Split Run158 measured pre-handoff wall into worker calls and gaps."""
import json, statistics as st
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'evidence/20260925_loop045_boundary/run158'
bench=json.loads((base/'measured12.json').read_text())
client_start=min(r['start'] for r in bench['requests'])
assert bench['summary']['success']==12 and bench['summary']['fail']==0
rows=[]
for rank in range(8):
 b=json.loads((base/f'boundary/rank{rank}_cohort5.json').read_text())
 handoff=b['handoff_ns']/1e9
 files=list((base/'exec_timing').glob(f'rank{rank}_pid*.jsonl'))
 assert files,rank
 events=[]
 for p in files:
  events += [json.loads(line) for line in p.read_text().splitlines() if line]
 events=[e for e in events if client_start <= e['start_ns']/1e9 < handoff]
 events.sort(key=lambda e:e['start_ns'])
 execute=[e for e in events if e['method']=='execute_model']
 sample=[e for e in events if e['method']=='sample_tokens']
 # The final execute call contains the full Extreme Runtime run and
 # begins just before handoff; exclude it from pre-handoff call time.
 prefill=[e for e in execute if e['end_ns']/1e9 < handoff]
 final=[e for e in execute if e['end_ns']/1e9 >= handoff]
 assert len(final)==1 and len(prefill)>=10,(rank,len(final),len(prefill))
 first=execute[0]['start_ns']/1e9
 prefill_wall=sum((e['end_ns']-e['start_ns'])/1e9 for e in prefill)
 sample_wall=sum((min(e['end_ns']/1e9,handoff)-e['start_ns']/1e9) for e in sample)
 gap_to_next=[(execute[i+1]['start_ns']-execute[i]['end_ns'])/1e9 for i in range(len(prefill))]
 assert all(g>=-0.001 for g in gap_to_next)
 rows.append({'rank':rank,'span_s':handoff-first,'first_execute_vs_client_s':first-client_start,
  'prefill_execute_calls':len(prefill),'prefill_execute_wall_s':prefill_wall,
  'sample_calls':len(sample),'sample_wall_s':sample_wall,
  'gap_to_next_execute_sum_s':sum(gap_to_next),'gap_to_next_execute_median_s':st.median(gap_to_next),
  'gap_to_next_execute_max_s':max(gap_to_next),'final_handoff_preamble_s':handoff-final[0]['start_ns']/1e9,
  'prefill_call_rows':[{'tokens':e['scheduled_tokens'],'wall_s':(e['end_ns']-e['start_ns'])/1e9,'gap_to_next_s':gap_to_next[i]} for i,e in enumerate(prefill)]})
report={'run':'run158','contract':'8x910B3 DP1TP8 DSpark7, legal 48x1024 warmup + 12x1024 measured, diagnostic not formal E2E',
 'measured_summary':bench['summary'],'rank_rows':rows,
 'medians':{k:st.median(r[k] for r in rows) for k in ('span_s','first_execute_vs_client_s','prefill_execute_calls','prefill_execute_wall_s','sample_calls','sample_wall_s','gap_to_next_execute_sum_s','gap_to_next_execute_median_s','gap_to_next_execute_max_s','final_handoff_preamble_s')},
 'limits':'Python wall time in execute_model and sample_tokens; asynchronous NPU work may complete outside the measured Python call. Gap includes scheduler, RPC, tokenization, and any unmeasured worker methods. Not device time.'}
(base/'execute_timing_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'medians':report['medians'],'rank0_prefill':rows[0]['prefill_call_rows'],'limits':report['limits']},indent=2))
