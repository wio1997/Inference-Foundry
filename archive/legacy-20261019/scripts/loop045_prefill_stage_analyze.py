#!/usr/bin/env python3
"""Analyze Run161 measured cohort Host-visible prefill stage wall."""
import json, statistics as st
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'evidence/20260925_loop045_boundary/run161'
bench=json.loads((base/'measured12.json').read_text())
start=min(r['start'] for r in bench['requests'])
assert bench['summary']['success']==12 and bench['summary']['fail']==0
rows=[]
stages=['prep','bridge','forward_context','model_forward','forward_tail','postprocess']
for rank in range(8):
 b=json.loads((base/f'boundary/rank{rank}_cohort5.json').read_text())
 handoff=b['handoff_ns']/1e9
 files=list((base/'exec_timing').glob(f'rank{rank}_pid*.jsonl'))
 assert len(files)==1,(rank,files)
 events=[json.loads(line) for line in files[0].read_text().splitlines() if line]
 prefill=[e for e in events if e['method']=='execute_model' and start<=e['start_ns']/1e9 and e['end_ns']/1e9<handoff]
 assert len(prefill)>=9,(rank,len(prefill))
 per=[]
 for e in prefill:
  marks={k:t for k,t in e['stage_marks']}
  names=['prep_start','prep_end','forward_context_start','model_forward_start','model_forward_end','post_start']
  assert list(marks)==names,(rank,e['scheduled_tokens'],marks)
  boundaries=[e['start_ns'],marks['prep_start'],marks['prep_end'],marks['forward_context_start'],marks['model_forward_start'],marks['model_forward_end'],marks['post_start'],e['end_ns']]
  assert boundaries==sorted(boundaries)
  deltas=[(boundaries[i+1]-boundaries[i])/1e9 for i in range(len(boundaries)-1)]
  per.append({'tokens':e['scheduled_tokens'],'entry_s':deltas[0],**{stage:val for stage,val in zip(stages,deltas[1:])},'wall_s':(e['end_ns']-e['start_ns'])/1e9})
 sums={k:sum(p[k] for p in per) for k in ['entry_s',*stages,'wall_s']}
 assert abs(sum(sums[k] for k in ['entry_s',*stages])-sums['wall_s'])<1e-6
 rows.append({'rank':rank,'calls':len(per),'sums_s':sums,'prefill_calls':per})
keys=['entry_s',*stages,'wall_s']
medians={k:st.median(r['sums_s'][k] for r in rows) for k in keys}
report={'run':'run161','source':'8 rank per-call Run161 entry/exit and source stage marks, measured cohort5',
 'rank_rows':rows,'median_stage_sums_s':medians,
 'limits':'Python wall only. Device work can be asynchronous. Model-forward wall may include launch, HCCL blocking, synchronization, or device wait; direct NPU event/profile needed before device attribution.'}
(base/'stage_analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'medians':medians,'rank0_calls':rows[0]['prefill_calls'],'limits':report['limits']},indent=2))
