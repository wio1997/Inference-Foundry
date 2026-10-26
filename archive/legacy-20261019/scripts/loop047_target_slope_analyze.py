#!/usr/bin/env python3
"""Analyze Run177 synced Stock FULL target scope and Run175 tail census."""
import json,statistics,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'evidence/20260925_loop047_tail/run177'
census=json.loads((root/'evidence/20260925_loop047_tail/run175/active_tail_census.json').read_text())
summary={}
for n in [12,8,6,4,1]:
 byrank={}
 for rank in range(8):
  f=p/'slope'/f'rank{rank}_size{8*n}.jsonl'
  rows=[json.loads(x) for x in f.read_text().splitlines()]
  assert len(rows)==24 and {x['call_index'] for x in rows}==set(range(64,88))
  assert all(x['rank']==rank and x['size']==8*n and x['num_reqs']==n and x['num_actual_tokens']==8*n and x['mode']=='FULL' for x in rows)
  byrank[rank]={x['call_index']:x for x in rows}
 maxima=[max(byrank[r][i]['event_ms'] for r in range(8)) for i in range(64,88)]
 median_critical=statistics.median(maxima)
 b=json.loads((p/f'measured{n}.json').read_text())['summary'];w=json.loads((p/f'warmup{n}.json').read_text())['summary']
 assert b['n']==n and b['success']==n and b['fail']==0 and b['max_tokens']==1024
 assert w['n']==n and w['success']==n and w['fail']==0 and w['max_tokens']==1024
 summary[str(n)]={'token_size':8*n,'rank_count':8,'calls_per_rank':24,'graph_mode':'FULL','critical_event_ms_median':median_critical,'critical_event_ms_min':min(maxima),'critical_event_ms_max':max(maxima),'rank_event_ms_medians':{str(r):statistics.median(byrank[r][i]['event_ms'] for i in range(64,88)) for r in range(8)},'min_computed_tokens':min(byrank[r][i]['min_computed_tokens'] for r in range(8) for i in range(64,88)),'max_computed_tokens':max(byrank[r][i]['max_computed_tokens'] for r in range(8) for i in range(64,88)),'measured_success':b['success'],'warmup_success':w['success']}
ms={int(k):v['critical_event_ms_median'] for k,v in summary.items()}
def bucket(active):
 return 12 if active>=9 else 8 if active>=7 else 6 if active>=5 else 4 if active>=2 else 1
cohorts=[]
for c in range(1,6):
 x=next(z for z in census['rows'] if z['cohort']==c and z['rank']==0)
 hist={int(k):v for k,v in x['active_hist'].items()}
 screen=sum(count*(ms[12]-ms[bucket(active)])/1000 for active,count in hist.items())
 endpoint=sum(count*(ms[12]-ms[1])/1000 for active,count in hist.items() if active<12)
 cohorts.append({'cohort':c,'cycles':x['cycles'],'partially_active_cycles':sum(count for active,count in hist.items() if active<12),'bucket_mapped_target_screen_s':screen,'all_partial_cycles_at_c1_endpoint_s':endpoint,'parked_fraction':x['parked_fraction'],'active_hist':hist})
source=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
restore=json.loads((p/'patch_restore.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==restore['restored_sha256']
out={'run':'run177','scope':'Stock-only same-service c12,c8,c6,c4,c1 legal warmed 1024-output diagnostics; 24 explicit pre/post-synchronized _model_forward FULL graph calls per size per rank; no Extreme handoff or formal E2E','size_summary':summary,'c12_to_c1_critical_delta_ms':ms[12]-ms[1],'one_second_threshold_per_partially_active_cycle_ms':1000/cohorts[-1]['partially_active_cycles'],'tail_screens':cohorts,'borrowed_model_runner_restored':True,'limits':['_model_forward includes model, graph-param update and hidden/aux gather but excludes outer context/RoPE and compute_logits; compare contemporaneous sizes only.','Explicit synchronization perturbs cadence; event intervals are diagnostic, not unprofiled target-stage latency.','Sizes ran sequentially with no repeated c12 at end; temporal/routing/thermal drift remains possible.','Stock target graph and Extreme bound closure may differ in metadata and graph update; same-size event deltas are a screen only.','Bucket-mapped target screen assumes c12 event baseline and existing graph sizes can be switched with zero packing/state cost, and uses c12 for active9-11; it is not achieved speedup or a formal bound.','All-partial-at-c1 endpoint ignores active-count constraints and is an impossible optimistic ceiling.']}
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'size_critical_ms':{k:v['critical_event_ms_median'] for k,v in summary.items()},'measured_tail_screen_s':cohorts[-1]['bucket_mapped_target_screen_s'],'measured_tail_c1_endpoint_s':cohorts[-1]['all_partial_cycles_at_c1_endpoint_s'],'source_restored':True},indent=2))
