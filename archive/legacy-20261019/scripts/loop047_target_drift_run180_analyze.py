#!/usr/bin/env python3
"""Check legal same-service Stock target-size slope against repeated c12."""
import hashlib,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=root/'evidence/20260925_loop047_tail/run180'
census=json.loads((root/'evidence/20260925_loop047_tail/run175/active_tail_census.json').read_text())
summary={}
for epoch in ['12a','8a','6a','4a','1a','12b']:
 n=int(epoch[:-1]);tag=epoch[-1];byrank={}
 for rank in range(8):
  f=p/'slope'/f'rank{rank}_size{8*n}_{tag}.jsonl'
  rows=[json.loads(x) for x in f.read_text().splitlines()]
  assert len(rows)==24 and {x['call_index'] for x in rows}==set(range(64,88)),(epoch,rank,len(rows))
  assert all(x['rank']==rank and x['size']==8*n and x['epoch']==tag and x['num_reqs']==n and x['num_actual_tokens']==8*n and x['mode']=='FULL' for x in rows),(epoch,rank)
  byrank[rank]={x['call_index']:x for x in rows}
 maxima=[max(byrank[r][i]['event_ms'] for r in range(8)) for i in range(64,88)]
 b=json.loads((p/f'measured{epoch}.json').read_text())['summary']
 w=json.loads((p/f'warmup{epoch}.json').read_text())['summary']
 for z in [b,w]:assert z['n']==n and z['success']==n and z['fail']==0 and z['max_tokens']==1024,(epoch,z)
 summary[epoch]={'token_size':8*n,'rank_count':8,'calls_per_rank':24,'mode':'FULL','critical_event_ms_median':statistics.median(maxima),'critical_event_ms_min':min(maxima),'critical_event_ms_max':max(maxima),'rank_event_ms_medians':{str(r):statistics.median(byrank[r][i]['event_ms'] for i in range(64,88)) for r in range(8)},'min_computed_tokens':min(byrank[r][i]['min_computed_tokens'] for r in range(8) for i in range(64,88)),'max_computed_tokens':max(byrank[r][i]['max_computed_tokens'] for r in range(8) for i in range(64,88)),'measured_success':b['success'],'warmup_success':w['success']}
ms={e:v['critical_event_ms_median'] for e,v in summary.items()}
def bucket(active):return '12a' if active>=9 else ('8a' if active>=7 else ('6a' if active>=5 else ('4a' if active>=2 else '1a')))
tails=[]
for c in range(1,6):
 x=next(z for z in census['rows'] if z['cohort']==c and z['rank']==0)
 hist={int(k):v for k,v in x['active_hist'].items()}
 screens={}
 for baseline_name,base in [('early_c12',ms['12a']),('late_c12',ms['12b']),('mean_c12',(ms['12a']+ms['12b'])/2)]:
  screens[baseline_name]=sum(count*(base-(base if active>=9 else ms[bucket(active)]))/1000 for active,count in hist.items())
 tails.append({'cohort':c,'cycles':x['cycles'],'partially_active_cycles':sum(v for k,v in hist.items() if k<12),'bucket_screen_s_by_baseline':screens,'parked_fraction':x['parked_fraction']})
source=Path('/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py')
restore=json.loads((p/'patch_restore.json').read_text())
assert hashlib.sha256(source.read_bytes()).hexdigest()==restore['restored_sha256']
out={'run':'run180','scope':'Stock-only same-service warmed c12a,c8,c6,c4,c1,c12b; 24 synchronized FULL _model_forward event calls per epoch/rank; legal 1024-output serving guard; diagnostic only','epochs':summary,'c12_drift_ms':ms['12b']-ms['12a'],'c12_to_c1_delta_ms_early':ms['12a']-ms['1a'],'c12_to_c1_delta_ms_late':ms['12b']-ms['1a'],'tail_screens':tails,'source_restored':True,'limitations':['Stock _model_forward is not exact Extreme target.execute scope or graph/metadata closure.','Explicit pre/post NPU synchronization perturbs normal cadence.','Cohorts run sequentially; repeated c12 controls only a two-point endpoint drift, not all temporal or routing differences.','Screens use Run155 active distribution and assume zero packing, graph switching, KV mapping, metadata, sync, and correctness costs; they are neither achieved speedup nor a formal bound.']}
(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'critical_ms':ms,'c12_drift_ms':out['c12_drift_ms'],'measured_tail_screens_s':tails[-1]['bucket_screen_s_by_baseline'],'source_restored':True},indent=2))
