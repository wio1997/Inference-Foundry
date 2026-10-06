"""Saved Run258 only: matched native boundaries, no live sampling/import torch."""
import collections,hashlib,json,statistics,struct
from pathlib import Path


def reduce(root):
 raw=(root/'native_boundary_raw.json').read_bytes();d=json.loads(raw);observer=json.loads((root/'observer_final.json').read_text())
 assert hashlib.sha256(raw).hexdigest()==observer['raw_sha256'] and len(raw)==observer['raw_bytes']
 assert len(d['events'])==9120
 assert all(x['enabled']==x['running'] for x in d['events'])
 assert d['lost']==0 and not any('lost' in x or 'other_record_type' in x for x in d['events'])
 result=json.loads((root/'native_boundaries_result.json').read_text());assert result['token_ids']==[785,1196,374,10156,264,3405,304,8452] and result['prompt_tokens']==2334 and result['finish_reason']=='length'
 assert not result['profiler_active'] and any('hits_total' in k and n==2334 for k,n in result['external_KV_delta'].items())
 assert json.loads((root/'guards_before.json').read_text())==json.loads((root/'guards_after.json').read_text())
 assert json.loads((root/'native_before.json').read_text())==json.loads((root/'native_after.json').read_text())
 assert json.loads((root/'probe_cleanup.json').read_text())['removed']
 plan=json.loads((root/'observer_plan.json').read_text());pid_to_rank={x['pid']:x['rank_name'] for x in plan['workers']};rows=[];outside=[];pairs=collections.Counter();negative=collections.Counter()
 for pid in pid_to_rank:
  events=[x for x in d['events'] if x['tid']==pid];assert events and all(x['pid']==x['tid']==x['target_tid'] for x in events)
  assert all(struct.unpack_from('<i',bytes.fromhex(x['raw']),4)[0]==pid for x in events)
  assert all(a['task_cpu_ns']<=b['task_cpu_ns'] for a,b in zip(events,events[1:]))
  active={};native=None
  for x in events:
   name,side=x['event'].rsplit('_',1)
   if side=='enter':
    assert name not in active,(name,'reentrant');active[name]=x
    if name.startswith('native_'):assert native is None;native=dict(name=name,enter=x,children={})
   else:
    a=active.pop(name);pair=dict(enter=a,exit=x);pairs[(pid,name)]+=1
    if name.startswith('native_'):
     assert native and native['name']==name;kind=name.removeprefix('native_');children=native['children'];assert set(children)=={'predicate','workspace_'+kind},children
     pred=children['predicate'];work=children['workspace_'+kind];marks=[a,pred['enter'],pred['exit'],work['enter'],work['exit'],x]
     assert all(z['wall_ns']<=y['wall_ns'] and z['task_cpu_ns']<=y['task_cpu_ns'] for z,y in zip(marks,marks[1:]))
     phases={}
     for phase,left,right in zip(['entry_to_predicate','predicate','predicate_to_workspace','workspace','workspace_to_return'],marks,marks[1:]):
      wall=right['wall_ns']-left['wall_ns'];cpu=right['task_cpu_ns']-left['task_cpu_ns'];residual=wall-cpu
      phases[phase]=dict(wall_ns=wall,task_cpu_ns=cpu,signed_wall_minus_task_cpu_ns=residual)
      if residual<0:negative[(pid,phase)]+=1
     wall=x['wall_ns']-a['wall_ns'];cpu=x['task_cpu_ns']-a['task_cpu_ns'];assert sum(y['wall_ns'] for y in phases.values())==wall and sum(y['task_cpu_ns'] for y in phases.values())==cpu
     rows.append(dict(pid=pid,rank_name=pid_to_rank[pid],kind=kind,enter_ns=a['wall_ns'],exit_ns=x['wall_ns'],native_wall_ns=wall,native_task_cpu_ns=cpu,phases=phases));native=None
    elif native is not None:
     assert name not in native['children'];native['children'][name]=pair
    else:outside.append(dict(pid=pid,name=name,wall_ns=x['wall_ns']-a['wall_ns'],task_cpu_ns=x['task_cpu_ns']-a['task_cpu_ns']))
  assert not active and native is None
 assert not outside,outside
 for pid in pid_to_rank:
  expected=['native_dispatch','native_combine']*380
  ordered=[x['kind'] for x in rows if x['pid']==pid]
  assert ordered==[name.removeprefix('native_') for name in expected]
 summaries=[]
 for pid in pid_to_rank:
  for kind in ['dispatch','combine']:
   xs=[x for x in rows if x['pid']==pid and x['kind']==kind];assert len(xs)==380
   phase={}
   for name in xs[0]['phases']:
    phase[name]={key:dict(sum_ms=sum(x['phases'][name][key] for x in xs)/1e6,median_us=statistics.median(x['phases'][name][key] for x in xs)/1e3,min_us=min(x['phases'][name][key] for x in xs)/1e3,max_us=max(x['phases'][name][key] for x in xs)/1e3) for key in ['wall_ns','task_cpu_ns','signed_wall_minus_task_cpu_ns']}
   total=sum(x['native_task_cpu_ns'] for x in xs);pred=sum(x['phases']['predicate']['task_cpu_ns'] for x in xs)
   summaries.append(dict(pid=pid,rank_name=pid_to_rank[pid],kind=kind,calls=len(xs),native_wall_ms=sum(x['native_wall_ns'] for x in xs)/1e6,native_task_cpu_ms=total/1e6,predicate_task_cpu_fraction=pred/total,phases=phase))
 envelopes=[]
 for pid in pid_to_rank:
  marks=[x for x in d['events'] if x['tid']==pid and x['event'].startswith('native_')]
  assert len(marks)==1520
  spans=collections.defaultdict(list)
  for a,b in zip(marks,marks[1:]):
   if a['event'].endswith('_enter'):
    assert b['event']==a['event'].removesuffix('_enter')+'_exit'
    phase='native_dispatch_or_combine'
   elif a['event']=='native_dispatch_exit':
    assert b['event']=='native_combine_enter';phase='routed_MLP_dispatch_return_to_combine_entry'
   else:
    assert a['event']=='native_combine_exit' and b['event']=='native_dispatch_enter'
    phase='interlayer_or_interround_combine_return_to_next_dispatch_entry'
   spans[phase].append((b['wall_ns']-a['wall_ns'],b['task_cpu_ns']-a['task_cpu_ns']))
  phases={n:dict(count=len(xs),wall_ms=sum(x[0] for x in xs)/1e6,task_cpu_ms=sum(x[1] for x in xs)/1e6,median_task_cpu_us=statistics.median(x[1] for x in xs)/1e3) for n,xs in spans.items()}
  wall=(marks[-1]['wall_ns']-marks[0]['wall_ns'])/1e6;cpu=(marks[-1]['task_cpu_ns']-marks[0]['task_cpu_ns'])/1e6
  assert abs(sum(x['wall_ms'] for x in phases.values())-wall)<1e-8 and abs(sum(x['task_cpu_ms'] for x in phases.values())-cpu)<1e-8
  envelopes.append(dict(rank_name=pid_to_rank[pid],wall_ms=wall,task_cpu_ms=cpu,phases=phases,limits='One stock request measured task-clock envelope, not source-phase allocation or removable budget. Includes four inter-round intervals; excludes prefix before first dispatch and suffix after last combine.'))
 round_summaries=[]
 for pid in pid_to_rank:
  xs=[x for x in rows if x['pid']==pid]
  for step in range(5):
   window=xs[step*152:(step+1)*152];assert len(window)==152
   total=sum(x['native_task_cpu_ns'] for x in window)/1e6;pred=sum(x['phases']['predicate']['task_cpu_ns'] for x in window)/1e6
   round_summaries.append(dict(pid=pid,rank_name=pid_to_rank[pid],round_by_76_MoE_calls=step+1,calls=152,native_task_cpu_ms=total,predicate_task_cpu_ms=pred,remaining_task_cpu_ms=total-pred,first_dispatch_ns=window[0]['enter_ns'],last_combine_return_ns=window[-1]['exit_ns'],limits='Source-count ordinal grouping, not exact ModelRunner/step boundary markers.'))
 return dict(native_first_to_last_envelopes=envelopes,round_summaries=round_summaries,run='GLM-RUN-0258',raw_sha256=hashlib.sha256(raw).hexdigest(),events=len(d['events']),lost=d['lost'],summaries=summaries,calls=rows,outside_native=outside,negative_residual_counts={str(k):v for k,v in negative.items()},request=result,verdict='COMPLETE_BOUNDARIES_AVAILABILITY_DOMINATES_MC2_NATIVE',limits=['Same stock request, same-TID scheduled task-clock group samples and wall timestamps; no Run249/profileOFF subtraction.','Task CPU includes probe and kernel overhead; CPU fixture median reference error4.730us, cold max177.840us. Signed wall-minus-task-clock is retained, not silently clamped.','Entry-to-predicate contains allocations/checks; predicate-to-workspace contains dynamic descriptors/cache/determinism preparation; workspace-to-return contains allocation/submission/cleanup. These are source-bounded intervals, not exact function costs.','No consumer runnable/idle or full Scheduler/Executor/ModelRunner entry/exit claim.','Instrumented golden8 request validates scope/output/KV; its latency is diagnostic, not matched code/E2E gain.'])


if __name__=='__main__':
 here=Path(__file__).resolve().parent;root=here.parents[1]/'runs/GLM-RUN-0258';out=reduce(root)
 detail=dict(calls=out.pop('calls'),outside_native=out.pop('outside_native'));import gzip
 (root/'boundary_calls.json.gz').write_bytes(gzip.compress(json.dumps(detail,separators=(',',':')).encode(),mtime=0))
 out['detail_file']='boundary_calls.json.gz';(root/'native_boundaries_reduced.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({k:out[k] for k in ['events','lost','negative_residual_counts']}))
 for s in out['summaries']:
  print(s['rank_name'],s['kind'],s['calls'],s['native_wall_ms'],s['native_task_cpu_ms'],s['predicate_task_cpu_fraction'],{k:dict(wall_us=v['wall_ns']['median_us'],cpu_us=v['task_cpu_ns']['median_us']) for k,v in s['phases'].items()})
