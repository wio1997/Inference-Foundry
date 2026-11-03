#!/usr/bin/env python3
"""Compare previously matched db852bff MC2 endpoints; no NPU access."""
import hashlib,json
from decimal import Decimal
from pathlib import Path
p=Path(__file__).resolve().parents[2]
src=p/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
want='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
assert hashlib.sha256(src.read_bytes()).hexdigest()==want
r=[]; signatures=set(); neighbors=[]
for ln in src.open():
 x=json.loads(ln)
 if x.get('kind')!='rank':continue
 d=x['data']
 for layer in d['layers']:
  if not layer['scheduler_digest'].startswith('db852bff') or layer['layer'] not in [15,36]:continue
  assert layer['trace_scope_count']==1 and layer['sidecar_thread_matches_trace']
  candidates=[g for g in layer['operator_groups'] if g['operator_name']=='DispatchFFNCombine']
  assert len(candidates)==1 and candidates[0]['task_count']==1
  g=candidates[0];t=g['tasks'][0];c=t['csv']
  if layer['layer']==15 and d['ep'] in (0,3):
   start=Decimal(t['device_start_us']); same=[]
   for ng in layer['operator_groups']:
    for nt in ng['tasks']:
     a=Decimal(nt['device_start_us']); nc=nt['csv']
     if a<start and nc['Stream ID']==c['Stream ID']:
      same.append((a,ng['operator_name'],nt))
   same.sort(key=lambda v:v[0],reverse=True)
   a,name,nt=same[0];end=a+Decimal(nt['csv']['Duration(us)'])
   neighbors.append({'ep':d['ep'],'layer':15,'last_matched_name':name,'last_matched_task':nt['csv']['Task ID'],'last_matched_start_us':str(a),'last_matched_duration_us':nt['csv']['Duration(us)'],'interval_to_mc2_us':str(start-end)})
  signatures.add(json.dumps([g['physical_shape_dtype_format'],c['Type'],c['Accelerator Core']],sort_keys=True))
  assert c['Name']=='DispatchFFNCombine' and Decimal(t['duration_abs_difference_us']) <= Decimal('0.001')
  r.append(dict(ep=d['ep'],rank=d['rank'],layer=layer['layer'],scope=layer['layer_scope_id'],flow=t['flow_id'],csv_line=t['csv_line'],start_us=t['device_start_us'],duration_us=c['Duration(us)'],stream=c['Stream ID'],task=c['Task ID'],model=c['Model ID'],aic_cycles=c['aic_total_cycles'],aic_mac=c['aic_mac_ratio'],aiv_cycles=c['aiv_total_cycles'],aiv_vec=c['aiv_vec_ratio']))
assert len(signatures)==1 and len(r)==16 and len(set((x['ep'],x['layer']) for x in r))==16
assert len({x['ep'] for x in r if x['layer']==15})==8 and {x['ep'] for x in r if x['layer']==15}=={x['ep'] for x in r if x['layer']==36}
summary={}
for layer in [15,36]:
 q=[x for x in r if x['layer']==layer]
 den=sum(Decimal(x['aic_cycles']) for x in q);num=sum(Decimal(x['aic_cycles'])*Decimal(x['aic_mac']) for x in q)
 summary[str(layer)]={'cycles':str(den),'cycles_times_display_ratio':str(num),'weighted_ratio':str(num/den),'duration_min_us':min(Decimal(x['duration_us']) for x in q).__str__(),'duration_max_us':max(Decimal(x['duration_us']) for x in q).__str__()}
out={'source_sha256':want,'scope':'Existing profiled digest db852bff, two layer groups, same recorded physical signature; no effective-work equivalence asserted','summary':summary,'last_matched_same_scope_neighbors':neighbors,'rows':sorted(r,key=lambda x:(x['ep'],x['layer'])),'limits':['displayed ratios rounded; multiplied cycles not MAC counts','same physical shapes do not imply same expert distribution or mask','task durations are instrumented, not baseline serving latency','rank clocks and scheduling are not interchangeable']}
(p/'agents/diagnosis/mc2-db852-extremes.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
for x in out['rows']:print(x['ep'],x['layer'],x['stream'],x['task'],x['start_us'],x['duration_us'],x['aic_cycles'],x['aic_mac'])
