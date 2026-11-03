#!/usr/bin/env python3
"""Compare uniquely matched existing profiled tasks, CPU only."""
import collections, hashlib, json, statistics
from pathlib import Path
R=Path(__file__).resolve().parents[2]; src=R/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
sha=hashlib.sha256(src.read_bytes()).hexdigest();assert sha=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
scopes={}; wanted='1b826883b782ba77814d5bd40fcd3e46c759e380069db2e066ff0e9cb4bce89d'
with src.open() as f:
 for line in f:
  d=json.loads(line)
  if d.get('kind')!='rank' or d['data']['ep'] not in range(8,16):continue
  ep=d['data']['ep']
  for layer in d['data']['layers']:
   if layer['layer'] not in (14,39) or layer['scheduler_digest']!=wanted:continue
   scope=(ep,layer['layer']);assert scope not in scopes
   tasks=collections.defaultdict(list)
   for group in layer['operator_groups']:
    if len(group['tasks'])!=1:continue
    t=group['tasks'][0];c=t['csv'];core=c['Accelerator Core']
    if c['Stream ID']!='47' or core not in ('AI_CORE','AI_VECTOR_CORE','MIX_AIC'):continue
    assert c['Name']==group['operator_name']
    sig=group['physical_shape_dtype_format'];assert all(c[k]==v for k,v in sig.items())
    key=json.dumps([group['operator_name'],c['Type'],sig,core,c['Block Num'],c['Mix Block Num']],sort_keys=True,separators=(',',':'))
    duration=float(t['trace_duration_us']);assert duration>0
    tasks[key].append({'duration_us':duration,'flow_id':str(t['flow_id']),'task_id':c['Task ID']})
   scopes[scope]={'scope_id':layer['layer_scope_id'],'tasks':tasks}
assert len(scopes)==16
by=collections.defaultdict(list)
for ep in range(8,16):
 a,b=scopes[ep,14]['tasks'],scopes[ep,39]['tasks']
 for key in a.keys()&b.keys():
  if len(a[key])==len(b[key])==1:
   x,y=a[key][0],b[key][0];by[key].append({'ep':ep,'l14':x,'l39':y,'ratio':x['duration_us']/y['duration_us']})
rows=[]
for key,pairs in by.items():
 if len(pairs)!=8:continue
 name,typ,sig,core,block,mix=json.loads(key);ratios=[v['ratio'] for v in pairs]
 rows.append({'name':name,'type':typ,'core':core,'block':block,'mix_block':mix,'signature':sig,'pairs':pairs,'ratio_min':min(ratios),'ratio_median':statistics.median(ratios),'ratio_max':max(ratios)})
rows.sort(key=lambda x:(-x['ratio_median'],x['name'],json.dumps(x['signature'],sort_keys=True)))
mc2=[x for x in rows if x['name']=='DispatchFFNCombine'];assert len(rows)==18 and len(mc2)==1
others=[x for x in rows if x['name']!='DispatchFFNCombine'];assert len(others)==17
result={'source_sha256':sha,'selection':'EP8-15, one scheduler digest, layers 14 and 39, stream 47 per device; one unique task per exact exported name/type/signature/core/block in each of all eight same-EP pairs','matched_signatures':len(rows),'mc2_ratio_median':mc2[0]['ratio_median'],'other_signature_median_ratio_range':[min(x['ratio_median'] for x in others),max(x['ratio_median'] for x in others)],'other_individual_pair_ratio_range':[min(v['ratio'] for x in others for v in x['pairs']),max(v['ratio'] for x in others for v in x['pairs'])],'results':rows,'limits':['Selected named device records do not cover all tasks, streams, host work or MC2 internal work.','Physical signature does not prove identical dynamic workload, request or runtime identity.','Ratios are from heavily profiled different-layer tasks; do not imply causal explanation or online speedup.','Cross-stream AI_CPU tasks are outside this stream-47 selection.']}
p=R/'agents/diagnosis/mc2-l14-l39-matched-tasks.json';p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('matched_signatures','mc2_ratio_median','other_signature_median_ratio_range','other_individual_pair_ratio_range')},ensure_ascii=False));print('json_sha256',hashlib.sha256(p.read_bytes()).hexdigest())
