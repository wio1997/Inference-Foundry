#!/usr/bin/env python3
"""Bounded EP8 737-5: layer CPU scopes -> trace flows -> device X -> CSV.
Read only. JSON stdout. No profiler or device imports. Never emits request IDs.
"""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import collections, csv, hashlib, json
import ijson

R=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime')
rank=R/'diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt'
trace=rank/'ASCEND_PROFILER_OUTPUT/trace_view.json'
csvpath=rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
metadata=R/'metadata/events-737.jsonl'
assert all(p.is_file() and not p.is_symlink() for p in (trace,csvpath,metadata))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def dec(x):
 try:return Decimal(str(x))
 except (InvalidOperation,ValueError):return None

meta=collections.defaultdict(lambda:{'begin':[],'end':[]})
with metadata.open() as f:
 for line in f:
  e=json.loads(line);kind=e.get('kind');cid=e.get('call_id')
  if kind in ('call_begin','call_end') and isinstance(cid,str):
   meta[cid][kind[5:]].append({k:e.get(k) for k in ('label','pid','tid','layer_index','schedule')})
parentmeta=meta['737-5'];assert len(parentmeta['begin'])==len(parentmeta['end'])==1
assert parentmeta['begin'][0]['label']=='NPUModelRunner.execute_model'
schedule=parentmeta['begin'][0]['schedule'];assert isinstance(schedule,dict)
starts=collections.defaultdict(list);finishes=collections.defaultdict(list);scopes=[];device={}
counts=collections.Counter()
with trace.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict):continue
  name=e.get('name');ph=e.get('ph')
  if isinstance(name,str) and name.startswith('wio_diag:') and ph=='X' and str(e.get('pid'))=='737' and str(e.get('tid'))=='737':
   if name.startswith('wio_diag:DeepseekV41DecoderLayer.forward:') or name=='wio_diag:NPUModelRunner.execute_model:737-5':
    ts,dur=dec(e.get('ts')),dec(e.get('dur'))
    if ts is not None and dur is not None and dur>=0:scopes.append({'id':name.rsplit(':',1)[-1],'label':name.split(':')[1],'start':ts,'end':ts+dur})
  if name=='torch_to_npu' and e.get('cat')=='async_npu' and ph in ('s','f'):
   key=str(e.get('id'));ts=dec(e.get('ts'))
   if ts is not None:
    x={'pid':str(e.get('pid')),'tid':str(e.get('tid')),'ts':ts}
    (starts if ph=='s' else finishes)[key].append(x)
parents=[x for x in scopes if x['id']=='737-5'];assert len(parents)==1
parent=parents[0]; layers=[]; bad=[]
for x in scopes:
 if x['label']!='DeepseekV41DecoderLayer.forward' or not (parent['start']<=x['start'] and x['end']<=parent['end']):continue
 m=meta[x['id']]
 if len(m['begin'])!=1 or len(m['end'])!=1:bad.append({'id':x['id'],'reason':'unpaired_or_nonunique_metadata'});continue
 b=m['begin'][0];en=m['end'][0]
 if b['label']!=x['label'] or en['label']!=x['label'] or b['pid']!=737 or b['tid']!=737 or type(b['layer_index']) is not int:
  bad.append({'id':x['id'],'reason':'metadata_label_thread_or_layer_missing'});continue
 x['layer']=b['layer_index'];layers.append(x)
keys=collections.defaultdict(list)
for k in starts:
 if len(starts[k])!=1 or len(finishes[k])!=1:continue
 st=starts[k][0];fi=finishes[k][0]
 if (st['pid'],st['tid'])!=('737','737') or not(parent['start']<=st['ts']<=parent['end']):continue
 keys[(fi['pid'],fi['tid'],fi['ts'])].append(k)
with trace.open('rb') as f:
 for e in ijson.items(f,'item'):
  if not isinstance(e,dict) or e.get('ph')!='X':continue
  ts=dec(e.get('ts')); key=(str(e.get('pid')),str(e.get('tid')),ts)
  for k in keys.get(key,[]):
   device.setdefault(k,[]).append(e)
rows=[];ambiguous=[]
for k in keys.values():
 for flow in k:
  evs=device.get(flow,[])
  if len(evs)!=1 or evs[0].get('name')!='DispatchFFNCombine':continue
  e=evs[0]; st=starts[flow][0]['ts']; candidates=[x for x in layers if x['start']<=st<=x['end']]
  # True nested ranges can choose innermost; overlapping non-nested intervals
  # are ambiguous even if one happens to have shorter duration.
  chosen=None
  if len(candidates)==1:chosen=candidates[0]
  elif len(candidates)>1:
   choices=[a for a in candidates if all(b['start']<=a['start'] and a['end']<=b['end'] for b in candidates)]
   if len(choices)==1:chosen=choices[0]
  if not chosen:
   ambiguous.append({'flow_id':flow,'candidate_layers':[x['layer'] for x in candidates]});continue
  args=e.get('args') if isinstance(e.get('args'),dict) else {}
  rows.append({'layer':chosen['layer'],'layer_scope_id':chosen['id'],'flow_id':flow,'host_flow_start_us':str(st),'trace_pid':str(e.get('pid')),'trace_tid':str(e.get('tid')),'trace_start_us':str(dec(e.get('ts'))),'trace_duration_us':str(dec(e.get('dur'))),'trace_model_id':str(args.get('Model Id')),'trace_task_id':str(args.get('Task Id')),'trace_physical_stream_id':str(args.get('Physic Stream Id'))})
lookup=collections.defaultdict(list)
with csvpath.open(newline='') as f:
 for r in csv.DictReader(f):
  if r.get('Name')!='DispatchFFNCombine':continue
  ts=dec(r.get('Start Time(us)'))
  if ts is not None:lookup[(ts,str(r.get('Model ID')),str(r.get('Task ID')),str(r.get('Stream ID')))].append(r)
fields=('Device_id','Model ID','Task ID','Stream ID','Start Time(us)','Duration(us)','Accelerator Core','aicore_time(us)','aic_total_cycles','aic_mac_time(us)','aic_mac_ratio','aic_scalar_ratio','aic_mte2_ratio','aiv_time(us)','aiv_total_cycles','aiv_vec_time(us)','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','cube_utilization(%)')
for row in rows:
 key=(Decimal(row['trace_start_us']),row['trace_model_id'],row['trace_task_id'],row['trace_physical_stream_id'])
 candidates=lookup.get(key,[]);row['csv_match_count']=len(candidates)
 if len(candidates)==1:
  row['csv']={k:candidates[0].get(k) for k in fields}
  td=Decimal(row['trace_duration_us']);cd=dec(candidates[0].get('Duration(us)'))
  row['duration_difference_us']=str(abs(td-cd)) if cd is not None else None
rows.sort(key=lambda x:(x['layer'],Decimal(x['trace_start_us'])))
layercounts=collections.Counter(row['layer'] for row in rows)
result={'scope':'EP8 737-5','input_sha256':{'trace_view':sha(trace),'kernel_details':sha(csvpath),'metadata':sha(metadata)},'parent':{'start_us':str(parent['start']),'end_us':str(parent['end']),'schedule_digest':schedule.get('schedule_digest_sha256'),'scheduled_total':schedule.get('scheduled_total'),'request_count':len(schedule.get('requests',[]))},'layer_scope_count':len(layers),'layer_scope_layer_counts':dict(collections.Counter(x['layer'] for x in layers)),'bad_scopes':bad,'unique_flow_pair_count':sum(len(v) for v in keys.values()),'layer_attributed_dispatch_count':len(rows),'dispatch_layer_counts':dict(layercounts),'ambiguous_or_unattributed':ambiguous,'rows':rows,'limits':['Scope CPU containment determines submitting layer only when unique; device may execute after layer scope ends.','Flow is exported correlation, not direct tensor dependency, and graph replay may omit layer CPU scopes.','Only same-thread flow starts and chosen parent; missing/ambiguous events excluded, not zero work.','Profile task durations and counters are perturbed, task durations may overlap, counters are not whole-chip utilization.','No request IDs/tensor content exported; schedule digest is not guaranteed unique.']}
print(json.dumps(result,indent=2))
