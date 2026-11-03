#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
from collections import Counter
root=Path(__file__).resolve().parents[2];p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';want='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4';h=hashlib.sha256(p.read_bytes()).hexdigest()
if h!=want:raise ValueError('source hash mismatch')
digest='db852bff987d2b98d3dc0734e3731ec5528e37dcf6e874860d1ebb4fd12eefc8';out=[]
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('kind')!='rank':continue
  for s in x['data']['layers']:
   if s['scheduler_digest']!=digest or s['layer']!=36 or s['ep'] not in range(8):continue
   assert s['trace_scope_count']==1 and s['sidecar_thread_matches_trace'] is True
   d={'ep':s['ep'],'scope_id':s['layer_scope_id'],'worker_pid':s['worker_pid'],'trace_scope_count':s['trace_scope_count'],'sidecar_thread_matches_trace':s['sidecar_thread_matches_trace'],'tasks':{}}
   for kind,weight in [('q_shape','1280,5120'),('kv_shape','512,5120')]:
    cand=[]
    for g in s['operator_groups']:
     t=g['physical_shape_dtype_format']; arr=t['Input Shapes'].strip('"').split(';'); outsh=t['Output Shapes'].strip('"')
     if len(arr)!=2 or arr[1]!=weight or t['Input Data Types']!='DT_BF16;DT_BF16' or t['Output Data Types']!='DT_BF16':continue
     if not g['operator_name'].startswith('aclnnMatmul_'):continue
     for task in g['tasks']:cand.append((t,arr[0],outsh,task))
    if len(cand)!=1:raise ValueError('not unique '+str((s['ep'],kind,len(cand))))
    t,ins,outs,task=cand[0];dim=[int(y) for y in ins.split(',')];outdim=[int(y) for y in outs.split(',')];assert len(dim)==len(outdim)==2 and dim[1]==5120 and dim[0]==outdim[0]
    assert str(task['host_pid'])==str(s['worker_pid'])
    d['tasks'][kind]={'host_pid':task['host_pid'],'host_tid':task['host_tid'],'host_start_us':task['host_start_us'],'device_start_us':task['device_start_us'],'input_shape':dim,'output_shape':outdim,'weight_shape':weight,'flow_id':task['flow_id'],'task_id':task['csv']['Task ID'],'stream':task['device_track_tid'],'duration_us':task['trace_duration_us']}
   out.append(d)
assert len(out)==8 and set(x['ep'] for x in out)==set(range(8));out.sort(key=lambda x:x['ep']);qs=[x['tasks']['q_shape']['input_shape'][0] for x in out]; kvs=[x['tasks']['kv_shape']['input_shape'][0] for x in out]
assert qs==[1023]*7+[1019] and kvs==[8180]*8
print(json.dumps({'source_sha256':h,'scope':'EP0-7, layer36, one scheduler digest; shape-compatible independently matched host-flow-to-device tasks; no tensor identity or global request proof','rows':out,'q_rows_by_rank':qs,'kv_rows_by_rank':kvs,'sum_q_rows':sum(qs),'limitations':['Physical shape is not valid token count, identity, rank token ordering, or proof of pad slots.','Global KV tasks are not proven to consume the same gathered buffer as Q tasks.','Actual call chain beyond selected EP3, loaded binary and projection semantics remain unbound.','Conditional equal-size collectives need correct padding or a separately supported variable-size protocol.']},ensure_ascii=False,indent=2))
