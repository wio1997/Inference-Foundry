"""Independent native startup gate, with Graph evidence distinct from qualification."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/'runs/20261018T-L10-target-graph-startup-v3';receipt=json.loads((run/'receipt.json').read_text());assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values())
 capfile=ROOT/'results/sglang-C32-target-graph-fullmesh-pool-capacity-v2.json';cap=json.loads(capfile.read_text());assert cap['pass']
 result={'project_id':'glm53','framework':'sglang','pass':True,'shared_run_id':'20261018T-L10-sglang-C32-target-graph-fullmesh-v2','runtime_CIDs':{k:v['container_id'] for k,v in cap['providers'].items()},'nodes':{},'external_model_requests':0,'native_internal_warmup_separate':True,'semantic_and_formal_qualification':'PENDING','source_evidence_sha256':{str(capfile.relative_to(ROOT)):sha(capfile)}}
 total_ops=0;total_graph=0
 for node in [34,35,36,41]:
  p=run/('node-'+str(node)+'.json');wire=json.loads(p.read_text());assert wire['rc']==0;o=json.loads(wire['stdout']);assert o['startup_wait']['completed'] and not o['startup_wait']['timeout']
  model=o['model'];assert model['id']==cap['providers'][str(node)]['container_id'] and model['state']['Running'] and o['supervisor_status']['phase']=='MODEL_RUNNING' and o['provider_status']['phase']=='READY' and 'model_exit_code' not in o['supervisor_status']
  met=o['/metrics'];assert met['status']==200 and hashlib.sha256(met['body'].encode()).hexdigest()==met['sha256'];gauges=[float(l.rsplit(' ',1)[1]) for l in met['body'].splitlines() if l.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and not any(gauges)
  events=[]
  for x in o['persistent_audit'].values():
   raw=''.join(json.dumps(y,sort_keys=True)+'\n' for y in x['rows']).encode();assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256'];events+=x['rows']
  initial={x['pid']:x for x in events if x['event']=='initialize_complete'};assert len(initial)==16 and all(x['rc']==0 and x['protocol']=='SDMA' and x['protocol_value']==1 for x in initial.values())
  layouts=[x for x in events if x['event']=='model_state_registered'];assert len(layouts)==16
  for x in layouts:
   v=x['metadata'];assert v['hidden_kv_layers']==78 and v['draft_kv_layers']==1 and v['kv_buf_groups']==3 and len(v['kv_item_lens'])==237 and v['kv_cache_dtype_str']=='bfloat16'
  logs=model['logs'];markers=[l for stream in ['stdout','stderr'] for l in logs[stream+'_metadata_markers']];assert not any(x in l for l in markers for x in ['GLM53_METADATA_HOOK_FATAL','GLM53_C32_ALIGNMENT_FATAL','GLM53_C32_OPS_HOOK_FATAL','GLM53_C32_PACKED_DISPATCH_']);assert not any(logs[s+'_archived_candidate_markers'] for s in ['stdout','stderr'])
  if node in [36,41]:assert not any('GLM53_DEEPEP_METADATA_BATCH_FIRST_USE=' in l for l in markers),'NativeOps must not execute normal metadata route'
  align=[]
  for x in o['alignment_audit'].values():
   raw=''.join(json.dumps(y,separators=(',',':'))+'\n' for y in x['rows']).encode();assert len(raw)==x['bytes'] and hashlib.sha256(raw).hexdigest()==x['sha256'];align+=x['rows']
  per={}
  if node in [36,41]:
   for pid in initial:
    rows=[x for x in align if x['pid']==pid];assert rows and all(x['revision']=='glm53-C32-target-graph-fullmesh-v2' for x in rows)
    for method in ['_npu_low_latency_dispatch','_npu_low_latency_combine']:
     v=[x for x in rows if x['event']=='ops_first_return' and x['method']==method];assert len(v)==1;v=v[0];assert v['group_size']==32 and v['comm_alg']=='fullmesh_v1'
     if method.endswith('dispatch'):
      assert v['quant_mode']==2 and v['inputs']['x']['shape']==[32,6144] and v['inputs']['topk_weights'] is None and v['inputs']['x']['dtype']=='torch.bfloat16' and v['outputs'][0]['dtype']=='torch.int8'
     else:assert v['comm_quant_mode']==0 and v['inputs']['topk_weights']['shape']==[32,8] and v['outputs']['dtype']=='torch.bfloat16'
    assert any(x['event']=='draft_graph_disabled' for x in rows)
    replays=[x for x in rows if x['event']=='target_graph_replay_return'];decisions=[x for x in rows if x['event']=='target_graph_decision'];assert replays,'Actual internal warmup target replay required; capture config alone insufficient'
    assert all(not x['is_draft'] and x['capture_request_width']==6 for x in replays)
    keys={x['key_size'] for x in replays};assert keys<=set([1,2,3,4]) or keys<=set([6,12,18,24]),'Pinned runner uses request buckets or compact-ragged token buckets'
    per[str(pid)]={'rank':next(x['rank'] for x in rows if x['event']=='ops_first_return'),'replay_returns':len(replays),'keys':sorted(keys),'key_unit_from_pinned_runner':'requests' if keys<=set([1,2,3,4]) else 'compact_ragged_tokens','raw_bs':sorted({x['raw_bs'] for x in replays}),'eligible_false':sum(x['eligible'] is False for x in decisions),'modes':sorted({x['mode'] for x in replays}),'backend_types':sorted({x['backend_type'] for x in replays})}
   assert len(per)==16 and len({x['rank'] for x in per.values()})==16;total_ops+=16;total_graph+=16
  else:assert not align
  capacity=None
  if node in [34,36]:
   assert o['/get_server_info'].get('status')==200;selected={x['path']:x['value'] for x in o['/get_server_info']['selected']}
   for k,v in [('tp_size',32),('ep_size',32),('dp_size',4 if node==34 else 16),('speculative_num_steps',5),('speculative_num_draft_tokens',6),('kv_cache_dtype','bfloat16'),('deepep_mode','normal' if node==34 else 'low_latency')]:assert selected[k]==v,(node,k,selected.get(k))
   capacity=selected['max_total_num_tokens'];assert capacity>=(254848 if node==34 else 161280)
   if node==36:assert selected['cuda_graph_config']['decode']['backend']=='full' and selected['cuda_graph_config']['decode']['bs']==[1,2,3,4]
  errors=[l for s in ['stdout','stderr'] for l in logs[s+'_error_lines']];pattern=r'^\d{4}-\d{2}-\d{2} [0-9:.]+ ERROR \[\d+-\d+\]\[AccLink acc_tcp_port_util\.cpp:(?:43|62) BindTcpPortV[46]\] bind ipv[46] tcp port failed, port=\d+, errno=98, errstr=Address already in use$';preserved=[x for x in errors if re.fullmatch(pattern,x)];assert len(preserved)==len(errors),errors
  assert o['npu']['rc']==model['top']['rc']==0;owned={int(l.split()[0]) for l in model['top']['stdout'].splitlines()[1:] if l.split() and l.split()[0].isdigit()};contexts=[]
  for line in o['npu']['stdout'].split('Process id',1)[-1].splitlines():
   cols=[v.strip() for v in line.split('|')]
   if len(cols)>=5 and len(cols[1].split())==2 and all(v.isdigit() for v in cols[1].split()) and cols[2].isdigit():contexts.append(int(cols[2]))
  assert len(contexts)==18 and set(contexts)<=owned
  result['nodes'][str(node)]={'container_id':model['id'],'native_SDMA_workers':16,'registered_state_workers':16,'idle_gauges':gauges,'reported_KV_capacity':capacity,'graph_ops_workers':per,'preserved_initial_bind_errors':len(preserved)};result['source_evidence_sha256'][str(p.relative_to(ROOT))]=sha(p)
 assert total_ops==total_graph==32;result['all32_Ops_workers']=32;result['all32_target_graph_warmup_replay_workers']=32;result['draft_graphs_disabled_workers']=32
 p=ROOT/'results/sglang-C32-target-graph-fullmesh-startup-v3.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'actual_target_graph_workers':32,'Ops_workers':32,'formal_qualified':False,'sha256':sha(p)}))
if __name__=='__main__':main()
