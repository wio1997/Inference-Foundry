"""Independent native startup gate, with Graph evidence distinct from qualification."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/'runs/20261018T-L10-target-graph-startup-completion-v5';receipt=json.loads((run/'receipt.json').read_text());assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values())
 capfile=ROOT/'results/sglang-C32-target-graph-fullmesh-pool-capacity-v3.json';cap=json.loads(capfile.read_text());assert cap['pass']
 result={'project_id':'glm53','framework':'sglang','pass':True,'shared_run_id':'20261018T-L10-sglang-C32-target-graph-fullmesh-v3','runtime_CIDs':{k:v['container_id'] for k,v in cap['providers'].items()},'nodes':{},'external_model_requests':0,'native_internal_warmup_separate':True,'semantic_and_formal_qualification':'PENDING','source_evidence_sha256':{str(capfile.relative_to(ROOT)):sha(capfile)}}
 evidence=[ROOT/'runs/20261017T1944-sglang-native-graph-source-v1/source/srt__hardware_backend__npu__graph_runner__npu_graph_runner.py',ROOT/'runs/20261018T-C32-alignment-installed-source-v1/source/sglang/sgl-workspace__sglang__python__sglang__srt__model_executor__model_runner.py',ROOT/'runs/20261018T-C32-graph-native-metric-source-v1/metrics_collector.py']
 for p in evidence:result['source_evidence_sha256'][str(p.relative_to(ROOT))]=sha(p)
 npu=evidence[0].read_text();mr=evidence[1].read_text();metrics=evidence[2].read_text()
 assert 'class NPUGraphRunner(DecodeCudaGraphRunner)' in npu and '    def execute(' in npu and 'self.backend.replay(graph_key, forward_batch)' in npu and 'NPUCudaGraphBackend' in npu
 assert 'if can_run_graph:' in mr and 'ret = self.decode_cuda_graph_runner.execute(' in mr and 'can_run_graph=can_run_graph' in mr
 assert 'def increment_decode_cuda_graph_pass' in metrics and 'self.cuda_graph_passes_total.labels(**self.labels, mode=mode).inc(1)' in metrics
 result['graph_evidence_provenance']='Native per-rank positive decode graph passes plus productive per-PID TARGET_VERIFY decisions, completed native warmup, healthy idle and pinned NPU execute replay path. Generic execute hook bypassed by NPUGraphRunner override; actual replay bucket keys unavailable.'
 result['configured_request_capture_bins']=[1,2,3,4]
 total_ops=0;total_graph=0
 for node in [34,35,36,41]:
  p=run/('node-'+str(node)+'.json');wire=json.loads(p.read_text());assert wire['rc']==0;o=json.loads(wire['stdout']);assert o['startup_wait']['completed'] and not o['startup_wait'].get('timeout',False)
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
  native_graph={}
  if node in [36,41]:
   for line in met['body'].splitlines():
    if not line.startswith('sglang:cuda_graph_passes_total{'):continue
    label_text,value=line.split('} ',1);labels=dict(re.findall(r'(\w+)="([^"]*)"',label_text));rank=int(labels['moe_ep_rank'])
    assert labels['mode']=='decode_cuda_graph' and labels['engine_type']=='decode' and labels['model_name']=='glm53-w8a8' and labels['pp_rank']=='0'
    assert int(labels['tp_rank'])==rank and int(labels['dp_rank'])==rank//2 and rank not in native_graph
    native_graph[rank]=float(value);assert native_graph[rank]>0
   assert set(native_graph)==set(range(0,16) if node==36 else range(16,32))
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
    replays=[x for x in rows if x['event']=='target_graph_replay_return'];decisions=[x for x in rows if x['event']=='target_graph_decision']
    productive=[x for x in decisions if x['eligible'] is True and x['mode']=='TARGET_VERIFY' and x['raw_bs']>0 and not x['is_draft']];assert productive
    assert not replays,'Generic execute hook is bypassed by pinned NPU override; preserve actual rows'
    rank=next(x['rank'] for x in rows if x['event']=='ops_first_return')
    counter=native_graph[rank];assert counter>0
    per[str(pid)]={'rank':rank,'native_decode_graph_passes':counter,'generic_helper_replay_rows':len(replays),'actual_graph_key_sizes':None,'productive_target_verify_decisions':len(productive),'productive_raw_bs':sorted({x['raw_bs'] for x in productive}),'eligible_false':sum(x['eligible'] is False for x in decisions),'idle_decisions':sum(x['mode']=='IDLE' for x in decisions),'source_backend_type':'NPUCudaGraphBackend'}
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
 p=ROOT/'results/sglang-C32-target-graph-fullmesh-startup-v5.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'actual_target_graph_workers':32,'Ops_workers':32,'formal_qualified':False,'sha256':sha(p)}))
if __name__=='__main__':main()
