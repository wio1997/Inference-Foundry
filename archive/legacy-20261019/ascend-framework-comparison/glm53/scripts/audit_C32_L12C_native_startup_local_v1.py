"""Independent new-P/retained-D native startup identity and hard capacity audit."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 capfile=ROOT/'results/sglang-C32-L12C-selected-P-pool-capacity-v1.json';cap=json.loads(capfile.read_text());assert cap['pass'] and cap['native_82_kernel_fixture_qualified'] and cap['selected_variant']=='candidate'
 root=ROOT/'runs/20261019T-L12C-sglang-native-startup-v1';receipt=json.loads((root/'receipt.json').read_text());assert receipt['pass'] and len(receipt['nodes'])==4 and all(v==0 for v in receipt['nodes'].values())
 nodes={};all_D_ranks=set();sources={str(capfile.relative_to(ROOT)):sha(capfile)}
 for n in [34,35,36,41]:
  p=root/('node-'+str(n)+'.json');w=json.loads(p.read_text());assert w['rc']==0;o=json.loads(w['stdout']);sources[str(p.relative_to(ROOT))]=sha(p);assert o['container_id']==cap['providers'][str(n)]['container_id'] and o['supervisor']['phase']=='MODEL_RUNNING' and 'model_exit_code' not in o['supervisor'] and o['provider_phase']=='READY'
  rows=[]
  for item in o['persistent_audit'].values():
   raw=''.join(json.dumps(x,sort_keys=True)+'\n' for x in item['rows']).encode();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];rows+=item['rows']
  init={x['pid']:x for x in rows if x['event']=='initialize_complete'};layout=[x for x in rows if x['event']=='model_state_registered'];assert len(init)==len(layout)==16
  assert all(x['rc']==0 and x['protocol']=='SDMA' and x['protocol_value']==1 for x in init.values())
  for x in layout:
   v=x['metadata'];assert v['hidden_kv_layers']==78 and v['draft_kv_layers']==1 and v['kv_buf_groups']==3 and len(v['kv_item_lens'])==237 and v['kv_cache_dtype_str']=='bfloat16'
  assert len(o['owned_npu_process_rows'])==18;ns={x['namespace_pid'] for x in o['owned_npu_process_rows']};assert ns==({o['supervisor']['provider_pid'],o['supervisor']['model_pid']}|set(init))
  assert o['/health']['status']==o['/metrics']['status']==200
  if n in [34,36]:assert o['/get_server_info']['status']==200
  else:assert o['/get_server_info']=={'error':"<HTTPError 404: 'Not Found'>"}
  met=o['/metrics'];assert hashlib.sha256(met['body'].encode()).hexdigest()==met['sha256'];gauges=[float(x.rsplit(' ',1)[1]) for x in met['body'].splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and not any(gauges)
  leader=n if n in [34,36] else (34 if n==35 else 36)
  leader_wire=json.loads((root/('node-'+str(leader)+'.json')).read_text());assert leader_wire['rc']==0;leader_native=json.loads(leader_wire['stdout']);assert leader_native['/get_server_info']['status']==200
  info=json.loads(leader_native['/get_server_info']['body']);assert info['tp_size']==info['ep_size']==32 and info['dp_size']==(4 if n in [34,35] else 16);assert info['speculative_num_steps']==5 and info['speculative_num_draft_tokens']==6 and info['kv_cache_dtype']=='bfloat16' and info['disable_overlap_schedule']
  capacity=info['max_total_num_tokens'];assert capacity>=(254848 if n in [34,35] else 161280)
  logs=list(o['logs'].values());errors=[];markers=[]
  for log in logs:
   assert log['full_archive_complete'];assert log['error_line_count']==len(log['error_lines']);errors+=log['error_lines'];markers+=log['metadata_marker_lines']
   assert log['native_sdma']['copy_calls']==0 or log['native_sdma']['copy_status_values']==[0]
  benign=r'^\d{4}-\d{2}-\d{2} [0-9:.]+ ERROR \[\d+-\d+\]\[AccLink acc_tcp_port_util\.cpp:(?:43|62) BindTcpPortV[46]\] bind ipv[46] tcp port failed, port=\d+, errno=98, errstr=Address already in use$'
  assert all(re.fullmatch(benign,line) for line in errors),errors
  assert not any('FATAL' in line or 'GLM53_C32_PACKED_DISPATCH_' in line for line in markers)
  if n in [34,35]:
   assert info['hicache_host_memory_mode']=='buffer_only' and info['hicache_size']==9
   assert len(o['native_optional_extension_worker_maps'])==16
   assert {x['namespace_pid'] for x in o['native_optional_extension_worker_maps']}==set(init)
   reused={};installed={}
   for line in markers:
    if 'GLM53_C32_NATIVE_MC_MAPPING_REUSED=' in line:
     x=json.loads(line.split('GLM53_C32_NATIVE_MC_MAPPING_REUSED=',1)[1]);assert x['pid'] in init and x['existing_linear_mapping'] and not x['explicit_second_MC_registration']
     assert x['component'] not in reused.setdefault(x['pid'],{});reused[x['pid']][x['component']]=x
    if 'GLM53_C32_NATIVE_HICACHE_INSTALLED=' in line:
     x=json.loads(line.split('GLM53_C32_NATIVE_HICACHE_INSTALLED=',1)[1]);installed.setdefault(x['pid'],set()).add(x['module'])
   assert set(reused)==set(init)
   for pid in init:
    assert set(reused[pid])=={'k_buffer','v_buffer','index_k_buffer'}
    assert installed[pid]=={'sglang.srt.mem_cache.pool_host.mla','sglang.srt.mem_cache.storage.npu_memcache.npu_memcache_store'}
    for name,shape,size in [('k_buffer',[634,79,128,1,512],6564872192),('v_buffer',[634,79,128,1,64],820609024),('index_k_buffer',[634,78,128,1,128],1620443136)]:assert reused[pid][name]['shape']==shape and reused[pid][name]['bytes']==size
  else:
   assert info['cuda_graph_config']['decode']['backend']=='full' and info['cuda_graph_config']['decode']['bs']==[1,2,3,4]
   aligned=[]
   for item in o['alignment_audit'].values():
    raw=''.join(json.dumps(x,separators=(',',':'))+'\n' for x in item['rows']).encode();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];aligned+=item['rows']
   for pid in init:
    records=[x for x in aligned if x['pid']==pid]
    for method in ['_npu_low_latency_dispatch','_npu_low_latency_combine']:
     found=[x for x in records if x['event']=='ops_first_return' and x['method']==method];assert len(found)==1 and found[0]['group_size']==32 and found[0]['comm_alg']=='fullmesh_v1';all_D_ranks.add(found[0]['rank'])
    assert any(x['event']=='draft_graph_disabled' for x in records)
   counts={}
   for line in met['body'].splitlines():
    if line.startswith('sglang:cuda_graph_passes_total{'):
     labels=dict(re.findall(r'(\w+)="([^"]*)"',line))
     if labels['mode']=='decode_cuda_graph':counts[int(labels['moe_ep_rank'])]=float(line.rsplit(' ',1)[1])
   assert set(counts)==set(range(16) if n==36 else range(16,32)) and all(x>0 for x in counts.values())
  nodes[str(n)]={'container_id':o['container_id'],'actual_SDMA_workers':len(init),'state_workers':len(layout),'KV_capacity':capacity,'KV_capacity_source_node':leader,'KV_capacity_scope':'Role-global native primary with distributedworker registrations; secondaryAPI404preserved','new_native_extension_worker_maps':len(o['native_optional_extension_worker_maps']),'preserved_bind_error_lines':len(errors),'healthy_idle':True}
 assert all_D_ranks==set(range(32))
 out={'project_id':'glm53','framework':'sglang','pass':True,'shared_run_id':'20261019T-L12C-candidate-v1','runtime_CIDs':{k:v['container_id'] for k,v in cap['providers'].items()},'nodes':nodes,'all64_actual_SDMA_and_BF16_layout_verified':True,'all32_P_native_IO_and_existing_MC_mapping_reuse_verified':True,'all32_D_nativeOps_fullmesh_and_targetGraph_retained':True,'D_graph_counter_scope':'Cumulative current retained-worker counters; not a new experiment delta. Fresh productive Graph evidence mandatory during functional/formal.','combined_MC_and_native_82_kernel_prerequisite_pass':True,'native_P_model_cache_semantics':'PENDING_SHORT_AND_TWO80K_CACHE_SDMA','external_model_requests':0,'formal_qualified':False,'source_evidence_sha256':sources}
 p=ROOT/'results/sglang-C32-L12C-native-startup-v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'actual_SDMA_workers':64,'P_native_maps':32,'D_retained':True,'KV_capacities':{n:v['KV_capacity'] for n,v in nodes.items()},'formal_qualified':False}))
if __name__=='__main__':main()
