"""Independent local native startup audit; no inference/SDK/resource actions."""
from pathlib import Path
import collections,hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def main():
 r=ROOT/'runs/20261017T1769-vllm-shared-startup-checkpoint-v2';nodes={};all_init=[];all_reg=[];D_caps={};P_caps={};source={}
 for n,engines in {30:{0,1},31:{2,3},43:set(range(8)),45:set(range(8,16))}.items():
  p=r/f'node-{n}.json';outer=json.loads(p.read_text());assert outer['rc']==0;data=json.loads(outer['stdout']);source[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
  assert data['supervisor']['phase']=='MODEL_RUNNING' and 'model_exit_code' not in data['supervisor'] and 'error' not in data['actual_model_identity'] and data['provider_phase']=='READY'
  assert data['/health']['status']==data['/metrics']['status']==data['/v1/models']['status']==200 and not data['profiler_artifacts']
  assert len(data['owned_npu_process_rows'])==17 and all(v['full_archive_complete'] and not v['error_lines'] for v in data['logs'].values())
  lines=data['/metrics']['body'].splitlines();cache={};gauges=[]
  for line in lines:
   if line.startswith('vllm:cache_config_info{'):
    dp=int(re.search(r'engine="(\d+)"',line)[1]);tokens=int(re.search(r'kv_cache_size_tokens="(\d+)"',line)[1]);cache[dp]=tokens
   if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')):gauges.append(float(line.rsplit(' ',1)[1]))
  assert set(cache)==engines and gauges and all(v==0 for v in gauges)
  if n in [30,31]:P_caps.update(cache)
  else:D_caps.update(cache);assert all(v>=2*80640 for v in cache.values())
  rows=[v for f in data['persistent_audit'].values() for v in f['rows']];init=[v for v in rows if v.get('event')=='initialize_complete'];reg=[v for v in rows if v.get('event')=='register_complete'];libs=[v for v in rows if v.get('event')=='loaded_libraries']
  assert len(init)==len(reg)==len(libs)==16 and {v['device_id'] for v in init}==set(range(16))
  assert all(v['rc']==0 and v['protocol']=='SDMA' and v['protocol_value']==1 and v['role']==('Prefill' if n in [30,31] else 'Decode') for v in init)
  assert all(v['rc']==0 and v['allocations']==180 and v['allocation_bytes']>0 for v in reg)
  for row in libs:
   assert any('/opt/glm53/sdk13/memfabric_hybrid/lib/libmf_smem.so'==p for p in row['paths'])
   assert any('/opt/glm53/sdk13/memcache_hybrid/lib/libmf_memcache.so'==p for p in row['paths'])
  all_init+=init;all_reg+=reg
  capacity=[line for log in data['logs'].values() for line in log['capacity_lines']];weight=[l for l in capacity if 'Loading model weights took' in l];assert len(weight)==16
  graph_dp={int(re.search(r'Worker_DP(\d+)_TP0',l)[1]) for l in capacity if 'Graph capturing finished' in l and re.search(r'Worker_DP(\d+)_TP0',l)}
  if n in [43,45]:assert graph_dp==engines
  nodes[str(n)]={'container_id':data['container_id'],'actual_model_identity':data['actual_model_identity'],'health':200,'SDK_SDMA_init_count':16,'SDK_register_count':16,'cache_tokens_by_global_DP':cache,'idle_native_gauges':gauges,'weightload_completed_workers':16,'target_graph_capture_completed_DP':sorted(graph_dp),'profiler_artifacts':[],'native_log_full_archives':{k:{f:v[f] for f in ['archive','archive_sha256','sha256','bytes','full_archive_complete']} for k,v in data['logs'].items()}}
 assert set(P_caps)==set(range(4)) and set(D_caps)==set(range(16)) and len(all_init)==len(all_reg)==64
 out={'project_id':'glm53','framework':'vllm','shared_run_id':'20261017T1766-vllm-shared-pool-v4','pass':True,'classification':'NATIVE_FULL_STARTUP_ALL4_HEALTH_ALL64_SDMA_INIT_REGISTER_TARGET_GRAPH_COMPLETE_IDLE','nodes':nodes,'P_tokens_per_DP':P_caps,'D_tokens_per_DP':D_caps,'D_full80640_slots_by_DP':{k:v//80640 for k,v in D_caps.items()},'total_model_SDK_SDMA_init':64,'total_model_KV_buffer_registers':64,'registered_buffers_per_worker':180,'buffer_count_scope':'Main79K+79V+22indexer layout; auxiliary actualtransfer coverage stillrequired byshort/longrequest audit.','model_requests':0,'short_long_correctness_pending':True,'formal_requests':0,'baseline_accepted':False,'source_evidence_sha256':source}
 p=ROOT/'results/vllm-shared-startup-v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'all4health':200,'SDMA_init_register':64,'P_tokens_per_DP':P_caps,'D_tokens_per_DP':D_caps,'model_requests':0,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
