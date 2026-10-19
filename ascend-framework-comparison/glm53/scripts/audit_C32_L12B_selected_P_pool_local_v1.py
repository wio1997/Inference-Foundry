"""Independent actual remount/retained-D/primitive audit before selected P model start."""
import collections,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/20261019T-L12B-controlled-P-owner-empty-and-pool-v1'
def read(name):
 p=RUN/name;r=json.loads(p.read_text());assert r['rc']==0,(name,r.get('stderr'));return json.loads(r['stdout'])
def main():
 receipt=json.loads((RUN/'receipt.json').read_text());assert receipt['pass'] and receipt['phase']=='POOLS_READY_AWAITING_INDEPENDENT_CAPACITY_AUDIT' and not receipt['models_started'];kind=receipt['selected_variant'];assert kind in ['candidate','rollback']
 before=read('meta-before.json');after=read('meta-after.json');assert before['container_id']==after['container_id'] and before['Meta_unchanged'] and after['Meta_unchanged']
 delta=list((collections.Counter(after['critical_lines'])-collections.Counter(before['critical_lines'])).elements());mounts={}
 for line in delta:
  a=re.search(r'(?<![A-Za-z])Mount bm on loc\{rank=(\d+),media=DRAM\}, capacity:(\d+)\s+successfully',line)
  if a and int(a[2]):
   rank,size=map(int,a.groups());assert rank not in mounts;mounts[rank]={'bytes':size,'native_line':line}
 native=read('fixture-owner-empty-34.json');fixture=native.get('fixture_result');native_qualified=False
 if kind=='candidate':
  assert native['pass'] and native['fixture_rc']==0 and native['original_NPU_contexts_unchanged'] and native['owned_NPU_context_PIDs_before']==native['owned_NPU_context_PIDs_after']==[]
  assert fixture and fixture['pass'] and fixture['native_kernel_launches']==82 and fixture['native_return_codes']==[0] and fixture['all_kernel_calls_device0']
  assert fixture['target_layers']==78 and fixture['draft_layers']==1 and fixture['target_indexer_rows']==78 and fixture['BF16'] and fixture['single_layer_isolation_all_components']
  assert len(fixture['checks'])==10 and all(x['whole_tensor_equal'] for x in fixture['checks'])
  assert fixture['optional_extension_sha256']=='3e87649f4d2ea85936cd94b439afd3473973bd62abcc06f777c9b875cf1a9d46' and fixture['compatibility_sha256']=='50d7cc8040b9956092dd40fe204bc5d6d3633bf04f1ba6937a8273fc72ac0b2f'
  assert fixture['SDK_bm_init_calls']==0 and fixture['SDK_offload_init_calls']==1 and fixture['model_requests']==0;native_qualified=True
 else:assert not native['pass'] and native['fixture_rc']!=0
 old=json.loads((ROOT/'results/sglang-C32-target-graph-fullmesh-pool-capacity-v3.json').read_text());providers={};native_preflight={}
 for n,size in [(34,136),(35,112)]:
  stopped=read('stop-'+str(n)+'.json');assert stopped['resource_shutdown_complete'] and stopped['provider_closed0'] and stopped['all_NPUs_empty'] and stopped['complete_profile_preserved']['retained_in_place']
  out=read(kind+'-pool-'+str(n)+'.json');assert out['ready'] and out['run_rc']==0;status=out['provider_status'];sup=out['supervisor_status'];assert status['phase']=='READY' and sup['phase']=='POOL_READY_WAITING_MODEL_SIGNAL' and not sup['model_started']
  event=next(e for e in status['events'] if e['event']=='READY');sid=event['local_service_id'];assert mounts[sid]['bytes']==size*1024**3
  providers[str(n)]={'container_id':out['container_id'],'name':out['name'],'local_service_id':sid,'dram_GiB':size,'remote_root':'/data/wio/inference-foundry/glm53/runs/20261019T-L12B-'+kind+'-v1','run_id':'20261019T-L12B-'+kind+'-v1'}
  pre=read(kind+'-preflight-'+str(n)+'.json');assert pre['query_rc']==0;value=json.loads(pre['stdout'].split('GLM53_SHARED_PREFLIGHT=')[-1]);assert value['torch_device_count']==16 and value['ACL_count']==[16,0] and value['tp']==32 and value['dp']==4 and value['MTP_steps']==5 and value['hicache_size_decimal_GB']==9 and value['hicache_host_memory_mode']=='buffer_only';native_preflight[str(n)]=value
 for n in [36,41]:
  first=read('health-'+str(n)+'.json');last=read('retained-D-health-'+str(n)+'.json');assert first['container_id']==last['container_id']==old['providers'][str(n)]['container_id'];assert first['NPU_context_PIDs']==last['NPU_context_PIDs'] and len(last['NPU_context_PIDs'])==18 and last['healthy_idle']
  providers[str(n)]=old['providers'][str(n)]|{'remote_root':'/data/wio/inference-foundry/glm53/runs/'+old['shared_run_id'],'run_id':old['shared_run_id'],'retained_native_capacity_source':'results/sglang-C32-target-graph-fullmesh-pool-capacity-v3.json'}
 assert len(mounts)==2 and sum(x['bytes'] for x in mounts.values())==248*1024**3
 assert not ({providers[str(n)]['local_service_id'] for n in [34,35]} & {providers[str(n)]['local_service_id'] for n in [36,41]})
 sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob('*.json')}
 out={'project_id':'glm53','pass':True,'classification':'P_ONLY_REMOUNT_568GiB_RETAINED_D_META_BEFORE_P_MODELS','selected_variant':kind,'actual_MC_bytes':568*1024**3,'fresh_P_native_mounts':mounts,'retained_D_native_capacity_GiB':320,'providers':providers,'native_preflight':native_preflight,'meta_container_id':after['container_id'],'Meta_retained':True,'D_identities_and_contexts_retained':True,'new_P_models_started':False,'native_82_kernel_fixture_qualified':native_qualified,'native_fixture_actual':fixture,'raw_fixture_failure_preserved':not native_qualified,'model_requests':0,'source_evidence_sha256':sources,'model_qualification_pending':True,'formal_performance_qualified':False}
 p=ROOT/'results/sglang-C32-L12B-selected-P-pool-capacity-v1.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'variant':kind,'MC_GiB':568,'native82_qualified':native_qualified,'D_Meta_retained':True,'new_P_models_started':False}))
if __name__=='__main__':main()
