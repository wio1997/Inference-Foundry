"""Independent native remount qualification for tested L10 P rollback."""
import collections,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/20261019T-L12B-reject-native-candidate-restore-L10-P-pools-v1'
def read(name):
 p=RUN/name;w=json.loads(p.read_text());assert w['rc']==0;return json.loads(w['stdout'])
def main():
 receipt=json.loads((RUN/'receipt.json').read_text());assert receipt['pass'] and receipt['selected_variant']=='rollback' and not receipt['models_started'] and receipt['phase']=='ROLLBACK_P_POOLS_READY_AWAIT_INDEPENDENT_CAPACITY_AUDIT'
 before=read('rollback-meta-before.json');after=read('rollback-meta-after.json');assert before['container_id']==after['container_id'] and before['Meta_unchanged'] and after['Meta_unchanged']
 delta=list((collections.Counter(after['critical_lines'])-collections.Counter(before['critical_lines'])).elements());mounts={}
 for line in delta:
  match=re.search(r'(?<![A-Za-z])Mount bm on loc\{rank=(\d+),media=DRAM\}, capacity:(\d+)\s+successfully',line)
  if match and int(match[2]):
   rank,size=map(int,match.groups());assert rank not in mounts;mounts[rank]={'bytes':size,'native_line':line}
 assert len(mounts)==2 and sum(x['bytes'] for x in mounts.values())==248*1024**3
 old=json.loads((ROOT/'results/sglang-C32-L12B-selected-P-pool-capacity-v1.json').read_text());providers={};preflight={}
 for n,size in [(34,136),(35,112)]:
  stopped=read('stop-candidate-'+str(n)+'.json');assert stopped['provider_closed0'] and stopped['resource_shutdown_complete'] and stopped['all_NPUs_empty'] and stopped['D_Meta_retained']
  started=read('rollback-pool-'+str(n)+'.json');assert started['ready'] and started['run_rc']==0;provider=started['provider_status'];sup=started['supervisor_status'];assert provider['phase']=='READY' and not sup['model_started'] and sup['phase']=='POOL_READY_WAITING_MODEL_SIGNAL'
  event=next(e for e in provider['events'] if e['event']=='READY');sid=event['local_service_id'];assert mounts[sid]['bytes']==size*1024**3
  providers[str(n)]={'container_id':started['container_id'],'name':started['name'],'local_service_id':sid,'dram_GiB':size,'run_id':'20261019T-L12B-rollback-v1','remote_root':'/data/wio/inference-foundry/glm53/runs/20261019T-L12B-rollback-v1'}
  obj=read('rollback-preflight-'+str(n)+'.json');assert obj['query_rc']==0;v=json.loads(obj['stdout'].split('GLM53_SHARED_PREFLIGHT=')[-1]);assert v['torch_device_count']==16 and v['ACL_count']==[16,0] and v['tp']==32 and v['dp']==4 and v['MTP_steps']==5 and v['hicache_size_decimal_GB']==9 and v['hicache_host_memory_mode']=='buffer_only';preflight[str(n)]=v
 for n in [36,41]:providers[str(n)]=old['providers'][str(n)]
 assert not ({providers[str(n)]['local_service_id'] for n in [34,35]} & {providers[str(n)]['local_service_id'] for n in [36,41]})
 sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.glob('*.json')}
 result={'project_id':'glm53','framework':'sglang','pass':True,'selected_variant':'rollback','classification':'TESTED_L10_P_POOL_RESTORED_NEW_MODELS_NOT_STARTED','actual_MC_bytes':568*1024**3,'fresh_P_native_mounts':mounts,'retained_D_native_capacity_GiB':320,'providers':providers,'native_preflight':preflight,'meta_container_id':after['container_id'],'Meta_retained':True,'D_retained_health_asserted_during_P_shutdown':True,'D_native_identity_recheck_required_before_model_requests':True,'new_P_models_started':False,'native_compatibility_enabled':False,'native_primitive_previously_qualified':True,'candidate_model_rejected':True,'candidate_failure_capture':'runs/20261019T-L12B-P-startup-failure-capture-v1','model_requests':0,'source_evidence_sha256':sources,'new_P_model_qualification_pending':True}
 p=ROOT/'results/sglang-C32-L12B-restored-L10-P-pool-capacity-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'MC_GiB':568,'variant':'rollback','models_started':False}))
if __name__=='__main__':main()
