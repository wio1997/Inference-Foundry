"""Independent local audit of frozen native pool evidence; zero resource actions."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 r=ROOT/'runs/20261017T1757-vllm-shared-pool-v3';receipt=json.loads((r/'receipt.json').read_text());assert receipt['pool_kept_running'] and receipt['cleanup']=={} and len(receipt['native_preflight'])==4 and all(x==0 for x in receipt['native_preflight'].values())
 meta_path=ROOT/'runs/20261017T1759-vllm-shared-meta-mounts-v1/node-30.json';outer=json.loads(meta_path.read_text());assert outer['rc']==0;meta=json.loads(outer['stdout'])
 mounts={};lines=[]
 for f in meta['files']:
  for line in f['critical_lines_complete']:
   lines.append(line);match=re.search(r'Mount bm on loc\{rank=(\d+),media=DRAM\}, capacity:(\d+)\s+successfully',line)
   if match:
    sid,size=map(int,match.groups());assert sid not in mounts;mounts[sid]={'bytes':size,'native_line':line,'source_sha256':f['sha256']}
 assert set(mounts)==set(range(4)) and sum(v['bytes'] for v in mounts.values())==568*1024**3
 for key,value in [('ock.mmc.meta.lease_ttl_ms','60000'),('ock.mmc.evict_threshold_high','99'),('ock.mmc.evict_threshold_low','98')]:assert any(f'{key} = {value}' in line for line in lines)
 providers={};native={}
 for n,size in {30:192,31:192,43:128,45:56}.items():
  y=json.loads(receipt['providers'][str(n)]['stdout']);status=y['provider_status'];assert y['ready'] and status['phase']=='READY' and y['supervisor_status']['model_started'] is False
  assert status['configuration']['dram_size']==f'{size}GB' and status['configuration']['max_dram_size']=='192GB' and status['sdk_module'].startswith('/opt/glm53/sdk13/')
  event=next(e for e in status['events'] if e['event']=='READY');sid=event['local_service_id'];assert mounts[sid]['bytes']==size*1024**3
  providers[str(n)]={'container_id':y['container_id'],'name':y['name'],'local_service_id':sid,'provider_container_pid':status['pid'],'dram_GiB':size}
  p=r/f'native-preflight-{n}.json';x=json.loads(p.read_text());assert x['rc']==0;y=json.loads(x['stdout']);assert y['query_rc']==0 and not y['supervisor_after']['model_started']
  hw=json.loads(next(line.split('=',1)[1] for line in y['stdout'].splitlines() if line.startswith('GLM53_SHARED_HARDWARE=')));assert hw['torch_device_count']==16 and hw['ACL_count']==[16,0]
  graphs=[json.loads(line.split('=',1)[1]) for line in y['stdout'].splitlines() if line.startswith('GLM53_GRAPH_PREFLIGHT=')];assert len(graphs)==4 and {g['node'] for g in graphs}=={30,31,43,45}
  native[str(n)]=hw|{'native_all4_CLI_validated':True,'graphs':graphs,'profiler_inactive':True}
 assert len({x['local_service_id'] for x in providers.values()})==4
 out={'project_id':'glm53','run':str(r.relative_to(ROOT)),'pass':True,'classification':'FOUR_READY_NATIVE16_DEVICES_AND_568GiB_META_CAPACITY_MODELS_NOT_STARTED','actual_MC_bytes':568*1024**3,'native_mounts':{str(k):v for k,v in mounts.items()},'meta_lease_ttl_ms':60000,'meta_evict_high':99,'meta_evict_low':98,'providers':providers,'native_preflight':native,'meta':json.loads((r/'meta-start.json').read_text()),'model_requests':0,'model_started':False,'source_evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [r/'receipt.json',r/'live-meta-snapshot.json',meta_path,*[r/f'native-preflight-{n}.json' for n in [30,31,43,45]]]},'model_qualification_pending':True,'source_selected_tail_limit':'Originalselectedtail100 omittedearlymounts/TTL;1759readscompletecritical7linesfromsameSHA-frozenlogcopy. No SDK/Meta/modelrerun.'}
 p=ROOT/'results/vllm-shared-pool-capacity-v3.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'pass':True,'actual_MC_GiB':568,'native_mounts_GiB':{k:v['bytes']//1024**3 for k,v in mounts.items()},'native_all4_torch16_ACL16_CLI':True,'models_started':False,'result_sha256':sha(p)}))
if __name__=='__main__':main()
