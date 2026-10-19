"""Independent native mount/device/CLI/overlay audit; local artifacts only."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/'runs/20261018T-L09-sglang-C32-packed-normal-L04-rollback-v1'
 receipt=json.loads((run/'receipt.json').read_text())
 assert receipt['pool_kept_running'] and receipt['cleanup']=={} and receipt['model_requests']==0
 assert len(receipt['native_preflight'])==4 and all(x==0 for x in receipt['native_preflight'].values())
 outer=json.loads((run/'live-meta-snapshot.json').read_text());assert outer['rc']==0
 meta=json.loads(outer['stdout']);assert meta['copy_rc']==0
 mounts={};lines=[]
 for f in meta['files']:
  for line in f['critical_lines_complete']:
   lines.append(line);match=re.search(r'Mount bm on loc\{rank=(\d+),media=DRAM\}, capacity:(\d+)\s+successfully',line)
   if match:
    sid,size=map(int,match.groups());assert sid not in mounts
    mounts[sid]={'bytes':size,'native_line':line,'source_sha256':f['sha256']}
 assert set(mounts)==set(range(4)) and sum(x['bytes'] for x in mounts.values())==568*1024**3
 for key,value in [('ock.mmc.meta.lease_ttl_ms','60000'),('ock.mmc.evict_threshold_high','99'),('ock.mmc.evict_threshold_low','98')]:assert any(f'{key} = {value}' in line for line in lines)
 providers={};native={}
 for n,size in {34:136,35:112,36:160,41:160}.items():
  outer=receipt['providers'][str(n)];assert outer['rc']==0;y=json.loads(outer['stdout']);status=y['provider_status']
  assert y['ready'] and status['phase']=='READY' and not y['supervisor_status']['model_started']
  assert status['configuration']['dram_size']==f'{size}GB' and status['configuration']['max_dram_size']=='160GB'
  assert status['sdk_version']=='1.3.0' and status['sdk_module'].startswith('/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/')
  sid=next(e for e in status['events'] if e['event']=='READY')['local_service_id'];assert mounts[sid]['bytes']==size*1024**3
  assert len(y['container_id'])==64 and y['name']==f'glm53-sglang-C32-packed-normal-L04-rollback-v1-20261018-{n}'
  providers[str(n)]={'container_id':y['container_id'],'name':y['name'],'local_service_id':sid,'provider_container_pid':status['pid'],'dram_GiB':size}
  outer=json.loads((run/f'native-preflight-{n}.json').read_text());assert outer['rc']==0;y=json.loads(outer['stdout'])
  assert y['query_rc']==0 and y['container_id']==providers[str(n)]['container_id'] and not y['supervisor_after']['model_started']
  assert 'GLM53_BUFFER_DEDUP_' not in y['stdout'],'RestoredL04mustnotloadarchivedPcandidate'
  hw=json.loads(next(l.split('=',1)[1] for l in y['stdout'].splitlines() if l.startswith('GLM53_SHARED_PREFLIGHT=')))
  assert hw['torch_device_count']==16 and hw['ACL_count']==[16,0] and hw['tp']==hw['ep']==32 and hw['dp']==(4 if n in [34,35] else 16) and hw['MTP_steps']==5
  installed=[json.loads(l.split('=',1)[1]) for l in y['stdout'].splitlines() if l.startswith('GLM53_DEEPEP_METADATA_BATCH_INSTALLED=')]
  assert bool(installed)==(n in [36,41])
  if installed:
   assert len(installed)==1 and installed[0]['node']==str(n) and installed[0]['source_sha256']=='48fbc94dd60c7c37e198935717357d2479150099487ee0d9a749ed41083a7f84'
  native[str(n)]=hw|{'real_native_DeepEP_module_import_patch_verified':bool(installed),'models_not_started':True}
 assert len({x['local_service_id'] for x in providers.values()})==4
 paths=[run/'receipt.json',run/'live-meta-snapshot.json',*[run/f'native-preflight-{n}.json' for n in [34,35,36,41]]]
 result={'project_id':'glm53','framework':'sglang','shared_run_id':'20261018T-L09-sglang-C32-packed-normal-L04-rollback-v1','pass':True,'actual_MC_bytes':568*1024**3,'native_mounts':mounts,'providers':providers,'native_preflight':native,'meta':json.loads((run/'meta-start.json').read_text()),'model_requests':0,'model_started':False,'qualification':'POOL_AND_REAL_MODULE_IMPORT_ONLY; runtime correctness/performance pending','source_evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths}}
 p=ROOT/'results/sglang-C32-packed-L04-rollback-pool-capacity-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pass':True,'actual_MC_GiB':568,'result_sha256':sha(p)}))
if __name__=='__main__':main()
