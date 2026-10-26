"""Reconcile actual stopped SG1720 identities; preserve exit1; close Meta if safe."""
import ast,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1]
dep=P.parent/'stop_sglang_1720_for_metadata_v1.py';assert hashlib.sha256(dep.read_bytes()).hexdigest()=='2e7fb55190be91704433b94b00d451810c5c38c0367c8389f0e415d3c4cbdf18'
s=importlib.util.spec_from_file_location('prior_stop',dep);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
RUN=ROOT/'runs/20261017T1953-sglang-stop-closeout-v1'
HOST=r'''
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1953-sglang-stop-closeout-v1');root.mkdir(parents=True,exist_ok=False)
d=owned(CID,NAME);shared=pathlib.Path(OLD_ROOT)
supervisor=json.loads((shared/'supervisor-status.json').read_text());provider=json.loads((shared/'provider-status.json').read_text())
npu=cmd(['npu-smi','info'])
out={'node':NODE,'container_id':CID,'state':d['State'],'supervisor':supervisor,'provider':provider,'npu':npu,'model_requests':0,'SDK_init_calls':0,'SDK_logs':[],'model_log_manifests':[],'profile_preservation':None}
assert not d['State']['Running'] and d['State']['Pid']==0,'No additional stop action authorized'
assert npu.count('No running processes')==8,'Residual NPU process: no further mutation'
assert supervisor['phase']=='STOPPED' and provider['phase']=='CLOSE_COMPLETE' and provider['events'][-1]['rc']==0,'Provider close not complete: preserve Meta'
for directory in ['/var/log/memcache_hybrid','/var/log/memfabric_hybrid']:
 dst=root/directory.rsplit('/',1)[1]
 r=subprocess.run(['docker','cp',CID+':'+directory,str(dst)],capture_output=True,text=True,timeout=120)
 assert r.returncode==0,(directory,r.stderr)
 for p in dst.rglob('*'):
  if not p.is_file():continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(1048576),b''):h.update(block)
  row={'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
  if p.suffix=='.dat':
   assert row['bytes']<=8388608;row['text']=p.read_text(errors='strict')
  out['SDK_logs'].append(row)
for p in shared.glob('model.*.txt'):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-65536));tail=f.read().decode(errors='replace')
 out['model_log_manifests'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'last_64KiB':tail})
if NODE in [36,41]:
 preservation=pathlib.Path(REMOTE)/'profile-preservation.json';out['profile_preservation']=json.loads(preservation.read_text())
 assert pathlib.Path(out['profile_preservation']['destination']).exists() and not pathlib.Path(out['profile_preservation']['source']).exists()
out['native_failure_preserved']=d['State']['ExitCode']!=0
out['safe_to_release_Meta_after_all_four_checked']=True
print(json.dumps(out))
'''
def main():
 prior=json.loads((ROOT/'runs/20261017T1952-stop-sglang-1720-for-metadata-v1/receipt.json').read_text());assert prior['nodes']=={'34':0,'35':0,'36':0,'41':1} and prior['meta_rc'] is None
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'nodes':{},'meta_rc':None,'original_D41_exit1_preserved':True,'model_requests':0,'SDK_init_calls':0}
 try:
  for n in m.IDS:
   result=m.invoke(n,HOST,600);(RUN/f'node-{n}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['nodes'][str(n)]=result['rc'];assert result['rc']==0
  result=m.invoke(34,m.STOP_META,150);(RUN/'meta-stop.json').write_text(json.dumps(result,indent=2)+'\n');receipt['meta_rc']=result['rc'];assert result['rc']==0
  receipt['all_four_stopped_empty_NPU_and_provider_close0']=True
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
