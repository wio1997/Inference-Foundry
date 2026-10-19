"""Recover only already completed original long-client artifacts; no new requests."""
import hashlib,importlib.util,json,ast,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'scripts/run_sglang_C32_target_graph_long_v4.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='de2461b739c537f9c06679bcef31a50dac01b0c40d92dec3d3dae803d2d3fb9b'
s=importlib.util.spec_from_file_location('long_source',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
HOST=r'''import pathlib,json,hashlib,subprocess
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-target-graph-long-v3');assert root.is_dir()
name='glm53-sglang-C32-target-graph-long-v3-20261018-43'
check=subprocess.run(['docker','container','inspect',name],capture_output=True,text=True,timeout=20)
assert check.returncode!=0 and 'No such container' in check.stderr,'Original client still present or inspect failed unexpectedly; preserve and stop'
status=json.loads((root/'semantic-phase-status.json').read_text());assert status['client_rc']==0 and not status.get('error') and len(status['gates'])==2 and all(g['rc']==0 and g['pass'] for g in status['gates'].values())
report_raw=(root/'semantic-result.json').read_bytes();assert len(report_raw)<=16*1024**2 and hashlib.sha256(report_raw).hexdigest()=='bd62ee57a10c7168ade442230879b4b385caa12c8f7bb1eaa67f52e9ade75968'
report=json.loads(report_raw);assert report['semantic_pass'] and len(report['cases'])==6 and report['external_backend_HTTP_attempts']==report['successful_backend_inferences']==12 and report['actual_successful_output_tokens']==2408
artifacts={}
for path in sorted(root.iterdir()):
 if path.is_file():
  digest=hashlib.sha256();total=0
  with path.open('rb') as f:
   for block in iter(lambda:f.read(1024**2),b''):digest.update(block);total+=len(block)
  artifacts[path.name]={'sha256':digest.hexdigest(),'bytes':total}
  if path.name in ('semantic-result.json','semantic-phase-status.json') or path.name.startswith('metadata-gate-'):
   assert total<=16*1024**2;artifacts[path.name]['text']=path.read_text()
print('GLM53_LONG_ARTIFACTS='+json.dumps(artifacts),flush=True)
print('GLM53_CLIENT_EXIT='+json.dumps(status),flush=True)
print('GLM53_RECOVERY_IDENTITY='+json.dumps({'original_client_container_absent':True,'original_result_sha256':hashlib.sha256(report_raw).hexdigest(),'original_client_rc':status['client_rc'],'model_requests':0,'SDK_init_calls':0}),flush=True)
'''
def main():
 run=ROOT/'runs/20261018T-L10-long-completed-artifact-recovery-v1';run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write('once')
 old=ROOT/'runs/20261018T-L10-sglang-target-graph-long-v3';dispatch=json.loads((old/'dispatch.json').read_text());assert len(dispatch['model_precheck'])==4 and all(x==0 for x in dispatch['model_precheck'].values())
 assert not (old/'receipt.json').exists(),'Original completed receipt exists; preserve rather than recover twice'
 proc=subprocess.run(['pgrep','-f','^.*python[^ ]* .*run_sglang_C32_target_graph_long_v4[.]py$'],capture_output=True,text=True);assert proc.returncode==1 and not proc.stdout,'Original collector still active'
 result=m.invoke(43,HOST,timeout=120);(run/'node-43.json').write_text(json.dumps(result,indent=2)+'\n');assert result['rc']==0
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rc':0,'model_precheck':dispatch['model_precheck'],'original_run_id':'20261018T-L10-sglang-target-graph-long-v3','collection_only_recovered':True,'original_local_collector_absent':True,'original_controller_receipt_absent':True,'original_result_verified':True,'original_client_rc':0,'new_model_requests':0,'new_SDK_init_calls':0,'original_dispatched_source_sha256':dispatch['launcher_sha256'],'original_dispatch_sha256':hashlib.sha256((old/'dispatch.json').read_bytes()).hexdigest(),'cause_of_local_collector_exit':'UNDETERMINED; original raw local log empty, original client completed remotely. Do not overwrite or fabricate old receipt.'}
 (run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'rc':0,'recovered_actual_backend_successes':12,'new_model_requests':0}))
if __name__=='__main__':main()
