"""Matching-stage SG admission; no all-three-vLLM certificate dependency."""
import hashlib,importlib.util,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CONTROLLER_SHA='6810044962d74782b013543486eecaf3d589a1eb1a5740df43416065b1ae9de2'
CLIENT_SHA='935a4a1c3188b3c7d2041f91e946423838dece7bcc5669daee6ae81c6e6c56a3'
def checked(root,relative,sha):
 assert re.fullmatch('[0-9a-f]{64}',sha)
 path=(root/relative).resolve();assert path.is_relative_to(root.resolve()) and hashlib.sha256(path.read_bytes()).hexdigest()==sha
 return path
def validate_static(root,relative,sha):
 settings=json.loads(checked(root,relative,sha).read_text())
 assert settings['project_id']=='glm53' and settings['framework']=='sglang' and settings['execution_policy_revision']=='paired-stage-v1'
 assert settings['status']=='FROZEN_READY_FOR_SINGLE_EXECUTION' and settings['harness_revision']=='v2-uniform-fresh-upstream-HTTP-no-retries'
 C=settings['concurrency'];assert C in [16,32,64];rid=settings['run_id'];assert Path(rid).name==rid and f'-sglang-formal-C{C}-' in rid and settings['namespace']==f'glm53-formal-{rid}-C{C}'
 required={f'vllm-C{C}':('vllm',C)}|{f'sglang-C{c}':('sglang',c) for c in [16,32,64] if c<C}
 assert set(settings['predecessor_formal_certificates'])==set(required),'Matching vLLM stage and lower SG stages only'
 for key,(fw,c) in required.items():
  item=settings['predecessor_formal_certificates'][key];proof=json.loads(checked(root,item['path'],item['sha256']).read_text())
  assert proof['project_id']=='glm53' and proof['framework']==fw and proof['concurrency']==c and proof['measurement_valid'] is True
 proof=json.loads(checked(root,settings['native_qualification_path'],settings['native_qualification_sha256']).read_text())
 assert proof['project_id']=='glm53' and proof['framework']=='sglang' and proof['pass'] is True and proof['shared_run_id']==settings['shared_run_id']
 assert proof['runtime_CIDs']=={n:item['container_id'] for n,item in settings['runtime_containers'].items()}
 for path,sha in proof['source_evidence_sha256'].items():checked(root,path,sha)
 assert settings['client_files_sha256']['glm53_formal_ais_client_v1.py']==CLIENT_SHA
 checked(root,settings['client_files']['glm53_formal_ais_client_v1.py'],CLIENT_SHA)
 transfer_path=checked(root,settings['prefix_transfer_helper'],settings['prefix_transfer_sha256'])
 spec=importlib.util.spec_from_file_location('runtime_bound_transfer',transfer_path);transfer=importlib.util.module_from_spec(spec);spec.loader.exec_module(transfer)
 _,destination=transfer.render('sglang',C,settings['input_manifest_sha256'],rid,'CPU43_STATIC_VALIDATE','P34_STATIC_VALIDATE')
 assert destination==settings['shared_remote_root']+'/formal-gates/'+rid+'/data','Prefix transfer destination must match actual qualified runtime root'
 assert settings['resource_controller_sha256']==CONTROLLER_SHA
 return settings,checked(root,'scripts/glm53_formal_resource_controller_paired_v1.py',CONTROLLER_SHA)
def validate_no_overlap(root,settings):
 for prior in settings.get('complete_before_start',[]):
  rid=prior['run_id'];assert Path(rid).name==rid
  rpc=json.loads((root/'runs'/rid/'controller-rpc-state.json').read_text())
  assert rpc['done_received'] and rpc['connection_rc'] is not None,'Previous benchmark connection still active'
  capture=root/'runs'/prior['native_capture_run'];receipt=json.loads((capture/'receipt.json').read_text());assert receipt['nodes'] and all(value==0 for value in receipt['nodes'].values())
  # Quiescence is a resource-exclusion check, not a new performance certificate.
  for node in prior['nodes']:
   wire=json.loads((capture/f'node-{node}.json').read_text());assert wire['rc']==0;raw=wire['stdout'].encode();transport=wire['native_output_transport'];assert len(raw)==transport['raw_bytes'] and hashlib.sha256(raw).hexdigest()==transport['raw_sha256'];o=json.loads(raw)
   assert o['provider_phase']=='READY' and o['/health']['status']==200
   prefixes=('vllm:num_requests_running{','vllm:num_requests_waiting{') if prior['framework']=='vllm' else ('sglang:num_running_reqs{','sglang:num_queue_reqs{')
   gauges=[float(line.rsplit(' ',1)[1]) for line in o['/metrics']['body'].splitlines() if line.startswith(prefixes)];assert gauges and all(value==0 for value in gauges),'Previous workload remains active'
 return True
def execute(relative,sha):
 settings,path=validate_static(ROOT,relative,sha);validate_no_overlap(ROOT,settings)
 run=ROOT/'runs'/settings['run_id'];run.mkdir(parents=True,exist_ok=True)
 with (run/'execution-once.lock').open('x') as file:file.write(str(__import__('os').getpid()))
 spec=importlib.util.spec_from_file_location('paired_controller',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 result=module.execute(settings);print('GLM53_PAIRED_FORMAL_COMPLETE='+json.dumps(result),flush=True);return result
