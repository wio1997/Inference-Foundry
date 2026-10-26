"""Controller-only full orchestration fixture; never calls SSH or a real SDK."""
import ast,base64,hashlib,importlib.util,json,shlex,tempfile,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(file):
 spec=importlib.util.spec_from_file_location('fixture_'+file[:-3],ROOT/'scripts'/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 m=load('glm53_formal_resource_controller_v1.py');base=load('inventory_readonly.py');gate=load('glm53_formal_gate_remote_v1.py');transfer=load('glm53_prefix_transfer_v5.py');real_load=m.load
 cap=json.loads((ROOT/'results/vllm-shared-pool-capacity-v4.json').read_text());CIDs={n:v['container_id'] for n,v in cap['providers'].items()};seen=[];reports=[]
 files=['glm53_formal_ais_client_v1.py','glm53_formal_host_controller_v2.py','aisbench_glm53_token_ids.py','glm53_request_terminal_v1.py','glm53_vllm_pd_gateway_v3.py','glm53_sglang_pd_gateway_v3.py','glm53_closed_loop_v1.py','glm53_aisbench_native_steady_v1.py','glm53_runtime_metrics_v1.py','glm53_storage_proof_v1.py']
 with tempfile.TemporaryDirectory() as temporary:
  cert=Path(temporary)/'fixture-certificate.json';cert.write_text(json.dumps({'project_id':'glm53','framework':'vllm','shared_run_id':'20261017T1766-vllm-shared-pool-v4','runtime_CIDs':CIDs,'pass':True,'LOCAL_FIXTURE_ONLY':True}))
  rid='20261017T1776-local-fixture-vllm-formal-C16-v1';run=ROOT/'runs'/rid;assert not run.exists();run.mkdir()
  settings={'project_id':'glm53','framework':'vllm','concurrency':16,'run_id':rid,'namespace':'glm53-formal-'+rid+'-C16','native_qualification_path':str(cert),'native_qualification_sha256':sha(cert),'shared_run_id':'20261017T1766-vllm-shared-pool-v4','runtime_containers':cap['providers'],'model_image':'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83','model_identity_helper':'scripts/glm53_vllm_shared_model_identity_v3.py','model_identity_sha256':sha(ROOT/'scripts/glm53_vllm_shared_model_identity_v3.py'),'gate_renderer_sha256':sha(ROOT/'scripts/glm53_formal_gate_remote_v1.py'),'prefix_transfer_helper':'scripts/glm53_prefix_transfer_v5.py','prefix_transfer_sha256':sha(ROOT/'scripts/glm53_prefix_transfer_v5.py'),'input_manifest_sha256':'dff4e3943c14f8d12c1964f0aad9ed419441e6b94274f40507d2fdd34f4e3ab9','client_files':{f:'scripts/'+f for f in files},'client_files_sha256':{f:sha(ROOT/'scripts'/f) for f in files},'gate_files':{'glm53_formal_cache_gate_v1.py':'scripts/glm53_formal_cache_gate_v1.py'},'gate_files_sha256':{'glm53_formal_cache_gate_v1.py':sha(ROOT/'scripts/glm53_formal_cache_gate_v1.py')},'expected_sdk_root':'/opt/glm53/sdk13/','meta_service_url':'tcp://172.16.12.30:29680','config_store_url':'tcp://172.16.12.30:29681','native_client_config_path':'config/vllm-correctness-v10/local.conf','native_client_config_sha256':sha(ROOT/'config/vllm-correctness-v10/local.conf'),'max_dram_size':'192GB','shared_remote_root':'/data/wio/inference-foundry/glm53/runs/20261017T1766-vllm-shared-pool-v4','client_container_name':'glm53-LOCAL-FIXTURE-ONLY'}
  expected={'project_id':'glm53','framework':'vllm','namespace':settings['namespace'],'concurrency':16,'manifest_sha256':settings['input_manifest_sha256']}
  def fake_invoke(commands,code=None,timeout=120):
   seen.append((commands,code))
   if 'kubectl get pod' in commands[0]:
    pod=shlex.split(commands[0])[5];node=next(n for n,item in base.NODES.items() if item[0]==pod);return {'rc':0,'stdout':f'172.16.12.{node}','stderr':''}
   if code is not None:
    ast.parse(code)
    if "root/'host-settings.json'" in code:return {'rc':0,'stdout':json.dumps({'settings_sha256':'0'*64}),'stderr':''}
    if "control/'gate-controller.once'" in code:
     report=expected|{'prefix_count':16,'pass':True,'all_expected_keys_present':True,'actual_payload_samples_read':True,'close_rc':0,'LOCAL_FIXTURE_ONLY':True};reports.append(report)
     return {'rc':0,'stdout':json.dumps({'query_rc':0,'report':report,'report_sha256':hashlib.sha256((json.dumps(report,indent=2)+'\n').encode()).hexdigest()}),'stderr':''}
    return {'rc':0,'stdout':'{}','stderr':''}
   pipe=shlex.split(commands[0]);assert pipe[:4]==['bash','-o','pipefail','-c'];parts=shlex.split(pipe[4]);assert parts.count('|')==1
   for index,item in enumerate(parts):
    if item=='-c':ast.parse(parts[index+1])
   return {'rc':0,'stdout':json.dumps({'manifest_sha256':settings['input_manifest_sha256'],'files':{f'fixture{i}':'0'*64 for i in range(17)}}),'stderr':''}
  base.invoke=fake_invoke
  def fake_rpc(command,root,check,callback):
   assert check==expected and len(seen)>10 and command[:len(base.SSH)]==base.SSH
   response=callback(expected);assert response['pass'];return {'LOCAL_FIXTURE_ONLY':True,'client_baseline_accepted':True}
  def patched_load(name,path,digest):
   if name=='inventory':return base
   if name=='formal_rpc':return types.SimpleNamespace(rpc=fake_rpc)
   return real_load(name,path,digest)
  m.load=patched_load
  try:
   receipt=m.execute(settings);assert receipt['phase']=='CLIENT_COMPLETE_NATIVE_POST_AUDIT_REQUIRED' and len(reports)==1
   generated=[code for _,code in seen if code is not None];assert len(generated)==11
   assert all('loadavg' in code and 'owned_NPU_host_PIDs' in code for code in generated[:4])
   assert any('gate-controller.once' in code for code in generated)
   before=len(seen);bad=settings|{'runtime_containers':{}};failed=False
   try:m.execute(bad)
   except AssertionError:failed=True
   assert failed and len(seen)==before
   failed=False
   try:m.execute(settings|{'meta_service_url':'tcp://172.16.12.30:29780'})
   except AssertionError:failed=True
   assert failed and len(seen)==before
   certdata=json.loads(cert.read_text());certdata['framework']='sglang';cert.write_text(json.dumps(certdata));bad=settings|{'framework':'sglang','native_qualification_sha256':sha(cert)};failed=False
   try:m.execute(bad)
   except KeyError:failed=True
   assert failed and len(seen)==before
   out={'project_id':'glm53','pass':True,'local_fixture_only':True,'real_resource_calls':0,'real_model_calls':0,'real_SDK_calls':0,'generated_HOST_AST':len(generated),'prefix_sender_receiver_AST':2,'full_orchestration_and_gate_callback':True,'wrong_metadata_port_rejected_before_resource_calls':True,'CID_mismatch_rejected_before_resource_calls':True,'SGLang_missing_vLLM_predecessors_rejected_before_resource_calls':True,'resource_controller_sha256':sha(ROOT/'scripts/glm53_formal_resource_controller_v1.py'),'gate_renderer_sha256':settings['gate_renderer_sha256'],'prefix_transfer_sha256':settings['prefix_transfer_sha256']}
   (ROOT/'results/formal-resource-controller-local-fixture-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
  finally:
   import shutil;shutil.rmtree(run)
if __name__=='__main__':main()
