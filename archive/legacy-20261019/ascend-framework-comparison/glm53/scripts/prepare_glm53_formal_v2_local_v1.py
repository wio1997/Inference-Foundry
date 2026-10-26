"""Freeze one uniform fresh-HTTP baseline run locally; never executes resources."""
import argparse,ast,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CLIENT_SHA='935a4a1c3188b3c7d2041f91e946423838dece7bcc5669daee6ae81c6e6c56a3'
CONTROLLER_SHA='d070d50197036420fb07f05a2a595155db3360de736d6c93cfc0847c40fa3404'
REGRESSION_SHA='0d482fb212a53b94bbf18ab7e535b9a1dc0be675b14790ab13ae63b24b0c4443'
SGLAUNCH_SHA='c130c7c858aa69d3cd2c132ab6e62d4f1eea5d7709eb92742ce9d24038a0651e'
SGCAPTURE_SHA='89373b845e5623f41fa444d1052389bda4385d99a4bc81fc48e511a3ea25a71a'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(relative,digest):
 path=ROOT/relative;assert sha(path)==digest;return path
def wrapper(module,digest,body):
 return ('"""Frozen v2 operation; actual resource execution through ZCode only."""\n'
  'import hashlib,importlib.util\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\n'
  'def main():\n'
  f' path=ROOT/{module!r};assert hashlib.sha256(path.read_bytes()).hexdigest()=={digest!r}\n'
  " spec=importlib.util.spec_from_file_location('operation',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n"
  f' {body}\n'
  "if __name__=='__main__':main()\n")
def prepare(fw,C):
 assert fw in ['vllm','sglang'] and C in [16,32,64]
 require('scripts/glm53_formal_ais_client_v2.py',CLIENT_SHA);require('scripts/glm53_formal_resource_controller_v1.py',CONTROLLER_SHA)
 regression=json.loads(require('results/upstream-http-connection-regression-v1.json',REGRESSION_SHA).read_text());assert regression['pass'] and regression['automatic_retries']==0
 original=ROOT/f'config/formal-runs-v1/{fw}-C{C}{".draft" if fw=="sglang" else ""}.json'
 settings=json.loads(original.read_text());settings['status']='FROZEN_READY_FOR_SINGLE_EXECUTION';settings['harness_revision']='v2-uniform-fresh-upstream-HTTP-no-retries'
 index=[16,32,64].index(C);number=(1860 if fw=='vllm' else 1890)+10*index;rid=f'20261017T{number}-{fw}-formal-C{C}-v2';capture_id=f'20261017T{number+4}-{fw}-formal-C{C}-native-v4'
 settings['run_id']=rid;settings['namespace']=f'glm53-formal-{rid}-C{C}';settings['client_container_name']=f'glm53-{fw}-formal-C{C}-v2-20261017-43'
 # Keep the host-controller entrypoint name; bytes and repository source are v2.
 settings['client_files']['glm53_formal_ais_client_v1.py']='scripts/glm53_formal_ais_client_v2.py';settings['client_files_sha256']['glm53_formal_ais_client_v1.py']=CLIENT_SHA
 settings['client_entrypoint_compatibility_name']='glm53_formal_ais_client_v1.py staged with pinned v2 source bytes'
 settings.pop('prefix_data_staged_receipt',None);settings.pop('prefix_data_staged_receipt_sha256',None)
 prior=([('vllm',v) for v in [16,32,64]] if fw=='sglang' else [('vllm',v) for v in [16,32,64] if v<C])+([('sglang',v) for v in [16,32,64] if v<C] if fw=='sglang' else [])
 settings['predecessor_formal_certificates']={}
 for previous_fw,previous_C in prior:
  relative=f'results/{previous_fw}-formal-C{previous_C}-v2-measurement-v1.json';path=ROOT/relative;proof=json.loads(path.read_text());assert proof['project_id']=='glm53' and proof['framework']==previous_fw and proof['concurrency']==previous_C and proof['measurement_valid'] is True
  settings['predecessor_formal_certificates'][f'{previous_fw}-C{previous_C}']={'path':relative,'sha256':sha(path)}
 proof=json.loads(require(settings['native_qualification_path'],settings['native_qualification_sha256']).read_text());assert proof['pass'] is True and proof['framework']==fw and proof['runtime_CIDs']=={n:item['container_id'] for n,item in settings['runtime_containers'].items()}
 for relative,digest in proof['evidence_sha256' if fw=='vllm' else 'source_evidence_sha256'].items():require(relative,digest)
 relative=f'config/formal-runs-v2/{fw}-C{C}.json';path=ROOT/relative;path.parent.mkdir(exist_ok=True);assert not path.exists()
 raw=(json.dumps(settings,indent=2)+'\n').encode();settings_sha=hashlib.sha256(raw).hexdigest()
 if fw=='vllm':
  launch_source=f'''"""Once-only uniform v2 vLLM baseline; no profiling."""
import hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 path=ROOT/{relative!r};assert hashlib.sha256(path.read_bytes()).hexdigest()=={settings_sha!r};settings=json.loads(path.read_text());assert settings['harness_revision']=='v2-uniform-fresh-upstream-HTTP-no-retries'
 cert=ROOT/settings['native_qualification_path'];assert hashlib.sha256(cert.read_bytes()).hexdigest()==settings['native_qualification_sha256'];proof=json.loads(cert.read_text());assert proof['pass']
 for file,sha in proof['evidence_sha256'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==sha
 path=ROOT/'scripts/glm53_formal_resource_controller_v1.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=={CONTROLLER_SHA!r}
 run=ROOT/'runs'/settings['run_id'];run.mkdir(parents=True,exist_ok=True)
 with (run/'execution-once.lock').open('x') as file:file.write(str(__import__('os').getpid()))
 spec=importlib.util.spec_from_file_location('controller',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);print('GLM53_FORMAL_RESOURCE_COMPLETE='+json.dumps(m.execute(settings)),flush=True)
if __name__=='__main__':main()
'''
  template=require('scripts/inspect_vllm_formal_C64_native_v3.py','5eb80af4929aea8a74dde630ba780a6f37c34d6cff89ab14491094634243ae63').read_text()
  post_source=template.replace('20261017T1810-vllm-formal-C64-v1',rid).replace('20261017T1814-vllm-formal-C64-native-v3',capture_id).replace("report['concurrency']==64",f"report['concurrency']=={C}")
  post_source=post_source.replace('2097153','8388609').replace('2097152','8388608').replace("sum(item['bytes'] for item in out)<=64*1024**2","sum(item['bytes'] for item in out)<=128*1024**2")
 else:
  require('scripts/glm53_sglang_formal_launch_v1.py',SGLAUNCH_SHA);require('scripts/glm53_sglang_formal_native_capture_v1.py',SGCAPTURE_SHA)
  launch_source=wrapper('scripts/glm53_sglang_formal_launch_v1.py',SGLAUNCH_SHA,f'm.execute({relative!r},{settings_sha!r})')
  post_source=wrapper('scripts/glm53_sglang_formal_native_capture_v1.py',SGCAPTURE_SHA,f'm.execute({relative!r},{settings_sha!r},{capture_id!r})')
 ast.parse(launch_source);ast.parse(post_source)
 launch=ROOT/f'scripts/run_{fw}_formal_C{C}_v2.py';post=ROOT/f'scripts/inspect_{fw}_formal_C{C}_native_v4.py';assert not launch.exists() and not post.exists()
 with path.open('xb') as file:file.write(raw)
 with launch.open('x') as file:file.write(launch_source)
 with post.open('x') as file:file.write(post_source)
 (ROOT/'runs'/rid).mkdir(parents=True,exist_ok=True);(ROOT/'runs'/capture_id).mkdir(parents=True,exist_ok=True)
 result={'project_id':'glm53','framework':fw,'concurrency':C,'run_id':rid,'settings':relative,'settings_sha256':settings_sha,'launcher':str(launch.relative_to(ROOT)),'launcher_sha256':sha(launch),'native_capture_launcher':str(post.relative_to(ROOT)),'native_capture_launcher_sha256':sha(post),'native_capture_run':capture_id,'harness_revision':'v2','source_client_sha256':CLIENT_SHA,'known_reused_POST_disconnect_regression_passed':True,'all_predecessors_are_uniform_v2':True,'model_requests':0,'SDK_init_calls':0,'actual_resource_calls':0,'dispatched':False}
 (ROOT/f'results/{fw}-formal-C{C}-v2-frozen-review-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('framework',choices=['vllm','sglang']);parser.add_argument('concurrency',type=int,choices=[16,32,64]);a=parser.parse_args();prepare(a.framework,a.concurrency)
