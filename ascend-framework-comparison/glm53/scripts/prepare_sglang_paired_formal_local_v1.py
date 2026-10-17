"""Prepare one SG stage using its actual matching vLLM v2 certificate."""
import argparse,ast,hashlib,importlib.util,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LAUNCH_SHA='b227e29c24f0f85e4462457db3623a421c35cd1d3a165392e91f96bfd91f077f'
CONTROLLER_SHA='6810044962d74782b013543486eecaf3d589a1eb1a5740df43416065b1ae9de2'
CAPTURE_SHA='89373b845e5623f41fa444d1052389bda4385d99a4bc81fc48e511a3ea25a71a'
CLIENT_SHA='935a4a1c3188b3c7d2041f91e946423838dece7bcc5669daee6ae81c6e6c56a3'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def wrapper(module,digest,call):
 return ('"""Frozen paired-stage SG operation; actual execution via ZCode only."""\n'
  'import hashlib,importlib.util\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\n'
  'def main():\n'
  f' path=ROOT/{module!r};assert hashlib.sha256(path.read_bytes()).hexdigest()=={digest!r}\n'
  " spec=importlib.util.spec_from_file_location('operation',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n"
  f' {call}\n'
  "if __name__=='__main__':main()\n")
def prepare(C):
 assert C in [16,32,64]
 source=ROOT/'scripts/glm53_sglang_paired_launch_v1.py';assert sha(source)==LAUNCH_SHA
 assert sha(ROOT/'scripts/glm53_sglang_formal_native_capture_v1.py')==CAPTURE_SHA
 spec=importlib.util.spec_from_file_location('paired_validation',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 settings=json.loads((ROOT/f'config/formal-runs-v1/sglang-C{C}.draft.json').read_text());number={16:1890,32:1900,64:1910}[C];rid=f'20261017T{number}-sglang-formal-C{C}-v2';capture_id=f'20261017T{number+4}-sglang-formal-C{C}-native-v4'
 settings.update(status='FROZEN_READY_FOR_SINGLE_EXECUTION',harness_revision='v2-uniform-fresh-upstream-HTTP-no-retries',execution_policy_revision='paired-stage-v1',run_id=rid,namespace=f'glm53-formal-{rid}-C{C}',client_container_name=f'glm53-sglang-formal-C{C}-v2-20261017-43',resource_controller_sha256=CONTROLLER_SHA)
 settings['client_files']['glm53_formal_ais_client_v1.py']='scripts/glm53_formal_ais_client_v2.py';settings['client_files_sha256']['glm53_formal_ais_client_v1.py']=CLIENT_SHA
 settings['client_entrypoint_compatibility_name']='glm53_formal_ais_client_v1.py staged with pinned v2 source bytes'
 settings.pop('prefix_data_staged_receipt',None);settings.pop('prefix_data_staged_receipt_sha256',None)
 prior=[('vllm',C)]+[('sglang',c) for c in [16,32,64] if c<C];settings['predecessor_formal_certificates']={}
 for fw,c in prior:
  relative=f'results/{fw}-formal-C{c}-v2-measurement-v1.json';path=ROOT/relative;proof=json.loads(path.read_text())
  assert proof['project_id']=='glm53' and proof['framework']==fw and proof['concurrency']==c and proof['measurement_valid'] is True
  settings['predecessor_formal_certificates'][f'{fw}-C{c}']={'path':relative,'sha256':sha(path)}
 # Resource exclusion only: SG16 waits for the already-running C32 to finish,
 # not for its performance certificate, and never for vLLM C64.
 preceding={16:{'framework':'vllm','run_id':'20261017T1870-vllm-formal-C32-v2','native_capture_run':'20261017T1874-vllm-formal-C32-native-v4','nodes':[30,31,43,45]},32:{'framework':'sglang','run_id':'20261017T1890-sglang-formal-C16-v2','native_capture_run':'20261017T1894-sglang-formal-C16-native-v4','nodes':[34,35,36,41]},64:{'framework':'vllm','run_id':'20261017T1880-vllm-formal-C64-v2','native_capture_run':'20261017T1884-vllm-formal-C64-native-v4','nodes':[30,31,43,45]}}[C]
 settings['complete_before_start']=[preceding]
 relative=f'config/formal-runs-v2/sglang-C{C}.json';path=ROOT/relative;path.parent.mkdir(exist_ok=True);assert not path.exists(),'Never overwrite frozen settings'
 raw=(json.dumps(settings,indent=2)+'\n').encode();digest=hashlib.sha256(raw).hexdigest()
 with tempfile.NamedTemporaryFile(dir=ROOT/'config',prefix='.paired-static-',suffix='.json',delete=False) as file:file.write(raw);temporary=Path(file.name)
 try:module.validate_static(ROOT,str(temporary.relative_to(ROOT)),digest)
 finally:temporary.unlink()
 launch=ROOT/f'scripts/run_sglang_formal_C{C}_paired_v1.py';post=ROOT/f'scripts/inspect_sglang_formal_C{C}_native_v4.py';assert not launch.exists() and not post.exists()
 launcher_source=wrapper('scripts/glm53_sglang_paired_launch_v1.py',LAUNCH_SHA,f'm.execute({relative!r},{digest!r})')
 post_source=wrapper('scripts/glm53_sglang_formal_native_capture_v1.py',CAPTURE_SHA,f'm.execute({relative!r},{digest!r},{capture_id!r})')
 ast.parse(launcher_source);ast.parse(post_source)
 with path.open('xb') as file:file.write(raw)
 with launch.open('x') as file:file.write(launcher_source)
 with post.open('x') as file:file.write(post_source)
 (ROOT/'runs'/rid).mkdir(parents=True,exist_ok=True);(ROOT/'runs'/capture_id).mkdir(parents=True,exist_ok=True)
 result={'project_id':'glm53','framework':'sglang','concurrency':C,'execution_policy_revision':'paired-stage-v1','run_id':rid,'settings':relative,'settings_sha256':digest,'launcher':str(launch.relative_to(ROOT)),'launcher_sha256':sha(launch),'native_capture_launcher':str(post.relative_to(ROOT)),'native_capture_launcher_sha256':sha(post),'native_capture_run':capture_id,'required_measurement_certificates':list(settings['predecessor_formal_certificates']),'complete_before_start':[preceding],'static_qualification_passed':True,'execution_waits_for_preceding_run_and_native_quiescence':True,'no_C64_certificate_dependency_for_SG_C16':C==16,'actual_resource_calls':0,'model_requests':0,'SDK_init_calls':0,'dispatched':False}
 (ROOT/f'results/sglang-formal-C{C}-v2-paired-frozen-review-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('concurrency',type=int,choices=[16,32,64]);a=parser.parse_args();prepare(a.concurrency)
