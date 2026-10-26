"""Freeze one SG run locally after actual predecessor certificates exist."""
import argparse,ast,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LAUNCH_SHA='c130c7c858aa69d3cd2c132ab6e62d4f1eea5d7709eb92742ce9d24038a0651e'
CAPTURE_SHA='89373b845e5623f41fa444d1052389bda4385d99a4bc81fc48e511a3ea25a71a'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def wrapper(module,sha,body):
 return ('"""Once-only frozen SG resource operation; execute through ZCode only."""\n'
  'import hashlib,importlib.util\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\n'
  'def main():\n'
  f' path=ROOT/{module!r};assert hashlib.sha256(path.read_bytes()).hexdigest()=={sha!r}\n'
  " spec=importlib.util.spec_from_file_location('operation',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n"
  f' {body}\n'
  "if __name__=='__main__':main()\n")
def prepare(C):
 assert C in [16,32,64]
 helper=ROOT/'scripts/glm53_sglang_formal_launch_v1.py';assert digest(helper)==LAUNCH_SHA
 capture=ROOT/'scripts/glm53_sglang_formal_native_capture_v1.py';assert digest(capture)==CAPTURE_SHA
 spec=importlib.util.spec_from_file_location('launch_validation',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 settings=json.loads((ROOT/f'config/formal-runs-v1/sglang-C{C}.draft.json').read_text());settings['status']='FROZEN_READY_FOR_SINGLE_EXECUTION'
 prior=[('vllm',c) for c in [16,32,64]]+[('sglang',c) for c in [16,32,64] if c<C];certs={}
 for fw,c in prior:
  relative=f'results/{fw}-formal-C{c}-measurement-v1.json';path=ROOT/relative;proof=json.loads(path.read_text())
  assert proof['project_id']=='glm53' and proof['framework']==fw and proof['concurrency']==c and proof['measurement_valid'] is True
  certs[f'{fw}-C{c}']={'path':relative,'sha256':digest(path)}
 settings['predecessor_formal_certificates']=certs
 relative=f'config/formal-runs-v1/sglang-C{C}.json';path=ROOT/relative;assert not path.exists(),'Never replace frozen settings'
 raw=(json.dumps(settings,indent=2)+'\n').encode();settings_sha=hashlib.sha256(raw).hexdigest()
 # Validate the final bytes in an owned temporary config before freezing paths.
 import tempfile
 with tempfile.NamedTemporaryFile(prefix='.sglang-formal-check-',suffix='.json',dir=ROOT/'config',delete=False) as file:file.write(raw);temporary=Path(file.name)
 try:m.validate(ROOT,str(temporary.relative_to(ROOT)),settings_sha)
 finally:temporary.unlink()
 launch=ROOT/f'scripts/run_sglang_formal_C{C}_v1.py';post=ROOT/f'scripts/inspect_sglang_formal_C{C}_native_v1.py';run=ROOT/'runs'/settings['run_id'];capture_id={16:'20261017T1830-sglang-formal-C16-native-v1',32:'20261017T1840-sglang-formal-C32-native-v1',64:'20261017T1850-sglang-formal-C64-native-v1'}[C]
 assert not launch.exists() and not post.exists() and not (run/'execution-once.lock').exists()
 launch_source=wrapper('scripts/glm53_sglang_formal_launch_v1.py',LAUNCH_SHA,f'm.execute({relative!r},{settings_sha!r})')
 post_source=wrapper('scripts/glm53_sglang_formal_native_capture_v1.py',CAPTURE_SHA,f'm.execute({relative!r},{settings_sha!r},{capture_id!r})')
 ast.parse(launch_source);ast.parse(post_source)
 with path.open('xb') as file:file.write(raw)
 with launch.open('x') as file:file.write(launch_source)
 with post.open('x') as file:file.write(post_source)
 run.mkdir(parents=True,exist_ok=True);(ROOT/'runs'/capture_id).mkdir(parents=True,exist_ok=True)
 result={'project_id':'glm53','concurrency':C,'run_id':settings['run_id'],'settings':relative,'settings_sha256':settings_sha,'launcher':str(launch.relative_to(ROOT)),'launcher_sha256':digest(launch),'native_capture_launcher':str(post.relative_to(ROOT)),'native_capture_launcher_sha256':digest(post),'native_capture_run':capture_id,'all_predecessor_certificates_validated_before_freeze':True,'actual_resource_calls':0,'model_requests':0,'SDK_init_calls':0,'dispatched':False}
 (ROOT/f'results/sglang-formal-C{C}-frozen-review-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('concurrency',type=int,choices=[16,32,64]);a=parser.parse_args();prepare(a.concurrency)
