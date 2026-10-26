"""Freeze one L04 trial from actual new-runtime qualification; local files only."""
import argparse,ast,hashlib,importlib.util,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LAUNCH_SHA='c57a105a1803ca1d28bcd098f6ea722c77883757ecbd3180c00ae5798b8a4b2d'
CAPTURE_SHA='520bb2ef5cca591218912fccea2a494cbbb6b6eec9b8e54d3a152fb8342dba14'
IDENTITY_SHA='d334af2192b4140fd47229d2461e66ee15a655f7a2b21b5425d282c6631ff542'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(relative,digest):
 p=ROOT/relative;assert sha(p)==digest
 spec=importlib.util.spec_from_file_location('reviewed_module',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def wrapper(module,digest,call):
 return ('"""Frozen L04 trial operation; actual execution via ZCode only."""\n'
  'import hashlib,importlib.util\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\n'
  'def main():\n'+f' path=ROOT/{module!r};assert hashlib.sha256(path.read_bytes()).hexdigest()=={digest!r}\n'
  " spec=importlib.util.spec_from_file_location('operation',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n"
  f' {call}\n'+"if __name__=='__main__':main()\n")
def prepare(C):
 assert C in [16,32,64]
 paired=load('scripts/glm53_sglang_metadata_launch_v1.py',LAUNCH_SHA)
 assert sha(ROOT/'scripts/glm53_sglang_metadata_native_capture_v1.py')==CAPTURE_SHA
 assert sha(ROOT/'scripts/glm53_metadata_model_identity_v2.py')==IDENTITY_SHA
 settings=json.loads((ROOT/'config/formal-runs-v2/sglang-C16.json').read_text())
 runtime=json.loads((ROOT/'config/optimization-overlays/routing-metadata-v1/runtime.json').read_text())
 for key in ['shared_run_id','shared_remote_root','runtime_containers','model_image','meta_container_id']:settings[key]=runtime[key]
 qualification='results/sglang-metadata-native-readiness-v1.json';proof=json.loads((ROOT/qualification).read_text())
 assert proof['pass'] and proof['all32_D_actual_first_use_verified'] and proof['all_DP_short_and_80K_semantic_verified'] and proof['native_cache_and_SDMA_verified']
 settings.update(prefix_transfer_helper='scripts/glm53_metadata_prefix_transfer_v1.py',prefix_transfer_sha256='db7cc8bb5603051cbd03f710d4ad232851581206686c4a125ec6b40be5767185',native_qualification_path=qualification,native_qualification_sha256=sha(ROOT/qualification),model_identity_helper='scripts/glm53_metadata_model_identity_v2.py',model_identity_sha256=IDENTITY_SHA)
 number={16:1976,32:1990,64:2000}[C];rid=f'20261017T{number}-sglang-formal-C{C}-L04-v2';capture=f'20261017T{number+2}-sglang-L04-C{C}-native-v2'
 settings.update(status='FROZEN_READY_FOR_SINGLE_EXECUTION',concurrency=C,run_id=rid,namespace=f'glm53-formal-{rid}-C{C}',client_container_name=f'glm53-sglang-L04-C{C}-v1-20261017-43')
 input_manifest=json.loads((ROOT/f'runs/20261017T0615-formal-inputs-v3/C{C}-manifest.json').read_text());settings['input_manifest_sha256']=input_manifest['sha256']
 settings['predecessor_formal_certificates']={}
 for fw,c in [('vllm',C)]+[('sglang',c) for c in [16,32,64] if c<C]:
  relative=f'results/vllm-formal-C{c}-v2-measurement-v1.json' if fw=='vllm' else f'results/sglang-L04-C{c}-v1-measurement-v1.json'
  p=ROOT/relative;previous=json.loads(p.read_text());assert previous['measurement_valid'] is True and previous['framework']==fw and previous['concurrency']==c
  settings['predecessor_formal_certificates'][f'{fw}-C{c}']={'path':relative,'sha256':sha(p)}
 # Native qualification requires the actual functional clients/capture closed.
 # Resource exclusion from the latest preceding formal cohort is separate.
 if C==16:preceding={'framework':'vllm','run_id':'20261017T1870-vllm-formal-C32-v2','native_capture_run':'20261017T1874-vllm-formal-C32-native-v4','nodes':[30,31,43,45]}
 elif C==32:preceding={'framework':'sglang','run_id':'20261017T1976-sglang-formal-C16-L04-v2','native_capture_run':'20261017T1978-sglang-L04-C16-native-v2','nodes':[34,35,36,41]}
 else:preceding={'framework':'vllm','run_id':'20261017T1880-vllm-formal-C64-v2','native_capture_run':'20261017T1884-vllm-formal-C64-native-v4','nodes':[30,31,43,45]}
 settings['complete_before_start']=[preceding]
 relative=f'config/optimization-overlays/routing-metadata-v1/formal-C{C}-v2.json';path=ROOT/relative;assert not path.exists(),'Preserve frozen trial settings'
 raw=(json.dumps(settings,indent=2)+'\n').encode();digest=hashlib.sha256(raw).hexdigest()
 with tempfile.NamedTemporaryFile(dir=ROOT/'config',prefix='.L04-static-',suffix='.json',delete=False) as f:f.write(raw);temporary=Path(f.name)
 try:paired.validate_static(ROOT,str(temporary.relative_to(ROOT)),digest);paired.validate_no_overlap(ROOT,settings)
 finally:temporary.unlink()
 launcher=ROOT/f'scripts/run_sglang_metadata_formal_C{C}_v2.py';post=ROOT/f'scripts/inspect_sglang_metadata_formal_C{C}_v2.py';assert not launcher.exists() and not post.exists()
 launch_source=wrapper('scripts/glm53_sglang_metadata_launch_v1.py',LAUNCH_SHA,f'm.execute({relative!r},{digest!r})')
 post_source=wrapper('scripts/glm53_sglang_metadata_native_capture_v1.py',CAPTURE_SHA,f'm.execute({relative!r},{digest!r},{capture!r})')
 ast.parse(launch_source);ast.parse(post_source)
 path.write_bytes(raw);launcher.write_text(launch_source);post.write_text(post_source)
 (ROOT/'runs'/rid).mkdir(exist_ok=True);(ROOT/'runs'/capture).mkdir(exist_ok=True)
 result={'project_id':'glm53','concurrency':C,'run_id':rid,'settings':relative,'settings_sha256':digest,'launcher':str(launcher.relative_to(ROOT)),'launcher_sha256':sha(launcher),'native_capture_launcher':str(post.relative_to(ROOT)),'native_capture_launcher_sha256':sha(post),'native_capture_run':capture,'qualification_sha256':sha(ROOT/qualification),'static_validation_pass':True,'actual_resource_calls':0,'model_requests':0,'dispatched':False}
 (ROOT/f'results/sglang-L04-C{C}-v1-frozen-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('concurrency',type=int,choices=[16,32,64]);a=parser.parse_args();prepare(a.concurrency)
