from pathlib import Path
import datetime,hashlib,importlib.metadata,json,os,subprocess
R=Path(__file__).resolve().parent;os.chdir(R)
c=json.loads((R/'candidate.json').read_text());p=json.loads((R/'actual_image_source_prerequisite.json').read_text());assert c['image_id']==p['image_id']
for e in p['selected_files']:
 f=Path('/vllm-workspace/vllm-ascend')/e['path'];assert f.stat().st_size==e['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==e['sha256'],str(f)
assert hashlib.sha256(Path('/models/ds41/config.json').read_bytes()).hexdigest()=='43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7'
versions={n:importlib.metadata.version(n) for n in ['vllm','vllm-ascend','torch','torch-npu']};assert versions['vllm-ascend']=='0.32.0rc1' and versions['torch-npu']=='2.10.0.post7'
os.environ.update(c['environment']);os.environ['PYTORCH_NPU_ALLOC_CONF']='expandable_segments:True';os.environ['TORCHINDUCTOR_CACHE_DIR']=str(R/'cache/inductor');os.environ['TORCHAIR_CACHE_HOME']=str(R/'cache/torchair')
(R/'actual_native_launch.json').write_text(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':c['argv'],'environment':c['environment'],'versions':versions,'all113_actual_source_pins_match':True,'native_source_overlay':False,'runtime_validated':False},indent=2)+'\n')
with (R/'native_cli_help.log').open('x') as log:r=subprocess.run(['vllm','serve','--help=all'],cwd=R,stdout=log,stderr=subprocess.STDOUT,timeout=180)
assert r.returncode==0,('NativeCLIhelp/importfailed beforemodelstartup',r.returncode)
os.execvpe(c['argv'][0],c['argv'],os.environ)
