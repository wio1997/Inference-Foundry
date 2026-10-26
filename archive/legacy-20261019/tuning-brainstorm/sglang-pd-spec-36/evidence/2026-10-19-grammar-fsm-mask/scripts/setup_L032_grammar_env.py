#!/usr/bin/env python3
"""Isolated grammar dependencies, reuse task-owned CPU/HF packages read-only, no weights."""
import datetime,hashlib,json,pathlib,subprocess,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];s=json.loads((R/'specs/L032-grammar-env.json').read_text());O=R/'runs/L032-R003-grammar-env';O.mkdir(parents=True,exist_ok=False);V=R/'.venv-grammar';assert not V.exists()
def run(cmd,name,timeout=400):
 with (O/(name+'.stdout.log')).open('w') as out,(O/(name+'.stderr.log')).open('w') as err:p=subprocess.run(cmd,stdout=out,stderr=err,timeout=timeout)
 assert p.returncode==0,(name,p.returncode)
def pathof(p):return subprocess.run([str(p),'-c','import sysconfig;print(sysconfig.get_paths()["purelib"])'],capture_output=True,text=True,check=True).stdout.strip()
old_paths=[pathlib.Path(pathof(R/n/'bin/python')) for n in ['.venv-cpu','.venv-front']];protected=[p/'torch/__init__.py' for p in old_paths if (p/'torch/__init__.py').exists()];before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
run([str(R/'.venv-cpu/bin/python'),'-m','venv',str(V)],'venv');pure=pathlib.Path(pathof(V/'bin/python'));(pure/'lf32_readonly_task_dependencies.pth').write_text('\n'.join(map(str,old_paths))+'\n')
meta=json.loads((R/'notes/L032-xgrammar-pypi-preflight.json').read_text());chosen=next(x for x in meta['distributions'] if x['filename']==s['xgrammar_wheel']);assert meta['version']=='0.2.7'
with urllib.request.urlopen(chosen['url'],timeout=60) as response:wheel=response.read(5_000_000)
assert hashlib.sha256(wheel).hexdigest()==chosen['sha256'];w=O/chosen['filename'];w.write_bytes(wheel)
with urllib.request.urlopen('https://pypi.org/pypi/jsonschema/json',timeout=40) as response:jdata=response.read(5_000_000)
jver=json.loads(jdata)['info']['version'];(O/'selected-packages.json').write_text(json.dumps({'xgrammar':chosen,'jsonschema':{'version':jver,'metadata_sha256':hashlib.sha256(jdata).hexdigest()},'TVM_FFI':'0.1.10','readonly_reused_package_roots':list(map(str,old_paths))},indent=2)+'\n')
run([str(V/'bin/python'),'-m','pip','--isolated','install','--disable-pip-version-check','--no-cache-dir','--no-deps',str(w)],'xgrammar')
run([str(V/'bin/python'),'-m','pip','--isolated','install','--disable-pip-version-check','--no-cache-dir','--index-url','https://pypi.org/simple','apache-tvm-ffi==0.1.10','jsonschema=='+jver],'dependencies')
probe='import sys,json,importlib.metadata as m,torch,xgrammar,transformers,tokenizers; print(json.dumps({"python":sys.version,"torch":torch.__version__,"torch_path":torch.__file__,"xgrammar":m.version("xgrammar"),"xgrammar_path":xgrammar.__file__,"apache-tvm-ffi":m.version("apache-tvm-ffi"),"jsonschema":m.version("jsonschema"),"transformers":transformers.__version__,"tokenizers":tokenizers.__version__}))'
p=subprocess.run([str(V/'bin/python'),'-c',probe],capture_output=True,text=True,timeout=90);(O/'import.stdout.log').write_text(p.stdout);(O/'import.stderr.log').write_text(p.stderr);assert p.returncode==0;env=json.loads(p.stdout);assert env['torch'].split('+')[0]=='2.8.0' and env['xgrammar']=='0.2.7' and env['transformers']=='5.12.1' and env['tokenizers']=='0.22.2'
p=subprocess.run([str(V/'bin/python'),'-m','pip','freeze','--all'],capture_output=True,text=True,check=True,timeout=30);(O/'pip-freeze.txt').write_text(p.stdout)
assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in before.items())
# Optional exact historical config recovery; a mismatch never becomes model identity proof.
url='https://huggingface.co/zai-org/GLM-5.3/resolve/ba32f8c19c8738e2c57c71798c601d477a7bf844/config.json';config={'url':url,'status':'UNAVAILABLE','expected_sha256':s['config_sha256']}
try:
 with urllib.request.urlopen(url,timeout=50) as response:data=response.read(200_000)
 got=hashlib.sha256(data).hexdigest();config['sha256']=got;config['matches_historical']=got==s['config_sha256']
 if config['matches_historical']:
  q=R/'fixtures/L032-exact-model-metadata/config.json';q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data);x=json.loads(data);config.update(status='MATCHED',selected_fields={k:x.get(k) for k in ['vocab_size','eos_token_id','bos_token_id','architectures','model_type']})
except Exception as e:config.update(error_type=type(e).__name__,error=str(e)[:200])
(O/'model-config-identity.json').write_text(json.dumps(config,indent=2)+'\n');v={'status':'PASS','environment':env,'old_CPU_Torch_protected_hashes_unchanged':True,'source_pyproject_Torch':'2.13.0 declared only','historical_image_Torch':'2.10.0+cpu observed earlier','installed_target_xgrammar':'UNKNOWN','local_only':True,'import_only_no_FSM_execution_credit':True,'model_config_identity':config,'native_executed':False,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(O/'result.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
