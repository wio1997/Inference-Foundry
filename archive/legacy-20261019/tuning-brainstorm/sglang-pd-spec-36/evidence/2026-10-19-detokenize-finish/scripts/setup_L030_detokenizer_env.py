#!/usr/bin/env python3
"""Create a task-owned tokenizer-only environment; never modify the old CPU env."""
import datetime,hashlib,json,pathlib,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L030-R003-tokenizer-env';O.mkdir(parents=True,exist_ok=False);V=R/'.venv-detok';assert not V.exists()
spec=json.loads((R/'specs/L030-tokenizer-env.json').read_text());commands=[[str(R/'.venv-cpu/bin/python'),'-m','venv',str(V)],[str(V/'bin/python'),'-m','pip','--isolated','install','--disable-pip-version-check','--no-cache-dir','--index-url','https://pypi.org/simple']+spec['packages']]
for n,cmd in enumerate(commands):
 with (O/f'{n}.stdout.log').open('w') as out,(O/f'{n}.stderr.log').open('w') as err:p=subprocess.run(cmd,stdout=out,stderr=err,timeout=500)
 if p.returncode:raise RuntimeError('environment command '+str(n)+' failed: '+str(p.returncode))
p=subprocess.run([str(V/'bin/python'),'-m','pip','freeze','--all'],capture_output=True,text=True,timeout=30);assert p.returncode==0;(O/'pip-freeze.txt').write_text(p.stdout)
p=subprocess.run([str(V/'bin/python'),'-c','import transformers,tokenizers,sys,json; print(json.dumps({"python":sys.version,"transformers":transformers.__version__,"tokenizers":tokenizers.__version__}))'],capture_output=True,text=True,timeout=30);assert p.returncode==0;versions=json.loads(p.stdout);assert versions['transformers']=='5.12.1' and versions['tokenizers']=='0.22.2'
result={'status':'PASS','environment':versions,'task_owned_venv':'.venv-detok','prior_cpu_environment_modified':False,'native_executed':False,'packages':spec['packages'],'pip_freeze_sha256':hashlib.sha256((O/'pip-freeze.txt').read_bytes()).hexdigest(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
