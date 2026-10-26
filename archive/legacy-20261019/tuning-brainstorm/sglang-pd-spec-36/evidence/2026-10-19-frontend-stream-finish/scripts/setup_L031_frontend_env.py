#!/usr/bin/env python3
"""Task-owned protocol-only environment, public index and saved exact dependency lock."""
import datetime,hashlib,json,pathlib,subprocess,sys,urllib.request
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L031-R002-frontend-env';O.mkdir(parents=True,exist_ok=False);V=R/'.venv-front';assert not V.exists();s=json.loads((R/'specs/L031-frontend-env.json').read_text());packages=[];metadata=[]
for n,v in s['packages'].items():
 if v=='PUBLIC_INDEX_CURRENT':
  url='https://pypi.org/pypi/'+n+'/json'
  with urllib.request.urlopen(url,timeout=40) as r:data=r.read(8_000_000)
  x=json.loads(data);v=x['info']['version'];metadata.append({'package':n,'source':url,'selected_version':v,'metadata_sha256':hashlib.sha256(data).hexdigest()})
 packages.append(n+'=='+v)
(O/'selected-packages.json').write_text(json.dumps({'scope':'Local CPU protocol stack; NOT target image version equivalence','packages':packages,'public_index_metadata':metadata},indent=2)+'\n')
commands=[[str(R/'.venv-detok/bin/python'),'-m','venv',str(V)],[str(V/'bin/python'),'-m','pip','--isolated','install','--disable-pip-version-check','--no-cache-dir','--index-url','https://pypi.org/simple']+packages]
for i,cmd in enumerate(commands):
 with (O/f'{i}.stdout.log').open('w') as out,(O/f'{i}.stderr.log').open('w') as err:p=subprocess.run(cmd,stdout=out,stderr=err,timeout=600)
 if p.returncode:raise RuntimeError('environment command '+str(i)+' failed:'+str(p.returncode))
p=subprocess.run([str(V/'bin/python'),'-m','pip','freeze','--all'],capture_output=True,text=True,timeout=30);assert p.returncode==0;(O/'pip-freeze.txt').write_text(p.stdout)
result={'status':'PASS','packages':packages,'environment_scope':'Task-owned local protocol fixture; not target image','prior_envs_modified':False,'native_executed':False,'pip_freeze_sha256':hashlib.sha256((O/'pip-freeze.txt').read_bytes()).hexdigest(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(O/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
