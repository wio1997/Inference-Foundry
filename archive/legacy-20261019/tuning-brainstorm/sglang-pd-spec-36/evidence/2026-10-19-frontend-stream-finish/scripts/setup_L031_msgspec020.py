#!/usr/bin/env python3
"""One isolated dependency contrast; old venv/failed run are preserved."""
import datetime,hashlib,json,pathlib,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L031-R006-msgspec020';O.mkdir(parents=True,exist_ok=False);target=R/'.deps/L031-msgspec020';assert not target.exists();target.parent.mkdir(exist_ok=True)
cmd=[str(R/'.venv-front/bin/python'),'-m','pip','--isolated','install','--disable-pip-version-check','--no-cache-dir','--index-url','https://pypi.org/simple','--no-deps','--only-binary=:all:','--target',str(target),'msgspec==0.20.0']
with (O/'pip.stdout.log').open('w') as out,(O/'pip.stderr.log').open('w') as err:p=subprocess.run(cmd,stdout=out,stderr=err,timeout=180)
assert p.returncode==0
public={}
for name,resource in [('pull','repos/msgspec/msgspec/pulls/836'),('files','repos/msgspec/msgspec/pulls/836/files')]:
 p=subprocess.run(['gh','api',resource],capture_output=True,text=True,timeout=40);assert p.returncode==0;x=json.loads(p.stdout)
 if name=='pull':public={'url':x['html_url'],'title':x['title'],'merged':x['merged'],'merge_commit_sha':x['merge_commit_sha'],'body':x['body']}
 else:public['files']=[{k:y.get(k) for k in ['filename','status','sha','patch']} for y in x]
(O/'upstream-pr836.json').write_text(json.dumps(public,indent=2)+'\n')
manifest=[{'path':str(p.relative_to(target)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(target.rglob('*')) if p.is_file() and '__pycache__' not in p.parts];(O/'installed-files.json').write_text(json.dumps(manifest,indent=2)+'\n');v={'status':'PASS','overlay':'.deps/L031-msgspec020','package':'msgspec==0.20.0','previous_venv_modified':False,'purpose':'Version-only compatibility contrast for memoryview extension, not native deployment or speedup','upstream_support_url':public['url'],'installed_file_count':len(manifest),'native_executed':False,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(O/'result.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
