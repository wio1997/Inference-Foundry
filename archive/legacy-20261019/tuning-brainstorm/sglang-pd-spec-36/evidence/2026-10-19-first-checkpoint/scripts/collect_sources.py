#!/usr/bin/env python3
"""Read selected installed source without importing SGLang or touching device state."""
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys,urllib.request,concurrent.futures
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/L001-R002-sources';OUT.mkdir(parents=True,exist_ok=False)
catalog=json.loads((ROOT.parent/'repo-export/tuning-brainstorm/sglang-pd-spec-36/mechanisms.json').read_text())
paths=set()
for m in catalog:
 for s in m['sources']:
  if '/blob/' in s['url']:paths.add(s['url'].split('/blob/')[1].split('/',1)[1].split('#')[0])
paths.update(['python/sglang/srt/server_args.py','python/sglang/srt/managers/scheduler.py','python/sglang/srt/models/glm_moe_dsa.py','python/sglang/srt/models/deepseek_v2.py','python/sglang/srt/speculative/eagle_info_v2.py','python/sglang/srt/disaggregation/ascend/conn.py','python/sglang/srt/disaggregation/ascend/transfer.py','python/sglang/srt/model_executor/forward_batch_info.py'])
INNER=r'''
import pathlib,subprocess,json,hashlib,sys
root=pathlib.Path('/sgl-workspace/sglang')
paths=json.loads(sys.argv[1])
def run(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=30);return {'command':a,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
v={'source_root':str(root),'git_head':run(['git','-C',str(root),'rev-parse','HEAD']),'git_status':run(['git','-C',str(root),'status','--porcelain']),'files':{},'all_paths':[]}
for p in (root/'python/sglang/srt').rglob('*.py'):v['all_paths'].append(str(p.relative_to(root)))
for n in paths:
 p=root/n
 if p.is_file():
  b=p.read_bytes();v['files'][n]={'sha256':hashlib.sha256(b).hexdigest(),'content':b.decode()}
 else:v['files'][n]={'missing':True}
print(json.dumps(v))
'''
REMOTE=r'''
import subprocess,json,pathlib,sys
j=json.loads(sys.stdin.read())
p=subprocess.run(['docker','inspect','--format','{{json .}}','sglang-learning37-glm53-20261017'],capture_output=True,text=True,timeout=30)
assert p.returncode==0,p.stderr
x=json.loads(p.stdout)
meta={k:x[k] for k in ['Id','Image','Name','Created','State']}
meta['labels']={k:v for k,v in (x['Config'].get('Labels') or {}).items() if any(t in k.lower() for t in ['task','owner','project','org.opencontainers.image.revision'])}
meta['mounts']=[{k:m.get(k) for k in ['Type','Source','Destination','RW']} for m in x['Mounts']]
p=subprocess.run(['docker','exec','sglang-learning37-glm53-20261017','python3','-c',j['inner'],json.dumps(j['paths'])],capture_output=True,text=True,timeout=100)
assert p.returncode==0,p.stderr
v=json.loads(p.stdout);v['container']=meta
v['model_files']=[{'name':f.name,'size':f.stat().st_size} for f in pathlib.Path('/data/wio/models/GLM-5.3-w8a8c8').iterdir() if f.is_file() and (f.suffix=='.json' or f.suffix=='.jinja')]
print(json.dumps(v))
'''
# Use the same reviewed hostctl route, with input passed through stdin.
JUMP=r'''
import json,subprocess,sys
j=json.load(sys.stdin);pod='sglang-tuning-access-37-20261013'
p=subprocess.run(['kubectl','get','pod','-n','default',pod,'-o','json'],capture_output=True,text=True,timeout=30)
assert p.returncode==0,p.stderr
x=json.loads(p.stdout);assert x['status']['hostIP']=='172.16.12.37' and x['status']['phase']=='Running'
p=subprocess.run(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',j['remote']],input=json.dumps(j),capture_output=True,text=True,timeout=150)
sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
'''
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=subprocess.run(cmd,input=json.dumps({'remote':REMOTE,'inner':INNER,'paths':sorted(paths)}),capture_output=True,text=True,timeout=180)
(OUT/'installed.json').write_text(p.stdout);(OUT/'stderr.log').write_text(p.stderr)
if p.returncode==0:
 j=json.loads(p.stdout)
 for n,f in j['files'].items():
  if 'content' in f:
   q=ROOT/'sources/installed'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(f['content']);assert hashlib.sha256(q.read_bytes()).hexdigest()==f['sha256']
commit='57db5089c207ae4ce189cc092a89a2027c8aea29'
def fetch(n):
 url='https://raw.githubusercontent.com/sgl-project/sglang/'+commit+'/'+n
 try:
  with urllib.request.urlopen(url,timeout=45) as r:b=r.read()
  q=ROOT/'sources/reference'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
  return n,{'url':url,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
 except Exception as e:return n,{'url':url,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:refs=dict(pool.map(fetch,sorted(paths)))
(OUT/'reference-manifest.json').write_text(json.dumps(refs,indent=2)+'\n')
receipt={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'command':cmd,'exit_code':p.returncode,'read_only_remote':True,'services_started':False,'artifacts':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['installed.json','stderr.log','reference-manifest.json']}}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(OUT/'receipt.json'),'exit_code':p.returncode,'reference_count':len(refs)}));sys.exit(p.returncode)
