#!/usr/bin/env python3
import ast,datetime,hashlib,json,pathlib,shlex,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'runs/L003-R004-transfer-sources';OUT.mkdir(parents=True,exist_ok=False)
helper=ROOT/'scripts/read_source_batch.py';const={}
for n in ast.parse(helper.read_text()).body:
 if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['INNER','JUMP']:const[n.targets[0].id]=ast.literal_eval(n.value)
paths=json.loads((ROOT/'specs/L003-source-batch-v2.json').read_text());assert all(p.startswith('python/sglang/srt/') and '..' not in p for p in paths)
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',const['JUMP']])]
started=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'inner':const['INNER'],'paths':paths}),capture_output=True,text=True,timeout=150)
(OUT/'source.json').write_text(p.stdout);(OUT/'stderr.log').write_text(p.stderr)
if p.returncode==0:
 for n,f in json.loads(p.stdout).items():
  q=ROOT/'sources/installed'/n;q.parent.mkdir(parents=True,exist_ok=True)
  if q.exists():assert hashlib.sha256(q.read_bytes()).hexdigest()==f['sha256'],'Concurrent source drift: '+n
  else:q.write_text(f['content'])
  assert hashlib.sha256(q.read_bytes()).hexdigest()==f['sha256']
v={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'exit_code':p.returncode,'read_only':True,'inputs':{str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [pathlib.Path(__file__),helper,ROOT/'specs/L003-source-batch-v2.json']},'artifacts':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['source.json','stderr.log']}}
(OUT/'receipt.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode}));sys.exit(p.returncode)
