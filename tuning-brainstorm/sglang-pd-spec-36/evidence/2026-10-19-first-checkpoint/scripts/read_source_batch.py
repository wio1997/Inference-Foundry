#!/usr/bin/env python3
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
paths=json.loads((ROOT/'source-batch-extra.json').read_text())
assert all(p.startswith('python/sglang/srt/') and '..' not in p for p in paths)
OUT=ROOT/'runs/L001-R004-source-extra';OUT.mkdir(parents=True,exist_ok=False)
INNER=r'''
import pathlib,json,hashlib,sys
root=pathlib.Path('/sgl-workspace/sglang');v={}
for n in json.loads(sys.argv[1]):
 p=root/n;b=p.read_bytes();v[n]={'sha256':hashlib.sha256(b).hexdigest(),'content':b.decode()}
print(json.dumps(v))
'''
JUMP=r'''
import subprocess,json,sys
j=json.load(sys.stdin)
outer='import subprocess,sys; p=subprocess.run(["docker","exec","sglang-learning37-glm53-20261017","python3","-c",'+repr(j['inner'])+','+repr(json.dumps(j['paths']))+'],capture_output=True,text=True,timeout=90);sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)'
p=subprocess.run(['kubectl','exec','-i','-n','default','sglang-tuning-access-37-20261013','-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',outer],capture_output=True,text=True,timeout=120)
sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
'''
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
started=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'inner':INNER,'paths':paths}),capture_output=True,text=True,timeout=150)
(OUT/'source.json').write_text(p.stdout);(OUT/'stderr.log').write_text(p.stderr)
if p.returncode==0:
 for n,f in json.loads(p.stdout).items():
  q=ROOT/'sources/installed'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(f['content']);assert hashlib.sha256(q.read_bytes()).hexdigest()==f['sha256']
receipt={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'paths_sha256':hashlib.sha256((ROOT/'source-batch-extra.json').read_bytes()).hexdigest(),'command':cmd,'exit_code':p.returncode,'read_only':True,'artifacts':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['source.json','stderr.log']}}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(OUT/'receipt.json'),'exit_code':p.returncode}));sys.exit(p.returncode)
