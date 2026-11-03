#!/usr/bin/env python3
from pathlib import Path
from hashlib import sha256
import json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_v41_config_sources.py';OUT=T/'agents/diagnosis/layer0-v41-config-sources.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
containers={'baseline':'wio-vllm-v41-repro169-20261028','stage':'wio-vllm-v41-repro169-stage-20261028'}
if OUT.exists():raise RuntimeError('STOP output exists')
res={}
for name,container in containers.items():
 cmd=shlex.join(['docker','exec','-e','PYTHONDONTWRITEBYTECODE=1',container,'python3','-c',S.read_text()]);r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=120)
 if r.returncode:raise RuntimeError('STOP '+name+' '+r.stderr[-1200:])
 if len(r.stdout)>4*1024*1024:raise RuntimeError('STOP oversized response')
 res[name]=json.loads(r.stdout)
for name,a in res['baseline']['sources'].items():
 b=res['stage']['sources'].get(name)
 if not b or a['status']!=b['status'] or a.get('sha256')!=b.get('sha256'):raise RuntimeError('STOP container source mismatch '+name)
result={'query_sha256':sha256(S.read_bytes()).hexdigest(),'source_comparison':'IDENTICAL','sources':res['baseline']['sources'],
 'containers':containers,'no_device_work':True,'scope':'current matching static files, not runtime object'}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(OUT),'sha256':sha256(OUT.read_bytes()).hexdigest()}))
