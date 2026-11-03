#!/usr/bin/env python3
"""Run bounded metadata/source read in the two existing 169 containers."""
from hashlib import sha256
from pathlib import Path
import json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_layer0_hash_config.py'
OUT=T/'agents/diagnosis/layer0-hash-config-inventory.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
containers={'baseline':'wio-vllm-v41-repro169-20261028','stage':'wio-vllm-v41-repro169-stage-20261028'}
if OUT.exists():raise RuntimeError('STOP output exists')
results={}
for name,container in containers.items():
 cmd=shlex.join(['docker','exec','-e','PYTHONDONTWRITEBYTECODE=1',container,'python3','-c',S.read_text()])
 r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=120)
 if r.returncode!=0:raise RuntimeError('STOP '+name+' query failed: '+r.stderr[-2000:])
 if len(r.stdout)>1000000:raise RuntimeError('STOP excessive response')
 results[name]=json.loads(r.stdout)
a,b=results['baseline'],results['stage']
source_map=lambda x:{s['path']:s['sha256'] for s in x['config_sources']}
comparison={'same_config_sources':source_map(a)==source_map(b),'same_package_roots':a['package_roots']==b['package_roots'],
 'same_model_metadata':a['model_metadata']==b['model_metadata']}
out={'query_sha256':sha256(S.read_bytes()).hexdigest(),'containers':results,'comparison':comparison,'no_device_work':True}
OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','output':str(OUT),'sha256':sha256(OUT.read_bytes()).hexdigest(),'comparison':comparison}))
