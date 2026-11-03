#!/usr/bin/env python3
"""Fixed EP0 read-only host-link lookup using the stage190 criteria."""
from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_draft_ep0_mc2_host_links.py';P=T/'agents/diagnosis/draft-six-mc2-preceding-csv.json';OUT=T/'agents/diagnosis/draft-ep0-mc2-host-links.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028';R='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41';D='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
script=S.read_bytes();qsha=hashlib.sha256(script).hexdigest()
if qsha!='6ef37ca817fa18d94aa423b7c6c91a19bdd7472a6c7d10abcabf2a040d7c5330':raise RuntimeError('script hash changed')
if hashlib.sha256(P.read_bytes()).hexdigest()!='5de5efc398e82f21c28f1aabde8bd88141afb3343bf42d52289eb2126cdb7c42':raise RuntimeError('prior hash changed')
if OUT.exists() or OUT.is_symlink():raise FileExistsError('output exists')
cmd=shlex.join(['docker','exec',C,'python3','-c',script.decode(),'--task-root',R,'--trace-root',D]);r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only query failed with exit '+str(r.returncode))
x=json.loads(r.stdout)
if len(x['targets'])!=6 or x['input_sha256']['db']!='a6c626753b5e21cd066ede0cceffe104e8603c01a79918eb79a270dcb4ad0a09':raise RuntimeError('unexpected result')
x['script_sha256']=qsha;x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();x['no_device_work']=True
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':'QUERY_COMPLETED','output':str(OUT),'targets':len(x['targets']),'sha256':hashlib.sha256(body).hexdigest()}))
