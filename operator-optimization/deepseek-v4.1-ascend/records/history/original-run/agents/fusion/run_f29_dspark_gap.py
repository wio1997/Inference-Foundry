#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/fusion/query_f29_dspark_gap.py';SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
R='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41';D='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',S.read_text(),'--task-root',R,'--trace-root',D])],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only query failed, exit '+str(r.returncode)+' '+r.stderr[-2000:])
x=json.loads(r.stdout);x['script_sha256']=hashlib.sha256(S.read_bytes()).hexdigest();x['no_device_work']=True
out=T/'agents/fusion/f29-dspark-gap-evidence.json';body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
if out.is_symlink():raise RuntimeError('output is symlink')
if out.exists():
 if out.read_bytes()!=body:raise RuntimeError('existing result differs; refusing overwrite')
else:
 with out.open('xb') as f:f.write(body)
print(json.dumps({'status':'READ_ONLY_QUERY_COMPLETED','matches':len(x['matches']),'neighbor_events':len(x['neighbor_events']),'sha256':hashlib.sha256(body).hexdigest()},indent=2))
