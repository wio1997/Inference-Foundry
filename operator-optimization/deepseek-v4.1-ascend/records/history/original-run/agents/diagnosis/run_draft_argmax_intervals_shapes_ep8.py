#!/usr/bin/env python3
"""Run fixed 80-task shape lookup in the dedicated CPU staging container."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_draft_argmax_intervals_shapes_ep8.py';A=T/'agents/diagnosis/draft-argmax-between-ep8.json';OUT=T/'agents/diagnosis/draft-argmax-intervals-shapes-ep8.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
TASK='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
TRACE='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
script=S.read_bytes();qsha=hashlib.sha256(script).hexdigest();assert qsha=='839b983ca0fd8fbd4088c5844a8e23f480a817c1851e7d5c888bc0037d4fa03d'
anchors=A.read_bytes();assert hashlib.sha256(anchors).hexdigest()=='a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666'
if OUT.exists() or OUT.is_symlink():raise FileExistsError('fixed output exists; refusing overwrite')
cmd=shlex.join(['docker','exec',C,'python3','-c',script.decode(),'--task-root',TASK,'--trace-root',TRACE,'--anchors-json',anchors.decode()])
r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only query failed with exit code '+str(r.returncode))
x=json.loads(r.stdout);assert len(x['rows'])==80 and all(r['matches'] in (0,1) for r in x['rows'])
assert x['anchors_sha256']=='a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666'
x['query_sha256']=qsha;x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();x['no_device_work']=True
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':'READ_ONLY_QUERY_COMPLETED','output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),
 'rows':len(x['rows']),'matched':sum(r['matches'] for r in x['rows'])},ensure_ascii=False))
