#!/usr/bin/env python3
"""Run fixed EP8 four-type timeline query in dedicated CPU staging container."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2]; S=T/'agents/diagnosis/query_draft_verify_timeline_ep8.py'; OUT=T/'agents/diagnosis/draft-verify-timeline-ep8.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'; C='wio-vllm-v41-repro169-stage-20261028'
TASK='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
TRACE='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
script=S.read_bytes();qsha=hashlib.sha256(script).hexdigest();assert qsha=='78fad6da80246adccc2320a4fa97dca0dba021b5cb9e6878e5b587a67820b977'
if OUT.exists() or OUT.is_symlink():raise FileExistsError('fixed output exists; refusing overwrite')
cmd=shlex.join(['docker','exec',C,'python3','-c',script.decode(),'--task-root',TASK,'--trace-root',TRACE])
r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only query failed with exit code '+str(r.returncode))
x=json.loads(r.stdout);assert len(x['windows'])==2 and {w['ordinal'] for w in x['windows']}=={6,7}
for w in x['windows']:
 assert len(w['strict_draft_brackets'])==3 and len(w['events'])<=40
 assert all(len(e['relation_to_strict_draft_brackets'])==3 for e in w['events'])
x['query_sha256']=qsha;x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();x['no_device_work']=True
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':'READ_ONLY_QUERY_COMPLETED','output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),
 'counts':{str(w['ordinal']):w['types_count'] for w in x['windows']}},ensure_ascii=False))
