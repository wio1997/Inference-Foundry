#!/usr/bin/env python3
"""Run fixed two-event F16 DB lookup in the dedicated CPU staging container."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2]
S=T/'agents/fusion/query_f16_ep8_task_metadata.py'
OUT=T/'agents/fusion/f16-ep8-task-metadata.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
C='wio-vllm-v41-repro169-stage-20261028'
DB='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db'
script=S.read_bytes();digest=hashlib.sha256(script).hexdigest()
assert digest=='ecdb84bac1753084cdbbe39ab06f4fb09ab966f51307e0b0d483a4a5cbe98c23'
if OUT.exists() or OUT.is_symlink():raise FileExistsError('fixed output exists; refusing overwrite')
cmd=shlex.join(['docker','exec',C,'python3','-c',script.decode(),'--db',DB])
r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only query failed with exit code '+str(r.returncode))
x=json.loads(r.stdout);assert len(x['rows'])==2 and {z['label'] for z in x['rows']}=={'first','second'}
assert x['db_sha256']=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
x['query_sha256']=digest;x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();x['no_device_work']=True
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':'READ_ONLY_QUERY_COMPLETED','output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),
                  'counts':{z['label']:z['exact_task_count_up_to_3'] for z in x['rows']}},ensure_ascii=False))
