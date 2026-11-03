#!/usr/bin/env python3
"""Run fixed one-attribute DB lookup in the dedicated CPU staging container."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2]
S=T/'agents/fusion/query_f16_batch_split_exact.py'
OUT=T/'agents/fusion/f16-ep8-batch-split-exact.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
CONTAINER='wio-vllm-v41-repro169-stage-20261028'
DB='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db'
script=S.read_bytes();digest=hashlib.sha256(script).hexdigest()
assert digest=='abe6bd64583d4f8bb7f85d82de39516d26db2ea6e2bdb172dcb29376cb1fb4b8'
if OUT.exists() or OUT.is_symlink():raise FileExistsError('fixed output already exists; refusing overwrite')
cmd=shlex.join(['docker','exec',CONTAINER,'python3','-c',script.decode(),'--db',DB])
r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only query failed with exit code '+str(r.returncode))
x=json.loads(r.stdout)
assert x['db_sha256']=='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'
assert x['status'] in {'VALUE_AVAILABLE','TASK_ABSENT_OR_NONUNIQUE','COMPUTE_ABSENT_OR_NONUNIQUE','ATTRIBUTE_STRING_ABSENT','ATTRIBUTE_HASH_MISMATCH','KEY_ABSENT_OR_NONUNIQUE','VALUE_NOT_PLAIN_INTEGER'}
if x['status']=='VALUE_AVAILABLE':assert x['task_matches_up_to_3']==1 and x['compute_matches_up_to_3']==1 and x['attr_hash_matches_stage236'] and x['key_matches']==1 and isinstance(x['value'],int)
else:assert x['value'] is None
x['query_sha256']=digest;x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();x['no_device_work']=True
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':'READ_ONLY_QUERY_COMPLETED','output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),'lookup_status':x['status'],'value':x['value']},ensure_ascii=False))
