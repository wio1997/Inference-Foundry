#!/usr/bin/env python3
"""One fixed, read-only EP8 profiler query in the dedicated staging container."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_draft_api_outer_ep8.py';A=T/'agents/diagnosis/draft-api-candidate-fields-ep8.json';OUT=T/'agents/diagnosis/draft-api-outer-ep8.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028';DB='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db'
script=S.read_bytes();sha=hashlib.sha256(script).hexdigest()
if sha!='7f8bb9b948684166d3917c86f64a4e0db9997f2169974f2f5607f4cea2e425c8':raise RuntimeError('script hash mismatch')
anchors=A.read_bytes()
if hashlib.sha256(anchors).hexdigest()!='e257d11a0bf04ca9e68eab6a7bbdadccd2c0e4b8ba6c4cae4d668b834e15125a':raise RuntimeError('prior hash mismatch')
if OUT.exists() or OUT.is_symlink():raise FileExistsError('fixed output exists')
cmd=shlex.join(['docker','exec',C,'python3','-c',script.decode(),'--db',DB,'--prior-json',anchors.decode()]);r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('query failed: exit '+str(r.returncode))
x=json.loads(r.stdout)
if x['db_sha256']!='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58' or x['prior_sha256']!='e257d11a0bf04ca9e68eab6a7bbdadccd2c0e4b8ba6c4cae4d668b834e15125a' or x['candidate_tasks']!=[52765,52769,52770,52772,52776]:raise RuntimeError('unexpected output')
x['query_sha256']=sha;x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();x['no_device_work']=True
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':x['status'],'output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),'shown_count':x['shown_count'],'truncated':x['truncated']}))
