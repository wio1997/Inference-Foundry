#!/usr/bin/env python3
"""Fixed read-only five-candidate EP8 query, dedicated container only."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_draft_api_candidate_fields_ep8.py';A=T/'agents/diagnosis/draft-argmax-api-links-ep8.json';OUT=T/'agents/diagnosis/draft-api-candidate-fields-ep8.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028';DB='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db'
script=S.read_bytes();sha=hashlib.sha256(script).hexdigest()
if sha!='585ab7c0d2b1effead7dd337acf452a3b66d206a87943b75d586ebd56980f3ce':raise RuntimeError('script hash mismatch')
anchors=A.read_bytes()
if hashlib.sha256(anchors).hexdigest()!='8eab586ca22e79ddce8b88a5b304ba54d0f18a4aca840f0979e584b9e6e01007':raise RuntimeError('prior hash mismatch')
if OUT.exists() or OUT.is_symlink():raise FileExistsError('fixed output exists')
cmd=shlex.join(['docker','exec',C,'python3','-c',script.decode(),'--db',DB,'--prior-json',anchors.decode()]);r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('query failed: exit '+str(r.returncode))
x=json.loads(r.stdout)
if len(x['samples'])!=5 or x['db_sha256']!='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58' or x['prior_sha256']!='8eab586ca22e79ddce8b88a5b304ba54d0f18a4aca840f0979e584b9e6e01007':raise RuntimeError('unexpected result')
x['query_sha256']=sha;x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();x['no_device_work']=True
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':x['status'],'output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),'samples':len(x['samples']),'truncated':x['all_rows_truncated']}))
