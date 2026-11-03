#!/usr/bin/env python3
"""Run fixed two-database schema inventory in staging container, once."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_mc2_qli_analysis_schema.py';IDX=T/'profile-index.json';OUT=T/'agents/diagnosis/mc2-qli-analysis-schema.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
script=S.read_bytes();sha=hashlib.sha256(script).hexdigest()
if sha!='7a580f3fa734366eb71f0cbf1846edb9b4bf804026db07e0b831fadc65347f30':raise RuntimeError('script hash mismatch')
idx=IDX.read_bytes();idxsha=hashlib.sha256(idx).hexdigest()
if idxsha!='2573efcaba3bb4c8babee95f1198c4c55b22d9aeafa72463a486c6d25df9ebac':raise RuntimeError('profile index changed')
x=json.loads(idx); sizes={}
for e in x['files']:
 path=e.get('id','')
 if path.endswith('/ASCEND_PROFILER_OUTPUT/analysis.db') and ('_ep0_' in path or '_ep8_' in path):sizes['ep0' if '_ep0_' in path else 'ep8']=e['size_bytes']
if sizes!={'ep0':323584,'ep8':290816}:raise RuntimeError('unexpected index entries')
if OUT.exists() or OUT.is_symlink():raise FileExistsError('fixed output exists')
cmd=shlex.join(['docker','exec',C,'python3','-c',script.decode()]);r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('query exit '+str(r.returncode))
y=json.loads(r.stdout)
if [s['rank'] for s in y['samples']]!=['ep0','ep8']:raise RuntimeError('unexpected result')
y['profile_index_sha256']=idxsha;y['query_sha256']=sha;y['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();y['no_device_work']=True
body=(json.dumps(y,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':'SCHEMA_INVENTORY_COMPLETE','output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),'samples':len(y['samples']),'statuses':[s['status'] for s in y['samples']]}))
