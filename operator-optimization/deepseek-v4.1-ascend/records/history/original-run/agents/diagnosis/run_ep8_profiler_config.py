#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];s=T/'agents/diagnosis/query_ep8_profiler_config.py';out=T/'agents/diagnosis/indexer-ep8-profile-collection.json'
if out.exists():raise RuntimeError('refuse overwrite')
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';c='wio-vllm-v41-repro169-stage-20261028';trace='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
r=subprocess.run([ssh,shlex.join(['docker','exec',c,'python3','-c',s.read_text(),'--trace-root',trace])],capture_output=True,text=True,timeout=120)
if r.returncode:raise RuntimeError('read-only query failed '+r.stderr[-3000:])
x=json.loads(r.stdout);x['query_sha256']=hashlib.sha256(s.read_bytes()).hexdigest();out.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
