#!/usr/bin/env python3
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2];q=T/'agents/diagnosis/query_engram_boundary_mainstream_counters.py';out=T/'agents/diagnosis/engram-boundary-mainstream-counters.json';ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';c='wio-vllm-v41-repro169-stage-20261028';task='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41';trace='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
if out.exists():raise RuntimeError('output exists')
r=subprocess.run([ssh,shlex.join(['docker','exec',c,'python3','-c',q.read_text(),'--task-root',task,'--trace-root',trace])],capture_output=True,text=True,timeout=240)
if r.returncode:raise RuntimeError('read-only query failed '+r.stderr[-1500:])
x=json.loads(r.stdout);x['query_sha256']=hashlib.sha256(q.read_bytes()).hexdigest();x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'OK','output':str(out),'rows':len(x['rows'])}))
