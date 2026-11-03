#!/usr/bin/env python3
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2];q=T/'agents/diagnosis/query_qli_compute_provenance_exact.py';out=T/'agents/diagnosis/qli-compute-provenance-exact.json'
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';c='wio-vllm-v41-repro169-stage-20261028';db='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db'
if out.exists():raise RuntimeError('output exists')
r=subprocess.run([ssh,shlex.join(['docker','exec',c,'python3','-c',q.read_text(),'--db',db])],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only query failed '+r.stderr[-1500:])
x=json.loads(r.stdout);x['script_sha256']=hashlib.sha256(q.read_bytes()).hexdigest();x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':'OK','out':str(out),'decoded':x['decoded_fields'],'global_id_task_count':x['global_id_task_count'],'schema_tables':len(x['potential_schema_tables'])}))
