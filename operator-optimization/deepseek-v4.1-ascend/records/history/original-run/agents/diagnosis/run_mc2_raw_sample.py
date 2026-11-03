"""Run reviewed read-only SQLite query in the user's CPU stage container."""
from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];R='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41';SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028';src=T/'agents/diagnosis/query_mc2_raw_event_evidence.py'
DB='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db'
expected='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58';p=R+'/agents/diagnosis/query_mc2_raw_event_evidence.py'
code='from pathlib import Path;import hashlib;p=Path('+repr(p)+');p.parent.mkdir(parents=True,exist_ok=True);b='+repr(src.read_bytes())+';assert not p.exists() or p.read_bytes()==b;p.write_bytes(b);f=Path('+repr(DB)+');h=hashlib.sha256();\nwith f.open("rb") as q:\n for b in iter(lambda:q.read(4194304),b""):h.update(b)\nassert h.hexdigest()=='+repr(expected)
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',code])],capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3',p,'--db',DB])],capture_output=True,text=True,timeout=180);assert r.returncode==0,r.stderr
x=json.loads(r.stdout);x['query_sha256']=hashlib.sha256(src.read_bytes()).hexdigest();x['database_sha256']=expected;x['no_device_work']=True
out=T/'agents/diagnosis/mc2-raw-sample-ep8.json';assert not out.exists();out.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'QUERY_COMPLETED','task':x['target'],'candidate_tables':len(x['candidate_table_schemas']),'pmu_entries':len(x['exact_task_pmu']),'mstx_temporal_overlap_only':x['mstx_temporal_overlap_only']},ensure_ascii=False))
