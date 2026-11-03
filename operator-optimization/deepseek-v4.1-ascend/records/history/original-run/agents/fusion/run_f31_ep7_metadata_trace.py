#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];S=T/'agents/fusion/query_f31_ep7_metadata_trace.py';SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
R='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41';P='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp0_pp0_tp7_dcp0_ep7_rank7_4869_20261028020238581_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv'
r=subprocess.run([SSH,shlex.join(['docker','exec',C,'python3','-c',S.read_text(),'--task-root',R,'--csv',P])],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only query failed, exit '+str(r.returncode))
x=json.loads(r.stdout);x['script_sha256']=hashlib.sha256(S.read_bytes()).hexdigest();x['no_device_work']=True
out=T/'agents/fusion/f31-ep7-metadata-trace.json';body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
if out.is_symlink():raise RuntimeError('output is symlink')
if out.exists():
 if out.read_bytes()!=body:raise RuntimeError('existing result differs; refusing overwrite')
else:
 with out.open('xb') as f:f.write(body)
print(json.dumps({'status':'READ_ONLY_QUERY_COMPLETED','counts_whole_csv':x['counts_whole_csv'],
 'windows':[{'label':w['label'],'same_model_counts':w['same_model_counts'],'all_counts':w['all_in_window_counts']} for w in x['windows']],
 'sha256':hashlib.sha256(body).hexdigest()},indent=2))
