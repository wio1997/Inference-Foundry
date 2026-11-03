#!/usr/bin/env python3
"""Run fixed inter-ArgMax CSV query on dedicated staging container."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2];S=T/'agents/diagnosis/query_draft_argmax_between_ep8.py';A=T/'agents/diagnosis/draft-verify-timeline-ep8.json';OUT=T/'agents/diagnosis/draft-argmax-between-ep8.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';C='wio-vllm-v41-repro169-stage-20261028'
TASK='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41'
TRACE='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
script=S.read_bytes();qsha=hashlib.sha256(script).hexdigest();assert qsha=='bd0a2e3e23a40bce5fe308b0b4f2d825aaf2383940f63b28613ea6519419e500'
anchors=A.read_bytes();assert hashlib.sha256(anchors).hexdigest()=='6270f9cb785b75cdfbe2442498631a5b0933941b552257c5220046ac55eaf5ce'
if OUT.exists() or OUT.is_symlink():raise FileExistsError('fixed output exists; refusing overwrite')
cmd=shlex.join(['docker','exec',C,'python3','-c',script.decode(),'--task-root',TASK,'--trace-root',TRACE,'--anchors-json',anchors.decode()])
r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=180)
if r.returncode:raise RuntimeError('read-only query failed with exit code '+str(r.returncode))
x=json.loads(r.stdout);assert len(x['matched_anchor_rows'])==10
assert len(x['intervals'])==8 and {(g['ordinal'],g['gap_index']) for g in x['intervals']}=={(n,i) for n in (6,7) for i in (1,2,3,4)}
for g in x['intervals']:
 assert len(g['events'])<=80 and g['all_rows_union_ns']+g['uncovered_by_csv_rows_ns']==g['gap_ns']
x['query_sha256']=qsha;x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();x['no_device_work']=True
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f:f.write(body)
print(json.dumps({'status':'READ_ONLY_QUERY_COMPLETED','output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),
 'intervals':len(x['intervals']),'total_rows':sum(len(g['events']) for g in x['intervals'])},ensure_ascii=False))
