#!/usr/bin/env python3
"""Run only fixed CPU read-only SQLite query in dedicated staging container."""
import hashlib,json,shlex,subprocess
from pathlib import Path
T=Path(__file__).resolve().parents[2]
S=T/'agents/diagnosis/query_draft_argmax_api_links_ep8.py'; A=T/'agents/diagnosis/draft-argmax-between-ep8.json'; OUT=T/'agents/diagnosis/draft-argmax-api-links-ep8.json'
SSH='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'; CONTAINER='wio-vllm-v41-repro169-stage-20261028'
DB='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_727_20261028020238580_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_8.db'
script=S.read_bytes(); sha=hashlib.sha256(script).hexdigest()
if sha!='fad9ade7b3b198ea4d09da12cfd1204ae54c4a8f5d122ffd26ebe103c165494e': raise RuntimeError('script hash changed')
anchors=A.read_bytes()
if hashlib.sha256(anchors).hexdigest()!='a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666': raise RuntimeError('anchor hash changed')
if OUT.exists() or OUT.is_symlink(): raise FileExistsError('fixed output exists')
cmd=shlex.join(['docker','exec',CONTAINER,'python3','-c',script.decode(),'--db',DB,'--anchors-json',anchors.decode()])
r=subprocess.run([SSH,cmd],capture_output=True,text=True,timeout=180)
if r.returncode: raise RuntimeError('read-only query failed with exit code '+str(r.returncode))
x=json.loads(r.stdout)
if len(x['rows'])!=90 or x['anchors_sha256']!='a20fd5761c9ca88b5161ea624c267c248342a21d230bb69a7f83ee589fdc0666' or x['db_sha256']!='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58': raise RuntimeError('unexpected result')
x['query_sha256']=sha; x['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); x['no_device_work']=True
body=(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()
with OUT.open('xb') as f: f.write(body)
print(json.dumps({'status':'READ_ONLY_QUERY_COMPLETED','output':str(OUT),'sha256':hashlib.sha256(body).hexdigest(),'rows':len(x['rows']),'unique_task_matches':sum(r['task_count_shown']==1 and not r['task_truncated'] for r in x['rows'])}))
