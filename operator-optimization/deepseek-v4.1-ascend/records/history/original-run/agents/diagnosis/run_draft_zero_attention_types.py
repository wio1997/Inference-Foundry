#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];dp=T/'eager-draft-mainstream-attribution-v5/draft_call_metrics.json';cp=T/'agents/diagnosis/indexer-attention-paired-runs.json';script=T/'agents/diagnosis/query_draft_zero_attention_types.py';out=T/'agents/diagnosis/draft-zero-attention-types.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={dp:'6473445dda048c157cf7e451bf959b973c5138a05e0272172a5ea98cfef21923',cp:'d8d9a7e651a82616b5387557f4568dba96ed78b30da17cccf575da9b602de0d3'}
if any(sha(p)!=h for p,h in expected.items()):raise RuntimeError('source changed')
if out.exists():raise RuntimeError('refuse overwrite')
z=[z for z in json.loads(dp.read_text()) if z['attention_count_allstreams']==0]
windows=[{'rank':a['rank'],'start':a['start_ns'],'end':a['end_ns'],'model':str(a['model_id']),'stream':str(a['main_stream_id']),'mc2_task_id':str(a['mc2_task_id']),'expected_num':a['aic_mac_proxy_numerator'],'expected_den':a['aic_cycles_denominator'],'expected_skipped':a['skipped_no_positive_cycles']} for a in z]
if len(windows)!=126 or len({a['rank'] for a in windows})!=16:raise RuntimeError('scope mismatch')
hashes=json.loads(cp.read_text())['inputs']['csv_sha256_by_rank']
if set(hashes)!={a['rank'] for a in windows}:raise RuntimeError('rank hash mismatch')
payload={'windows':windows,'csv_sha256_by_rank':hashes,'inputs':{str(p.relative_to(T)):h for p,h in expected.items()}}
code=script.read_text().replace('FROZEN_INPUT',repr(json.dumps(payload,separators=(',',':'))));ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';container='wio-vllm-v41-repro169-stage-20261028';trace='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
r=subprocess.run([ssh,shlex.join(['docker','exec',container,'python3','-c',code,'--trace-root',trace])],capture_output=True,text=True,timeout=240)
if r.returncode:raise RuntimeError('remote read-only query failed: '+r.stderr[-3000:])
x=json.loads(r.stdout);x['query_sha256']=sha(script);x['wrapper_sha256']=sha(Path(__file__));out.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'sha256':sha(out)}))
