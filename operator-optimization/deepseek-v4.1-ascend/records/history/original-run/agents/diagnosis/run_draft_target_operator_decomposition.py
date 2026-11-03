#!/usr/bin/env python3
from pathlib import Path
import csv,hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];base=T/'eager-draft-mainstream-attribution-v2';dp=base/'draft_call_metrics.json';mp=base/'module_metrics.csv';wp=T/'captured-layer-attribution/module-call-metrics.json';cp=T/'agents/diagnosis/indexer-attention-paired-runs.json';script=T/'agents/diagnosis/query_draft_target_operator_decomposition.py';out=T/'agents/diagnosis/draft-target-operator-decomposition.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={dp:'0e6f0cb4d810fdd9f7ec99f7b9a7dbe2123a7db7e5a927115e721b8fd8982c48',mp:'9e96d35b57fe4652da6e955e3a0f1318f8638b5094f42b12c9c2299c3b43fe23',wp:'d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a',cp:'d8d9a7e651a82616b5387557f4568dba96ed78b30da17cccf575da9b602de0d3'}
if any(sha(p)!=h for p,h in expected.items()):raise RuntimeError('source changed')
if out.exists():raise RuntimeError('refuse overwrite')
draft=json.loads(dp.read_text());rankset={z['rank'] for z in draft};window=json.loads(wp.read_text());metric=list(csv.DictReader(mp.open())); paired=json.loads(cp.read_text())
if len(draft)!=162 or len(rankset)!=15:raise RuntimeError('draft scope')
windows=[]
for z in draft:windows.append({'rank':z['rank'],'kind':'draft','start':z['start_ns'],'end':z['end_ns'],'model':str(z['model_id']),'stream':str(z['main_stream_id']),'expected_num':z['aic_mac_proxy_numerator'],'expected_den':z['aic_cycles_denominator'],'expected_skipped':z['skipped_no_positive_cycles']})
for z in window:
 if z['rank'] in rankset and z['layer']==0 and z['run_in_model'] in (1,2,3):windows.append({'rank':z['rank'],'kind':'target','start':z['window_start_ns'],'end':z['window_end_ns'],'model':str(z['model_id']),'stream':str(z['main_stream_id'])})
expected_target={z['rank']:{'num':z['aic_mac_proxy_numerator'],'den':z['aic_cycles_denominator'],'skipped':int(z['skipped_no_positive_cycles'])} for z in metric if z['rank'] in rankset and z['module']=='target.layers.0.forward'}
if len(windows)!=207 or len(expected_target)!=15:raise RuntimeError('target scope')
payload={'windows':windows,'target_expected_by_rank':expected_target,'csv_sha256_by_rank':paired['inputs']['csv_sha256_by_rank'],'inputs':{str(p.relative_to(T)):h for p,h in expected.items()}}
code=script.read_text().replace('FROZEN_INPUT',repr(json.dumps(payload,separators=(',',':'))));ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';container='wio-vllm-v41-repro169-stage-20261028';trace='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
r=subprocess.run([ssh,shlex.join(['docker','exec',container,'python3','-c',code,'--trace-root',trace])],capture_output=True,text=True,timeout=240)
if r.returncode:raise RuntimeError('remote read-only query failed: '+r.stderr[-3000:])
x=json.loads(r.stdout);x['query_sha256']=sha(script);x['wrapper_sha256']=sha(Path(__file__));out.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'sha256':sha(out)}))
