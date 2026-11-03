#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shlex,subprocess
T=Path(__file__).resolve().parents[2];p=T/'agents/diagnosis/indexer-layer-window-occupancy.json';prior=T/'agents/diagnosis/indexer-attention-paired-runs.json';script=T/'agents/diagnosis/query_indexer_fixed_metrics.py';out=T/'agents/diagnosis/indexer-fixed-metrics-ep8.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if sha(p)!='ca1d1dabed7cede2a95bc1ac117d24ad68f8fd2368f32c3aa225583cb3e9afc8' or sha(prior)!='d8d9a7e651a82616b5387557f4568dba96ed78b30da17cccf575da9b602de0d3':raise RuntimeError('source SHA mismatch')
if out.exists():raise RuntimeError('refuse overwrite')
w=json.loads(p.read_text());r=json.loads(prior.read_text());rows=[z for z in w['rows'] if '_ep8_' in z['rank'] and z['layer'] in (2,24)]
if len(rows)!=6 or len({z['rank'] for z in rows})!=1:raise RuntimeError('unexpected target set')
rank=rows[0]['rank'];targets=[{'layer':z['layer'],'run':z['run'],'start':z['indexer']['start_ns'],'end':z['indexer']['end_ns'],'model':z['model_id'],'stream':z['main_stream_id']} for z in rows]
payload={'rank':rank,'targets':targets,'prior_sha256':sha(p),'csv_sha256':r['inputs']['csv_sha256_by_rank'][rank]};code=script.read_text().replace('FROZEN_INPUT',repr(json.dumps(payload,separators=(',',':'))))
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';container='wio-vllm-v41-repro169-stage-20261028';trace='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
r=subprocess.run([ssh,shlex.join(['docker','exec',container,'python3','-c',code,'--trace-root',trace])],capture_output=True,text=True,timeout=120)
if r.returncode:raise RuntimeError('remote read-only query failed: '+r.stderr[-3000:])
x=json.loads(r.stdout);x['query_sha256']=sha(script);x['wrapper_sha256']=sha(Path(__file__));out.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'sha256':sha(out)}))
