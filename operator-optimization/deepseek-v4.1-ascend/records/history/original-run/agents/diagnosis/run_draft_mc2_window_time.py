#!/usr/bin/env python3
"""Verify frozen inputs, submit read-only query in dedicated container, aggregate locally."""
from pathlib import Path
import hashlib,json,shlex,subprocess
from collections import defaultdict
from decimal import Decimal
T=Path(__file__).resolve().parents[2]; dp=T/'eager-draft-mainstream-attribution-v5/draft_call_metrics.json'; cp=T/'agents/diagnosis/indexer-attention-paired-runs.json'; query=T/'agents/diagnosis/query_draft_mc2_window_time.py'; out=T/'agents/diagnosis/draft-mc2-window-time.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={dp:'6473445dda048c157cf7e451bf959b973c5138a05e0272172a5ea98cfef21923',cp:'d8d9a7e651a82616b5387557f4568dba96ed78b30da17cccf575da9b602de0d3'}
if any(sha(p)!=h for p,h in expected.items()):raise RuntimeError('source changed')
if out.exists():raise RuntimeError('refuse overwrite')
rows=json.loads(dp.read_text()); windows=[]
for a in rows:
 c=a['attention_count_allstreams']
 if c not in (0,1):raise RuntimeError('bad group')
 windows.append({'rank':a['rank'],'start':a['start_ns'],'end':a['end_ns'],'model':str(a['model_id']),'stream':str(a['main_stream_id']),'mc2_task_id':str(a['mc2_task_id']),'group':'strict' if c==1 else 'unclassified'})
if len(windows)!=288 or sum(w['group']=='strict' for w in windows)!=162 or len({w['rank'] for w in windows})!=16:raise RuntimeError('scope mismatch')
hashes=json.loads(cp.read_text())['inputs']['csv_sha256_by_rank']
if set(hashes)!={w['rank'] for w in windows}:raise RuntimeError('rank hash mismatch')
payload={'windows':windows,'csv_sha256_by_rank':hashes,'inputs':{str(p.relative_to(T)):h for p,h in expected.items()}}
code=query.read_text().replace('FROZEN_INPUT',repr(json.dumps(payload,separators=(',',':'))));ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh';container='wio-vllm-v41-repro169-stage-20261028';trace='/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces'
r=subprocess.run([ssh,shlex.join(['docker','exec',container,'python3','-c',code,'--trace-root',trace])],capture_output=True,text=True,timeout=240)
if r.returncode:raise RuntimeError('read-only query failed: '+r.stderr[-3000:])
x=json.loads(r.stdout);groups=defaultdict(list)
for v in x['records']:
 groups[(v['group'],'all')].append(v);groups[(v['group'],v['rank'])].append(v)
summary=[]
for (g,rank),vals in sorted(groups.items()):
 md=sum((Decimal(v['mc2_duration_ns']) for v in vals),Decimal(0));wd=sum((Decimal(v['window_duration_ns']) for v in vals),Decimal(0))
 summary.append({'group':g,'rank':rank,'windows':len(vals),'mc2_sum_ns':str(md),'window_sum_ns':str(wd),'time_share':str(md/wd)})
x['summary']=summary;x['query_sha256']=sha(query);x['wrapper_sha256']=sha(Path(__file__));out.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'status':'QUERY_COMPLETED','output':str(out),'sha256':sha(out),'groups':[s for s in summary if s['rank']=='all']}))
