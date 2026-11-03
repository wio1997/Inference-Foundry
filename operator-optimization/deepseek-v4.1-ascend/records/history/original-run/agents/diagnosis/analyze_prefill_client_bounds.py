#!/usr/bin/env python3
"""Decode existing client timing arrays, not text; no network or device use."""
import hashlib,io,json,math,sqlite3,statistics
from pathlib import Path
import numpy as np

T=Path(__file__).resolve().parents[2]
B=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/results')
baseline=T/'evidence/baseline-result.json';x=json.loads(baseline.read_text())
P=Path('/Users/wio/work/173-sglang-matched-172-benchmark/package/tools/aisbench-7bcb6440da11')
src={
 'plugins/prefix_cache/ais_bench_prefix_cache/models/vllm_prefix_cache_api.py':'cb0424af9e1ad9dd694aff5e7a0249a31cff40d77ff37b5d8caadc23d04a266e',
 'ais_bench/benchmark/summarizers/default_perf.py':'e33857dd2a55030bd594d554d30598030fc4358489dc539ebe50dd54b3a2fd43',
 'ais_bench/benchmark/models/output.py':'ecf7600edf599cae33f8024fac0d2d1b285050110b4a45d7f6c6ec0d03c4836d',
 'ais_bench/benchmark/calculators/default_perf_metric_calculator.py':'d7411d3ba02ab530ecd733b48c49f067ab6c92e1179d2054316487012c51cdd6'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(P/name)==expected for name,expected in src.items())
def union(spans):
 result=[]
 for a,b in sorted(spans):
  if not result or a>result[-1][1]:result.append([a,b])
  else:result[-1][1]=max(result[-1][1],b)
 return sum(b-a for a,b in result)
rows=[]
for r in x['rounds']:
 phase=r['phase'];matches=list((B/phase).rglob('*_details.jsonl'));assert len(matches)==1
 p=matches[0];objs=[json.loads(line) for line in p.open()];assert len(objs)==8 and all(o['success'] for o in objs)
 dbs=list(p.parent.glob('db_data/*.db'));assert len(dbs)==1;db=dbs[0]
 con=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);con.execute('PRAGMA query_only=ON')
 times=[];ids=set()
 for o in objs:
  i=o['time_points']['__db_ref__'];assert i not in ids;ids.add(i)
  rec=con.execute('SELECT arr_blob FROM numpy_store WHERE id=?',(i,)).fetchone();assert rec is not None
  arr=np.load(io.BytesIO(rec[0]),allow_pickle=False);assert arr.ndim==1 and len(arr)>=2 and np.isfinite(arr).all() and (np.diff(arr)>=0).all()
  times.append((float(arr[0]),float(arr[1]),float(arr[-1])))
 con.close();starts=[z[0] for z in times];first=[z[1] for z in times];ends=[z[2] for z in times];lat=[1000*(z[1]-z[0]) for z in times]
 duration=max(ends)-min(starts);assert abs(duration-r['duration_seconds'])<1e-6
 assert abs(statistics.mean(lat)-r['mean_ttft_ms'])<0.051
 q=r['server_counter_deltas']['prefix_cache_queries_total'];h=r['server_counter_deltas']['prefix_cache_hits_total']
 assert q>=h>=0
 rows.append({'phase':phase,'requests':len(times),'reported_input_tokens_total':sum(o['input_tokens'] for o in objs),
     'prefix_queries_total':q,'prefix_hits_total':h,'prefix_query_minus_hit':q-h,
     'client_full_batch_span_s':duration,'start_spread_ms':1000*(max(starts)-min(starts)),
     'first_data_spread_ms':1000*(max(first)-min(first)),
     'earliest_start_to_latest_first_data_ms':1000*(max(first)-min(starts)),
     'union_send_to_first_data_intervals_ms':1000*union([(s,f) for s,f,_ in times]),
     'first_data_latency_ms':{'min':min(lat),'median':statistics.median(lat),'mean':statistics.mean(lat),'max':max(lat)},
     'details_sha256':sha(p),'timing_db_sha256':sha(db)})
out={'scope':'Two unprofiled formal baseline client rounds; successful requests only. Timing points are sent and received SSE data events, not server-prefill timestamps.',
     'baseline_sha256':sha(baseline),'source_expected_sha256':src,'script_sha256':sha(Path(__file__)), 'rounds':rows,
     'limitations':['First nonempty noncomment non-DONE data event is not necessarily first event with generated content.',
       'Client send-to-first-data includes client/server queue, inference, sampling and response; it is not prefill-only timing.',
       'Prefix query minus hit is a counter difference, not proven actual recomputed tokens or per-request work.',
       'Different client and profile recordings have not been joined by request identity or clock anchors.',
       'No raw prompt, response content, request identifier, or absolute timing is written.']}
path=T/'agents/diagnosis/prefill-client-evidence.json';assert not path.exists();path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'COMPLETED','output':str(path),'rounds':rows},ensure_ascii=False))
