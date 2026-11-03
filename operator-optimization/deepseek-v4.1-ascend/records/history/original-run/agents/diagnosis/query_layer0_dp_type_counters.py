#!/usr/bin/env python3
"""Read frozen rank CSVs, summarize attributed layer0 task types by DP.

Run only in own staging container. No prompt text, environment or device APIs.
The caller supplies paths to already saved files. Prints JSON to stdout.
"""
import argparse,csv,hashlib,json,re
from collections import defaultdict
from decimal import Decimal,InvalidOperation
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);args=ap.parse_args()
T=Path(args.task_root);R=Path(args.trace_root)
call_path=T/'captured-layer-attribution/module-call-metrics.json'
manifest_path=T/'evidence/TRACE-EXPORT-SHA256.json'
EXPECTED_CALLS='d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a'
EXPECTED_MANIFEST='45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(call_path)!=EXPECTED_CALLS or sha(manifest_path)!=EXPECTED_MANIFEST:raise RuntimeError('STOP frozen input identity')
expected={r['rank']:r['kernel_csv_sha256'] for r in json.loads(manifest_path.read_text())['exports']}
calls=[r for r in json.loads(call_path.read_text()) if r['layer']==0 and r['run_in_model'] in (1,2,3)]
if len(calls)!=48 or len({r['rank'] for r in calls})!=16:raise RuntimeError('STOP coverage')
byrank=defaultdict(list)
for c in calls:byrank[c['rank']].append(c)
num=lambda s: Decimal(s)
def dec(s):
 try:
  x=Decimal(s);return x if x.is_finite() else None
 except (InvalidOperation,TypeError):return None

def accumulator():
 return {'rows':0,'task_duration_ns':0,'aic_den':Decimal(0),'aic_num':Decimal(0),'aiv_den':Decimal(0),'aiv_num':Decimal(0),'aic_nonpositive_or_missing':0,'aiv_nonpositive_or_missing':0,'shapes':defaultdict(int),'main_rows':0,'other_rows':0}
agg=defaultdict(accumulator); coverage=[];mc2_rows=[];sha_by_rank={}
for rank,windows in sorted(byrank.items()):
 if len(windows)!=3 or sorted(w['run_in_model'] for w in windows)!=[1,2,3]:raise RuntimeError('STOP run coverage')
 if not re.match(r'^dp[01]_',rank):raise RuntimeError('STOP rank label')
 dp=rank.split('_')[0];p=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 if sha(p)!=expected.get(rank):raise RuntimeError('STOP CSV hash mismatch')
 sha_by_rank[rank]=sha(p);counts=defaultdict(int)
 with p.open(newline='') as f:
  for row in csv.DictReader(f):
   if row['Model ID']!='49' or 'metadata' in row['Type'].lower():continue
   start=int(num(row['Start Time(us)'].strip())*1000);end=start+int(num(row['Duration(us)'].strip())*1000)
   selected=[c for c in windows if c['window_start_ns']<=start and end<=c['window_end_ns']]
   if len(selected)>1:raise RuntimeError('STOP overlapping windows')
   if not selected:continue
   c=selected[0]; counts[c['run_in_model']]+=1
   typ=row['Type'] if re.fullmatch(r'[A-Za-z0-9_.-]{1,120}',row['Type']) else 'UNSAFE_OR_UNKNOWN_NAME'
   shape=row['Input Shapes']
   if len(shape)>300 or not re.fullmatch(r'[0-9,;"\s\-]*',shape):shape='UNAVAILABLE_OR_UNSAFE'
   a=agg[(dp,typ)];a['rows']+=1;a['task_duration_ns']+=end-start;a['shapes'][shape]+=1
   if typ=='DispatchFFNCombine':
    mc2_rows.append({'rank':rank,'run':c['run_in_model'],'stream':row['Stream ID'],'task_id':row['Task ID'],'duration_ns':end-start,'input_shape':shape,'aic_cycles':row['aic_total_cycles'],'aic_mac_ratio':row['aic_mac_ratio'],'aiv_cycles':row['aiv_total_cycles'],'aiv_vec_ratio':row['aiv_vec_ratio']})
   if row['Stream ID']==c['main_stream_id']:a['main_rows']+=1
   else:a['other_rows']+=1
   for engine,unit in (('aic','mac'),('aiv','vec')):
    cyc=dec(row[engine+'_total_cycles']);ratio=dec(row[engine+'_'+unit+'_ratio'])
    if cyc is None or cyc<=0:a[engine+'_nonpositive_or_missing']+=1;continue
    if ratio is None or not 0<=ratio<=1:raise RuntimeError('STOP positive cycle missing/invalid ratio')
    a[engine+'_den']+=cyc;a[engine+'_num']+=cyc*ratio
 for w in windows:
  if counts[w['run_in_model']]!=w['attributed_rows']:raise RuntimeError('STOP attributed row mismatch')
  coverage.append({'rank':rank,'run':w['run_in_model'],'rows':counts[w['run_in_model']]})
if len(coverage)!=48 or len(mc2_rows)!=48 or len({(r['rank'],r['run']) for r in mc2_rows})!=48:raise RuntimeError('STOP final coverage')
out=[]
for (dp,typ),a in sorted(agg.items()):
 out.append({'dp':dp,'type':typ,'rows':a['rows'],'main_stream_rows':a['main_rows'],'other_stream_rows':a['other_rows'],
  'task_duration_ns_sum':a['task_duration_ns'],'aic_cycles':str(a['aic_den']),'aic_mac_cycle_proxy':str(a['aic_num']),
  'aiv_cycles':str(a['aiv_den']),'aiv_vec_cycle_proxy':str(a['aiv_num']),
  'aic_nonpositive_or_missing_rows':a['aic_nonpositive_or_missing'],'aiv_nonpositive_or_missing_rows':a['aiv_nonpositive_or_missing'],
  'reported_input_shapes':dict(a['shapes'])})
print(json.dumps({'scope':'Existing 48 layer0 windows, per-rank runs 1/2/3; same model, fully enclosed, metadata excluded',
 'inputs':{'calls':EXPECTED_CALLS,'manifest':EXPECTED_MANIFEST,'csv_sha256_by_rank':sha_by_rank},'window_coverage':coverage,'mc2_by_window':mc2_rows,'task_types':out,
 'limitations':['CSV task rows and reported cycle proxies, not physical independent work or overall utilization',
 'overlapping rows and AIC/AIV counters cannot be added to yield time or peak utilization',
 'shape fields describe physical reported buffers, not actual valid tokens or same workload',
 'rank/run ordinals are not proven same request or collective', 'absence of telemetry is not proof of zero activity',
 'duration and cycle sums per DP include overlapping ranks/streams and are not end-to-end time'], 'no_device_work':True},ensure_ascii=False,indent=2))
