#!/usr/bin/env python3
"""Stage262 follow-up: frozen historical CSV only; aggregate, never export rows."""
import bisect,csv,hashlib,json,re,stat
from collections import Counter,defaultdict
from decimal import Decimal,InvalidOperation
from pathlib import Path

T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
R=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
EXPECT={'captured-layer-attribution/module-call-metrics.json':'d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a','evidence/TRACE-EXPORT-SHA256.json':'45ac6c7d22cc20601df2592ce0d78406df24a612f4aa56045566354d9ff46139','captured-layer-engine-proxies/summary.json':'010499ffc68cd21e042407cf075b339736c5f2d547dd538547b9eef261a5ed9f'}
ALLOWED=set('Cast DequantSwigluQuant DispatchFFNCombine DynamicQuant Equal Fill ForeachAddList HcPost HcPre InplacePartialRotaryMul MatMulV2 MatMulV3 MoeGatingTopKHash QuantBatchMatmulV3 RmsNorm RmsNormCast ScatterNdUpdateSk SelectV2 SparseFlashMla Transpose TransposeBatchMatMul hcom_allGather_ hcom_alltoall_ hcom_reduceScatter_'.split())
def sha(f):
 f.seek(0);h=hashlib.sha256()
 for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def fixed(rel):
 p=T/rel
 with p.open('rb') as f:
  if sha(f)!=EXPECT[rel]:raise RuntimeError('fixed input hash mismatch')
  f.seek(0);return json.load(f)
def dec(s):
 try:
  x=Decimal(s);return x if x.is_finite() else None
 except (InvalidOperation,TypeError):return None
def ns(x):return int(Decimal(x.strip())*1000)

calls=fixed('captured-layer-attribution/module-call-metrics.json')
manifest=fixed('evidence/TRACE-EXPORT-SHA256.json')
summary=fixed('captured-layer-engine-proxies/summary.json')
exports=manifest['exports'];expected={x['rank']:x['kernel_csv_sha256'] for x in exports}
if len(exports)!=16 or len(expected)!=16:raise RuntimeError('manifest coverage')
byrank=defaultdict(list)
for c in calls:
 if c['run_in_model'] in (1,2,3):byrank[c['rank']].append(c)
if set(byrank)!=set(expected):raise RuntimeError('rank coverage')
agg=defaultdict(lambda:{'rows':0,'main_stream_rows':0,'other_stream_rows':0,'aiv_cycles':Decimal(0),'aiv_vec_num':Decimal(0),'aiv_valid_ratio_rows':0})
all_groups=Counter();all_streams=Counter();dup_full=0;dup_identity=0;coverage=0
anum=Decimal(0);aden=Decimal(0);vnum=Decimal(0);vden=Decimal(0)
for rank in sorted(expected):
 if not re.fullmatch(r'dp[01]_pp[0-9]+_tp[0-9]+_dcp[0-9]+_ep[0-9]+_rank[0-9]+_[0-9]+_[0-9]+_ascend_pt',rank):raise RuntimeError('bad rank')
 windows=sorted(byrank[rank],key=lambda x:x['window_start_ns'])
 if len(windows)!=120 or len({(x['run_in_model'],x['layer']) for x in windows})!=120:raise RuntimeError('window coverage')
 if any(x['model_id']!='49' for x in windows) or any(a['window_end_ns']>b['window_start_ns'] for a,b in zip(windows,windows[1:])):raise RuntimeError('window model or overlap')
 starts=[x['window_start_ns'] for x in windows];counts=Counter();full=Counter();identity=Counter()
 path=R/rank/'ASCEND_PROFILER_OUTPUT/kernel_details.csv'
 with path.open('rb') as raw:
  pre=raw.fileno();s1=__import__('os').fstat(pre)
  if not stat.S_ISREG(s1.st_mode) or s1.st_size>25000000 or sha(raw)!=expected[rank]:raise RuntimeError('CSV identity')
  raw.seek(0)
  import io
  f=io.TextIOWrapper(raw,encoding='utf-8',newline='',write_through=True)
  for row in csv.DictReader(f):
   if row['Model ID']!='49' or 'metadata' in row['Type'].lower():continue
   st=ns(row['Start Time(us)']);en=st+ns(row['Duration(us)']);i=bisect.bisect_right(starts,st)-1
   if i<0 or en>windows[i]['window_end_ns']:continue
   w=windows[i];counts[i]+=1
   if w['layer']!=0:continue
   typ=row['Type']
   if typ not in ALLOWED:raise RuntimeError('unexpected task type')
   ac=dec(row['aic_total_cycles']);vc=dec(row['aiv_total_cycles']);ar=dec(row['aic_mac_ratio']);vr=dec(row['aiv_vec_ratio'])
   ap=ac is not None and ac>0;vp=vc is not None and vc>0
   if (ap and (ar is None or not 0<=ar<=1)) or (vp and (vr is None or not 0<=vr<=1)):raise RuntimeError('positive-cycle ratio invalid')
   group=('aic_positive' if ap else 'aic_nonpositive_or_missing')+'__'+('aiv_positive' if vp else 'aiv_nonpositive_or_missing')
   stream='main' if row['Stream ID']==w['main_stream_id'] else 'other'
   all_groups[group]+=1;all_streams[(group,stream)]+=1
   z=agg[(group,typ)];z['rows']+=1;z[stream+'_stream_rows']+=1
   if ap:aden+=ac;anum+=ac*ar
   if vp:vden+=vc;vnum+=vc*vr;z['aiv_cycles']+=vc;z['aiv_vec_num']+=vc*vr;z['aiv_valid_ratio_rows']+=1
   full[hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).digest()]+=1
   identity[(row['Model ID'],row['Stream ID'],row['Task ID'],row['Start Time(us)'],row['Duration(us)'])]+=1
  f.detach()
  s2=__import__('os').fstat(pre)
  if (s1.st_dev,s1.st_ino,s1.st_size,s1.st_mtime_ns)!=(s2.st_dev,s2.st_ino,s2.st_size,s2.st_mtime_ns) or sha(raw)!=expected[rank]:raise RuntimeError('CSV changed')
 if any(counts[i]!=w['attributed_rows'] for i,w in enumerate(windows)):raise RuntimeError('per-window attribution mismatch')
 coverage+=sum(w['layer']==0 for w in windows)
 dup_full+=sum(n-1 for n in full.values() if n>1);dup_identity+=sum(n-1 for n in identity.values() if n>1)
layer=next(x for x in summary['layers'] if x['layer']==0)
a=layer['metrics']['aic_mac_ratio'];v=layer['metrics']['aiv_vec_ratio']
if coverage!=48 or sum(all_groups.values())!=1824 or layer['rows']!=1824:raise RuntimeError('layer0 count mismatch')
if (aden,anum,vden,vnum)!=(Decimal(a['denominator']),Decimal(a['numerator']),Decimal(v['denominator']),Decimal(v['numerator'])):raise RuntimeError('counter summary mismatch')
if sum(n for k,n in all_groups.items() if k.startswith('aic_nonpositive'))!=a['no_positive_cycles'] or sum(n for k,n in all_groups.items() if k.endswith('aiv_nonpositive_or_missing'))!=v['no_positive_cycles']:raise RuntimeError('cycle count mismatch')
result=[]
for (group,typ),z in sorted(agg.items()):
 result.append({'group':group,'type':typ,'rows':z['rows'],'main_stream_rows':z['main_stream_rows'],'other_stream_rows':z['other_stream_rows'],'aiv_positive_cycle_rows_with_valid_ratio':z['aiv_valid_ratio_rows'],'aiv_cycle_sum':str(z['aiv_cycles']),'aiv_vec_weighted_numerator':str(z['aiv_vec_num']),'aiv_vec_weighted_ratio':str(z['aiv_vec_num']/z['aiv_cycles']) if z['aiv_cycles'] else None})
print(json.dumps({'scope':'16 rank, 48 target layer0 windows; source selection same model=49, all streams, fully contained; metadata excluded','coverage_windows':coverage,'rows':sum(all_groups.values()),'groups':[{'group':g,'rows':n,'main_stream_rows':all_streams[(g,'main')],'other_stream_rows':all_streams[(g,'other')]} for g,n in sorted(all_groups.items())],'group_by_type':result,'duplicate_extra_full_row_occurrences_within_rank':dup_full,'duplicate_extra_task_identity_occurrences_within_rank':dup_identity,'input_hashes':EXPECT,'manifest_csv_count':len(expected),'limitations':['Counts are exported rows, preserved with duplicates and overlapping tasks; no inference about independent physical operations, overall utilization or elapsed time.','Positive cycles and valid ratio are recorded exporter fields, not proof of collection completeness or hardware activity.','Main stream is relative to each window marker; other streams are retained by original attribution rule and do not prove ownership.','Current ratio fields are rounded and historical exporter internals not established.'],'no_device_work':True},ensure_ascii=False,indent=2))
