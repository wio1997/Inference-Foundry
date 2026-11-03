"""CPU-only grouped read of saved profiler rows; no cross-rank timestamp use."""
from pathlib import Path
from decimal import Decimal
from collections import defaultdict,Counter
import bisect,csv,json,statistics
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'q-projection-allranks';O.mkdir(exist_ok=False)
target=json.loads((T/'captured-layer-attribution/module-call-metrics.json').read_text())
draft=json.loads((T/'eager-draft-mainstream-attribution-v2/draft_call_metrics.json').read_text())
windows=defaultdict(list)
for x in target:
 if x['run_in_model'] in [1,2,3]:windows[(x['rank'],x['model_id'],x['main_stream_id'])].append((x['window_start_ns'],x['window_end_ns'],'CAPTURED_TARGET_INTERIOR',x['layer']))
for x in draft:windows[(x['rank'],x['model_id'],x['main_stream_id'])].append((x['start_ns'],x['end_ns'],'STRICT_EAGER_DRAFT',None))
for a in windows.values():a.sort();assert all(x[1]<=y[0] for x,y in zip(a,a[1:]))
allrows=[]
for rp in sorted(ROOT.glob('*_ascend_pt')):
 rank=rp.name
 for z in csv.DictReader((rp/'ASCEND_PROFILER_OUTPUT/kernel_details.csv').open()):
  if z['Type']!='QuantBatchMatmulV3' or '1024,80,16,32' not in z['Input Shapes']:continue
  shapes=z['Input Shapes'].strip('"').split(';');m=int(shapes[0].split(',')[0]);assert shapes[0]==f'{m},1280' and shapes[1]=='1024,80,16,32'
  assert z['Input Data Types']=='INT8;INT8;DT_BF16;FLOAT' and z['Input Formats']=='ND;FRACTAL_NZ;ND;ND'
  start=int(Decimal(z['Start Time(us)'].strip())*1000);end=start+int(Decimal(z['Duration(us)'])*1000)
  spans=windows.get((rank,z['Model ID'],z['Stream ID']),[]);ix=bisect.bisect_right(spans,(start,10**30,'',0))-1
  scope='OTHER_UNATTRIBUTED';layer=None
  if ix>=0 and spans[ix][0]<=start and end<=spans[ix][1]:_,_,scope,layer=spans[ix]
  c=Decimal(z['aic_total_cycles'] or '0')
  if c>0: assert z['aic_mac_ratio'] and z['aic_mte2_ratio'], 'missing PMU ratio with positive cycles'
  mac=Decimal(z['aic_mac_ratio'] or '0');mte=Decimal(z['aic_mte2_ratio'] or '0');assert c>=0 and 0<=mac<=1 and 0<=mte<=1
  allrows.append({'rank':rank,'scope':scope,'layer':layer,'m':m,'duration_us':float(z['Duration(us)']),'aic_cycles':int(c),'mac_cycles_proxy':float(c*mac),'mte2_cycles_proxy':float(c*mte)})

def agg(rs):
 ds=sorted(x['duration_us'] for x in rs);c=sum(x['aic_cycles'] for x in rs)
 return {'calls':len(rs),'median_duration_us':statistics.median(ds),'p05_duration_us':ds[int((len(ds)-1)*0.05)],'p95_duration_us':ds[int((len(ds)-1)*0.95)],'aic_cycle_weighted_mac_proxy':sum(x['mac_cycles_proxy'] for x in rs)/c if c else None,'aic_cycle_weighted_mte2_proxy':sum(x['mte2_cycles_proxy'] for x in rs)/c if c else None,'aic_cycles':c}
buckets=defaultdict(list)
for x in allrows:buckets[(x['rank'],x['scope'],x['m'])].append(x)
rows=[{'rank':a,'scope':b,'m':c,**agg(v)} for (a,b,c),v in sorted(buckets.items())]
with (O/'rank-scope-shape.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups=defaultdict(list)
for x in allrows:groups[(x['scope'],x['m'])].append(x)
summary=[]
for (scope,m),rs in sorted(groups.items()):
 per=[x for x in rows if x['scope']==scope and x['m']==m]
 summary.append({'scope':scope,'m':m,'rank_count':len(per),'rank_median_us_min':min(x['median_duration_us'] for x in per),'rank_median_us_max':max(x['median_duration_us'] for x in per),**agg(rs),'conditional_compute_only_ideal_us':m*41943040/353894400000000*1e6})
out={'status':'SAME_SHAPE_16RANK_TARGET_DRAFT_SENSITIVITY','rows':len(allrows),'rank_count':len(list(ROOT.glob('*_ascend_pt'))),'groups':summary,'metric_limits':['Same shape/type is not by itself proof of Qb semantics; target/draft grouping uses pre-existing source-invariant brackets, same model ID and same stream; other calls remain unattributed.','AIC-cycle-weighted ratios are counter proxies, not device utilization or MFU; MTE2 and MAC pipes can overlap.','Conditional ideal uses the existing nominal INT8 Cube assumptions and independent dense full-output domain; it is not a physical-unconditional bound or predicted speedup.','Kernel profiler changes execution and target/draft/bracket samples differ; pooled per-rank distributions are not request-level latency.'],'no_device_work':True}
assert len(json.dumps(out).encode())<32768
(O/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'rows':len(allrows),'groups':summary},indent=2))
