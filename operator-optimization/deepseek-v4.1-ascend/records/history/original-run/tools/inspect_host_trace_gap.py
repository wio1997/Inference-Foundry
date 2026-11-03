"""Query bounded profiler CPU APIs around an observed inter-forward gap.

Temporal proximity is not treated as request ownership, CPU causation, or a
clock-synchronization certificate. Reads SQLite in readonly mode.
"""
from pathlib import Path
from decimal import Decimal,InvalidOperation
import csv,json,sqlite3,statistics
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'host-gap-trace-inspection';O.mkdir(exist_ok=False)
def number(s):
 try:return int(Decimal(str(s)))
 except (ValueError,InvalidOperation,TypeError):return None
def union(rs):
 end=None;v=0
 for a,b in sorted(rs):
  if end is None or a>end:v+=b-a;end=b
  elif b>end:v+=b-end;end=b
 return v
gaps=list(csv.DictReader((T/'inter-forward-gap-analysis-v2/rank_gaps.csv').open()))
out=[]
for ep in [0,8]:
 sel=[r for r in gaps if '_ep'+str(ep)+'_' in r['rank'] and r['target_forward_ordinal']=='7'];assert len(sel)==1
 r=sel[0];begin=int(r['previous_target_layer39_mc2_end_ns']);end=int(r['next_target_layer0_start_ns']);db=next((ROOT/r['rank']/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'))
 c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
 names=[]
 for start,finish,typ,tid,name in c.execute('SELECT p.startNs,p.endNs,p.type,p.globalTid,s.value FROM PYTORCH_API p LEFT JOIN STRING_IDS s ON p.name=s.id'):
  start=number(start);finish=number(finish)
  if start is None or finish is None or start>=end or finish<=begin:continue
  lo=max(begin,start);hi=min(end,finish)
  names.append({'type':typ,'name':name,'global_tid':tid,'start_relative_us':(start-begin)/1000,
                'duration_us':(finish-start)/1000,'clipped_start_ns':lo,'clipped_end_ns':hi})
 cann=c.execute('SELECT COUNT(*) FROM CANN_API').fetchone()[0]
 bounds=c.execute('SELECT min(CAST(startNs AS INTEGER)),max(CAST(endNs AS INTEGER)) FROM PYTORCH_API').fetchone()
 c.close();names.sort(key=lambda x:x['clipped_end_ns']-x['clipped_start_ns'],reverse=True)
 top=[]
 for x in names[:20]:top.append({k:v for k,v in x.items() if k not in ['clipped_start_ns','clipped_end_ns']})
 covered=union([(x['clipped_start_ns'],x['clipped_end_ns']) for x in names])
 out.append({'ep':ep,'rank':r['rank'],'previous39mc2_to_next0_start_us':(end-begin)/1000,
             'db_api_timestamp_min_max_ns':bounds,'window_has_host_records':bool(names),
             'cpu_api_events_intersecting_window':len(names),'cpu_api_interval_union_us':covered/1000,
             'cann_api_table_rows':cann,'longest_20_intersections':top})
doc={'status':'BOUNDED_HOST_WINDOW_CORRELATION_NOT_CAUSAL_ATTRIBUTION','windows':out,
     'warning':'Profiler API intervals may overlap or run on unrelated threads; timestamps, connection IDs and queue semantics are not independently calibrated to each replay. Interval union is not CPU utilization and cannot establish the cause of a cross-DP delay.',
     'no_device_work':True}
assert len(json.dumps(doc).encode())<32768
(O/'summary.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2))
