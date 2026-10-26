#!/usr/bin/env python3
"""Exact Host CPU-op begin→native stream47 flow-gap census for Run611."""
from __future__ import annotations
import hashlib,json
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry');BASE=ROOT/'evidence/20260928_loop081_bound'
M=BASE/'run611/offline_parse/manifest.json';G=BASE/'run611/native_graph_replay.json'
OUT=BASE/'run646/stream47_flow_gap.json'
SUMMARY=BASE/'run646/summary.json'
M_SHA='b51eac01dcdc9f183544d51d20040f371f5ffb23a686ded7372b2cb366c84e83'
G_SHA='9518ee739ece657c8cff467ff476d92e7cc939048945abccc1e9f7e5a009336d'
def d(v):return Decimal(str(v).strip())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,s):
 if not x:raise AssertionError(s)
def main():
 need(sha(M)==M_SHA and sha(G)==G_SHA,'source identity')
 m=json.loads(M.read_text());g=json.loads(G.read_text());need(len(m['parsed'])==8 and len(g['rows'])==24,'all8x3')
 graph={(x['rank'],x['ordinal_in_profile']):x for x in g['rows']}
 need(set(graph)=={(r,i) for r in range(8) for i in range(3)},'graph bijection')
 rows=[]
 for rank,s in enumerate(m['parsed']):
  need(rank==s['rank'],'rank order')
  p=ROOT/s['trace_view_json'];need(sha(p)==s['trace_view_sha256'],'trace SHA')
  trace=json.loads(p.read_text());cpu=defaultdict(list)
  for x in trace:
   if x.get('cat')=='cpu_op' and x.get('ph')=='X':cpu[(x['pid'],x['tid'],x['ts'])].append(x)
  starts=defaultdict(list);finishes=defaultdict(list)
  for x in trace:
   if x.get('cat')!='async_npu' or x.get('name')!='torch_to_npu':continue
   if x.get('ph')=='s':starts[x['id']].append(x)
   elif x.get('ph')=='f':finishes[(x['pid'],x['tid'],x['ts'])].append(x)
  for i in (0,1):
   begin=d(graph[rank,i]['last_task_end_us']);end=d(graph[rank,i+1]['first_task_start_us'])
   native=sorted((x for x in trace if x.get('ph')=='X' and
     x.get('args',{}).get('Model Id')==4294967295 and
     x.get('args',{}).get('Physic Stream Id')==47 and
     begin<=d(x['ts'])<end and x['args'].get('Task Type')!='MODEL_EXECUTE'),key=lambda x:d(x['ts']))
   need(len(native) in (754,771),'stream47 task population')
   host_starts=[]
   for x in native:
    found=finishes[(x['pid'],x['tid'],x['ts'])]
    need(len(found)==1,'native flow finish uniqueness')
    flow=found[0];host=starts[flow['id']]
    need(len(host)==1,'flow start uniqueness')
    h=host[0];owner=cpu[(h['pid'],h['tid'],h['ts'])]
    need(len(owner)==1,'flow start CPU op identity')
    ts=d(h['ts']);need(ts<=d(x['ts']),'host-before-device')
    host_starts.append((ts,owner[0]['name']))
   gap_total=Decimal(0);host_op_not_begun=Decimal(0);remainder=Decimal(0)
   gap_count=0;late_gap_count=0;max_gap=Decimal(0);rank_rows=[];late_by_cpu_op=defaultdict(Decimal)
   for j in range(1,len(native)):
    prior_end=d(native[j-1]['ts'])+d(native[j-1]['dur']);current_start=d(native[j]['ts'])
    gap=max(Decimal(0),current_start-prior_end)
    if gap==0:continue
    h,name=host_starts[j]
    late=min(gap,max(Decimal(0),h-prior_end))
    gap_total+=gap;host_op_not_begun+=late;remainder+=gap-late
    gap_count+=1;late_gap_count+=int(late>0);max_gap=max(max_gap,gap)
    late_by_cpu_op[name]+=late
    rank_rows.append({'next_native_task_id':native[j]['args']['Task Id'],
      'previous_native_end_us':str(prior_end),'next_native_start_us':str(current_start),
      'next_cpu_op_begin_us':str(h),'next_cpu_op_name':name,'gap_us':str(gap),'cpu_op_not_begun_us':str(late),
      'next_native_task_type':native[j]['args']['Task Type']})
   need(gap_total==host_op_not_begun+remainder,'gap partition')
   rows.append({'rank':rank,'transition':f'{i}->{i+1}',
    'stream47_flow_joined_native_task_count':len(native),'positive_gap_count':gap_count,
    'positive_late_cpu_op_gap_count':late_gap_count,'stream47_between_task_gap_sum_us':str(gap_total),
    'cpu_op_not_yet_begun_before_next_native_us':str(host_op_not_begun),
    'unclassified_gap_remainder_us':str(remainder),'max_single_gap_us':str(max_gap),
    'cpu_op_not_begun_by_name_us':dict(sorted((k,str(v)) for k,v in late_by_cpu_op.items() if v>0)),
    'gap_rows':rank_rows,'trace_view_sha256':s['trace_view_sha256']})
 result={'status':'all8_two_transition_exact_stream47_flow_gap_pass',
  'scope':'Run611 instrumented same W0. Exact torch_to_npu s/f and native pid/tid/Decimal timestamp join. Flow s coincides with CPU op begin; positive s-after-prior-native-end proves that next CPU op had not begun at gap start, not when it was actually enqueued or whether that delay is legally removable. Remainder may include late enqueue, dependency, queue/resource wait. Stream47 gaps are not all-rank Product critical-path time; Host/native clock mapping remains conditional.',
  'manifest_sha256':sha(M),'replay_sha256':sha(G),'rows':rows,
  'framework_only_whole_product_tps_bound':None}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n')
 compact={k:v for k,v in result.items() if k!='rows'}
 compact['raw_sha256']=sha(OUT)
 compact['rows']=[{k:v for k,v in row.items() if k!='gap_rows'} for row in rows]
 SUMMARY.write_text(json.dumps(compact,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'rank7':[{k:v for k,v in x.items() if k in ('transition','stream47_between_task_gap_sum_us','cpu_op_not_yet_begun_before_next_native_us','unclassified_gap_remainder_us','positive_gap_count')} for x in rows if x['rank']==7]}))
if __name__=='__main__':main()
