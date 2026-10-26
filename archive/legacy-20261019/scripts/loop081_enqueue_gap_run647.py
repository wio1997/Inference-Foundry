#!/usr/bin/env python3
"""Join Run646 stream47 gaps to actual async-task enqueue/dequeue records."""
from __future__ import annotations
import hashlib,json
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry');BASE=ROOT/'evidence/20260928_loop081_bound'
M=BASE/'run611/offline_parse/manifest.json';RAW=BASE/'run646/stream47_flow_gap.json'
OUT=BASE/'run647/enqueue_gap_summary.json'
M_SHA='b51eac01dcdc9f183544d51d20040f371f5ffb23a686ded7372b2cb366c84e83'
RAW_SHA='1aaa460bbc43f1c89baaf69767c6eefe1d44837a2905d12b8cf3d71c3e7dc968'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def d(x):return Decimal(str(x).strip())
def need(ok,s):
 if not ok:raise AssertionError(s)
def main():
 need(sha(M)==M_SHA and sha(RAW)==RAW_SHA,'source SHA')
 m=json.loads(M.read_text());r=json.loads(RAW.read_text())
 need(len(m['parsed'])==8 and len(r['rows'])==16,'all8x2')
 rows=[]
 for rank,s in enumerate(m['parsed']):
  need(s['rank']==rank,'rank order')
  p=ROOT/s['trace_view_json'];need(sha(p)==s['trace_view_sha256'],'trace SHA')
  trace=json.loads(p.read_text());cpu=defaultdict(list);enqueues=defaultdict(list);dequeues=defaultdict(list);scopes=[]
  for x in trace:
   if x.get('cat')=='cpu_op' and x.get('ph')=='X':
    cpu[(x['name'],x['ts'])].append(x)
    if x['name'].startswith('extreme::'):scopes.append(x)
   elif x.get('cat')=='enqueue':enqueues[(x['pid'],x['tid'],x['name'])].append(x)
   elif x.get('cat')=='dequeue':dequeues[x['args']['correlation_id']].append(x)
  for row in (v for v in r['rows'] if v['rank']==rank):
   totals=defaultdict(Decimal);by_scope=defaultdict(lambda:defaultdict(Decimal))
   gap_count=0;enq_late_count=0;dequeue_after_native_count=0
   for z in row['gap_rows']:
    op=cpu[(z['next_cpu_op_name'],z['next_cpu_op_begin_us'])]
    need(len(op)==1,'CPU op exact uniqueness')
    op=op[0];op_start=d(op['ts']);op_end=op_start+d(op['dur'])
    en=[x for x in enqueues[(op['pid'],op['tid'],'Enqueue@'+op['name'])] if op_start<=d(x['ts'])<=op_end]
    need(len(en)==1,'CPU op enqueue uniqueness')
    en=en[0];de=dequeues[en['args']['correlation_id']]
    need(len(de)==1,'enqueue/dequeue correlation uniqueness')
    de=de[0]
    need(de['name']=='Dequeue@'+op['name'] and de['pid']==op['pid'],'enqueue/dequeue name and process identity')
    prior=d(z['previous_native_end_us']);native=d(z['next_native_start_us']);gap=d(z['gap_us'])
    need(native-prior==gap and op_start<=d(en['ts'])<=native,'CPU/enqueue/native milestone order')
    dequeue_after_native_count+=int(d(de['ts'])>native)
    no_enq=min(gap,max(Decimal(0),d(en['ts'])-prior))
    need(no_enq>=d(z['cpu_op_not_begun_us']),'enqueue start after CPU begin')
    matching=[q for q in scopes if d(q['ts'])<=op_start<=d(q['ts'])+d(q['dur'])]
    scope=min(matching,key=lambda q:d(q['dur']))['name'] if matching else 'unscoped'
    totals['gap_us']+=gap;totals['cpu_op_not_begun_us']+=d(z['cpu_op_not_begun_us'])
    totals['enqueue_not_begun_us']+=no_enq
    by_scope[scope]['gap_us']+=gap;by_scope[scope]['enqueue_not_begun_us']+=no_enq
    gap_count+=1;enq_late_count+=int(no_enq>0)
   need(gap_count==row['positive_gap_count'] and totals['gap_us']==d(row['stream47_between_task_gap_sum_us']),'Run646 match')
   rows.append({'rank':rank,'transition':row['transition'],'positive_gap_count':gap_count,
     'enqueue_after_previous_native_count':enq_late_count,
     'dequeue_start_after_native_start_count':dequeue_after_native_count,
     'stream47_gap_sum_us':str(totals['gap_us']),
     'next_cpu_op_not_begun_us':str(totals['cpu_op_not_begun_us']),
     'next_enqueue_not_begun_us':str(totals['enqueue_not_begun_us']),
     'by_enclosing_extreme_scope_us':{name:{k:str(v) for k,v in vals.items()} for name,vals in sorted(by_scope.items())},
     'trace_view_sha256':s['trace_view_sha256']})
 need(len(rows)==16,'all8 rows')
 result={'status':'all8_two_transition_enqueue_dequeue_exact_join_pass',
  'scope':'Run611 instrumented same W0. Each positive Run646 stream47 gap joins an exact CPU op, one contained Enqueue@op record and one correlation-ID Dequeue@op record. Some dequeue starts appear after matched native starts, so no cross-domain dequeue-time bound is asserted. Enqueue-start after prior native end means the next op was not yet enqueuing, subject to Host/native clock mapping. It does not prove why Host waited, actual enqueue completion, legal reordering, hardware idle or removable Product wall.',
  'manifest_sha256':sha(M),'run646_raw_sha256':sha(RAW),'rows':rows,
  'framework_only_whole_product_tps_bound':None}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'rank7':[{k:v for k,v in x.items() if k in ('transition','stream47_gap_sum_us','next_enqueue_not_begun_us','dequeue_start_after_native_start_count')} for x in rows if x['rank']==7]}))
if __name__=='__main__':main()
