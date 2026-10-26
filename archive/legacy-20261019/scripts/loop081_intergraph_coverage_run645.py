#!/usr/bin/env python3
"""Run611 same-W0 rank-local native task coverage between Target Graphs."""
from __future__ import annotations
import csv,hashlib,json
from collections import Counter,defaultdict
from decimal import Decimal
from pathlib import Path
ROOT=Path('/data/wio/Inference_Foundry');BASE=ROOT/'evidence/20260928_loop081_bound'
MANIFEST=BASE/'run611/offline_parse/manifest.json';REPLAY=BASE/'run611/native_graph_replay.json'
OUT=BASE/'run645/intergraph_native_coverage.json'
MANIFEST_SHA='b51eac01dcdc9f183544d51d20040f371f5ffb23a686ded7372b2cb366c84e83'
REPLAY_SHA='9518ee739ece657c8cff467ff476d92e7cc939048945abccc1e9f7e5a009336d'
PHYSICAL={'KERNEL_AICORE','KERNEL_MIX_AIC','KERNEL_AIVEC','KERNEL_MIX_AIV','KERNEL_AICPU','MEMCPY_ASYNC'}
WAIT={'EVENT_WAIT','NOTIFY_WAIT'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def d(v):return Decimal(str(v).strip())
def merge(items):
 out=[]
 for a,b in sorted(items):
  if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
  else:out.append((a,b))
 return out
def union(items):return sum((b-a for a,b in merge(items)),Decimal(0))
def main():
 assert sha(MANIFEST)==MANIFEST_SHA and sha(REPLAY)==REPLAY_SHA
 m=json.loads(MANIFEST.read_text());r=json.loads(REPLAY.read_text())
 assert len(m['parsed'])==8 and len(r['rows'])==24
 refs={(x['rank'],x['ordinal_in_profile']):x for x in r['rows']}
 assert set(refs)=={(rank,i) for rank in range(8) for i in range(3)}
 rows=[]
 for rank,s in enumerate(m['parsed']):
  assert s['rank']==rank
  path=ROOT/s['task_time_csv'];assert sha(path)==s['task_time_sha256']
  tasks=list(csv.DictReader(path.open()))
  for i in (0,1):
   begin=d(refs[rank,i]['last_task_end_us']);end=d(refs[rank,i+1]['first_task_start_us']);assert begin<end
   all_tasks=[];physical=[];wait=[];by_stream=defaultdict(Decimal);counts=Counter()
   for x in tasks:
    a,b=d(x['task_start(us)']),d(x['task_stop(us)'])
    if b<=begin or a>=end:continue
    aa,bb=max(a,begin),min(b,end);kind=x['kernel_type'];counts[kind]+=1
    all_tasks.append((aa,bb))
    if kind in PHYSICAL:
     physical.append((aa,bb));by_stream[str(x['stream_id'])]+=bb-aa
    elif kind in WAIT:wait.append((aa,bb))
   assert sum(counts.values())>700
   span=end-begin;covered=union(all_tasks);pc=union(physical);wc=union(wait)
   merged=merge(all_tasks)
   gaps=[merged[0][0]-begin]+[right[0]-left[1] for left,right in zip(merged,merged[1:])]+[end-merged[-1][1]]
   assert sum(gaps)==span-covered
   assert Decimal(0)<covered<=span and pc<=covered and wc<=covered
   rows.append({'rank':rank,'transition':f'{i}->{i+1}',
    'window_span_us':str(span),'all_exported_task_union_us':str(covered),
    'window_without_exported_task_us':str(span-covered),
    'max_single_task_free_gap_us':str(max(gaps)),'positive_task_free_gap_count':sum(x>0 for x in gaps),
    'physical_kernel_copy_union_us':str(pc),'wait_union_us':str(wc),
    'physical_service_by_stream_us':dict(sorted((k,str(v)) for k,v in by_stream.items())),
    'task_counts':dict(counts),'task_time_sha256':s['task_time_sha256']})
 result={'status':'all8_two_transition_native_task_coverage_pass',
  'scope':'Run611 Level0 instrumented rank-local windows, previous Model45 last exported task to next Model45 first task. No exported-task coverage is not certified hardware idle, Host overhead, disjoint service or removable time. Stream38 durations can include HCCL peer wait. No full Product transfer.',
  'manifest_sha256':sha(MANIFEST),'replay_sha256':sha(REPLAY),'rows':rows,
  'framework_only_whole_product_tps_bound':None}
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'status':result['status'],'rank7':[{k:v for k,v in x.items() if k in ('transition','window_span_us','window_without_exported_task_us','physical_kernel_copy_union_us')} for x in rows if x['rank']==7]}))
if __name__=='__main__':main()
