from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14')
b=(P/'frontend-CPU-trace-inventory-v14.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='4da035667b2a4dde966b884590ab8d8d0668bf408258ee172be04310329a09d3';x=json.loads(b)
OUT=P/'frontend-parent-work-v14.json';assert not OUT.exists()
files=[]
for f in x['files']:
 rs=[dict(r)for r in f['complete_events']if r.get('cat')=='python_function'];byid=defaultdict(list);children=defaultdict(list)
 for r in rs:byid[r['relation_args']['Python id']].append(r)
 violations=[]
 for r in rs:
  cs=byid.get(r['relation_args']['Python parent id'],[])
  if len(cs)==1:
   p=cs[0];same=r['pid']==p['pid']and r['tid']==p['tid'];inside=p['ts']<=r['ts']and r['ts']+r['dur']<=p['ts']+p['dur']+0.002
   if same and inside:children[p['trace_event_index']].append(r)
   else:violations.append(dict(child=r['trace_event_index'],parent=p['trace_event_index'],same_thread=same,contained=inside))
 def union(v):
  total=0;end=None
  for a,b in sorted(v):
   if end is None or a>end:total+=b-a;end=b
   elif b>end:total+=b-end;end=b
  return total
 groups=defaultdict(list)
 for r in rs:
  r['validated_direct_child_union_us']=union([(c['ts'],c['ts']+c['dur'])for c in children[r['trace_event_index']]])
  r['exclusive_recorded_wall_us']=max(0,r['dur']-r['validated_direct_child_union_us']);groups[r['name']].append(r)
 aggs=[dict(name=name,calls=len(v),inclusive_wall_us=sum(r['dur']for r in v),exclusive_recorded_wall_us=sum(r['exclusive_recorded_wall_us']for r in v),maximum_inclusive_us=max(r['dur']for r in v),pid_tids=sorted({(r['pid'],r['tid'])for r in v}))for name,v in groups.items()]
 aggs.sort(key=lambda r:r['exclusive_recorded_wall_us'],reverse=True)
 selected=[r for r in rs if ('encode_batch' in r['name'])or('DecodeStream' in r['name']and 'step 'in r['name'])or any(s in r['name']for s in ['_tokenize_prompt','process_outputs','detokenizer.py(96)','decode_next','_protected_step','model_dump_json'])]
 chains=[]
 for r in selected:
  chain=[];cur=r;seen=set()
  while cur is not None and cur['trace_event_index']not in seen:
   seen.add(cur['trace_event_index']);chain.append(dict(event_index=cur['trace_event_index'],name=cur['name'],dur=cur['dur'],pid=cur['pid'],tid=cur['tid']));ps=byid.get(cur['relation_args']['Python parent id'],[]);cur=ps[0]if len(ps)==1 else None
  chains.append(dict(event=r,parent_chain=chain))
 files.append(dict(path=f['path'],source_sha256=f['raw_sha256'],events=len(rs),Python_id_duplicate_groups=sum(len(v)>1 for v in byid.values()),validated_parent_edges=sum(len(v)for v in children.values()),parent_violations=violations,selected_event_full_parent_chains=chains,exclusive_wall_name_aggregates=aggs))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),files=len(files),Python_id_duplicate_groups=sum(f['Python_id_duplicate_groups']for f in files),parent_violations=sum(len(f['parent_violations'])for f in files),no_source_DB_or_device_work=True,overall_complete=False,
 limits=['Exclusive recorded wall subtracts only union of validated direct children; it is not thread CPU time or unprofiled cost. Missing/truncated parents and invalid edges retained.', 'Thread0/monitor/wait spans are not CPU occupancy; initial and later windows are distinct populations.'])
payload=(json.dumps(dict(summary=summary,files=files),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
interesting=[]
for f in files:
 selected=[a for a in f['exclusive_wall_name_aggregates']if any(s in a['name']for s in ['_tokenize_prompt','encode_batch','DecodeStream','process_outputs','detokenizer.py(96)','model_dump_json'])]
 expensive=[dict(name=r['event']['name'],duration_us=r['event']['dur'],parent_chain=r['parent_chain'][:8])for r in f['selected_event_full_parent_chains']if 'DecodeStream'in r['event']['name']and r['event']['dur']>1000]
 interesting.append(dict(path=f['path'],selected_aggregates=selected,expensive_DecodeStream_steps=expensive,parent_violation_examples=f['parent_violations'][:5]))
print(json.dumps(dict(summary=summary,selected=interesting,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
