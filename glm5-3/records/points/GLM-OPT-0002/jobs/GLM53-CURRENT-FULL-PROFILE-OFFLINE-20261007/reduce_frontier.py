from pathlib import Path
import json,hashlib,collections,re,gzip,sys
root=Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0273/profiles_167');job=Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007');dest=job/'frontier';assert not dest.exists();dest.mkdir()
def merge(iv):
 out=[]
 for s,e in sorted(iv):
  if e<=s:continue
  if out and s<=out[-1][1]:out[-1][1]=max(out[-1][1],e)
  else:out.append([s,e])
 return out
index=[]
for path in sorted(root.glob('*_tp*_*/ASCEND_PROFILER_OUTPUT/trace_view.json')):
 rank=int(re.search(r'_tp(\d+)_',str(path)).group(1));raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest();obj=json.loads(raw);del raw;es=obj['traceEvents'] if isinstance(obj,dict) else obj
 meta={e['pid']:e.get('args',{}).get('name') for e in es if e.get('ph')=='M' and e.get('name')=='process_name'}
 hw=next(p for p,n in meta.items() if n=='Ascend Hardware');py=next(p for p,n in meta.items() if n=='Python');cann=next(p for p,n in meta.items() if n=='CANN')
 complete=[(i,e,float(e['ts']),float(e.get('dur',0))) for i,e in enumerate(es) if e.get('ph')=='X']
 hardware=[r for r in complete if r[1]['pid']==hw and r[1].get('args',{}).get('Task Type') not in ('PROFILING_ENABLE','PROFILING_DISABLE')]
 calls=sorted([r for r in complete if r[1]['pid']==cann and ('aclmdlRIExecute' in r[1]['name'] or 'aclrtLaunchCallback' in r[1]['name'])],key=lambda r:r[2])
 graph=[r for r in calls if 'aclmdlRIExecute' in r[1]['name']];cpu=[r for r in complete if r[1]['pid']==py and r[1].get('cat')=='cpu_op'];scope=[r for r in complete if r[1]['pid']==py and any(x in r[1]['name'] for x in ('prepare input','sample_token','draft_token','post process','async_state','execute_model','rejection'))]
 apis=collections.Counter(r[1]['name'] for r in complete if r[1]['pid']==cann and any(w in r[1]['name'].lower() for w in ('graph','aclmdlri','synchronize')))
 raw_event=lambda r:dict(index=r[0],event=r[1])
 row=dict(rank=rank,path=str(path),sha256=sha,events=len(es),graph_api_names=dict(apis),graph_execute=[raw_event(r) for r in graph],host_scopes=[raw_event(r) for r in scope],hardware_task_types=dict(collections.Counter(r[1].get('args',{}).get('Task Type','?') for r in hardware)),periods=[])
 if graph:
  for k,(i,e,lo,d) in enumerate(graph):
   if k+1>=len(graph):break
   hi=graph[k+1][2];active=[r for r in hardware if r[2]<hi and r[2]+r[3]>lo];union=merge([(max(lo,r[2]),min(hi,r[2]+r[3])) for r in active]);gaps=[];last=lo
   for a,b in union:
    if a>last:gaps.append([last,a])
    last=b
   if last<hi:gaps.append([last,hi])
   types={t:merge([(max(lo,s),min(hi,s+du)) for _,ee,s,du in active if ee.get('args',{}).get('Task Type')==t]) for t in row['hardware_task_types']}
   within=[r for r in complete if r[2]<hi and r[2]+r[3]>lo]
   sync=[r for r in within if r[1]['pid']==cann and 'synchroniz' in r[1]['name'].lower()];ctype=collections.defaultdict(float)
   for _,ee,s,du in cpu:
    if s<hi and s+du>lo:ctype[ee['name']]+=min(hi,s+du)-max(lo,s)
   period=dict(ordinal=k,lo_us=lo,hi_us=hi,extent_us=hi-lo,hardware_union_us=sum(b-a for a,b in union),hardware_gap_us=sum(b-a for a,b in gaps),largest_gaps=sorted(gaps,key=lambda z:z[1]-z[0],reverse=True)[:10],hardware_type_union_us={t:sum(b-a for a,b in iv) for t,iv in types.items()},sync=[raw_event(r) for r in sync],cpu_top_inclusive=sorted(ctype.items(),key=lambda z:-z[1])[:20])
   row['periods'].append(period)
   if rank in (0,13,15) and 2<=k<=3:
    with gzip.open(dest/('rank%d_period%d_events.json.gz'%(rank,k)),'wt') as f:json.dump([raw_event(r) for r in within if r[1]['pid'] in (hw,cann,py) or r[1]['name'].startswith('hcom_')],f)
 if rank==0:
  row['main_cpu_tid_top']=collections.Counter(r[1]['tid'] for r in cpu).most_common(8)
  row['graph_like_cpu_names']={n:c for n,c in collections.Counter(r[1]['name'] for r in cpu).items() if any(s in n.lower() for s in ('graph','sample','reject','prepare','forward','record','synchron'))}
 p=dest/('rank%d.json'%rank);p.write_text(json.dumps(row,indent=2)+'\n');index.append(dict(rank=rank,trace_sha256=sha,graph_execute_count=len(graph),periods=len(row['periods']),scope_count=len(scope),events=len(es)));print(json.dumps(index[-1]),flush=True)
 del obj,es,complete,hardware,cpu
(dest/'index.json').write_text(json.dumps(index,indent=2)+'\n')
assert len(index)==16 and {z['rank'] for z in index}==set(range(16))
