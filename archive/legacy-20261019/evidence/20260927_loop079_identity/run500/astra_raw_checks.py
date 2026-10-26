import json,hashlib,collections,re,types,sys
from pathlib import Path
root=Path('/data/wio/Inference_Foundry');b=root/'evidence/20260927_loop079_identity/run494/b_candidate'
mods={}
for name,fn in [('helper','loop079_target_frontier_run494.py'),('validator','loop079_target_frontier_validate_run494.py')]:
 s=(root/'scripts'/fn).read_text();assert s.count("'MEMCPY','MEMSET'}")==1
 m=types.ModuleType(name);exec(compile(s.replace("'MEMCPY','MEMSET'}","'MEMCPY','MEMSET','MEMCPY_ASYNC'}"),fn,'exec'),m.__dict__);mods[name]=m
rows=[]
for p in sorted((b/'graph_dump').glob('rank*_cohort5_acl_graph.json')):
 raw=p.read_bytes();nodes=json.loads(raw);a=mods['helper'].graph_task_metadata(nodes);v=mods['validator'].native_tasks(nodes);assert a==v
 kinds=collections.Counter(n['args']['Task Type'] for n in nodes);streams=collections.defaultdict(list);events={k:{} for k in ['RECORD','WAIT','RESET']};edges=collections.defaultdict(set)
 for i,n in enumerate(nodes):
  ar=n['args'];streams[ar['Stream Id']].append((ar['Task Id'],i));kind=ar['Task Type']
  if kind.startswith('EVENT_'):
   match=re.fullmatch(kind+r'_(\d+)',n['name']);assert match
   events[kind[6:]].setdefault(int(match[1]),[]).append(i)
 for st,items in streams.items():
  items.sort()
  for (_,i),(_,j) in zip(items,items[1:]):edges[i].add(j)
 rec,wait,reset=[events[k] for k in ['RECORD','WAIT','RESET']]
 for ev in set(rec)&set(wait):
  for i in rec[ev]:
   for j in wait[ev]:edges[i].add(j)
 indeg=[0]*len(nodes)
 for ii,jj in edges.items():
  for j in jj:indeg[j]+=1
 queue=collections.deque(i for i,d in enumerate(indeg) if d==0);seen=[]
 while queue:
  i=queue.popleft();seen.append(i)
  for j in edges[i]:
   indeg[j]-=1
   if indeg[j]==0:queue.append(j)
 rev=collections.defaultdict(set)
 for i,js in edges.items():
  for j in js:rev[j].add(i)
 terminal=[i for i,n in enumerate(nodes) if n['args']['Task Type']=='NOTIFY_RECORD']
 reached=set(terminal);q=list(terminal)
 while q:
  i=q.pop()
  for j in rev[i]:
   if j not in reached:reached.add(j);q.append(j)
 mem=[n for n in nodes if n['args']['Task Type']=='MEMCPY_ASYNC']
 rows.append(dict(file=str(p.relative_to(root)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),schema_after_only_memcpy_async=True,task_meta=a,types=dict(kinds),
  memcpy_async=dict(count=len(mem),streams=dict(collections.Counter(n['args']['Stream Id'] for n in mem)),arg_keys=sorted({tuple(sorted(n['args'])) for n in mem})),
  per_stream_contiguous={str(s):[x[0] for x in sorted(items)]==list(range(len(items))) for s,items in streams.items()},
  event_counts={k:len(d) for k,d in events.items()},events_duplicate_ids={k:sum(len(v)>1 for v in d.values()) for k,d in events.items()},wait_without_record=sorted(set(wait)-set(rec)),record_without_wait_count=len(set(rec)-set(wait)),reset_without_record=sorted(set(reset)-set(rec)),record_without_reset=sorted(set(rec)-set(reset)),
  record_wait_stream_pairs=dict(collections.Counter(str((nodes[rec[e][0]]['args']['Stream Id'],nodes[wait[e][0]]['args']['Stream Id'])) for e in set(rec)&set(wait))),
  stream_task_plus_record_wait_skeleton=dict(acyclic=len(seen)==len(nodes),terminal_notify=len(terminal),tasks_reaching_notify=len(reached),unreached_types=dict(collections.Counter(nodes[i]['args']['Task Type'] for i in range(len(nodes)) if i not in reached)))))
# Validate all existing warmup rows against unchanged detailed v3 gate.
sys.path.insert(0,str(root/'scripts'));import loop079_target_frontier_validate_run494 as validator
warm=list((b/'capture').glob('rank*_cohort*.json'));warmok=[]
for p in warm:
 d=json.loads(p.read_text());validator.validate_detail(d);warmok.append((d['rank'],d['cohort']))
negative={}
memnode={'name':'MEMCPY_ASYNC','args':{'Model Id':45,'Stream Id':1,'Task Id':0,'Task Type':'MEMCPY_ASYNC'}}
for label,nodes in [('valid_common_only',[memnode]),('missing_task',[{'name':'MEMCPY_ASYNC','args':{'Model Id':45,'Stream Id':1,'Task Type':'MEMCPY_ASYNC'}}]),('unknown_similar_kind',[{'name':'MEMCPY_ASYNC_X','args':{**memnode['args'],'Task Type':'MEMCPY_ASYNC_X'}}]),('duplicate',[memnode,memnode])]:
 negative[label]={}
 for name,fun in [('helper',mods['helper'].graph_task_metadata),('validator',mods['validator'].native_tasks)]:
  try:fun(nodes);negative[label][name]='accepted'
  except (RuntimeError,ValueError):negative[label][name]='rejected'
 assert all(v==('accepted' if label=='valid_common_only' else 'rejected') for v in negative[label].values())
print(json.dumps(dict(minimal_extension_cpu=negative,raw=rows,warmup_detail_pass=len(warmok),capture_keys=sorted(warmok),meta_count=len(list((b/'graph_dump').glob('*.meta.json')))),indent=2))
