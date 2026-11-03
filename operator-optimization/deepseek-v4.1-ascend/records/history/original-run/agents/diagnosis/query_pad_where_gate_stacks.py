#!/usr/bin/env python3
"""Bind selected Eq/Fill/Select/gate flows to saved host callchains; no NPU."""
import argparse,hashlib,json,sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);a=ap.parse_args();root=Path(a.task_root)
inputs=[('diagnostic-20261029/target-layer-all-kernels-16rank.jsonl','0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'),('agents/diagnosis/int64-pad-gating-scope.json','49e4028342df962d0724c7f95aac4537b4804f7227049ffa42bf22040829c218'),('agents/diagnosis/pad-where-gap.json','822f20c03e7711793e4e196d36437d5f75b3523404b2879be97b0280371fd446')]
for name,sha in inputs:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
pads={(r['ep'],r['scope']):r for r in json.loads((root/inputs[1][0]).read_text())['rows']}; mids={(r['ep'],r['scope']):r for r in json.loads((root/inputs[2][0]).read_text())['rows']};assert len(pads)==len(mids)==320 and set(pads)==set(mids)
api_map={'aclnnEqScalar_EqualAiCore_Equal':('aclnnEqScalar','fused_topk_router.py(176): _compute_routing'),'aclnnInplaceFillScalar_FillAiCore_Fill':('aclnnInplaceFillScalar','fused_topk_router.py(176): _compute_routing'),'aclnnSWhere_SelectV2AiCore_SelectV2':('aclnnSWhere','fused_topk_router.py(176): _compute_routing'),'MoeGatingTopKHash':('aclnnMoeGatingTopKHash','fused_topk_router.py(204): _compute_routing')}
summary=defaultdict(lambda:defaultdict(int));selected=defaultdict(lambda:defaultdict(int));ranks=[]
with (root/inputs[0][0]).open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('kind')!='rank' or x['data']['ep'] not in range(8):continue
  d=x['data'];ep=d['ep'];rank=d['rank'];assert d['status']=='completed' and '/' not in rank and '..' not in rank
  entries=[]
  for l in d['layers']:
   key=(ep,l['layer_scope_id'])
   if key not in pads:continue
   r=pads[key];m=mids[key];wanted=[(t['name'],t['task']) for t in m['tasks']]+[('MoeGatingTopKHash',r['gate']['task'])]
   assert len(wanted)==4
   for opname,taskid in wanted:
    gs=[g for g in l['operator_groups'] if g['operator_name']==opname];assert len(gs)==1
    ts=[t for t in gs[0]['tasks'] if t['csv']['Task ID']==taskid and t['csv']['Stream ID']==r['pad']['stream']];assert len(ts)==1
    t=ts[0];ns=Decimal(t['host_start_us'])*1000;assert ns==int(ns)
    tid=(int(t['host_pid'])<<32)|int(t['host_tid']);api,frame=api_map[opname];entries.append((str(int(ns)),tid,api,frame,opname))
  assert len(entries)==160 and len({(e[0],e[1],e[2]) for e in entries})==160 and len({e[1] for e in entries})==1
  db=root/'diagnostic-20261029/runtime/diagnostic-traces'/rank/'ASCEND_PROFILER_OUTPUT'/f'ascend_pytorch_profiler_{ep}.db';c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA automatic_index=OFF')
  qs=','.join('?' for _ in entries);sql='SELECT a.startNs,a.globalTid,a.callchainId,s.value FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name WHERE a.startNs IN ('+qs+') AND a.globalTid=?'
  rows=c.execute(sql,tuple(e[0] for e in entries)+(entries[0][1],)).fetchall();by=defaultdict(list)
  for ns,tid,cid,name in rows:by[(str(ns),tid,name)].append(cid)
  ids=[]
  for ns,tid,api,frame,op in entries:
   selected[op]['selected']+=1;hits=by[(ns,tid,api)]
   if len(hits)!=1 or hits[0] is None:summary[op]['missing_or_ambiguous']+=1;continue
   summary[op]['unique_api']+=1;ids.append((hits[0],frame,op))
  qs=','.join('?' for _ in ids);frames=c.execute('SELECT pc.id,s.value FROM PYTORCH_CALLCHAINS pc LEFT JOIN STRING_IDS s ON s.id=pc.stack WHERE pc.id IN ('+qs+')',tuple(x[0] for x in ids)).fetchall();c.close();st=defaultdict(list)
  for cid,frame in frames:
   if frame:st[cid].append(frame)
  for cid,required,op in ids:
   summary[op]['required_stack' if any(required in frame for frame in st[cid]) else 'stack_missing']+=1
  ranks.append(ep)
assert sorted(ranks)==list(range(8))
for op in api_map:assert selected[op]['selected']==320
print(json.dumps({'scope':'one digest; eight rank databases; selected 320 target scopes','counts':{op:{**selected[op],**summary[op]} for op in api_map},'limitations':['Exact host start/thread/name matched to pre-associated device flow','No tensor identity or history binary/source hash proof','Does not identify other scopes, prove critical path, or compare host/device clocks']},indent=2))
