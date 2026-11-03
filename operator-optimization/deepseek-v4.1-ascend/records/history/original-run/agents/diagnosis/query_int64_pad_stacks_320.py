#!/usr/bin/env python3
"""Bind 320 saved Pad task host starts to profiler call stacks, read-only."""
import argparse,hashlib,json,sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);a=ap.parse_args();root=Path(a.task_root)
p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl';h='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4';assert hashlib.sha256(p.read_bytes()).hexdigest()==h
want=('prepare_finalize.py(338): pad_and_split_input_ids','moe_comm_method.py(327): pad_and_split_input_ids','fused_topk_router.py(169): _compute_routing','model.py(1592): forward')
ranks=[]
with p.open() as f:
 for line in f:
  x=json.loads(line)
  if x.get('kind')!='rank' or x['data']['ep'] not in range(8):continue
  data=x['data'];ep=data['ep'];assert data['status']=='completed';rank=data['rank'];assert '/' not in rank and '..' not in rank
  layers=[l for l in data['layers'] if l['scheduler_digest'].startswith('7ddf7ae98a00')];assert len(layers)==40 and {l['layer'] for l in layers}==set(range(40))
  entries=[]
  for l in layers:
   gs=[g for g in l['operator_groups'] if g['operator_name']=='aclnnConstantPadNd_PadV3AiCpu_PadV3'];assert len(gs)==1 and len(gs[0]['tasks'])==1
   t=gs[0]['tasks'][0];ns=Decimal(t['host_start_us'])*1000;assert ns==int(ns)
   tid=(int(t['host_pid'])<<32)|int(t['host_tid']);entries.append((str(int(ns)),tid,t['flow_id'],t['csv']['Task ID'],l['layer']))
  assert len(set((x[0],x[1]) for x in entries))==40 and len({x[1] for x in entries})==1
  db=root/'diagnostic-20261029/runtime/diagnostic-traces'/rank/'ASCEND_PROFILER_OUTPUT'/f'ascend_pytorch_profiler_{ep}.db'
  c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA automatic_index=OFF')
  marks=','.join('?' for _ in entries)
  sql='SELECT a.startNs,a.globalTid,a.connectionId,a.callchainId,s.value FROM PYTORCH_API a JOIN STRING_IDS s ON s.id=a.name WHERE a.startNs IN ('+marks+') AND a.globalTid=? AND s.value=?'
  apis=c.execute(sql,tuple(x[0] for x in entries)+(entries[0][1],'aclnnConstantPadNd')).fetchall()
  by=defaultdict(list)
  for r in apis:by[(str(r[0]),r[1])].append(r)
  assert all(len(by[(e[0],e[1])])==1 for e in entries)
  callids=[by[(e[0],e[1])][0][3] for e in entries];assert None not in callids and len(set(callids))==40
  marks=','.join('?' for _ in callids)
  frames=c.execute('SELECT pc.id,s.value FROM PYTORCH_CALLCHAINS pc LEFT JOIN STRING_IDS s ON s.id=pc.stack WHERE pc.id IN ('+marks+')',callids).fetchall();c.close()
  stacks=defaultdict(list)
  for cid,frame in frames:
   if frame:stacks[cid].append(frame)
  matches=sum(all(any(n in frame for frame in stacks[cid]) for n in want) for cid in callids)
  assert matches==40
  ranks.append({'ep':ep,'layer_count':40,'unique_host_api_count':40,'unique_callchain_count':len(set(callids)),'required_stack_count':matches})
assert sorted(x['ep'] for x in ranks)==list(range(8))
print(json.dumps({'input_sha256':h,'scope':'one saved profiled digest; 8 rank DBs; 40 selected target-layer records per rank','selected':320,'uniquely_matched':sum(x['unique_host_api_count'] for x in ranks),'required_stack_matches':sum(x['required_stack_count'] for x in ranks),'rank_summaries':sorted(ranks,key=lambda x:x['ep']),'limitations':['Host event matched via exact thread/start; existing source extractor establishes host-flow to device task','No tensor values or storage identity, no historical binary/source SHA, no cross-clock subtraction','No inference about the other 960 scopes or online savings']},indent=2))
