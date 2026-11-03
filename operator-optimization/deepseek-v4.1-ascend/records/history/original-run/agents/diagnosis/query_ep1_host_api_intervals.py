#!/usr/bin/env python3
"""Inspect same-column CANN thread IDs and three fixed host API intervals."""
import argparse,hashlib,json,sqlite3
from collections import Counter
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);prior=T/'agents/diagnosis/ep1-gap-host-links.json';rank='dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt';db=R/rank/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_1.db'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(prior)!='a422ed4011b0388515cd98e0d56582fbcd9433fc29fb6583234c1903be3f1f72' or sha(db)!='1d823cb13f49703ebd0a86ade38030ac8d6ef66b7085b23acf06f19ea42fa38b':raise RuntimeError('input mismatch')
x=json.loads(prior.read_text());c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
tables={t:[dict(r) for r in c.execute('PRAGMA table_info('+t+')')] for t in ('CANN_API','PYTORCH_API')}
can=[z['name'] for z in tables['CANN_API']];pyt=[z['name'] for z in tables['PYTORCH_API']]
thread_candidates=[k for k in ('globalTid','threadId','tid','globalThreadId') if k in can]
anchors=[]
for op in (402,405,408):
 targets={y['role']:y for y in x['targets'] if y['opId_local']==op}
 if not {'before','after'} <= set(targets):raise RuntimeError('missing boundary')
 item={'opId_local':op}
 for role in ('before','after'):
  target=targets[role];conn=target['task']['connectionId'];rows=[dict(z) for z in c.execute('SELECT * FROM CANN_API WHERE connectionId=? LIMIT 3',(conn,))]
  if len(rows)!=1:raise RuntimeError('anchor CANN not unique')
  if len(target['cann_same_connection'])!=1 or rows[0]['startNs']!=target['cann_same_connection'][0]['startNs'] or rows[0]['endNs']!=target['cann_same_connection'][0]['endNs']:raise RuntimeError('anchor changed')
  item[role]={'connectionId':conn,'startNs':rows[0]['startNs'],'endNs':rows[0]['endNs'],'thread_fields':{k:rows[0].get(k) for k in thread_candidates}}
 anchors.append(item)
eligible=[]
for key in thread_candidates:
 vals=[row[role]['thread_fields'][key] for row in anchors for role in ('before','after')]
 if all(v is not None for v in vals) and len(set(vals))==1:eligible.append(key)
chosen=eligible[0] if eligible else None
def union(rows,lo,hi):
 seg=sorted((max(lo,r['startNs']),min(hi,r['endNs'])) for r in rows if r['startNs']<hi and r['endNs']>lo and r['startNs']<r['endNs']);out=[]
 for a,b in seg:
  if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
  else:out.append([a,b])
 return sum(b-a for a,b in out)
outputs=[]
if chosen:
 value=anchors[0]['before']['thread_fields'][chosen]
 for item in anchors:
  lo=item['before']['endNs'];hi=item['after']['startNs']
  if hi<=lo:raise RuntimeError('invalid same-CANN interval')
  # chosen is from the fixed identifier whitelist; no untrusted SQL identifiers.
  sql='''SELECT a.startNs,a.endNs,a.connectionId,s.value AS name FROM CANN_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a."'''+chosen+'''"=? AND ((a.startNs<? AND a.endNs>?) OR (a.startNs>=? AND a.startNs<? AND a.endNs=a.startNs)) ORDER BY a.startNs LIMIT 5001'''
  rows=[dict(z) for z in c.execute(sql,(value,hi,lo,lo,hi))]
  if len(rows)>5000:raise RuntimeError('CANN row bound exceeded')
  cats=Counter();anon=[]
  for z in rows:
   name=str(z.pop('name',None));lower=name.lower();classes=[t for t in ('event','record','wait','notify','launch','kernel','memcpy','copy','synchronize','sync','alloc','free','stream','graph','model','task','callback','quant','matmul') if t in lower]
   for tag in classes:cats[tag]+=1
   anon.append({'start_relative_us':(z['startNs']-lo)/1000,'duration_us':(z['endNs']-z['startNs'])/1000,'connectionId':z['connectionId'],'name_sha256':hashlib.sha256(name.encode()).hexdigest(),'name_class':classes})
  outputs.append({'opId_local':item['opId_local'],'interval_us':(hi-lo)/1000,'row_count':len(rows),'same_field_api_union_us':union(rows,lo,hi)/1000,'class_counts_nonexclusive':dict(cats),'rows':anon})
c.close()
print(json.dumps({'scope':'Only three fixed EP1 CANN_API host intervals; names hashed and classified; no tensor/request text', 'db_sha256':sha(db),'prior_sha256':sha(prior),'table_columns':{k:[{'name':v['name'],'type':v['type']} for v in z] for k,z in tables.items()},'cann_thread_field_candidates':thread_candidates,'anchor_thread_fields':anchors,'selected_identical_nonnull_cann_field':chosen,'pytorch_thread_fields_present':[k for k in ('globalTid','threadId','tid','globalThreadId') if k in pyt],'intervals':outputs,'limits':['No API interval query if no non-null identical candidate thread field for all anchors.','Same column value is database grouping only until actual encoding and trace PID/TID semantics are verified; endpoint thread identity is conditional.','CANN_API intervals compare only values from this table, never host-device timestamps.','PYTORCH_API schema only: shared clock and identity versus CANN not established; no cross-table time overlap calculation.','API classes inferred by substrings; full names and input data omitted.','No matching API means no captured matching row, not CPU idle; uncovered activity and profiler blind spots remain.','No new NPU work.']},ensure_ascii=False,indent=2))
