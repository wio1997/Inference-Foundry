#!/usr/bin/env python3
"""One bounded scan for enclosing framework records around five EP8 candidates."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--db',required=True);ap.add_argument('--prior-json',required=True);a=ap.parse_args()
dbsha='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58'; prior='e257d11a0bf04ca9e68eab6a7bbdadccd2c0e4b8ba6c4cae4d668b834e15125a'
p=Path(a.db).resolve(strict=True);h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
if h.hexdigest()!=dbsha or hashlib.sha256(a.prior_json.encode()).hexdigest()!=prior:raise RuntimeError('source hash mismatch')
x=json.loads(a.prior_json); wanted=[(52765,128634),(52769,128636),(52770,128637),(52772,128638),(52776,128640)];samples=[]
if x['status']!='complete' or x['all_rows_truncated'] or len(x['samples'])!=5:raise RuntimeError('prior incomplete')
for task,group in wanted:
 z=[v for v in x['samples'] if v['task']==task and v['connection_set_id']==group]
 if len(z)!=1 or z[0]['match_count_shown']!=1:raise RuntimeError('prior candidate not unique')
 r=z[0]['matches'][0]
 if not r['matches_prior_name'] or not r['numeric_time'] or r['global_tid']!=3122441224919 or r['type']!=50001:raise RuntimeError('prior candidate changed')
 s,e=int(r['start_ns']),int(r['end_ns'])
 if s>=e:raise RuntimeError('invalid timestamp')
 samples.append((task,s,e))
paths=['/vllm_ascend/spec_decode/llm_base_proposer.py','/vllm_ascend/spec_decode/dspark_proposer.py','/vllm/model_executor/models/qwen3_dspark.py','/vllm/models/deepseek_v41/nvidia/dspark.py','/vllm/v1/spec_decode/llm_base_proposer.py']
funcs=['compute_draft_token_ids','_sample_draft_from_logits','markov_embed','markov_bias']
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
conditions=[];args=[3122441224919]
for _,s,e in samples:
 conditions.append('(CAST(a.startNs AS INTEGER)<=? AND CAST(a.endNs AS INTEGER)>=? AND (CAST(a.startNs AS INTEGER)<? OR CAST(a.endNs AS INTEGER)>?))');args.extend((s,e,s,e))
sql='''SELECT a.startNs,a.endNs,a.type,a.callchainId,n.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS n ON n.id=a.name WHERE a.globalTid=? AND a.startNs<>'' AND a.endNs<>'' AND a.startNs NOT GLOB '*[^0-9]*' AND a.endNs NOT GLOB '*[^0-9]*' AND ('''+ ' OR '.join(conditions)+''') ORDER BY a.rowid LIMIT 101'''
ticks=[0]
def budget():
 ticks[0]+=1
 return int(ticks[0]>2000)  # maximum roughly 20 million SQLite VM instructions
c.set_progress_handler(budget,10000)
try: rows=c.execute(sql,args).fetchall();status='complete'
except sqlite3.OperationalError:rows=[];status='interrupted_or_sqlite_error'
c.set_progress_handler(None,0);out=[]
for row in rows[:100]:
 s,e=int(row['startNs']),int(row['endNs']);name=row['name']
 applies=[task for task,ts,te in samples if s<=ts and e>=te and (s<ts or e>te)]
 out.append({'encloses_tasks':applies,'type':row['type'],'name_sha256':hashlib.sha256(str(name).encode()).hexdigest() if name is not None else None,'known_path_mentions':[v for v in paths if isinstance(name,str) and v in name],'known_function_mentions':[v for v in funcs if isinstance(name,str) and v in name],'callchain_present':row['callchainId'] is not None})
c.close();print(json.dumps({'scope':'one bounded EP8 same-thread enclosing API scan for five fixed candidates','db_sha256':dbsha,'prior_sha256':prior,'status':status,'candidate_tasks':[t for t,_,_ in samples],'shown_count':len(out),'truncated':len(rows)>100,'outer':out,'limits':['Same-table same-thread temporal containment is a candidate relation; it does not prove Python caller, request, tensor, source deployment, or causal device dependencies.','Missing/truncated trace or interrupted query is not proof that calls were absent.','Fixed path and function mentions are substring categories only; other names are hashed.','No user payload, token values, cross-domain time subtraction, target edits, or NPU execution.']},ensure_ascii=False,indent=2))
