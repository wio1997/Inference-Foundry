#!/usr/bin/env python3
"""Bounded API-row representation and profiler source provenance inspection."""
import argparse,hashlib,json,re,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);prior=T/'agents/diagnosis/ep1-gap-host-links.json';rank='dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt';db=R/rank/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_1.db'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(prior)!='a422ed4011b0388515cd98e0d56582fbcd9433fc29fb6583234c1903be3f1f72' or sha(db)!='1d823cb13f49703ebd0a86ade38030ac8d6ef66b7085b23acf06f19ea42fa38b':raise RuntimeError('input mismatch')
x=json.loads(prior.read_text());pairs=[]
for op in (402,405,408):
 z=[r for r in x['targets'] if r['opId_local']==op and r['role']=='after']
 if len(z)!=1:raise RuntimeError('missing after anchor')
 z=z[0];ids=[q['connection_set_id'] for q in z['pytorch_via_sets'] if q['rows']]
 if len(ids)!=1:raise RuntimeError('ambiguous pytorch connection')
 pairs.append({'opId_local':op,'cann_conn':z['task']['connectionId'],'pytorch_conn':ids[0]})
c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');samples=[]
for p in pairs:
 crows=[dict(r) for r in c.execute('SELECT startNs,endNs,globalTid,connectionId FROM CANN_API WHERE connectionId=? LIMIT 3',(p['cann_conn'],))]
 prows=[dict(r) for r in c.execute('SELECT startNs,endNs,globalTid,connectionId FROM PYTORCH_API WHERE connectionId=? LIMIT 3',(p['pytorch_conn'],))]
 if len(crows)!=1 or len(prows)!=1:raise RuntimeError('anchor cardinality')
 for q in prows:
  q['start_decimal_integer_text']=isinstance(q['startNs'],str) and bool(re.fullmatch(r'[0-9]{1,30}',q['startNs']))
  q['end_decimal_integer_text']=isinstance(q['endNs'],str) and bool(re.fullmatch(r'[0-9]{1,30}',q['endNs']))
  if not q['start_decimal_integer_text']:q['startNs']='NON_NUMERIC_REDACTED'
  if not q['end_decimal_integer_text']:q['endNs']='NON_NUMERIC_REDACTED'
 samples.append({'opId_local':p['opId_local'],'cann':crows,'pytorch':prows})
c.close()
root=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')
dirs=[root/'prof_view/prof_db_parse',root/'prof_common_func',root/'prof_parse']
pat=re.compile(r'global.?tid|get.*tid|startNs|endNs|syscnt|timestamp|time.?offset|local.?time|monotonic|db_time|convert.*time',re.I)
files=[];total=0
for directory in dirs:
 if not directory.is_dir():files.append({'root':str(directory),'missing':True});continue
 for p in sorted(directory.rglob('*.py')):
  if p.is_symlink():continue
  total+=1
  if total>500:raise RuntimeError('source file bound exceeded')
  raw=p.read_bytes()
  if len(raw)>1048576:raise RuntimeError('source file too large')
  lines=raw.decode('utf8').splitlines();hits=[{'line':i,'text':line.strip()[:300]} for i,line in enumerate(lines,1) if pat.search(line)]
  if hits:files.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'hits':hits[:80],'truncated':len(hits)>80})
print(json.dumps({'scope':'Three already-associated PYTORCH_API rows and limited installed torch_npu profiler source roots; no request content','db_sha256':sha(db),'prior_sha256':sha(prior),'samples':samples,'source_files':files,'python_files_checked':total,'limits':['Source search is keyword-limited; omitted files, native binaries, and dynamic exporter versions may control semantics.','Numeric-looking TEXT timestamps do not by themselves establish units, conversion, common clock, or event order across tables.','globalTid equality or bit-pattern similarity does not by itself establish encoding, thread identity or cross-table semantics.','No host-device time subtraction, no request texts, no new NPU work.']},ensure_ascii=False,indent=2))
