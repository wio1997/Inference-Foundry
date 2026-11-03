#!/usr/bin/env python3
"""Retrieve only fixed EP8 candidate types and same-table time/thread fields."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--db',required=True);ap.add_argument('--prior-json',required=True);a=ap.parse_args()
dbsha='67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58';prior='8eab586ca22e79ddce8b88a5b304ba54d0f18a4aca840f0979e584b9e6e01007'
p=Path(a.db).resolve(strict=True);h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
if h.hexdigest()!=dbsha or hashlib.sha256(a.prior_json.encode()).hexdigest()!=prior:raise RuntimeError('source hash mismatch')
x=json.loads(a.prior_json); targets=[(52765,128634),(52769,128636),(52770,128637),(52772,128638),(52776,128640)];expected={}
for task,group in targets:
 rr=[r for r in x['rows'] if r['target']['ordinal']==6 and r['target']['task']==task]
 if len(rr)!=1:raise RuntimeError('target not unique')
 gg=[g for l in rr[0]['links'] for g in l['pytorch_via_sets'] if g['connection_set_id']==group and g['candidate_count_shown']==1 and not g['truncated']]
 if len(gg)!=1:raise RuntimeError('candidate not unique')
 expected[group]=gg[0]['candidates'][0]['other_name_sha256']
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
ticks=[0]
def budget():
 ticks[0]+=1
 return int(ticks[0]>2000)
c.set_progress_handler(budget,10000)
try:
 rows=c.execute('SELECT a.connectionId,a.startNs,a.endNs,a.globalTid,a.type,a.callchainId,s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.connectionId IN (?,?,?,?,?) LIMIT 16',tuple(g for _,g in targets)).fetchall(); status='complete'
except sqlite3.OperationalError: rows=[];status='interrupted_or_sqlite_error'
c.set_progress_handler(None,0);out=[]
for task,g in targets:
 rr=[r for r in rows if r['connectionId']==g];matches=[]
 for r in rr:
  nh=hashlib.sha256(str(r['name']).encode()).hexdigest() if r['name'] is not None else None
  numeric=str(r['startNs']).isdigit() and str(r['endNs']).isdigit()
  matches.append({'name_sha256':nh,'matches_prior_name':nh==expected[g],'type':r['type'],'global_tid':r['globalTid'],'start_ns':r['startNs'] if numeric else None,'end_ns':r['endNs'] if numeric else None,'numeric_time':numeric,'callchain_present':r['callchainId'] is not None})
 out.append({'task':task,'connection_set_id':g,'match_count_shown':len(matches),'matches':matches})
c.close();print(json.dumps({'scope':'five fixed EP8 candidate groups only; no enclosing-record scan','db_sha256':dbsha,'prior_sha256':prior,'status':status,'all_rows_truncated':len(rows)>15,'samples':out,'limits':['Type and same-table timing/thread fields do not prove Python DSpark caller or request.','Unmatched or aborted query is not evidence of absent call.','No raw names, payloads, tokens, cross-domain time calculations, NPU work, or target-code edits.']},ensure_ascii=False,indent=2))
