#!/usr/bin/env python3
"""Inspect indexed EP7 profiler metadata and exact label candidates, never request text."""
import argparse,hashlib,json,re,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--db',required=True);ap.add_argument('--info',required=True);a=ap.parse_args()
T=Path(a.task_root);db=Path(a.db).resolve(strict=True);info=Path(a.info).resolve(strict=True);prior=T/'agents/diagnosis/draft-query-metadata-ep7.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
expected={db:'8b8e72a61e2521f0d91050bc53ab852ddf26dd4e40e2d5ec5d3973652fd807c6',info:'74cde5c6e1ece98eee3b06733d40eee2e5ee300b9d8a4f0c07443b39346a738d',prior:'91bd06a12087247de6265ae5c5db915af8f86973d9ff78089b22e973b9545cf0'}
for p,h in expected.items():
 if sha(p)!=h:raise RuntimeError('input hash mismatch: '+p.name)
windows=[x['window'] for x in json.loads(prior.read_text())['outputs']]
if len(windows)!=2:raise RuntimeError('expected two windows')
terms=('num_actual_tokens','cp_token_range','query_metadata','cache_metadata','AscendDSAV41CPImpl','AscendDSAV41PCPImpl','AscendDSAV41Impl','batch_id')
termkey=tuple(x.lower() for x in terms)
allowedclasses=set(terms[4:7]);metadata_fields=[]
def walk(obj,path=''):
 if isinstance(obj,dict):
  for k,v in obj.items():
   if not isinstance(k,str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}',k):continue
   q=(path+'.' if path else '')+k
   if any(t in k.lower() for t in termkey):
    safe=v if (type(v) is int and k.lower() in ('num_actual_tokens',)) or (isinstance(v,str) and v in allowedclasses) else None
    metadata_fields.append({'path':q,'value_type':type(v).__name__,'safe_value':safe})
   walk(v,q)
 elif isinstance(obj,list):
  for i,v in enumerate(obj[:100]):walk(v,path+'['+str(i)+']')
walk(json.loads(info.read_text()))
c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
tables=[r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")]
schemas=[]
for t in tables:
 for col in c.execute('PRAGMA table_info("'+t.replace('"','""')+'")'):
  if any(k in str(col['name']).lower() for k in termkey):schemas.append({'table':t,'column':col['name']})
labels=[]
for term in terms:
 matching=c.execute('SELECT id FROM STRING_IDS WHERE instr(lower(value),lower(?))>0 ORDER BY id LIMIT 101',(term,)).fetchall()
 ids=[int(r['id']) for r in matching[:100]];per=[]
 for w in windows:
  count=0
  if ids:
   placeholders=','.join('?'*len(ids))
   count=c.execute('SELECT COUNT(*) FROM PYTORCH_API WHERE name IN ('+placeholders+') AND CAST(startNs AS INTEGER)>=? AND CAST(startNs AS INTEGER)<?',(*ids,w['start_ns'],w['attention_end_ns'])).fetchone()[0]
  per.append({'window':w['label'],'labelled_api_rows_in_nominal_numeric_interval':count})
 labels.append({'term':term,'matched_string_id_count_at_least':len(ids),'string_ids_truncated':len(matching)>100,'windows':per})
meta_match_counts={term:0 for term in terms};meta_row_count=0
if 'META_DATA' in tables:
 for r in c.execute('SELECT name FROM META_DATA'):
  meta_row_count+=1
  name=r['name']
  if isinstance(name,str):
   for term in terms:
    if term.lower() in name.lower():meta_match_counts[term]+=1
print(json.dumps({'scope':'Only frozen EP7 DB/info and two prior selected windows; explicit schema and eight label terms',
 'input_sha256':{p.name:h for p,h in expected.items()},'profiler_info_matching_fields':metadata_fields,
 'matching_schema_columns':schemas,'meta_data_row_count':meta_row_count,'meta_data_fixed_term_counts':meta_match_counts,'labels':labels,
 'limits':['Matching labels are inventory candidates; neither label nor time coincidence proves same batch, request, backend object or execution branch.',
 'Host API timestamps are only a numeric search interval here; cross-domain clock equivalence and error have not been proven.',
 'No matching named fields does not rule out differently encoded, unindexed or other captured data.',
 'Profiler info JSON is session-level, not per-call; array inspection is capped at 100 items.',
 'Only fixed label terms and safe profiler metadata field names/anonymous numeric values are returned; no request content.']},ensure_ascii=False,indent=2))
c.close()
