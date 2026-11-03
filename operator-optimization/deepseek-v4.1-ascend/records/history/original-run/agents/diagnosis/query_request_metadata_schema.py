#!/usr/bin/env python3
"""Read-only inventory of possible request/phase/dynamic-value metadata, EP8 DB."""
import argparse,json,re,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--db',required=True);a=ap.parse_args()
db=Path(a.db).resolve(strict=True);c=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.row_factory=sqlite3.Row
tables=[r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
def quote(x):return '"'+x.replace('"','""')+'"'
schema={}
for t in tables:
 names=[r['name'] for r in c.execute('PRAGMA table_info('+quote(t)+')')]
 if re.search(r'request|phase|step|sequence|length|metadata|tensor',t,re.I) or any(re.search(r'request|phase|seq.*len|candidate.*len|query.*len|tensor.*value',n,re.I) for n in names):schema[t]=names
terms=['request_id','requestid','candidate_lengths','cache_seq_lens','query_start_loc','num_actual_tokens']
matches={}
for term in terms:
 rows=c.execute('SELECT value FROM STRING_IDS WHERE lower(value) LIKE ? AND length(value)<300 LIMIT 21',('%'+term.lower()+'%',)).fetchall()
 matches[term]={'values':[r[0] for r in rows[:20]],'truncated':len(rows)>20}
print(json.dumps({'database':str(db),'candidate_table_schemas':schema,'candidate_string_names':matches,
 'limits':['Table and string name existence does not establish tensor values, dynamic arguments, request mapping, or per-run phase.',
 'This is an inventory of one rank database; other trace exports may contain additional evidence.',
 'Values of user prompts and request payloads are not queried.']},ensure_ascii=False,indent=2))
c.close()
