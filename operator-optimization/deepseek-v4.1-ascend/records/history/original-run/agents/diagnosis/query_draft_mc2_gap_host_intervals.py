#!/usr/bin/env python3
"""Same-table same-globalTid intervals between six fixed stage190 host candidates."""
import argparse,base64,hashlib,json,sqlite3
from collections import Counter,defaultdict
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--trace-root',required=True);ap.add_argument('--prior-b64',required=True);a=ap.parse_args()
R=Path(a.trace_root);rank='dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt';db=R/rank/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_1.db'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
raw=base64.b64decode(a.prior_b64,validate=True);ph=hashlib.sha256(raw).hexdigest();dh=sha(db)
if ph!='2b07456e853931f2764b494ccc5e315ad6ed9af5745cfa87f0b2da19940b157e' or dh!='1d823cb13f49703ebd0a86ade38030ac8d6ef66b7085b23acf06f19ea42fa38b':raise RuntimeError('prior/DB mismatch')
x=json.loads(raw);c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
def merge(intervals,lo,hi):
 out=[]
 for a,b in sorted((max(lo,a),min(hi,b)) for a,b in intervals if a<hi and b>lo and a<b):
  if not out or a>out[-1][1]:out.append([a,b])
  else:out[-1][1]=max(out[-1][1],b)
 return sum(b-a for a,b in out)
def get_pyt(z):
 p=[r for group in z['pytorch_via_sets'] for r in group['rows']]
 if len(p)!=1:raise RuntimeError('unexpected pytorch anchor count')
 return p[0]
def get_cann(z):
 p=z['cann_same_connection']
 if len(p)!=1:raise RuntimeError('unexpected CANN anchor count')
 return p[0]
invalid={};samples=[]
for op in (402,405,408):
 a0=next(z for z in x['targets'] if z['opId_local']==op and z['role']=='gating');b0=next(z for z in x['targets'] if z['opId_local']==op and z['role']=='mc2')
 for table,get in [('CANN_API',get_cann),('PYTORCH_API',get_pyt)]:
  before=get(a0);after=get(b0);tid=before['globalTid']
  if tid is None or tid!=after['globalTid']:raise RuntimeError('thread mismatch')
  lo=int(before['endNs']);hi=int(after['startNs'])
  if not 0<hi-lo<1000000:raise RuntimeError('interval bound')
  # Exact candidate anchors still exist in the hash-verified same table.
  for z in (before,after):
   n=c.execute('SELECT count(*) FROM '+table+' WHERE globalTid=? AND connectionId=? AND startNs=? AND endNs=?',(tid,z['connectionId'],z['startNs'],z['endNs'])).fetchone()[0]
   if n!=1:raise RuntimeError('anchor count mismatch')
  # PYTORCH_API timestamps are decimal text; malformed rows cannot be located.
  key=table+':'+str(tid)
  if key not in invalid:
   invalid[key]=c.execute('''SELECT count(*) FROM '''+table+''' WHERE globalTid=? AND (startNs IS NULL OR endNs IS NULL OR CAST(startNs AS TEXT)='' OR CAST(endNs AS TEXT)='' OR CAST(startNs AS TEXT) GLOB '*[^0-9]*' OR CAST(endNs AS TEXT) GLOB '*[^0-9]*' OR length(CAST(startNs AS TEXT))>19 OR length(CAST(endNs AS TEXT))>19 OR CAST(endNs AS INTEGER)<CAST(startNs AS INTEGER))''',(tid,)).fetchone()[0]
  typecol='a.type'
  sql='''SELECT a.startNs,a.endNs,a.connectionId,'''+typecol+''' AS type,s.value AS name FROM '''+table+''' a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.globalTid=? AND a.startNs IS NOT NULL AND a.endNs IS NOT NULL AND CAST(a.startNs AS TEXT)<>'' AND CAST(a.endNs AS TEXT)<>'' AND CAST(a.startNs AS TEXT) NOT GLOB '*[^0-9]*' AND CAST(a.endNs AS TEXT) NOT GLOB '*[^0-9]*' AND length(CAST(a.startNs AS TEXT))<=19 AND length(CAST(a.endNs AS TEXT))<=19 AND CAST(a.endNs AS INTEGER)>=CAST(a.startNs AS INTEGER) AND ((CAST(a.startNs AS INTEGER)<? AND CAST(a.endNs AS INTEGER)>?) OR (CAST(a.startNs AS INTEGER)>=? AND CAST(a.startNs AS INTEGER)<? AND CAST(a.endNs AS INTEGER)=CAST(a.startNs AS INTEGER))) ORDER BY CAST(a.startNs AS INTEGER) LIMIT 5001'''
  rows=[dict(r) for r in c.execute(sql,(tid,hi,lo,lo,hi)).fetchall()]
  if len(rows)>5000:raise RuntimeError('row bound exceeded')
  anonym=[];groups=defaultdict(list);counts=Counter();allseg=[];without=[]
  for row in rows:
   s=int(row['startNs']);e=int(row['endNs']);name=str(row['name']);low=name.lower();tags=[v for v in ('enqueue','dequeue','queue','launch','kernel','event','record','wait','notify','memcpy','copy','synchronize','sync','alloc','free','stream','graph','model','task','callback','matmul','quant','moe','gating','dispatch','combine','forward') if v in low]
   typ=str(row['type'])
   outer=s<=lo and e>=hi;groups[typ].append((s,e));counts[typ]+=1;allseg.append((s,e))
   if not outer:without.append((s,e))
   anonym.append({'start_relative_ns':s-lo,'duration_ns':e-s,'connectionId':row['connectionId'],'type':row['type'],'name_sha256':hashlib.sha256(name.encode()).hexdigest(),'name_class':tags,'covers_entire_interval':outer})
  samples.append({'opId_local':op,'table':table,'globalTid':tid,'before_connection':before['connectionId'],'after_connection':after['connectionId'],'interval_start_ns':lo,'interval_end_ns':hi,'interval_ns':hi-lo,'row_count':len(rows),'outer_row_count':sum(z['covers_entire_interval'] for z in anonym),'union_ns':merge(allseg,lo,hi),'union_excluding_covering_rows_ns':merge(without,lo,hi),'by_type':[{'type':k,'count':counts[k],'union_ns':merge(v,lo,hi)} for k,v in sorted(groups.items())],'rows':anonym})
c.close()
print(json.dumps({'scope':'EP1 three stage190 pairs, same-table same-globalTid host intervals only','input_sha256':{'stage190':ph,'db':dh},'uninterpretable_time_rows_per_table_tid':invalid,'samples':samples,'limits':['Only same-table endpoint arithmetic; no host-device or cross-table time subtraction.','globalTid equality is database grouping, thread encoding and exporter provenance remain conditional.','One query per fixed interval, all API categories, max 5000 rows; invalid timestamp rows cannot be placed.','Rows spanning entire interval are called outer for this analysis only; excluding them is not measuring exclusive execution.','Union coverage is API-record coverage, neither CPU utilization nor proof of idle, causality, queue time or actual scheduler time.','API name classes are substrings, not official types; overlaps are not additive; API/other activities may not be collected.','No payload or tensor data returned, no NPU activity.']},ensure_ascii=False,indent=2))
