#!/usr/bin/env python3
"""Six fixed same-PYTORCH_API neighborhoods around queue marker endpoints."""
import argparse,hashlib,json,re,sqlite3
from collections import Counter,defaultdict
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);args=ap.parse_args()
T=Path(args.task_root);R=Path(args.trace_root);prior=T/'agents/diagnosis/ep1-fixed-queue-candidates.json';rank='dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt';db=R/rank/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_1.db'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if sha(prior)!='761064a054d3146e30bc634162e89bf7312b54858bbe3d3e8cb3812cb2d51360' or sha(db)!='1d823cb13f49703ebd0a86ade38030ac8d6ef66b7085b23acf06f19ea42fa38b':raise RuntimeError('input mismatch')
x=json.loads(prior.read_text());c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
anchors=[]
for sample in x['samples']:
 en=[r for r in sample['inside_anchor_same_globalTid'] if r['type']==50002 and 'enqueue' in r['name_class']];de=sample['same_correlation_other_globalTid_candidates']
 if len(en)!=1 or len(de)!=1 or 'dequeue' not in de[0]['worker_candidate']['name_class']:raise RuntimeError('ambiguous queue anchors')
 anchors.extend([{'opId_local':sample['opId_local'],'kind':'enqueue_end','row':en[0],'field':'endNs'}, {'opId_local':sample['opId_local'],'kind':'dequeue_start','row':de[0]['worker_candidate'],'field':'startNs'}])
if len(anchors)!=6:raise RuntimeError('bad anchor count')
def merge(intervals,lo,hi):
 out=[]
 for a,b in sorted((max(lo,a),min(hi,b)) for a,b in intervals if a<hi and b>lo and a<b):
  if not out or a>out[-1][1]:out.append([a,b])
  else:out[-1][1]=max(out[-1][1],b)
 return out
def gapmax(seq,lo,hi):
 last=lo;vals=[]
 for a,b in seq:
  if a>last:vals.append(a-last)
  last=b
 if last<hi:vals.append(hi-last)
 return max(vals,default=0)
invalid={}
for tid in sorted({a['row']['globalTid'] for a in anchors}):
 invalid[str(tid)]=c.execute('''SELECT count(*) FROM PYTORCH_API WHERE globalTid=? AND (startNs IS NULL OR endNs IS NULL OR startNs='' OR endNs='' OR startNs GLOB '*[^0-9]*' OR endNs GLOB '*[^0-9]*' OR length(startNs)>18+1 OR length(endNs)>18+1 OR CAST(endNs AS INTEGER)<CAST(startNs AS INTEGER))''',(tid,)).fetchone()[0]
outputs=[]
for a in anchors:
 anchor=a['row'];raw=anchor[a['field']]
 if not isinstance(raw,str) or not re.fullmatch('[0-9]{1,19}',raw):raise RuntimeError('non-numeric anchor')
 mid=int(raw);lo=mid-100000;hi=mid+100000;tid=anchor['globalTid']
 rows=[dict(r) for r in c.execute('''SELECT a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name
 WHERE a.globalTid=? AND a.startNs IS NOT NULL AND a.endNs IS NOT NULL AND a.startNs<>'' AND a.endNs<>'' AND a.startNs NOT GLOB '*[^0-9]*' AND a.endNs NOT GLOB '*[^0-9]*' AND length(a.startNs)<=19 AND length(a.endNs)<=19 AND CAST(a.endNs AS INTEGER)>=CAST(a.startNs AS INTEGER)
 AND ((CAST(a.startNs AS INTEGER)<? AND CAST(a.endNs AS INTEGER)>?) OR (CAST(a.startNs AS INTEGER)>=? AND CAST(a.startNs AS INTEGER)<? AND CAST(a.endNs AS INTEGER)=CAST(a.startNs AS INTEGER))) ORDER BY CAST(a.startNs AS INTEGER) LIMIT 2001''',(tid,hi,lo,lo,hi)).fetchall()]
 if len(rows)>2000:raise RuntimeError('row limit exceeded')
 groups=defaultdict(list);counts=Counter();points=Counter();anonymous=[]
 for row in rows:
  s=int(row['startNs']);e=int(row['endNs']);typ=row['type'];groups[str(typ)].append((s,e));counts[str(typ)]+=1
  if s==e:points[str(typ)]+=1
  name=str(row['name']);lower=name.lower();classes=[v for v in ('enqueue','dequeue','queue') if v in lower]
  item={'type':typ,'start_relative_us':(s-mid)/1000,'duration_us':(e-s)/1000,'connectionId':row['connectionId'],'name_sha256':hashlib.sha256(name.encode()).hexdigest(),'name_class':classes}
  if typ==50002 and row['connectionId'] is not None:
   linked=[r[0] for r in c.execute('SELECT connectionId FROM CONNECTION_IDS WHERE id=? LIMIT 101',(row['connectionId'],)).fetchall()]
   if len(linked)>100:raise RuntimeError('correlation bound exceeded')
   item['correlation_ids']=linked
  anonymous.append(item)
 allunion=merge([(int(r['startNs']),int(r['endNs'])) for r in rows],lo,hi)
 outputs.append({'opId_local':a['opId_local'],'anchor_kind':a['kind'],'anchor_connection_id':anchor['connectionId'],'globalTid':tid,'radius_us':100,'window_duration_us':200,'row_count':len(rows),'all_api_union_us':sum(b-a for a,b in allunion)/1000,'largest_uncovered_us':gapmax(allunion,lo,hi)/1000,'by_type':[{'type':k,'row_count':counts[k],'zero_duration_points':points[k],'union_us':sum(b-a for a,b in merge(v,lo,hi))/1000} for k,v in sorted(groups.items())],'rows':anonymous})
c.close()
print(json.dumps({'scope':'Six 200us same-PYTORCH_API neighborhoods, selected local globalTid only; all API types and queue correlation IDs','inputs':{'prior_sha256':sha(prior),'db_sha256':sha(db)},'uninterpretable_time_rows_per_selected_tid':invalid,'samples':outputs,'limits':['100us radius is preselected; it is not a CANN window, request boundary or evidence of causality.','All arithmetic uses PYTORCH_API fields; source-version and same-config time interpretation and precision remain conditional.','API type unions may overlap and cannot be added; neighborhoods may overlap or repeat rows.','Uncovered exported intervals do not mean CPU idle, scheduling time or missing real tasks.','Invalid time rows cannot be placed in windows and are counted per selected globalTid; captured APIs may be incomplete.','Only names hashed and enqueue/dequeue substrings disclosed; no request data, no cross-table arithmetic, no NPU activity.']},ensure_ascii=False,indent=2))
