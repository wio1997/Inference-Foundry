#!/usr/bin/env python3
"""Three exact long outer op rows; only metadata and allowlisted source paths."""
import argparse,hashlib,json,re,sqlite3
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--task-root',required=True);ap.add_argument('--trace-root',required=True);a=ap.parse_args()
T=Path(a.task_root);R=Path(a.trace_root);pr=T/'agents/diagnosis/ep1-fixed-queue-candidates.json';ne=T/'agents/diagnosis/ep1-queue-neighborhoods.json';rank='dp0_pp0_tp1_dcp0_ep1_rank1_744_20261028020238580_ascend_pt';db=R/rank/'ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_1.db'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for p,expected in [(pr,'761064a054d3146e30bc634162e89bf7312b54858bbe3d3e8cb3812cb2d51360'),(ne,'30810bf8927325678ad5a5ffd9ad909bb7519ff72bc1b5468096b0ed7ca47b87'),(db,'1d823cb13f49703ebd0a86ade38030ac8d6ef66b7085b23acf06f19ea42fa38b')]:
 if sha(p)!=expected:raise RuntimeError('input identity mismatch')
prior=json.loads(pr.read_text());neighbors=json.loads(ne.read_text());c=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON')
schema=[dict(z) for z in c.execute('PRAGMA table_info(PYTORCH_CALLCHAINS)')];out=[];target_hash='48c8a2e03124e213fefdf58913c3955cadd48c2249cff576e8b5e85e7a5fec1c'
for op in (402,405,408):
 q=[s for s in prior['samples'] if s['opId_local']==op];n=[s for s in neighbors['samples'] if s['opId_local']==op and s['anchor_kind']=='enqueue_end']
 if len(q)!=1 or len(n)!=1:raise RuntimeError('bad prior selection')
 en=[r for r in q[0]['inside_anchor_same_globalTid'] if r['type']==50002 and 'enqueue' in r['name_class']];outer=[r for r in n[0]['rows'] if r['name_sha256']==target_hash]
 if len(en)!=1 or len(outer)!=1:raise RuntimeError('ambiguous outer')
 anchor=int(en[0]['endNs']);start=anchor+round(outer[0]['start_relative_us']*1000);end=start+round(outer[0]['duration_us']*1000)
 matches=[dict(z) for z in c.execute('''SELECT a.startNs,a.endNs,a.globalTid,a.type,a.connectionId,a.callchainId,s.value AS name FROM PYTORCH_API a LEFT JOIN STRING_IDS s ON s.id=a.name WHERE a.globalTid=? AND a.startNs=? AND a.endNs=? AND a.type=50001 LIMIT 4''',(en[0]['globalTid'],str(start),str(end)))]
 matches=[z for z in matches if hashlib.sha256(str(z['name']).encode()).hexdigest()==target_hash]
 if len(matches)!=1:raise RuntimeError('outer DB match not unique')
 row=matches[0];name=str(row.pop('name'));low=name.lower();label=[v for v in ('vllm_ascend','vllm','torch_npu','torch','aten','forward','model','attention') if v in low]
 frames=[];stack_status='callchain_null' if row['callchainId'] is None else 'unsupported_schema'
 if row['callchainId'] is not None and len(schema)==3:
  # FwkApiDbParser saves [callchain id, string id, index]; verify all identifiers before SQL use.
  cols=[z['name'] for z in schema]
  if all(re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',z) for z in cols):
   sql='SELECT "'+cols[1]+'" AS frame_id,"'+cols[2]+'" AS frame_index FROM PYTORCH_CALLCHAINS WHERE "'+cols[0]+'"=? LIMIT 101'
   hits=[dict(z) for z in c.execute(sql,(row['callchainId'],)).fetchall()]
   if len(hits)>100:raise RuntimeError('callchain cap exceeded')
   stack_status='checked_allowlisted_paths'
   for hit in hits:
    values=c.execute('SELECT value FROM STRING_IDS WHERE id=? LIMIT 2',(hit['frame_id'],)).fetchall()
    if len(values)!=1:continue
    v=str(values[0][0]); allowed=[]
    for root in ('/vllm-workspace/vllm-ascend/','/vllm-workspace/vllm/','/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/','/usr/local/python3.12.13/lib/python3.12/site-packages/torch/'):
     allowed.extend(re.findall(re.escape(root)+r'[A-Za-z0-9_./-]+\.py(?=[:(\s]|$)',v))
    frames.append({'index':hit['frame_index'],'allowlisted_paths':sorted(set(allowed)),'frame_sha256':hashlib.sha256(v.encode()).hexdigest()})
 row['name_sha256']=target_hash;row['name_allowlist_classes']=label;row['stack_status']=stack_status;row['frames']=frames;out.append({'opId_local':op,'outer':row})
c.close();print(json.dumps({'scope':'Three frozen EP1 broad outer type=50001 rows only; no names or stack text','inputs':{'db':sha(db),'queue':sha(pr),'neighborhood':sha(ne)},'callchain_schema':[{'name':s['name'],'type':s['type']} for s in schema],'samples':out,'limits':['Name classes are substring checks, not proven model or function ownership.','Callchains and connections may be absent, truncated, asynchronous or lack coverage.','Matching parent duration is not CPU active time or evidence of bottleneck cause.','Only fixed allowlisted code root paths may appear; all other frame text is withheld.','No NPU work.']},ensure_ascii=False,indent=2))
