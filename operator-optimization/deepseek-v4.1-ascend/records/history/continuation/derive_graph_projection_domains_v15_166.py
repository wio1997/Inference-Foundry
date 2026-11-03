from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14')
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
idx=pin(P/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');g=pin(P/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
OUT=P/'graph-projection-domains-v15.json';assert not OUT.exists();rows=[]
def shapes(s):
 try:return [tuple(int(x.strip())for x in p.split(','))for p in s.strip('"').split(';')if p]
 except (ValueError,AttributeError):return []
for ref in idx['databases']:
 d=pin(Path(ref['path']),ref['sha256']);di=d['database_index'];tm={t['task_record_rowid']:t for t in d['TASK_records']};defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for graph in [v for v in g['graph_executions']if v['database_index']==di]:
  for block in graph['blocks']:
   candidates={'Q_A':[],'KV':[]}
   for i in block['task_record_rowids']:
    t=tm[i];c=defs[t['globalTaskId']];ss=shapes(c['inputShapes_text'])
    if c['opType_text']!='MatMulV2'or c['inputDataTypes_text']!='DT_BF16;DT_BF16'or len(ss)!=2:continue
    for role,w in [('Q_A',(1280,5120)),('KV',(512,5120))]:
     if ss[1]==w:candidates[role].append(dict(task_record_rowid=i,globalTaskId=t['globalTaskId'],input_shapes=c['inputShapes_text'],output_shapes=c['outputShapes_text'],M=ss[0][0]))
   rows.append(dict(database_index=di,graph_ordinal=graph['graph_ordinal'],block_ordinal=block['block_ordinal'],candidates=candidates))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),marker_blocks=len(rows),unique_Q_and_KV=sum(len(r['candidates']['Q_A'])==len(r['candidates']['KV'])==1 for r in rows),Q_M_counts=dict(Counter(c['M']for r in rows for c in r['candidates']['Q_A'])),KV_M_counts=dict(Counter(c['M']for r in rows for c in r['candidates']['KV'])),same_M_pairs=sum(len(r['candidates']['Q_A'])==len(r['candidates']['KV'])==1 and r['candidates']['Q_A'][0]['M']==r['candidates']['KV'][0]['M']for r in rows),no_source_DB_or_device_work=True,overall_complete=False,
 limits=['Roles are derived from unique per-marker-block op/dtype/weight shapes matching source-bound eager Q_A/KV interfaces; not tensor pointer identity or all-consumer proof.','Unequal M excludes a same-row direct concatenation preserving current row domains; does not exclude repartitioned or ragged joint algorithms.'])
b=(json.dumps(dict(summary=summary,records=rows),separators=(',',':'))+'\n').encode();OUT.write_bytes(b);print(json.dumps(dict(summary=summary,first_record=rows[:1],full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest())))
