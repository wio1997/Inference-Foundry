from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import subprocess, json, hashlib, socket

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
M=P/'native-PMU-complete-v18';E=P/'native-execution-identity-v14'
OUT=M/'export-route-localization-and-graph-attribute-reuse-v23.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
a=pin(M/'saved-COMPUTE-attribute-field-census-v22.json','46e59d2de1773f4e8f9fd0e717b5ae68a99094644986d1613a8cc8b9caf6f147')
gr=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
ar={(d['database_index'],r['globalTaskId']):r for d in a['databases']for r in d['attribute_records']}
ds={d['database_index']:pin(Path(d['path']),d['sha256'])for d in ix['databases']}
groups=Counter();byop=Counter();examples=[];total=0
for graph in gr['graph_executions']:
 di=graph['database_index'];d=ds[di];ts={t['TASK_rowid']:t for t in d['TASK_records']};cs={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 # Retain exact graph TASK order already recorded in the marker partition.
 rows=sorted([ts[k]for b in graph['blocks']for k in b['task_record_rowids']]+[ts[k]for k in graph['outside_marker_tasks']],key=lambda t:(t['startNs'],t['endNs'],t['TASK_rowid']))
 assert len(rows)==1650
 previous=None
 for t in rows:
  cur=ar[(di,t['globalTaskId'])];op=cs[t['globalTaskId']]['opType_text'];total+=1;byop[(op,cur['lookup_state'])]+=1
  if previous is not None:
   prev,prevop,prevt=previous
   if cur['lookup_state']=='NONEMPTY_STRING_CANDIDATE'and cur['raw_attrInfo']==prev['raw_attrInfo']:
    groups[(prevop,op)]+=1
    if len(examples)<8 and prevop!=op:examples.append(dict(database_index=di,previous_TASK=prevt['TASK_rowid'],TASK=t['TASK_rowid'],previous_op=prevop,op=op,raw_attrInfo=cur['raw_attrInfo'],raw_text=cur['string_table_value_at_exact_id']))
  previous=(cur,op,t)
assert total==132000
code=r'''from pathlib import Path
import json,hashlib,ast
B=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis')
paths=sorted(p for p in B.rglob('*.py')if 'export'in p.name.lower())
assert len(paths)<=12
out=[]
for p in paths:
 b=p.read_bytes();ls=b.decode().splitlines();hits=[i+1 for i,l in enumerate(ls)if any(k in l for k in ['msprof','--type','--export','subprocess','export_db','export_type','db_export'])]
 funcs=[]
 for node in ast.walk(ast.parse(b.decode())):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and any(node.lineno<=i<=node.end_lineno for i in hits):
   funcs.append(dict(name=node.name,start=node.lineno,end=node.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(node.lineno-1,node.end_lineno)]if node.end_lineno-node.lineno<=160 else None))
 out.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),hits=[dict(line=i,text=ls[i-1])for i in hits],functions=funcs))
print(json.dumps(dict(export_filename_candidates=[str(p)for p in paths],files=out)))'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),graph_compute_executions=total,
 adjacent_nonempty_same_attrInfo_op_pairs=[dict(previous_op=k[0],op=k[1],executions=v)for k,v in groups.items()],
 graph_op_attribute_lookup_states=[dict(op=k[0],state=k[1],executions=v)for k,v in byop.items()],examples=examples,
 installed_torch_export_route=json.loads(r.stdout),
 limit='Exact adjacent TASK-order attribute-ID equality only. Does not prove metadata ownership, exporter defect, parent API binding or actual own-op scalar values.',
 no_new_DB_model_import_or_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(graph_compute_executions=total,adjacent_nonempty_same_attrInfo_op_pairs=result['adjacent_nonempty_same_attrInfo_op_pairs'],export_route_files=[dict(path=f['path'],sha256=f['sha256'],hits=f['hits'],function_names=[q['name']for q in f['functions']])for f in result['installed_torch_export_route']['files']],full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
