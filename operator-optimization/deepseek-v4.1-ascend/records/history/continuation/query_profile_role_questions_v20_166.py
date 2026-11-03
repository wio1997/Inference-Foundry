from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14');OUT=P/'resource-role-profile-questions-v20.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
intake=pin(P/'resource-role-intake-v20.json','06892785a41ff3fa19c6ca86515c71f4d4d3ee21ee5f8272d98e266488d8a175')
idx=pin(P/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
graph=pin(P/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
wanted={0,1,46,55,56,57,58,59,60,61,105,126,127,128,129,154,171,172,173,174,175,203,213,217,218,219,227,228}
selected=[r for r in intake['leaves']if r['leaf_index']in wanted]
first=graph['graph_executions'][0];ref=idx['databases'][first['database_index']];d=pin(Path(ref['path']),ref['sha256']);tasks={r['task_record_rowid']:r for r in d['TASK_records']};defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']};blockmap={i:b['block_ordinal']for b in first['blocks']for i in b['task_record_rowids']}
ids=[i for b in first['blocks']for i in b['task_record_rowids']]+[r['task']['task_record_rowid']for r in first['outside_marker_tasks']];assert len(ids)==first['compute_executions']==1650
allrows=[];regimes=defaultdict(list)
for i in sorted(ids,key=lambda i:(tasks[i]['startNs'],tasks[i]['streamId'],tasks[i]['taskId'])):
 t=tasks[i];c=defs[t['globalTaskId']];sig={k:c.get(k)for k in ['opType_text','inputShapes_text','inputDataTypes_text','outputShapes_text','outputDataTypes_text','blockNum','mixBlockNum']};row=dict(task_record_rowid=i,globalTaskId=t['globalTaskId'],block_ordinal=blockmap.get(i),TASK=t,definition=c);allrows.append(row);regimes[json.dumps(sig,sort_keys=True)].append(row)
greg=[dict(regime_index=i,signature=json.loads(k),executions=len(rs),block_ordinals=dict(Counter(str(r['block_ordinal'])for r in rs)),example_task=rs[0]['TASK'])for i,(k,rs)in enumerate(regimes.items())]
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),selected_leaves=selected,first_graph_database_index=first['database_index'],first_graph_ordinal=first['graph_ordinal'],first_graph_rows=allrows,first_graph_regimes=greg,scope='Only saved caller context and one saved graph template;later full resource pass must retain all graphs/shape variations and unresolved roles.',no_new_source_DB_model_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
shortrows=[dict(task_record_rowid=r['task_record_rowid'],block_ordinal=r['block_ordinal'],definition=r['definition'])for r in allrows if r['block_ordinal']in[0,2,20]or r['block_ordinal']is None]
print(json.dumps(dict(selected_leaves=selected,first_graph_regimes=[{k:v for k,v in r.items()if k!='example_task'}for r in greg],selected_graph_rows=shortrows,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
