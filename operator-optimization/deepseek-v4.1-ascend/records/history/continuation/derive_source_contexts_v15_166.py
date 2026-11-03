from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14')
def pin(path,sha):
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
idx=pin(P/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
c=pin(P/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0')
OUT=P/'fullchain-source-contexts-v15.json';assert not OUT.exists();keys=defaultdict(set);ops=defaultdict(Counter);api_counts=Counter();multis=Counter()
for ref in idx['databases']:
 d=pin(Path(ref['path']),ref['sha256']);di=d['database_index'];ad={a['api_rowid']:a for a in c['databases'][di]['API_records']};defs={x['globalTaskId']:x for x in d['COMPUTE_definitions']}
 for a in ad.values():api_counts[a['chain_key']]+=1
 for t in d['TASK_records']:
  ids=t.get('native_linked_API_rowids',[])
  if not ids:continue
  ks={ad[i]['chain_key']for i in ids};multis[len(ks)]+=1
  for k in ks:keys[k].add((di,t['task_record_rowid']));ops[k][defs[t['globalTaskId']]['opType_text']]+=1
rows=[]
for k,fs in c['canonical_fullchains'].items():
 project=[r['frame']for r in fs if '/vllm_ascend/'in r['frame']or '/vllm/vllm/'in r['frame']]
 owners=[f for f in project if '/models/deepseek_v41/'in f or '/model_executor/models/deepseek_v41/'in f or '/spec_decode/'in f]
 contexts=[f for f in project if any(x in f for x in [': _dummy_run',': dummy_run',': _prepare_inputs',': _build_attention_metadata',': _propose',': _run_merged_draft',': execute_model'])]
 rows.append(dict(chain_key=k,API_records=api_counts[k],compute_executions=len(keys[k]),op_type_counts=dict(ops[k]),project_frames=project,model_spec_callers=owners,execution_context_frames=contexts))
rows.sort(key=lambda r:r['compute_executions'],reverse=True)
generic=[r for r in rows if r['project_frames']and any(s in r['project_frames'][0]for s in ['/ops/linear.py','/ops/layernorm.py','/parameter.py','base_device_communicator.py','/w8a8_dynamic.py','/vocab_parallel_embedding.py'])]
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),canonical_fullchains=len(rows),native_compute_chain_count_hist=dict(multis),generic_helper_chains=len(generic),no_source_DB_or_device_work=True,overall_complete=False,limits=['Counts per chain overlap if a native execution has several distinct API chains; preserve cardinality histogram,do notsumchaincountsasdisjointmodulecost.','Source role remains a semantic review; generic helper filename is not a projection role. All full project frames and execution keys retainedserver-side.'])
payload=(json.dumps(dict(summary=summary,chains=rows,execution_keys_by_chain={k:[list(v)for v in sorted(vs)]for k,vs in keys.items()}),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
public=dict(summary=summary,generic_helper_page=[{k:v for k,v in r.items()if k not in ['project_frames']}|dict(leaf=r['project_frames'][0])for r in generic[:35]],generic_helper_total=len(generic),full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())
if len(json.dumps(public).encode())>31000:
 public['generic_helper_page']=public['generic_helper_page'][:20];public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
