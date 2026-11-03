from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';OUT=E/'resource-role-intake-v20.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
ctx=pin(E/'fullchain-source-contexts-v15.json','1c003dbca3d72a216c646d209dc54bba38e936db32b0622b2efdf866b2b53ec9')
res=pin(P/'native-PMU-complete-v18/complete-PMU-time-identities-and-caller-resources-v18-v2.json','0d78be321528d347c4ce0c436021a36da5e9bc492ef33aba881043a9546a49e6')
leaves=defaultdict(list)
for r in ctx['chains']:
 assert r['compute_executions']==len(ctx['execution_keys_by_chain'][r['chain_key']])
 leaves[r['project_frames'][0]if r['project_frames']else'NO_PROJECT_FRAME'].append(r)
short=lambda s:s.replace('/vllm-workspace/vllm-ascend/vllm_ascend/','a/').replace('/vllm-workspace/vllm/vllm/','v/')
rows=[]
for i,(leaf,rr)in enumerate(sorted(leaves.items())):
 ops=Counter()
 for r in rr:ops.update(r['op_type_counts'])
 rows.append(dict(leaf_index=i,leaf=short(leaf),chains=len(rr),compute_executions=sum(r['compute_executions']for r in rr),op_type_counts=dict(ops),chain_samples=[dict(chain_key=r['chain_key'],compute_executions=r['compute_executions'],op_type_counts=r['op_type_counts'],project_frames=[short(f)for f in r['project_frames']if not any(x in f for x in ['/v1/executor/','/v1/worker/worker_base.py','/worker/worker.py','/compilation/decorators.py'])])for r in rr]))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),canonical_chains=len(ctx['chains']),leaf_locations=len(rows),native_direct_executions=sum(r['compute_executions']for r in rows),operator_types=len(res['op_reported_resource_summaries']),no_new_source_DB_model_device_or_service_work=True,overall_complete=False,limits=['Leaf/path/op names are an intake for semantic review,not automatic whole-module identity or tensor-consumer proof.','This native-direct population excludes graph132000 and queue-derived10 executions.','Counter-derived ratios are nonexclusive and not physical-peak utilization or additive latency.'])
assert summary['canonical_chains']==486 and summary['native_direct_executions']==154940
payload=(json.dumps(dict(summary=summary,leaves=rows,operator_resource_summaries=res['op_reported_resource_summaries']),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
public=[dict(leaf_index=r['leaf_index'],leaf=r['leaf'],chains=r['chains'],compute_executions=r['compute_executions'],op_type_counts=r['op_type_counts'])for r in rows]
print(json.dumps(dict(summary=summary,leaves=public,op_type_counts={r['op_type']:r['executions']for r in res['op_reported_resource_summaries']},full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
