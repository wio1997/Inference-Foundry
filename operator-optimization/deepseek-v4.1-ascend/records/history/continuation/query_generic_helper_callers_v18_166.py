from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14')
OUT=P/'generic-helper-full-caller-intake-v18.json';assert not OUT.exists()
b=(P/'fullchain-source-contexts-v15.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='1c003dbca3d72a216c646d209dc54bba38e936db32b0622b2efdf866b2b53ec9';ctx=json.loads(b)
generic=[r for r in ctx['chains']if r['project_frames']and any(s in r['project_frames'][0]for s in ['/ops/linear.py','/ops/layernorm.py','/parameter.py','base_device_communicator.py','/w8a8_dynamic.py','/vocab_parallel_embedding.py'])];assert len(generic)==70
rows=[]
for i,r in enumerate(generic):
 full=r['project_frames'];tags=[v for v in ['execute_dummy_batch','_dummy_run','dummy_run','_run_merged_draft','_propose','_prepare_inputs','_build_attention_metadata','_preprocess','_model_forward','sample_tokens']if any(': '+v in f for f in full)]
 trimmed=[f for f in full if not any(v in f for v in ['/v1/executor/','/v1/worker/worker_base.py','/worker/worker.py','/compilation/decorators.py'])]
 rows.append(dict(record_index=i,chain_key=r['chain_key'],compute_executions=r['compute_executions'],op_type_counts=r['op_type_counts'],full_project_frames=full,displayed_project_frames=trimmed,execution_context_tags=tags,full_execution_keys=ctx['execution_keys_by_chain'].get(r['chain_key'],[])))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),generic_chains=len(rows),generic_compute_executions=sum(r['compute_executions']for r in rows),native_chain_cardinality=ctx['summary']['native_compute_chain_count_hist'],leaf_helper_counts=dict(Counter(r['full_project_frames'][0]for r in rows)),no_new_source_DB_or_device_work=True,overall_complete=False,limits=['All full projectcaller frames andexecution keys retained. Only common executor/wrapper tail frames omitted from display,with execution-context tags retained.','Role still requires main semantic review,not generic-helper filename or first frame matching.','These are native direct-source chains,not all graph execution source or tensor-consumer proof.'])
payload=(json.dumps(dict(summary=summary,records=rows),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
short=lambda s:s.replace('/vllm-workspace/vllm-ascend/vllm_ascend/','a/').replace('/vllm-workspace/vllm/vllm/','v/')
public=[dict(record_index=r['record_index'],chain_key=r['chain_key'],compute_executions=r['compute_executions'],op_type_counts=r['op_type_counts'],project_frames=[short(f)for f in r['displayed_project_frames']],execution_context_tags=r['execution_context_tags'])for r in rows]
print(json.dumps(dict(summary=summary,records=public,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
