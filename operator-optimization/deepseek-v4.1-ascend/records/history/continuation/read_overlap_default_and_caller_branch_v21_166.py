from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');OUT=P/'native-PMU-complete-v18/shared-overlap-config-default-and-caller-branch-v21.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
ctx=pin(P/'native-execution-identity-v14/fullchain-source-contexts-v15.json','1c003dbca3d72a216c646d209dc54bba38e936db32b0622b2efdf866b2b53ec9')
rows=[];totals=Counter()
for r in ctx['chains']:
 fs=r['project_frames']
 if not fs or not '/ops/fused_moe/shared_experts.py'in fs[0]:continue
 branch='RUNNER_POST_ROUTED_FALLBACK_528'if any('/ops/fused_moe/fused_moe.py(528): _forward_impl'in f for f in fs)else'RUNNER_PRE_SUBMITTED_534'if any('/ops/fused_moe/fused_moe.py(534): _forward_impl'in f for f in fs)else'OTHER'
 totals[branch]+=r['compute_executions'];rows.append(dict(chain_key=r['chain_key'],compute_executions=r['compute_executions'],op_type_counts=r['op_type_counts'],runner_branch=branch,project_frames=fs,execution_keys=ctx['execution_keys_by_chain'][r['chain_key']]))
code=r'''from pathlib import Path
import ast,json,hashlib
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/ascend_config.py');s=p.read_text();ls=s.splitlines();tree=ast.parse(s);nodes=[n for n in ast.walk(tree)if isinstance(n,(ast.FunctionDef,ast.ClassDef))];points=set();hits=[]
for i,l in enumerate(ls):
 if 'multistream_overlap_shared_expert'in l:
  parents=sorted([n for n in nodes if n.lineno<=i+1<=n.end_lineno],key=lambda n:n.end_lineno-n.lineno)
  hits.append(dict(line=i+1,text=l,enclosing=[dict(name=n.name,start=n.lineno,end=n.end_lineno)for n in parents[:2]]));points.update(range(max(0,i-9),min(len(ls),i+12)))
assert hits and len(points)<=200
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),field_hits=hits,exact_field_context=[str(i+1)+': '+ls[i]for i in sorted(points)],no_module_import_or_device_work=True)))
'''
q=subprocess.run(['docker','exec','bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d','python3','-c',code],capture_output=True,text=True,timeout=80);assert q.returncode==0,q.stderr
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),config_source=json.loads(q.stdout),saved_native_shared_caller_branches=dict(totals),native_shared_chains=rows,scope='Exact one config field contexts andalready-saved shared operator call stacks;no model import,currentenv,device or service work.',overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
public=dict(result={k:v for k,v in result.items()if k!='native_shared_chains'},native_shared_chain_summary=[dict(chain_key=r['chain_key'],compute_executions=r['compute_executions'],op_type_counts=r['op_type_counts'],runner_branch=r['runner_branch'],project_frames=r['project_frames'][:7])for r in rows],full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest())
print(json.dumps(public,ensure_ascii=False))
