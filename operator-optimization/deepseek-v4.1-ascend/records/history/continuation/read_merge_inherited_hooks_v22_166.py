from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=P/'merge-inherited-scale-reduction-and-schema-v22.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib,re
specs=[('/vllm-workspace/vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py',None,{'_maybe_apply_routed_scale_to_output','_maybe_add_zero_expert_output','_maybe_pad_hidden_states','_fused_output_is_reduced'}),('/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py','2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4',{'_fused_output_is_reduced','_should_reduce_routed_before_combine','_get_shared_expert_parallel_mode'})]
rows=[]
for name,pin,wanted in specs:
 p=Path(name);s=p.read_text();sha=hashlib.sha256(s.encode()).hexdigest()
 if pin:assert sha==pin
 ls=s.splitlines();fs=[]
 for n in ast.walk(ast.parse(s)):
  if isinstance(n,ast.FunctionDef)and n.name in wanted:
   assert n.end_lineno-n.lineno<=140
   fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 rows.append(dict(path=name,sha256=sha,functions=fs,missing_requested_symbols=sorted(wanted-{f['name']for f in fs})))
p=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch/_C/_VariableFunctions.pyi');s=p.read_text();ls=s.splitlines();points=set();hits=[]
for i,l in enumerate(ls):
 if re.search(r'^def _foreach_add\(',l):hits.append(i+1);points.update(range(max(0,i-1),min(len(ls),i+8)))
assert hits and len(points)<=100
stub=dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),definition_lines=hits,exact_schema_context=[str(i+1)+': '+ls[i]for i in sorted(points)])
print(json.dumps(dict(files=rows,installed_torch_static_schema=stub,no_module_import_or_device_work=True)))
'''
q=subprocess.run(['docker','exec','bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d','python3','-c',code],capture_output=True,text=True,timeout=80);assert q.returncode==0,q.stderr
v=json.loads(q.stdout);v.update(created_utc=datetime.now(timezone.utc).isoformat(),scope='Only inheritedmergehelpers/exactrouted-reductionpredicate andinstalledforeach_add schema;no torch/modelimport ordevicework.',overall_complete=False)
b=(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=v,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
