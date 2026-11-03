from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'M15-normalized-hidden-identifier-use-and-row-contract-v31.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');rows=[]
specs=[(A/'models/deepseek_v41/model.py','101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91','function',856,[]),
       (A/'ops/layernorm.py',None,'function',99,[]),
       (A/'worker/model_runner_v1.py','568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c','identifier_contexts',2634,['hidden_states','sample_hidden_states','aux_hidden_states','logits_indices']),
       (A/'worker/model_runner_v1.py','568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c','identifier_contexts',2080,['hidden_states','target_hidden_states','aux_hidden_states','mtp_hidden_states'])]
for p,expected,mode,line,names in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if expected:assert sha==expected
 ls=b.decode().splitlines();tree=ast.parse(b.decode());parents={c:n for n in ast.walk(tree)for c in ast.iter_child_nodes(n)}
 fs=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.lineno<=line<=n.end_lineno];assert len(fs)==1,(str(p),line)
 f=fs[0];record=dict(path=str(p),sha256=sha,function=f.name,start=f.lineno,end=f.end_lineno,profile_or_prior_contract_line=line,mode=mode)
 if mode=='function':
  assert f.end_lineno-f.lineno<150,(str(p),f.name,'function too large;stop')
  record['source']=[str(j+1)+': '+ls[j]for j in range(f.lineno-1,f.end_lineno)]
 else:
  uses=[]
  for n in ast.walk(f):
   if not isinstance(n,ast.Name)or n.id not in names:continue
   anc=[];cur=n
   while cur in parents:
    cur=parents[cur];anc.append(cur)
    if cur is f:break
   statement=next((z for z in anc if isinstance(z,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Expr,ast.Return,ast.Raise))),None)
   branch=[dict(line=z.lineno,type=type(z).__name__,condition=ast.get_source_segment(b.decode(),z.test))for z in anc if isinstance(z,(ast.If,ast.IfExp,ast.While))]
   if statement is not None:
    span=statement.end_lineno-statement.lineno+1;assert span<=30,(f.name,n.id,n.lineno,'statement too large;stop')
    excerpt=[str(j+1)+': '+ls[j]for j in range(statement.lineno-1,statement.end_lineno)]
   else:excerpt=[str(n.lineno)+': '+ls[n.lineno-1]]
   uses.append(dict(identifier=n.id,context=type(n.ctx).__name__,line=n.lineno,statement_start=statement.lineno if statement else n.lineno,source=excerpt,enclosing_conditions=branch,nested_function_ancestors=[z.name for z in anc if isinstance(z,ast.FunctionDef)]))
  assert len(uses)<130,(f.name,'identifier context count too large;stop')
  record['uses']=uses
 rows.append(record)
print(json.dumps(dict(files=rows,no_torch_or_model_import=True,scope='Onlyprofile-selectedcollapse/normmethodsandidentifierusesinexactexecute/proposemethods;notwhole-methodorrepositoryaudit. Sameidentifierdoesnotprovecompletealiasanalysis.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
