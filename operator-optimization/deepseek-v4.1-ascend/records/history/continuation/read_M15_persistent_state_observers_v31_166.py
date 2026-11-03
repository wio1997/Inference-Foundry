from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=M/'M15-ExecuteModelState-observer-and-branch-arm-contract-v31.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py');b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha=='568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c'
text=b.decode();ls=text.splitlines();t=ast.parse(text);parents={c:n for n in ast.walk(t)for c in ast.iter_child_nodes(n)}
def contexts(n):
 anc=[];child=n;arms=[]
 while child in parents:
  parent=parents[child];anc.append(parent)
  if isinstance(parent,(ast.If,ast.IfExp,ast.While)):
   if isinstance(parent,ast.IfExp):arm='BODY'if child is parent.body else'ORELSE'if child is parent.orelse else'TEST'
   else:arm='BODY'if child in parent.body else'ORELSE'if child in parent.orelse else'TEST'
   arms.append(dict(line=parent.lineno,condition=ast.get_source_segment(text,parent.test),arm=arm))
  child=parent
  if isinstance(child,ast.FunctionDef):break
 stmt=next((a for a in anc if isinstance(a,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Expr,ast.Return,ast.Raise,ast.Assert))),None)
 if stmt is not None:
  assert stmt.end_lineno-stmt.lineno<34,(n.lineno,'statement too large;stop')
  excerpt=[str(j+1)+': '+ls[j]for j in range(stmt.lineno-1,stmt.end_lineno)]
 else:excerpt=[str(n.lineno)+': '+ls[n.lineno-1]]
 return dict(line=n.lineno,statement_start=stmt.lineno if stmt else n.lineno,source=excerpt,enclosing_arms=arms)
attrs=[n for n in ast.walk(t)if isinstance(n,ast.Attribute)and n.attr=='execute_model_state'];records=[];functions={}
for n in attrs:
 f=parents.get(n)
 while f is not None and not isinstance(f,ast.FunctionDef):f=parents.get(f)
 assert f is not None
 functions[f.name]=f;records.append(dict(function=f.name,attribute='execute_model_state',context=type(n.ctx).__name__,**contexts(n)))
classes=[n for n in ast.walk(t)if isinstance(n,ast.ClassDef)and n.name=='ExecuteModelState'];assert len(classes)==1;c=classes[0];assert c.end_lineno-c.lineno<75
uses=[]
for f in functions.values():
 aliases=set()
 for n in ast.walk(f):
  if isinstance(n,(ast.Assign,ast.AnnAssign))and isinstance(n.value,ast.Attribute)and n.value.attr=='execute_model_state':
   targets=n.targets if isinstance(n,ast.Assign)else[n.target]
   aliases.update(z.id for a in targets for z in ast.walk(a)if isinstance(z,ast.Name))
 names={'hidden_states','sample_hidden_states','aux_hidden_states'}|aliases
 for n in ast.walk(f):
  if isinstance(n,ast.Name)and n.id in names:uses.append(dict(function=f.name,identifier=n.id,context=type(n.ctx).__name__,**contexts(n)))
# Reuse the prior method location, adding explicit branch arms which the first extractor did not distinguish.
f=next(n for n in ast.walk(t)if isinstance(n,ast.FunctionDef)and n.name=='propose_draft_token_ids')
for n in ast.walk(f):
 if isinstance(n,ast.Name)and n.id in['hidden_states','target_hidden_states','aux_hidden_states','mtp_hidden_states']:uses.append(dict(function=f.name,identifier=n.id,context=type(n.ctx).__name__,**contexts(n)))
assert len(records)<24 and len(uses)<180,(len(records),len(uses))
print(json.dumps(dict(path=str(p),sha256=sha,state_class=dict(name=c.name,start=c.lineno,end=c.end_lineno,source=[str(j+1)+': '+ls[j]for j in range(c.lineno-1,c.end_lineno)]),exact_attribute_references=records,selected_identifier_uses=uses,scope='OnlyExecuteModelStateclass/exactattributeusesandnormalized/auxlocalidentifiercontextswithinowningmethods;explicitIfarmsretained;notexternalobserverorcompletealiasproof.',no_model_or_torch_import=True)))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
