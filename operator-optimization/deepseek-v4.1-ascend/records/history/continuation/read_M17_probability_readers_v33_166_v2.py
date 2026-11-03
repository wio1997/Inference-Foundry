from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=M/'M17-probability-variable-readers-and-greedy-return-control-v33-v2.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/sample/rejection_sampler.py');b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha=='bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda'
text=b.decode();ls=text.splitlines();t=ast.parse(text);par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};fs=[f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.name=='rejection_sample'];assert len(fs)==1;f=fs[0]
def ancestors(n):
 out=[];child=n
 while child in par:
  a=par[child]
  if isinstance(a,(ast.If,ast.IfExp,ast.While)):
   if isinstance(a,ast.IfExp):arm='BODY'if child is a.body else'ORELSE'if child is a.orelse else'TEST'
   else:arm='BODY'if child in a.body else'ORELSE'if child in a.orelse else'TEST'
   out.append(dict(line=a.lineno,condition=ast.get_source_segment(text,a.test),arm=arm))
  child=a
  if child is f:break
 return out
uses=[];statements={};callnames=set()
for z in ast.walk(f):
 if not isinstance(z,ast.Name)or z.id not in{'ori_target_probs','target_probs','draft_probs'}:continue
 aa=[];a=z
 while a in par:
  a=par[a];aa.append(a)
  if a is f:break
 stmt=next((a for a in aa if isinstance(a,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Expr,ast.Return,ast.Assert))),None)
 if stmt is None:
  control=next((a for a in aa if isinstance(a,(ast.If,ast.IfExp,ast.While))),None);assert control is not None,(z.id,z.lineno,'unsupportedcontext;stop')
  stmt=control.test
 assert stmt.end_lineno-stmt.lineno<75,(z.id,z.lineno,'too large;stop')
 calls=[a for a in aa if isinstance(a,ast.Call)]
 for c in ast.walk(stmt):
  if isinstance(c,ast.Call)and isinstance(c.func,ast.Name):callnames.add(c.func.id)
 statements[stmt.lineno]=dict(start=stmt.lineno,end=stmt.end_lineno,source=[str(j+1)+': '+ls[j]for j in range(stmt.lineno-1,stmt.end_lineno)],enclosing_arms=ancestors(stmt))
 uses.append(dict(identifier=z.id,line=z.lineno,context=type(z.ctx).__name__,statement_start=stmt.lineno,nearest_call=ast.unparse(calls[0].func)if calls else None,enclosing_arms=ancestors(z)))
returns=[]
for a in ast.walk(f):
 if isinstance(a,ast.If)and'all_greedy'in ast.unparse(a.test)and any(isinstance(z,ast.Return)for z in a.body):
  returns.append(dict(condition_line=a.lineno,condition=ast.unparse(a.test),enclosing_arms=ancestors(a),body=[str(j+1)+': '+ls[j]for z in a.body for j in range(z.lineno-1,z.end_lineno)]))
imports=[];locals_=[]
for a in ast.walk(t):
 if isinstance(a,ast.ImportFrom):
  for name in a.names:
   if(name.asname or name.name)in callnames:imports.append(dict(local_name=name.asname or name.name,module=a.module,level=a.level,symbol=name.name,line=a.lineno))
 if isinstance(a,ast.FunctionDef)and a.name in callnames:locals_.append(dict(name=a.name,start=a.lineno,end=a.end_lineno,path=str(p)))
assert sum(x['end']-x['start']+1 for x in statements.values())<250
print(json.dumps(dict(path=str(p),sha256=sha,function=f.name,start=f.lineno,end=f.end_lineno,identifier_uses=uses,selected_statements=list(statements.values()),early_return_guards=returns,resolved_imported_callees=imports,local_callee_locations=locals_,scope='Onlyprobabilityidentifieruses/theirsmallstatements/explicitcontrolarmsandgreedyreturn;notentire305-linebody.',no_model_or_torch_import=True)))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
