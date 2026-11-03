from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=M/'M17-probability-producers-temperature-truncation-RNG-and-branch-contract-v33-v2.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/sample/rejection_sampler.py');b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha=='bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda';text=b.decode();ls=text.splitlines();t=ast.parse(text);par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};f=next(z for z in ast.walk(t)if isinstance(z,ast.FunctionDef)and z.name=='rejection_sample')
names={'target_logits','using_block_verify','using_entropy_verify','synthetic_mode','uniform_probs','uniform_probs_for_greedy','is_greedy','global_vocab_size'};rows=[];callees=set()
def arms(z):
 out=[];child=z
 while child in par:
  a=par[child]
  if isinstance(a,ast.If):out.append(dict(line=a.lineno,condition=ast.unparse(a.test),arm='BODY'if child in a.body else'ORELSE'if child in a.orelse else'TEST'))
  child=a
  if child is f:break
 return out
for z in ast.walk(f):
 if not isinstance(z,(ast.Assign,ast.AnnAssign,ast.AugAssign)):continue
 targets=z.targets if isinstance(z,ast.Assign)else[z.target]
 written={x.id for target in targets for x in ast.walk(target)if isinstance(x,ast.Name)}
 if not(written&names):continue
 assert z.end_lineno-z.lineno<45
 rows.append(dict(start=z.lineno,end=z.end_lineno,variables=sorted(written),enclosing_arms=arms(z),source=[str(j+1)+': '+ls[j]for j in range(z.lineno-1,z.end_lineno)]))
 for call in ast.walk(z):
  if isinstance(call,ast.Call)and isinstance(call.func,ast.Name):callees.add(call.func.id)
for z in ast.walk(f):
 if isinstance(z,ast.Expr)and any(isinstance(a,ast.Name)and a.id=='target_logits'for a in ast.walk(z)):
  assert z.end_lineno-z.lineno<45
  rows.append(dict(start=z.lineno,end=z.end_lineno,variables=['target_logits_inplace_call'],enclosing_arms=arms(z),source=[str(j+1)+': '+ls[j]for j in range(z.lineno-1,z.end_lineno)]))
  for call in ast.walk(z):
   if isinstance(call,ast.Call)and isinstance(call.func,ast.Name):callees.add(call.func.id)
functions=[];locations=[]
for g in ast.walk(t):
 if not isinstance(g,ast.FunctionDef)or g.name not in callees:continue
 locations.append(dict(name=g.name,start=g.lineno,end=g.end_lineno))
 if g.name not in{'apply_sampling_params','generate_uniform_probs'}:continue
 assert g.end_lineno-g.lineno<120
 functions.append(dict(name=g.name,start=g.lineno,end=g.end_lineno,source=[str(j+1)+': '+ls[j]for j in range(g.lineno-1,g.end_lineno)]))
has_triton=[]
for z in ast.walk(t):
 if not isinstance(z,(ast.Assign,ast.AnnAssign)):continue
 targets=z.targets if isinstance(z,ast.Assign)else[z.target]
 if not any(isinstance(x,ast.Name)and x.id=='HAS_TRITON'for target in targets for x in ast.walk(target)):continue
 a=z;owner=None
 while a in par:
  a=par[a]
  if isinstance(a,ast.FunctionDef):owner=a.name;break
 has_triton.append(dict(line=z.lineno,owner_function=owner,source=ast.get_source_segment(text,z)))
assert sum(r['end']-r['start']+1 for r in rows)+sum(len(g['source'])for g in functions)<280
imports=[dict(module=z.module,symbol=a.name,local_name=a.asname or a.name,line=z.lineno)for z in ast.walk(t)if isinstance(z,ast.ImportFrom)for a in z.names if(a.asname or a.name)in(callees|{'HAS_TRITON'})]
for imp in imports:
 if imp['local_name']not in{'apply_sampling_params','generate_uniform_probs'}:continue
 module=imp['module'];assert module.startswith('vllm.'),(module,'unexpected utility module')
 pp=Path('/vllm-workspace/vllm/vllm').joinpath(*module.split('.')[1:]).with_suffix('.py');bb=pp.read_bytes();tt=ast.parse(bb);ll=bb.decode().splitlines();gg=[q for q in ast.walk(tt)if isinstance(q,ast.FunctionDef)and q.name==imp['symbol']];assert len(gg)==1;g=gg[0];assert g.end_lineno-g.lineno<180
 functions.append(dict(path=str(pp),sha256=hashlib.sha256(bb).hexdigest(),name=g.name,start=g.lineno,end=g.end_lineno,resolved_import=imp,source=[str(j+1)+': '+ll[j]for j in range(g.lineno-1,g.end_lineno)]))
error_guards=[]
for z in ast.walk(f):
 if isinstance(z,ast.If)and any(isinstance(q,ast.Name)and q.id in{'using_block_verify','synthetic_mode'}for q in ast.walk(z.test))and any(isinstance(q,(ast.Raise,ast.Assert))for q in z.body):
  assert z.end_lineno-z.lineno<12;error_guards.append(dict(start=z.lineno,end=z.end_lineno,source=[str(j+1)+': '+ls[j]for j in range(z.lineno-1,z.end_lineno)]))
print(json.dumps(dict(path=str(p),sha256=sha,producer_statements=rows,selected_local_helpers=functions,direct_local_callee_locations=locations,direct_imports=imports,HAS_TRITON_source_assignments=has_triton,error_guards=error_guards,scope='Onlyprobabilityproducer/control/RNGassignmentsandtwoexactlocalutilityhelpers;notwhole305-linehelperorfallbackbodies.',no_model_or_torch_import=True)))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
