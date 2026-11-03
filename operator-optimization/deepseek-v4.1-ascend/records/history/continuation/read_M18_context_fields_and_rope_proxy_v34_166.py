from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=M/'M18-context-object-field-handoff-and-RopeDataProxy-contract-v34.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');rows=[]
specs=[(A/'spec_decode/dflash_proposer.py','f3b736f44a59ec0831aa7f2849e2eae6d709951e955992b8016f49e772fce3ff',[('function',315),('named_method','set_inputs_first_pass')]),
       (A/'spec_decode/llm_base_proposer.py','458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31',[('field_uses',1481)]),
       (A/'ops/rope_dsv4.py','9cbcd2cb2a1123dcbf5afc3999025035d155dfd3b160f186c36aebb5aa7c706b',[('proxy_methods',None),('function',400)])]
for p,expected,selections in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha==expected;text=b.decode();ls=text.splitlines();t=ast.parse(text);par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};out=[]
 def arms(z,f):
  ret=[];child=z
  while child in par:
   a=par[child]
   if isinstance(a,ast.If):ret.append(dict(line=a.lineno,condition=ast.unparse(a.test),arm='BODY'if child in a.body else'ORELSE'if child in a.orelse else'TEST'))
   child=a
   if child is f:break
  return ret
 def full(f,mode):
  assert f.end_lineno-f.lineno<210,(str(p),f.name,'too large;stop')
  return dict(name=f.name,mode=mode,start=f.lineno,end=f.end_lineno,source=[str(j+1)+': '+ls[j]for j in range(f.lineno-1,f.end_lineno)])
 for mode,line in selections:
  if mode=='proxy_methods':
   cs=[z for z in ast.walk(t)if isinstance(z,ast.ClassDef)and z.name=='RopeDataProxy'];assert len(cs)==1
   for name in['__init__','__getitem__']:
    fs=[f for f in cs[0].body if isinstance(f,ast.FunctionDef)and f.name==name];assert len(fs)==1;out.append(full(fs[0],mode))
   continue
  if mode=='named_method':
   fs=[f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.name==line]
   out.append(dict(name=line,mode=mode,defined_here=len(fs),locations=[dict(start=f.lineno,end=f.end_lineno)for f in fs]))
   # Namedmethodmaybelarge:extracttarget_hidden_statesandcontextfieldusesonly.
   if not fs:continue
   assert len(fs)==1;f=fs[0]
  else:
   fs=[f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.lineno<=line<=f.end_lineno];assert len(fs)==1;f=fs[0]
  if mode=='function':out.append(full(f,mode));continue
  selected=[]
  for z in ast.walk(f):
   hit=isinstance(z,ast.Attribute)and any(x in z.attr for x in['context_hidden','context_state','context_position','context_slot','target_hidden'])
   hit=hit or(isinstance(z,ast.Name)and z.id in['target_hidden_states','context_hidden_states','context_states','context_positions','context_slot_mapping','model_inputs'])
   if not hit:continue
   aa=[];q=z
   while q in par:
    q=par[q];aa.append(q)
    if q is f:break
   st=next((q for q in aa if isinstance(q,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Expr,ast.Return,ast.Assert,ast.Raise))),None)
   if st is None:
    control=next((q for q in aa if isinstance(q,(ast.If,ast.While,ast.IfExp,ast.For))),None);assert control is not None,(str(p),z.lineno)
    st=control.iter if isinstance(control,ast.For)else control.test
   assert st.end_lineno-st.lineno<85
   selected.append(dict(field=ast.unparse(z),context=type(z.ctx).__name__ if hasattr(z,'ctx')else None,line=z.lineno,start=st.lineno,end=st.end_lineno,source=[str(j+1)+': '+ls[j]for j in range(st.lineno-1,st.end_lineno)],enclosing_arms=arms(z,f)))
  # Statementde-duplicationkeepsallfieldhitmetadatawithoutrepeatinglargerargumentlists.
  statements={r['start']:dict(start=r['start'],end=r['end'],source=r['source'],enclosing_arms=r['enclosing_arms'])for r in selected}
  out.append(dict(name=f.name,mode=mode,start=f.lineno,end=f.end_lineno,field_uses=[{k:v for k,v in r.items()if k not in['source']}for r in selected],statements=list(statements.values())))
 rows.append(dict(path=str(p),sha256=sha,functions=out))
assert sum(len(f.get('source',[]))+sum(len(s['source'])for s in f.get('statements',[]))for r in rows for f in r['functions'])<650
print(json.dumps(dict(files=rows,scope='Followexactcontext-objectfieldsafterfirstName-onlysourcepass;smallbuild-first-pass/RopeProxy/rotaryforwardandselectedset-input/merged-draftfieldstatements. Nootherproposerbodies.',no_model_or_torch_import=True)))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
