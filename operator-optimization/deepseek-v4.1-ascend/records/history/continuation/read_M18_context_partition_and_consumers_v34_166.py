from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=M/'M18-selected-context-parameter-partition-consumers-and-RoPE-query-contract-v34.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');V=Path('/vllm-workspace/vllm/vllm');rows=[]
specs=[(A/'models/deepseek_v41/dspark.py','fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601',[('function',282,[]),('owner_fields',282,['main_proj','main_norm','layers','use_sequence_parallel_moe','hidden_size']),('function',308,[])]),
       (A/'spec_decode/llm_base_proposer.py','458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31',[('uses',1068,['target_hidden_states','context_hidden_states','aux_hidden_states']),('uses',1481,['target_hidden_states','context_hidden_states','context_positions','context_slot_mapping'])]),
       (A/'spec_decode/dflash_proposer.py',None,[('uses',315,['target_hidden_states','context_hidden_states','context_states','context_positions','context_slot_mapping','num_context_tokens'])]),
       (A/'ops/rope_dsv4.py',None,[('function',185,[]),('frame_statement',400,[]),('frame_statement',402,[])]),
       (A/'ops/linear.py',None,[('function',485,[])]),
       (V/'model_executor/layers/linear.py',None,[('function',606,[])])]
for p,expected,selections in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if expected:assert sha==expected,str(p)
 text=b.decode();ls=text.splitlines();t=ast.parse(text);par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};out=[]
 def ancestors(n,f):
  seq=[];child=n
  while child in par:
   a=par[child]
   if isinstance(a,(ast.If,ast.IfExp,ast.While)):
    if isinstance(a,ast.IfExp):arm='BODY'if child is a.body else'ORELSE'if child is a.orelse else'TEST'
    else:arm='BODY'if child in a.body else'ORELSE'if child in a.orelse else'TEST'
    seq.append(dict(line=a.lineno,condition=ast.unparse(a.test),arm=arm))
   child=a
   if child is f:break
  return seq
 for mode,line,names in selections:
  fs=[f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.lineno<=line<=f.end_lineno];assert len(fs)==1,(str(p),line)
  f=fs[0];owners=[c for c in ast.walk(t)if isinstance(c,ast.ClassDef)and c.lineno<=f.lineno<=c.end_lineno]
  rec=dict(name=f.name,owners=[c.name for c in owners],mode=mode,start=f.lineno,end=f.end_lineno,selected_line=line)
  if mode=='function':
   assert f.end_lineno-f.lineno<180,(str(p),f.name,'too large;stop')
   rec['source']=[str(j+1)+': '+ls[j]for j in range(f.lineno-1,f.end_lineno)]
  elif mode=='owner_fields':
   assert len(owners)==1;ini=[z for z in owners[0].body if isinstance(z,ast.FunctionDef)and z.name=='__init__'];assert len(ini)==1;ini=ini[0];stmts=[]
   for z in ast.walk(ini):
    if not isinstance(z,(ast.Assign,ast.AnnAssign)):continue
    targets=z.targets if isinstance(z,ast.Assign)else[z.target]
    attrs={a.attr for target in targets for a in ast.walk(target)if isinstance(a,ast.Attribute)}
    if not(attrs&set(names)):continue
    assert z.end_lineno-z.lineno<45
    stmts.append(dict(start=z.lineno,end=z.end_lineno,attributes=sorted(attrs),source=[str(j+1)+': '+ls[j]for j in range(z.lineno-1,z.end_lineno)]))
   rec['constructor_fields']=stmts
  elif mode=='frame_statement':
   sts=[z for z in ast.walk(f)if isinstance(z,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Expr,ast.Return))and z.lineno<=line<=z.end_lineno];assert sts,(str(p),line)
   z=min(sts,key=lambda z:z.end_lineno-z.lineno);assert z.end_lineno-z.lineno<35
   rec.update(source=[str(j+1)+': '+ls[j]for j in range(z.lineno-1,z.end_lineno)],enclosing_arms=ancestors(z,f))
  else:
   statements={};uses=[]
   for z in ast.walk(f):
    if not isinstance(z,ast.Name)or z.id not in names:continue
    aa=[];q=z
    while q in par:
     q=par[q];aa.append(q)
     if q is f:break
    st=next((q for q in aa if isinstance(q,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Expr,ast.Return,ast.Assert,ast.Raise))),None)
    if st is None:
     control=next((q for q in aa if isinstance(q,(ast.If,ast.IfExp,ast.While,ast.For))),None);assert control is not None,(z.id,z.lineno,'unsupportedcontext')
     st=control.iter if isinstance(control,ast.For)else control.test
    assert st.end_lineno-st.lineno<80,(str(p),z.id,z.lineno,'statement too large;stop')
    statements[st.lineno]=dict(start=st.lineno,end=st.end_lineno,source=[str(j+1)+': '+ls[j]for j in range(st.lineno-1,st.end_lineno)],enclosing_arms=ancestors(z,f))
    uses.append(dict(identifier=z.id,context=type(z.ctx).__name__,line=z.lineno,statement_start=st.lineno,enclosing_arms=ancestors(z,f),nested_function_ancestors=[q.name for q in aa if isinstance(q,ast.FunctionDef)]))
   assert len(uses)<180,(str(p),f.name,'too many identifier uses;stop')
   rec.update(identifier_uses=uses,statements=list(statements.values()))
  out.append(rec)
 rows.append(dict(path=str(p),sha256=sha,functions=out))
total=sum(len(f.get('source',[]))+sum(len(z['source'])for z in f.get('constructor_fields',[]))+sum(len(z['source'])for z in f.get('statements',[]))for r in rows for f in r['functions']);assert total<1050,total
print(json.dumps(dict(files=rows,selected_source_lines=total,scope='Onlyprofile-selectedF27constructorfields/combine/linear,Ropequery+twoactualstatements,andcontextidentifierusesinexactproposer/build-first-passmethods. ReuseotherF29math/storeandtargetauxcontracts;notwholelargeproposers.',no_model_or_torch_import=True)))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
