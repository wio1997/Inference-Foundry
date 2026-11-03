from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=M/'M18-actual-DSpark-context-bank-copy-alias-and-layer-slot-contract-v34.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');rows=[]
specs=[(A/'spec_decode/dspark_proposer.py','70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c','fields')]
attrs={'_dflash_hidden_states','_dflash_num_context','_context_positions_buffer','_context_slot_mapping_buffers'}
for p,expected,mode in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha==expected;text=b.decode();ls=text.splitlines();t=ast.parse(text);par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};stmts={};uses=[];functions=[]
 def context(z):
  ancestors=[];child=z;guards=[];owner=None
  while child in par:
   q=par[child];ancestors.append(q)
   if isinstance(q,(ast.For,ast.AsyncFor)):guards.append(dict(line=q.lineno,condition=ast.unparse(q.target)+' in '+ast.unparse(q.iter),arm='LOOP'))
   if isinstance(q,ast.If):guards.append(dict(line=q.lineno,condition=ast.unparse(q.test),arm='BODY'if child in q.body else'ORELSE'if child in q.orelse else'TEST'))
   if isinstance(q,ast.FunctionDef):owner=q.name;break
   child=q
  return ancestors,guards,owner
 if mode=='fields':
  for z in ast.walk(t):
   hit=isinstance(z,ast.Attribute)and z.attr in attrs
   hit=hit or(isinstance(z,ast.Constant)and isinstance(z.value,str)and z.value in attrs)
   if not hit:continue
   anc,guards,owner=context(z);st=next((q for q in anc if isinstance(q,(ast.Assign,ast.AnnAssign,ast.AugAssign,ast.Expr,ast.Return,ast.Assert,ast.Raise))),None)
   if st is None:
    ctl=next((q for q in anc if isinstance(q,(ast.If,ast.For,ast.While))),None);assert ctl is not None,(str(p),z.lineno)
    st=ctl.iter if isinstance(ctl,ast.For)else ctl.test
   assert st.end_lineno-st.lineno<85
   stmts[st.lineno]=dict(start=st.lineno,end=st.end_lineno,owner=owner,enclosing_arms=guards,source=[str(j+1)+': '+ls[j]for j in range(st.lineno-1,st.end_lineno)])
   uses.append(dict(field=ast.unparse(z),line=z.lineno,statement_start=st.lineno,owner=owner,enclosing_arms=guards))
 elif mode=='wrapper':
  fs=[f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.lineno<=455<=f.end_lineno];assert len(fs)==1
  for f in fs:
   assert f.end_lineno-f.lineno<140;functions.append(dict(name=f.name,start=f.lineno,end=f.end_lineno,source=[str(j+1)+': '+ls[j]for j in range(f.lineno-1,f.end_lineno)]))
 elif mode=='KV_owner':
  for z in ast.walk(t):
   if not isinstance(z,(ast.Assign,ast.AnnAssign)):continue
   targets=z.targets if isinstance(z,ast.Assign)else[z.target];names={q.attr for target in targets for q in ast.walk(target)if isinstance(q,ast.Attribute)}
   if not(names&{'wkv','kv_norm','self_attn'}):continue
   anc,guards,owner=context(z);owners=[q.name for q in ast.walk(t)if isinstance(q,ast.ClassDef)and q.lineno<=z.lineno<=q.end_lineno]
   assert z.end_lineno-z.lineno<45
   stmts[z.lineno]=dict(start=z.lineno,end=z.end_lineno,owner=owner,class_owners=owners,attributes=sorted(names),enclosing_arms=guards,source=[str(j+1)+': '+ls[j]for j in range(z.lineno-1,z.end_lineno)])
 rows.append(dict(path=str(p),sha256=sha,mode=mode,class_headers=[dict(name=c.name,line=c.lineno,bases=[ast.unparse(b)for b in c.bases])for c in t.body if isinstance(c,ast.ClassDef)],function_headers=[dict(name=f.name,start=f.lineno,end=f.end_lineno,signature=ast.unparse(f.args))for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.name in {u['owner']for u in uses}],field_uses=uses,statements=list(stmts.values()),functions=functions))
assert sum(sum(len(q['source'])for q in r['statements'])+sum(len(f['source'])for f in r['functions'])for r in rows)<560
print(json.dumps(dict(files=rows,scope='OnlyactualDSparksubclassprofile-identifiedfirstpassandcontextpersistentfielduses/guards/classbases. BaseDFlash122had0profilehit;notassumedcurrentexecution. Noothermodel/allocatoraudit.',no_model_or_torch_import=True)))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
