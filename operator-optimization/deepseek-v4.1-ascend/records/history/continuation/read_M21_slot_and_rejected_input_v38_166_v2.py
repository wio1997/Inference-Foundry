from pathlib import Path
from datetime import datetime, timezone
import subprocess, json, hashlib, socket

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
M = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT = M/'M21-profile-selected-slot-and-rejected-input-contract-v38-v2.json'
assert not OUT.exists()
code = r'''
from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
specs=[
 (A/'ops/triton/compute_slot_mapping.py','slots'),
 (A/'worker/block_table.py','slot_owner'),
 (A/'spec_decode/llm_base_proposer.py','reject'),
 (A/'worker/model_runner_v1.py','call_sites'),
]
out=[]
for p,mode in specs:
 b=p.read_bytes();ls=b.decode().splitlines();t=ast.parse(b.decode())
 par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)}
 ff=[];ss={}
 def src(s,e):return [str(j+1)+': '+ls[j]for j in range(s-1,e)]
 def at(line):return min((f for f in ast.walk(t)if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))and f.lineno<=line<=f.end_lineno),key=lambda f:f.end_lineno-f.lineno)
 def take(f,limit):
  assert f.end_lineno-f.lineno<limit,(p,f.name,f.lineno,f.end_lineno)
  start=min([f.lineno]+[x.lineno for x in f.decorator_list])
  ff.append(dict(name=f.name,start=start,end=f.end_lineno,source=src(start,f.end_lineno)))
 def statement(line):
  z=min((z for z in ast.walk(t)if isinstance(z,ast.stmt)and not isinstance(z,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.If,ast.For,ast.While,ast.With,ast.Try))and z.lineno<=line<=z.end_lineno),key=lambda z:z.end_lineno-z.lineno)
  assert z.end_lineno-z.lineno<65,(p,line,z.end_lineno-z.lineno)
  q=z;guards=[]
  while q in par:
   a=par[q]
   if isinstance(a,ast.If):guards.append(dict(line=a.lineno,test=ast.unparse(a.test),arm='BODY'if q in a.body else'ORELSE'if q in a.orelse else'TEST'))
   if isinstance(a,(ast.FunctionDef,ast.AsyncFunctionDef)):break
   q=a
  ss[z.lineno]=dict(start=z.lineno,end=z.end_lineno,source=src(z.lineno,z.end_lineno),guards=guards)
 if mode=='slots':
  names=['_compute_slot_mapping_fused_groups_kernel','_compute_slot_mapping_fused_groups_adaptive_kernel','_select_slot_mapping_launch_config','compute_slot_mapping_fused_groups']
  for name in names:
   hits=[f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.name==name]
   assert len(hits)==1,(name,[f.name for f in t.body if isinstance(f,ast.FunctionDef)])
   take(hits[0],210)
 elif mode=='slot_owner':
  f=at(479);assert f.name=='compute_slot_mapping';take(f,150)
 elif mode=='reject':
  f=at(2414);assert f.name=='prepare_inputs_padded';take(f,170)
  # Only imports that identify the exact kernel called from the selected method.
  for z in t.body:
   if isinstance(z,(ast.Import,ast.ImportFrom))and 'prepare_inputs_padded' in ast.unparse(z):
    ss[z.lineno]=dict(start=z.lineno,end=z.end_lineno,source=src(z.lineno,z.end_lineno),guards=[])
 elif mode=='call_sites':
  for line in [1522,1599,1604,1606,1622,1645,1836,1837,2082]:statement(line)
 out.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),functions=ff,statements=list(ss.values())))
assert sum(sum(len(z['source'])for z in f['functions']+f['statements'])for f in out)<900
print(json.dumps(dict(files=out,scope='Only profile-positive 24 ordinary/64 adaptive slot mapping and 72 rejected-input kernel direct call/ownership contracts. Kernel import is localization only; no other SDK or model body.',no_model_or_torch_import=True)))
'''
r = subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=40)
assert r.returncode == 0, r.stderr
result=json.loads(r.stdout)
result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode()
OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
