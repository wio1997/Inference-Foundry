from pathlib import Path
from datetime import datetime, timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'M21-slot-helper-ownership-and-rejection-state-v38.json';assert not OUT.exists()
code=r'''
from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');B=Path('/vllm-workspace/vllm/vllm')
specs=[
 (A/'ops/triton/compute_slot_mapping.py','f349fdd683bc36ef3f13b78b91c0e84b930d4850f172220e2627e5ea98de996b','helper'),
 (A/'worker/block_table.py','215305c699a9df9c118f47c6f7b71d8457ef54e98b5b8571536fd373de7b21ed','banks'),
 (A/'ops/triton/spec_decode/utils.py',None,'reject_kernel'),
 (A/'spec_decode/utils.py',None,'state_update'),
 (B/'v1/worker/gpu_model_runner.py','87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51','input_ids'),
]
out=[]
for p,expected,mode in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if expected is not None:assert sha==expected,(p,sha)
 ls=b.decode().splitlines();t=ast.parse(b.decode());par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};ff=[];ss={}
 def src(s,e):return [str(j+1)+': '+ls[j]for j in range(s-1,e)]
 def at(line):return min((f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.lineno<=line<=f.end_lineno),key=lambda f:f.end_lineno-f.lineno)
 def take(f,limit):
  assert f.end_lineno-f.lineno<limit,(p,f.name,f.lineno,f.end_lineno)
  s=min([f.lineno]+[z.lineno for z in f.decorator_list]);ff.append(dict(name=f.name,start=s,end=f.end_lineno,source=src(s,f.end_lineno)))
 def collect(z):
  q=z;stmt=None;guards=[]
  while q in par:
   a=par[q]
   if stmt is None and isinstance(q,(ast.Assign,ast.AnnAssign,ast.Expr,ast.Return)):stmt=q
   if isinstance(a,ast.If):guards.append(dict(line=a.lineno,test=ast.unparse(a.test),arm='BODY'if q in a.body else'ORELSE'if q in a.orelse else'TEST'))
   if isinstance(a,ast.FunctionDef):break
   q=a
  if stmt is not None:
   assert stmt.end_lineno-stmt.lineno<90,(p,stmt.lineno)
   ss[stmt.lineno]=dict(start=stmt.lineno,end=stmt.end_lineno,source=src(stmt.lineno,stmt.end_lineno),guards=guards)
 if mode=='helper':
  take(next(f for f in t.body if isinstance(f,ast.FunctionDef)and f.name=='_compute_slot_mapping_request'),160)
 elif mode=='banks':
  # Only stores to fused-bank fields and the gate, plus initialization statements
  # containing these fields. No unrelated block-table implementation.
  for f in ast.walk(t):
   if isinstance(f,ast.FunctionDef)and f.name=='__init__':
    for z in ast.walk(f):
     if isinstance(z,ast.Attribute)and isinstance(z.ctx,ast.Store)and(z.attr.startswith('_fused_')or z.attr=='_can_fuse_slot_mapping'):collect(z)
 elif mode=='reject_kernel':
  take(next(f for f in t.body if isinstance(f,ast.FunctionDef)and f.name=='prepare_inputs_padded_kernel'),110)
 elif mode=='state_update':
  f=at(75);assert f.name=='update_num_computed_tokens_for_batch_change';take(f,130)
 elif mode=='input_ids':
  f=at(1896);assert f.name=='_prepare_input_ids';take(f,180)
 out.append(dict(path=str(p),sha256=sha,functions=ff,statements=list(ss.values())))
assert sum(sum(len(z['source'])for z in f['functions']+f['statements'])for f in out)<600
print(json.dumps(dict(files=out,no_model_or_torch_import=True,scope='Only direct helper from adaptive slot kernel, fused-bank field stores, exact72padded-input kernel,1224TASK previous-batch-state update and720TASK sampled-input packing function. No other module body.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=40);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
