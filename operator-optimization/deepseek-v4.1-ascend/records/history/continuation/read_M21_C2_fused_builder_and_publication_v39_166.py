from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'M21-existing-C2-fused-builder-publication-and-PAD-constant-v39.json';assert not OUT.exists()
code=r'''
from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');B=Path('/vllm-workspace/vllm/vllm')
specs=[(A/'ops/triton/c2_ring_metadata.py',None,'fused'),(A/'attention/dsa_v41.py','3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd','publication'),(B/'v1/attention/backends/utils.py',None,'pad_constant')]
out=[]
for p,expected,mode in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if expected is not None:assert sha==expected
 ls=b.decode().splitlines();t=ast.parse(b.decode());par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};ff=[];ss={}
 def src(s,e):return [str(j+1)+': '+ls[j]for j in range(s-1,e)]
 def take(f,limit):
  assert f.end_lineno-f.lineno<limit,(p,f.name,f.lineno,f.end_lineno)
  s=min([f.lineno]+[z.lineno for z in f.decorator_list]);ff.append(dict(name=f.name,start=s,end=f.end_lineno,source=src(s,f.end_lineno)))
 def collect(z):
  q=z;stmt=None;guards=[];fn=None
  while q in par:
   a=par[q]
   if stmt is None and isinstance(q,(ast.Assign,ast.AnnAssign,ast.Expr,ast.Return)):stmt=q
   if isinstance(a,ast.If):guards.append(dict(line=a.lineno,test=ast.unparse(a.test),arm='BODY'if q in a.body else'ORELSE'if q in a.orelse else'TEST'))
   if isinstance(a,ast.FunctionDef):fn=a.name;break
   q=a
  if stmt is not None:
   assert stmt.end_lineno-stmt.lineno<90
   ss[stmt.lineno]=dict(function=fn,start=stmt.lineno,end=stmt.end_lineno,source=src(stmt.lineno,stmt.end_lineno),guards=guards)
 if mode=='fused':
  wrapper=next(f for f in t.body if isinstance(f,ast.FunctionDef)and f.name=='build_c2_ring_metadata');take(wrapper,220)
  launches={z.func.value.id for z in ast.walk(wrapper)if isinstance(z,ast.Call)and isinstance(z.func,ast.Subscript)and isinstance(z.func.value,ast.Name)}
  assert launches
  for name in sorted(launches):take(next(f for f in t.body if isinstance(f,ast.FunctionDef)and f.name==name),280)
 elif mode=='publication':
  take(next(f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.name=='_publish_task'),100)
  for z in ast.walk(t):
   if isinstance(z,ast.Name)and isinstance(z.ctx,ast.Store)and(z.id.startswith('c2_')or z.id in['compressor_tokens','compressor_reqs']):collect(z)
 elif mode=='pad_constant':
  for z in t.body:
   if isinstance(z,(ast.Assign,ast.AnnAssign)):
    names={x.id for x in ast.walk(z)if isinstance(x,ast.Name)and isinstance(x.ctx,ast.Store)}
    if 'PAD_SLOT_ID'in names:collect(z)
  assert ss,'No literal PAD definition found;do not read other module body.'
 out.append(dict(path=str(p),sha256=sha,functions=ff,statements=list(ss.values())))
assert sum(sum(len(z['source'])for z in f['functions']+f['statements'])for f in out)<750
print(json.dumps(dict(files=out,no_model_or_torch_import=True,scope='Exact preexisting C2 fused wrapper/directlaunchedkernel,small publish helper/named bank aliases;one imported PADconstant. Reusev24consumer source withsamehash,noSDKorentirebuilder.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=40);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
