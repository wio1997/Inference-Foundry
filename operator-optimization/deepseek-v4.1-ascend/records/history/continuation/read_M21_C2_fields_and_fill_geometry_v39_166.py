from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'M21-C2-field-readers-and-initialization-geometry-v39.json';assert not OUT.exists()
code=r'''
from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
specs=[(A/'attention/dsa_v41.py','3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd','C2_fields'),(A/'worker/model_runner_v1.py','568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c','geometry'),(A/'worker/block_table.py','215305c699a9df9c118f47c6f7b71d8457ef54e98b5b8571536fd373de7b21ed','active_groups')]
out=[]
for p,expected,mode in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha==expected
 ls=b.decode().splitlines();t=ast.parse(b.decode());par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};ss={};sigs=[]
 def src(s,e):return [str(j+1)+': '+ls[j]for j in range(s-1,e)]
 def at(line):return min((f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.lineno<=line<=f.end_lineno),key=lambda f:f.end_lineno-f.lineno)
 def collect(z):
  q=z;stmt=None;guards=[];fn=None
  while q in par:
   a=par[q]
   if stmt is None and isinstance(q,(ast.Assign,ast.AnnAssign,ast.Expr,ast.Return,ast.Import,ast.ImportFrom)):stmt=q
   if isinstance(a,ast.If):guards.append(dict(line=a.lineno,test=ast.unparse(a.test),arm='BODY'if q in a.body else'ORELSE'if q in a.orelse else'TEST'))
   if isinstance(a,ast.FunctionDef):fn=a.name;break
   q=a
  if stmt is not None:
   assert stmt.end_lineno-stmt.lineno<150,(p,stmt.lineno,stmt.end_lineno)
   ss[stmt.lineno]=dict(function=fn,start=stmt.lineno,end=stmt.end_lineno,source=src(stmt.lineno,stmt.end_lineno),guards=guards)
 if mode=='C2_fields':
  build=at(1327);outer=par[build]
  while not isinstance(outer,ast.FunctionDef):outer=par[outer]
  c2names={'ring_meta','full_source_cos','full_source_sin','input_positions','skip_ring_update','num_actual_reqs','num_actual_tokens','num_input_tokens'}
  for z in ast.walk(t):
   if isinstance(z,ast.Attribute)and(z.attr.startswith('_c2_')or z.attr.startswith('c2_')):
    # The callback itself was already read; only external definitions/readers now.
    if not build.lineno<=z.lineno<=build.end_lineno:collect(z)
   elif isinstance(z,ast.Name)and z.id in c2names and isinstance(z.ctx,ast.Store)and outer.lineno<=z.lineno<=outer.end_lineno and not build.lineno<=z.lineno<=build.end_lineno:collect(z)
   elif isinstance(z,ast.keyword)and z.arg is not None and z.arg.startswith('c2_'):collect(z)
   elif isinstance(z,(ast.Import,ast.ImportFrom))and 'build_c2_ring_metadata'in ast.unparse(z):collect(z)
 elif mode=='geometry':
  f=at(3583);outer=par[f]
  while not isinstance(outer,ast.FunctionDef):outer=par[outer]
  sigs.append(dict(name=outer.name,line=outer.lineno,args=ast.unparse(outer.args)))
  for z in ast.walk(outer):
   if isinstance(z,ast.Name)and isinstance(z.ctx,ast.Store)and z.id in['num_tokens','num_tokens_padded','num_reqs','num_reqs_padded']:collect(z)
  call=next(z for z in ast.walk(t)if isinstance(z,ast.Call)and isinstance(z.func,ast.Attribute)and z.func.attr=='_build_attention_metadata'and z.lineno<=2495<=z.end_lineno)
  collect(call)
 elif mode=='active_groups':
  for z in ast.walk(t):
   if isinstance(z,ast.Name)and isinstance(z.ctx,ast.Store)and z.id=='active_block_tables':collect(z)
   elif isinstance(z,(ast.Assign,ast.AnnAssign,ast.Import,ast.ImportFrom))and 'PAD_SLOT_ID'in ast.unparse(z)and z.lineno<60:collect(z)
 out.append(dict(path=str(p),sha256=sha,statements=list(ss.values()),signatures=sigs))
assert sum(sum(len(z['source'])for z in f['statements'])for f in out)<800
print(json.dumps(dict(files=out,no_model_or_torch_import=True,scope='Only named C2 bank definitions/readers/returned fields andexact fused-builder import;only initializedbank geometry assignments/directcaller/PAD import/activegroupdefinition. No wholebuilder/SDK read.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=40);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
