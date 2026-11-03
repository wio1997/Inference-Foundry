from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'M21-exact-parallel-device-and-dynamic-dispatch-predicates-v42.json';assert not OUT.exists()
code=r'''
from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
specs=[('attention/utils.py','pcp'),('attention/context_parallel/common_cp.py','decode'),('ascend_config.py','config'),('device/device_op.py','device'),('utils.py','type'),('_build_info.py','build')]
out=[]
for rel,mode in specs:
 p=A/rel;b=p.read_bytes();ls=b.decode().splitlines();t=ast.parse(b.decode());par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};ff=[];ss={};headers=[]
 def src(s,e):return[str(i+1)+': '+ls[i]for i in range(s-1,e)]
 def take(f,limit=100):
  assert f.end_lineno-f.lineno<limit,(p,f.name,f.lineno,f.end_lineno)
  q=par.get(f);ff.append(dict(owner=q.name if isinstance(q,ast.ClassDef) else None,name=f.name,start=f.lineno,end=f.end_lineno,source=src(f.lineno,f.end_lineno)))
 def collect(z):
  q=z
  while q in par and not isinstance(q,(ast.Assign,ast.AnnAssign)):q=par[q]
  if isinstance(q,(ast.Assign,ast.AnnAssign)):
   assert q.end_lineno-q.lineno<70
   ss[q.lineno]=dict(start=q.lineno,end=q.end_lineno,source=src(q.lineno,q.end_lineno))
 if mode=='pcp':
  take(next(f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.name=='enable_pcp'))
 elif mode=='decode':
  take(next(f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.name=='is_pcp_decode_sharding_enabled'))
 elif mode=='config':
  for z in ast.walk(t):
   if isinstance(z,ast.ClassDef)and('DynamicSpec'in z.name):
    headers.append(dict(name=z.name,start=z.lineno,end=z.end_lineno,bases=[ast.unparse(q)for q in z.bases]))
    for f in z.body:
     if isinstance(f,(ast.Assign,ast.AnnAssign)):collect(f)
   elif isinstance(z,ast.Name)and isinstance(z.ctx,ast.Store)and z.id in['enable_dsa_cp','dynamic_spec_config']:collect(z)
   elif isinstance(z,ast.Attribute)and isinstance(z.ctx,ast.Store)and z.attr in['enable_dsa_cp','dynamic_spec_config']:collect(z)
 elif mode=='device':
  for cls in t.body:
   if not isinstance(cls,ast.ClassDef):continue
   headers.append(dict(name=cls.name,start=cls.lineno,end=cls.end_lineno,bases=[ast.unparse(q)for q in cls.bases]))
   for f in cls.body:
    if isinstance(f,ast.FunctionDef)and f.name in['get_dsv41_packed_cache_ops','__getattr__','__new__']:take(f)
  for f in t.body:
   if isinstance(f,ast.FunctionDef):headers.append(dict(function=f.name,start=f.lineno,end=f.end_lineno))
 elif mode=='type':
  for f in t.body:
   if isinstance(f,ast.FunctionDef)and f.name in['get_device_type','enable_dsa_cp']:take(f)
 elif mode=='build':
  assert len(b)<4096
  for z in t.body:
   if isinstance(z,(ast.Assign,ast.AnnAssign)):collect(z)
 out.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),functions=ff,statements=list(ss.values()),headers=headers))
assert sum(sum(len(z['source'])for z in f['functions']+f['statements'])for f in out)<700
print(json.dumps(dict(files=out,no_model_or_torch_import=True,scope='ExactnamedenablePCP/decode-shard/configfield/devicefactory/buildmetadata predicates. Missinglocalfreezehelperspoint-readonstage;no SDK/repo audit.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
