from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
out=M/'Engram-selected-kernel-full-write-and-history-priority-v30.json';assert not out.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');V=Path('/vllm-workspace/vllm/vllm');rows=[]
p=A/'models/deepseek_v41/engram/npu.py';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='2b2594a2126ef50ee6d77f508ab8e0d2f5b360d15ea700eea0d39923af8bc64f';tree=ast.parse(b.decode());imports=[ast.unparse(n)for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))and'engram'in ast.unparse(n)]
specs=[(p,['gather_dequantize_host_uva']),
       (A/'ops/triton/engram_hash.py',['_engram_req_index_kernel','_hash_ids_kernel']),
       (V/'models/deepseek_v41/common/engram.py',['_write_hash_cache_kernel'])]
# Resolve only the exact named UVA kernel from the already identified wrapper's import.
source_modules=[]
for n in tree.body:
 if isinstance(n,ast.ImportFrom)and any(a.name=='_engram_host_uva_gather_dequant_kernel'for a in n.names):source_modules.append(n.module)
for module in source_modules:
 assert module and module.startswith('vllm_ascend.')
 target=A/Path(*module.split('.')[1:]);specs.append((target.with_suffix('.py'),['_engram_host_uva_gather_dequant_kernel']))
# A module-level definition is also allowed, but never search the repository.
if any(isinstance(n,ast.FunctionDef)and n.name=='_engram_host_uva_gather_dequant_kernel'for n in tree.body):specs[0][1].append('_engram_host_uva_gather_dequant_kernel')
for p,names in specs:
 b=p.read_bytes();ls=b.decode().splitlines();tree=ast.parse(b.decode());fs=[]
 for name in names:
  matches=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name==name];assert len(matches)==1,(str(p),name,len(matches));n=matches[0]
  assert n.end_lineno-n.lineno<260,(str(p),name,'too large;stop')
  fs.append(dict(name=name,start=n.lineno,end=n.end_lineno,source=[str(j+1)+': '+ls[j]for j in range(n.lineno-1,n.end_lineno)]))
 rows.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),functions=fs))
assert any(f['name']=='_engram_host_uva_gather_dequant_kernel'for r in rows for f in r['functions']),'exactkernelimportunresolved;stop'
assert sum(len(f['source'])for r in rows for f in r['functions'])<700
print(json.dumps(dict(files=rows,wrapper_imports=imports,no_model_or_torch_import=True,scope='OnlyselectedUVAwrapper/exactimportedkernel andhashhistory/writekernels;noSDKorrepositorysearch.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
