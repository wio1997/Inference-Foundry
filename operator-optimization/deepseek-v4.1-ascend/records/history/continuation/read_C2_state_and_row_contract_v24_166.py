from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'C2-pool-ring-norm-and-row-source-contract-v24.json';assert not OUT.exists()
code=r'''from pathlib import Path
import json,hashlib,ast
B=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
specs=[('models/deepseek_v41/compressor.py','31516e5ba234e34fa01e73e864fbc88bd6473d4a5274a1d43da8a79bab3cf494',{'pool_projected'},[]),('ops/triton/compressor/compressor_triton.py','1c26f996fa91bcd3ae497e1ae33e88a2e0bf38c2e734f8af85baee06492a87ee',{'_pooled_blocked','_pool_kernel','_cache_update_kernel','compressor_from_projected'},[(16,84),(366,410)]),('attention/dsa_v41.py','3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd',{'_write_compressed_source'},[(195,242),(1283,1388)])]
files=[]
for rel,sha,names,ranges in specs:
 p=B/rel;b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;ls=b.decode().splitlines();tree=ast.parse(b.decode());fs=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.FunctionDef)and n.name in names:
   assert n.end_lineno-n.lineno<=140
   fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 points={i for a,z in ranges for i in range(a-1,min(z,len(ls)))}
 files.append(dict(path=str(p),sha256=sha,functions=fs,exact_row_contract_ranges=ranges,exact_row_contract_lines=[str(i+1)+': '+ls[i]for i in sorted(points)]))
print(json.dumps(dict(files=files,no_model_or_torch_import=True)))'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),scope='Only3pinned files/named C2functions and specific existing row-mapping ranges;not fullmodule/SDK audit.',no_model_import_or_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
