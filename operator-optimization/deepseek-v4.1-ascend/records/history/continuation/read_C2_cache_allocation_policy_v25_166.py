from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'C2-cache-allocation-and-binding-policy-v25.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
V=Path('/vllm-workspace/vllm/vllm');A=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
specs=[(V/'v1/core/kv_cache_utils.py','2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2',{'get_uniform_page_size','unify_kv_cache_spec_page_size','_get_packed_kv_cache_groups','get_kv_cache_groups','get_kv_cache_config_from_groups'},[]),(V/'v1/kv_cache_interface.py','1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb',{'unpadded_page_size_bytes'},[]),(A/'attention/dsa_v41.py','3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd',set(),[(790,825),(1427,1452)]),(A/'worker/model_runner_v1.py',None,set(),[(4770,4815)])]
files=[]
for p,expected,names,ranges in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if expected:assert sha==expected
 ls=b.decode().splitlines();tree=ast.parse(b.decode());fs=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.FunctionDef)and n.name in names:
   if p.name=='kv_cache_interface.py':
    # The exact inherited AttentionSpec byte formula,not other specs' overrides.
    owners=[c.name for c in ast.walk(tree)if isinstance(c,ast.ClassDef)and c.lineno<=n.lineno<=c.end_lineno]
    if owners!=['AttentionSpec']:continue
   assert n.end_lineno-n.lineno<=180,(str(p),n.name)
   fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 points={i for a,z in ranges for i in range(a-1,min(z,len(ls)))}
 files.append(dict(path=str(p),sha256=sha,functions=fs,exact_ranges=ranges,range_lines=[str(i+1)+': '+ls[i]for i in sorted(points)]))
print(json.dumps(dict(files=files,no_model_or_torch_import=True)))'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),scope='Exact inherited pageformula,KVgroup allocation routines andcurrentcachebackend/binding ranges;no repository audit or modelexecution.',no_model_import_or_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
