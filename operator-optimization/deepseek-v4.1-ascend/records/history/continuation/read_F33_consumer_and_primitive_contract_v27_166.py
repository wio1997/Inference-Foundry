from pathlib import Path
from datetime import datetime, timezone
import subprocess, json, hashlib, socket

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
out=M/'F33-consumer-and-native-primitive-point-contract-v27.json'
assert not out.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
specs=[('attention/dsa_v41.py','3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd',
        ['_write_compressed_source','_write_swa_cache','_write_index_cache','_write_long_cache','_apply_rotary']),
       ('ops/linear.py',None,['unquantized_gemm']),
       ('ops/cv_linear.py',None,['matmul']),
       ('ops/layernorm.py',None,['forward_oot'])]
rows=[]
for rel,expected,names in specs:
 p=A/rel;b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if expected: assert sha==expected
 ls=b.decode().splitlines();tree=ast.parse(b.decode());functions=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.FunctionDef)and n.name in names:
   assert n.end_lineno-n.lineno<180,(rel,n.name,'too large;stop rather than expand')
   owners=[c.name for c in ast.walk(tree)if isinstance(c,ast.ClassDef)and c.lineno<=n.lineno<=c.end_lineno]
   functions.append(dict(name=n.name,owners=owners,start=n.lineno,end=n.end_lineno,
     source=[str(i+1)+': '+ls[i] for i in range(n.lineno-1,n.end_lineno)]))
 assert functions,(rel,names)
 rows.append(dict(path=str(p),sha256=sha,requested_names=names,
    missing_names=sorted(set(names)-{f['name'] for f in functions}),functions=functions))
print(json.dumps(dict(files=rows,no_model_or_torch_import=True,
 scope='Named compressor/cache consumers and actual native frame primitive methods only;no repository audit.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45)
assert r.returncode==0,r.stderr
result=json.loads(r.stdout)
result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
