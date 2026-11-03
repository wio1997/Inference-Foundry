from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'embedding-and-dispatch-point-contracts-v16.json';assert not OUT.exists()
code="""from pathlib import Path
import ast,json,hashlib
specs=[(Path('/vllm-workspace/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py'),{'_forward_standard'}),(Path('/vllm-workspace/vllm/vllm/v1/cudagraph_dispatcher.py'),{'dispatch'})]
rows=[]
for p,names in specs:
 s=p.read_text();ls=s.splitlines();tree=ast.parse(s);fs=[]
 for n in ast.walk(tree):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name in names:
   fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 rows.append(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=fs,function_signatures=[dict(name=n.name,line=n.lineno,end=n.end_lineno)for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)]))
print(json.dumps(rows))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);payload=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),source=result,no_model_import_or_device_work=True,overall_complete=False),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
