from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'dispatcher-initialization-contract-v16.json';assert not OUT.exists()
code="""from pathlib import Path
import ast,json,hashlib
p=Path('/vllm-workspace/vllm/vllm/v1/cudagraph_dispatcher.py');s=p.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='51f632753af729aeee6ce4a802d9d74b64f22eef02a524f36e2a24b07ef0cf63';ls=s.splitlines();fs=[]
for c in ast.walk(ast.parse(s)):
 if isinstance(c,ast.ClassDef)and c.name=='CudagraphDispatcher':
  for n in c.body:
   if isinstance(n,ast.FunctionDef)and n.name in ['__init__','initialize_cudagraph_keys']:
    fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=fs)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);payload=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),source=result,no_model_import_or_device_work=True,overall_complete=False),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
