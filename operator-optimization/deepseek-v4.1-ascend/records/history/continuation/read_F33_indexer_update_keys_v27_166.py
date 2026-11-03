from pathlib import Path
from datetime import datetime, timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
out=M/'F33-indexer-update-keys-latent-consumer-v27.json';assert not out.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py');b=p.read_bytes();ls=b.decode().splitlines();tree=ast.parse(b.decode());fs=[]
for n in ast.walk(tree):
 if isinstance(n,ast.FunctionDef)and n.name=='update_keys':
  assert n.end_lineno-n.lineno<140
  fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
assert len(fs)==1
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),functions=fs,no_model_or_torch_import=True)))'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
