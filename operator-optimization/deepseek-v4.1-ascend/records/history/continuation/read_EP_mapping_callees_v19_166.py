from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'EP-Ascend-remap-and-fused-input-callees-v19.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
ar='/vllm-workspace/vllm-ascend/vllm_ascend/'
specs=[(ar+'ops/fused_moe/routed_experts.py',[(402,411),(493,578),(598,617),(638,691),(720,760)]),(ar+'ops/fused_moe/dataclass/fused_experts.py',[(1,30),(48,144)])]
rows=[]
for name,ranges in specs:
 p=Path(name);s=p.read_text();ls=s.splitlines();rows.append(dict(path=name,sha256=hashlib.sha256(s.encode()).hexdigest(),excerpts=[dict(start=a,end=b,lines=[str(i+1)+': '+ls[i]for i in range(a-1,b)])for a,b in ranges],weight_loader_override_definitions=[dict(name=n.name,start=n.lineno,end=n.end_lineno)for n in ast.walk(ast.parse(s))if isinstance(n,ast.FunctionDef)and n.name=='weight_loader']))
print(json.dumps(dict(files=rows,no_module_import_or_device_work=True)))
'''
q=subprocess.run(['docker','exec','bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d','python3','-c',code],capture_output=True,text=True,timeout=80);assert q.returncode==0,q.stderr
v=json.loads(q.stdout);v.update(created_utc=datetime.now(timezone.utc).isoformat(),scope='Two exact callees/imported Ascend override files for log2phy/loader inheritance and input payload; no repository traversal or model import.',overall_complete=False)
b=(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=v,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
