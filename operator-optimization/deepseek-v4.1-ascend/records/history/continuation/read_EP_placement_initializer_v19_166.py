from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'EP-placement-initializer-callee-v19.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/eplb/core/eplb_utils.py');s=p.read_text();ls=s.splitlines();tree=ast.parse(s);nodes=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='init_eplb_config'];assert len(nodes)==1
fs=[]
for n in nodes:
 assert n.end_lineno-n.lineno<=250
 fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,lines=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
imports=[str(i+1)+': '+l for i,l in enumerate(ls)if i<90 and ('import 'in l or 'expert_map'in l)]
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=fs,imports=imports,no_module_import_or_device_work=True)))
'''
q=subprocess.run(['docker','exec','bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d','python3','-c',code],capture_output=True,text=True,timeout=80);assert q.returncode==0,q.stderr
v=json.loads(q.stdout);v.update(created_utc=datetime.now(timezone.utc).isoformat(),scope='Exact init_eplb_config named callee of observed Ascend expert-selection mapping contract; no model import.',overall_complete=False)
b=(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=v,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
