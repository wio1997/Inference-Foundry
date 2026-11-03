from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,subprocess,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'attrInfo-export-route-leaf-source-v23.json';assert not OUT.exists()
p=M/'export-route-localization-and-graph-attribute-reuse-v23-v2.json';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='815649f17092d0ab895e2375f356d424845c96da7992a55acd2f29e26b65a2f9';d=json.loads(b)
stored=[dict(path=f['path'],sha256=f['sha256'],run_functions=[q for q in f['functions']if q['name']=='run'])for f in d['installed_torch_export_route']['files']]
code=r'''from pathlib import Path
import ast,json,hashlib
p=Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis/msprof/msprof.py');b=p.read_bytes();s=b.decode();ls=s.splitlines();tree=ast.parse(s)
terms=['is_cpp_enable','Cpp','CPP','cpp','--type','--export','export_type','export_db']
hits=[i+1 for i,l in enumerate(ls)if any(k in l for k in terms)];funcs=[]
for n in ast.walk(tree):
 if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and any(n.lineno<=i<=n.end_lineno for i in hits):
  funcs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]if n.end_lineno-n.lineno<=180 else None))
points={i for h in hits for i in range(max(0,h-4),min(len(ls),h+3))}
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),imports=[str(i+1)+': '+l for i,l in enumerate(ls)if l.startswith(('from ','import '))],exact_route_context=[str(i+1)+': '+ls[i]for i in sorted(points)],functions=funcs)))'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=25);assert r.returncode==0,r.stderr
leaf=json.loads(r.stdout)
result=dict(created_utc=datetime.now(timezone.utc).isoformat(),stored_torch_route=stored,one_hop_msprof_entry=leaf,scope='Only stored Torch export run functions and one exact msprof.py entry file selected by them. No whole-SDK source audit.',no_model_import_or_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
