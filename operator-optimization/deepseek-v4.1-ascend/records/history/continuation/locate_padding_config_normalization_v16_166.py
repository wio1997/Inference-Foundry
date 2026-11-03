from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'padding-config-normalization-symbols-v16.json';assert not OUT.exists()
code="""from pathlib import Path
import ast,json,hashlib
paths=['/vllm-workspace/vllm-ascend/vllm_ascend/platform.py','/vllm-workspace/vllm-ascend/vllm_ascend/utils.py','/vllm-workspace/vllm/vllm/config/compilation.py','/vllm-workspace/vllm/vllm/config/vllm.py','/vllm-workspace/vllm/vllm/config/speculative.py']
rows=[]
for name in paths:
 p=Path(name)
 if not p.exists():rows.append(dict(path=name,exists=False));continue
 s=p.read_text();ls=s.splitlines();tree=ast.parse(s);points=set();matching=[]
 for i,l in enumerate(ls):
  if 'cudagraph_capture_sizes'in l or 'cudagraph_capture_size_multiple'in l:
   points.update(range(max(0,i-4),min(len(ls),i+5)));matching.append(i+1)
 fs=[dict(name=n.name,start=n.lineno,end=n.end_lineno)for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and any(n.lineno<=i<=n.end_lineno for i in matching)]
 rows.append(dict(path=name,sha256=hashlib.sha256(s.encode()).hexdigest(),exact_matching_lines=matching,enclosing_function_locations=fs,context=[str(i+1)+': '+ls[i]for i in sorted(points)]))
print(json.dumps(rows))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);payload=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),source=result,no_model_import_or_device_work=True,overall_complete=False),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
