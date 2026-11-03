from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-communications-v15')
OUT=P/'C1-and-padding-adjustment-contracts-v16.json';assert not OUT.exists()
code="""from pathlib import Path
import ast,json,hashlib
specs=[('/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/compressor.py','31516e5ba234e34fa01e73e864fbc88bd6473d4a5274a1d43da8a79bab3cf494',{'__init__','forward'}),('/vllm-workspace/vllm/vllm/config/compilation.py','c9cec5c7200e8e559810ec8c30113ad61dab6780f9f7fb3c116bd0d9b5a43065',{'adjust_cudagraph_sizes_for_spec_decode'}),('/vllm-workspace/vllm/vllm/config/vllm.py','956b812e5a719bcbfa3a3958801361b38b9c928c96b8c073311bbb96376dfb7a',{'update_sizes_for_sequence_parallelism'})]
rows=[]
for name,sha,names in specs:
 p=Path(name);s=p.read_text();assert hashlib.sha256(s.encode()).hexdigest()==sha;ls=s.splitlines();fs=[]
 for n in ast.walk(ast.parse(s)):
  if isinstance(n,ast.FunctionDef)and n.name in names:fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 points=set()
 for i,l in enumerate(ls):
  if 'adjust_cudagraph_sizes_for_spec_decode('in l and 'def 'not in l:points.update(range(max(0,i-7),min(len(ls),i+8)))
 rows.append(dict(path=name,sha256=sha,functions=fs,exact_call_contexts=[str(i+1)+': '+ls[i]for i in sorted(points)]))
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/platform.py');s=p.read_text();sha=hashlib.sha256(s.encode()).hexdigest();assert sha=='0f4e619517058f36b700a521e93c449d78b269c1d29ed1a18cbc312f2d67b082';ls=s.splitlines();points=set(range(1273,1301))
for i,l in enumerate(ls):
 if 'adjust_cudagraph_sizes_for_spec_decode('in l:points.update(range(max(0,i-7),min(len(ls),i+8)))
rows.append(dict(path=str(p),sha256=sha,exact_branch_lines=[str(i+1)+': '+ls[i]for i in sorted(points)]))
print(json.dumps(rows))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);payload=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),source=result,no_model_import_or_device_work=True,overall_complete=False),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
