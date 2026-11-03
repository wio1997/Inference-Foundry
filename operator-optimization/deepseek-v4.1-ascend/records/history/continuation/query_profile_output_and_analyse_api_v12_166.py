from pathlib import Path
import json,hashlib,socket,subprocess
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
x=json.loads((P/'capture-index.json').read_text());files=x['files']
code="""from pathlib import Path
import ast,json,hashlib
p=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/profiler.py');s=p.read_text();tree=ast.parse(s);lines=s.splitlines();out=[]
for node in tree.body:
 if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='analyse':
  out.append(dict(name=node.name,start=node.lineno,end=node.end_lineno,lines=[str(i+1)+': '+lines[i] for i in range(node.lineno-1,node.end_lineno)]))
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=out)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
info=json.loads(r.stdout)
public=dict(file_count=len(files),all_database_paths=[f for f in files if f['path'].endswith('.db')],trace_file_examples=[f for f in files if 'trace_view' in f['path']][:2],all_file_names=sorted(set(Path(f['path']).name for f in files))[:100],SDK_analyse=info)
out=P/'analysis/profile-output-and-analyse-api-v12.json';assert not out.exists();payload=json.dumps(public,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(result=public,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
