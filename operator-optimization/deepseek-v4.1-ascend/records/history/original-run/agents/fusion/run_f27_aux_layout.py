#!/usr/bin/env python3
"""Read EagleModelMixin setter in the two authorized containers; no imports/device."""
import ast, hashlib, json, shlex, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]
remote = '''import ast,hashlib,json\nfrom pathlib import Path\np=Path("/vllm-workspace/vllm/vllm/model_executor/models/interfaces.py")\nb=p.read_bytes();s=b.decode("utf-8");t=ast.parse(s)\nc=[n for n in t.body if isinstance(n,ast.ClassDef) and n.name=="EagleModelMixin"]\nassert len(c)==1\nm=[n for n in c[0].body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name=="_cache_aux_pp_layout"]\nassert len(m)==1\nprint(json.dumps({"path":str(p),"source_sha256":hashlib.sha256(b).hexdigest(),"class_bases":[ast.unparse(x) for x in c[0].bases],"method":ast.get_source_segment(s,m[0]),"line":m[0].lineno},ensure_ascii=True))'''
ssh='/Users/wio/work/169-sglang-deepseek-v41/ssh169.sh'
containers=['wio-vllm-v41-repro169-stage-20261028']
result=[]
for name in containers:
 r=subprocess.run([ssh,shlex.join(['docker','exec',name,'python3','-c',remote])],capture_output=True,text=True,timeout=90)
 if r.returncode: raise RuntimeError('read-only query failed '+str(r.returncode)+' '+r.stderr[-1000:])
 v=json.loads(r.stdout);v['container']=name;result.append(v)
p=root/'agents/fusion/f27-aux-layout-evidence.json'
body=(json.dumps({'scope':'static AST extraction only; no import or NPU','result':result},ensure_ascii=False,indent=2)+'\n').encode()
if p.is_symlink():raise RuntimeError('refuse symlink')
if p.exists():
 if p.read_bytes()!=body:raise RuntimeError('refuse overwrite')
else:
 with p.open('xb') as f:f.write(body)
print(json.dumps({'output_sha256':hashlib.sha256(body).hexdigest(),'source_sha256':result[0]['source_sha256']}))
