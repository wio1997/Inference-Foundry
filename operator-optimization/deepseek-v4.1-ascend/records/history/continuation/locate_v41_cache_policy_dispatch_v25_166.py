from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'V41-tuple-cache-policy-dispatch-v25.json';assert not OUT.exists()
code=r'''from pathlib import Path
import subprocess,ast,json,hashlib
B=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
terms=['get_deepseek_v41_kv_cache_config','get_deepseek_v41_pool_bytes_per_block','group_cache_specs']
cmd=['grep','-R','-n','-F','--include=*.py']
for t in terms:cmd+=['-e',t]
cmd+=[str(B)]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=20);assert r.returncode in(0,1),r.stderr
hits=r.stdout.splitlines();assert len(hits)<=100
byfile={}
for l in hits:
 p,n,text=l.split(':',2);byfile.setdefault(p,[]).append(dict(line=int(n),text=text))
assert len(byfile)<=8
out=[]
for path,hh in byfile.items():
 p=Path(path);bb=p.read_bytes();ls=bb.decode().splitlines();tree=ast.parse(bb.decode());fs=[]
 for f in ast.walk(tree):
  if isinstance(f,ast.FunctionDef)and any(f.lineno<=h['line']<=f.end_lineno for h in hh):
   fs.append(dict(name=f.name,start=f.lineno,end=f.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(f.lineno-1,f.end_lineno)]if f.end_lineno-f.lineno<=180 else None))
 points={i for h in hh for i in range(max(0,h['line']-4),min(len(ls),h['line']+3))}
 # Retain module-level installation assignments/calls for the exact threehook names.
 for node in tree.body:
  if isinstance(node,(ast.Assign,ast.Expr))and any(t in ast.get_source_segment(bb.decode(),node)for t in terms):points.update(range(node.lineno-1,node.end_lineno))
 out.append(dict(path=path,sha256=hashlib.sha256(bb).hexdigest(),hits=hh,functions=fs,exact_hook_context=[str(i+1)+': '+ls[i]for i in sorted(points)]))
print(json.dumps(dict(search_terms=terms,files=out,no_model_or_torch_import=True)))'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),scope='Bounded exactcallee hook/localization andmatching smallfunctions only;no repository semantic audit.',no_model_import_or_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
