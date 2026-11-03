from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'C2-cache-reader-and-cache-spec-contract-v25.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib,subprocess
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');V=Path('/vllm-workspace/vllm/vllm')
def extract_classes(p,class_names,expected=None,whole_limit=170):
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest()
 if expected:assert sha==expected
 s=b.decode();ls=s.splitlines();tree=ast.parse(s);classes={c.name:c for c in ast.walk(tree)if isinstance(c,ast.ClassDef)};needed=set(class_names);pending=list(class_names)
 while pending:
  name=pending.pop();assert name in classes,(str(p),name)
  for base in classes[name].bases:
   bn=ast.unparse(base)
   if bn in classes and bn not in needed:needed.add(bn);pending.append(bn)
 result=[]
 wanted={'page_size_bytes','real_page_size_bytes','max_memory_usage_bytes','merge','copy_with_new_block_size','__post_init__'}
 for name in sorted(needed):
  node=classes[name];fs=[]
  for q in node.body:
   if isinstance(q,ast.FunctionDef)and(name in class_names or q.name in wanted):
    assert q.end_lineno-q.lineno<=180,(name,q.name)
    fs.append(dict(name=q.name,start=q.lineno,end=q.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(q.lineno-1,q.end_lineno)]))
  fields=[str(i+1)+': '+ls[i]for q in node.body if isinstance(q,(ast.AnnAssign,ast.Assign))for i in range(q.lineno-1,q.end_lineno)]
  result.append(dict(name=name,start=node.lineno,end=node.end_lineno,bases=[ast.unparse(x)for x in node.bases],fields=fields,functions=fs))
 return dict(path=str(p),sha256=sha,classes=result,imports=[str(i+1)+': '+l for i,l in enumerate(ls)if l.startswith(('from ','import '))])
files=[extract_classes(A/'attention/dsa_v41.py',{'DeepseekV41CacheLayer'},'3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd'),extract_classes(V/'v1/kv_cache_interface.py',{'CircularBufferSpec'}),extract_classes(A/'core/kv_cache_interface.py',{'AscendCircularBufferSpec'})]
p=A/'models/deepseek_v41/cache_config.py';b=p.read_bytes();ls=b.decode().splitlines();tree=ast.parse(b.decode());functions=[]
for q in ast.walk(tree):
 if isinstance(q,ast.FunctionDef)and q.name=='uses_a5_packed_cache':functions.append(dict(name=q.name,start=q.lineno,end=q.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(q.lineno-1,q.end_lineno)]))
points={i for j,l in enumerate(ls)if 'STATE_RING_ROWS'in l for i in range(max(0,j-2),min(len(ls),j+3))}
files.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),functions=functions,exact_constant_context=[str(i+1)+': '+ls[i]for i in sorted(points)]))
p=A/'models/deepseek_v41/compressor.py';b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='31516e5ba234e34fa01e73e864fbc88bd6473d4a5274a1d43da8a79bab3cf494'
files.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),state_cache_mentions=[dict(line=i+1,text=l)for i,l in enumerate(b.decode().splitlines())if 'state_cache'in l]))
roots=[str(A/'models/deepseek_v41'),str(A/'attention/dsa_v41.py'),str(A/'attention/context_parallel/dsa_v41_cp.py')]
q=subprocess.run(['grep','-R','-n','-E','--include=*.py','compressor\.state_cache|self\.state_cache|\.state_cache\.kv_cache',*roots],capture_output=True,text=True,timeout=20);assert q.returncode in(0,1),q.stderr;hits=q.stdout.splitlines();assert len(hits)<=160,('boundedstateobjectlocalization exceeded',len(hits))
print(json.dumps(dict(files=files,state_object_identifier_localization=dict(roots=roots,hits=hits,meaning='Identifier matches only;not whole-program alias or external-reader closure.'),no_model_or_torch_import=True)))'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=50);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),scope='Exact current compressor cache-object/class/spec dependency subtree and boundedstate identifier localization;not whole-repository audit.',no_model_import_or_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
