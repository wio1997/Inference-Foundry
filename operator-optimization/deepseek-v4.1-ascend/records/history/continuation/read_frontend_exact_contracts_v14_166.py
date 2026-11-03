from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
code="""from pathlib import Path
import ast,json,hashlib,re
specs=[('/vllm-workspace/vllm/vllm/renderers/base.py',{'_tokenize_prompt','_tokenize_singleton_prompt_async','tokenize_prompt_async'},None),('/vllm-workspace/vllm/vllm/v1/engine/detokenizer.py',set(),{96,169,211,224})]
results=[]
for name,names,starts in specs:
 p=Path(name);s=p.read_text();ls=s.splitlines();tree=ast.parse(s);functions=[]
 for node in ast.walk(tree):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and(node.name in names or (starts and node.lineno in starts)):
   functions.append(dict(name=node.name,start=node.lineno,end=node.end_lineno,decorators=[ast.unparse(d)for d in node.decorator_list],source=[str(i+1)+': '+ls[i]for i in range(node.lineno-1,node.end_lineno)]))
 related=[str(i+1)+': '+line for i,line in enumerate(ls)if re.search(r'ThreadPoolExecutor|max_workers|_thread_pool|_executor|make_async|lru_cache|cache_info',line)]
 results.append(dict(path=name,sha256=hashlib.sha256(s.encode()).hexdigest(),functions=functions,executor_cache_reference_lines=related))
print(json.dumps(results))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14')
OUT=P/'frontend-exact-contracts-v14.json';assert not OUT.exists();payload=(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),results=result,no_target_import_or_device_work=True),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
public=dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest());assert len(json.dumps(public).encode())<31000;print(json.dumps(public,ensure_ascii=False))
