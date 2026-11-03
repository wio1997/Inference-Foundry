from pathlib import Path
import json,subprocess,socket,hashlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
code="""from pathlib import Path
import ast,json,hashlib,re
files=[('/vllm-workspace/vllm-ascend/vllm_ascend/worker/block_table.py',{'commit_block_table','append_row','add_row','move_row','swap_row','clear_row'}),('/vllm-workspace/vllm/vllm/v1/utils.py',{'copy_to_gpu'})]
results=[]
for name,wanted in files:
 p=Path(name);s=p.read_text();ls=s.splitlines();tree=ast.parse(s);functions=[]
 for node in ast.walk(tree):
  if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in wanted:
   functions.append(dict(name=node.name,start=node.lineno,end=node.end_lineno,source=[str(i+1)+': '+ls[i] for i in range(node.lineno-1,node.end_lineno)]))
 results.append(dict(path=name,sha256=hashlib.sha256(s.encode()).hexdigest(),functions=functions,dirty_or_generation_symbols=[str(i+1)+': '+line for i,line in enumerate(ls) if re.search(r'dirty|generation|modified|changed',line,re.I)]))
print(json.dumps(results))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
result=json.loads(r.stdout)
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
out=P/'block-table-copy-contract-v13.json';assert not out.exists();payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
