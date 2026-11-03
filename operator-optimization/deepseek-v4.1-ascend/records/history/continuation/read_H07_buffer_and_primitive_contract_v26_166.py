from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'H07-persistent-buffer-and-embedding-primitive-contract-v26.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');V=Path('/vllm-workspace/vllm/vllm');rows=[]
p=V/'model_executor/layers/vocab_parallel_embedding.py';b=p.read_bytes();ls=b.decode().splitlines();tree=ast.parse(b.decode());fs=[]
for n in ast.walk(tree):
 if isinstance(n,ast.FunctionDef)and n.name=='embedding':
  owners=[c.name for c in ast.walk(tree)if isinstance(c,ast.ClassDef)and c.lineno<=n.lineno<=c.end_lineno];assert n.end_lineno-n.lineno<=90
  fs.append(dict(name=n.name,owners=owners,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
rows.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),functions=fs))
for p,expected in [(V/'v1/worker/gpu_model_runner.py','87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51'),(A/'worker/model_runner_v1.py','568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c')]:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha==expected;ls=b.decode().splitlines();tree=ast.parse(b.decode());hits=[];points=set()
 for i,l in enumerate(ls):
  if 'self.inputs_embeds'in l:
   owners=[dict(name=n.name,start=n.lineno,end=n.end_lineno)for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.lineno<=i+1<=n.end_lineno]
   hits.append(dict(line=i+1,text=l,enclosing_functions=owners));points.update(range(max(0,i-4),min(len(ls),i+5)))
 assert len(hits)<=100
 # Track the local handoff value only in the actual execute/sample methods.
 handoffs=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.FunctionDef)and n.name in ['execute_model','sample_tokens','propose_draft_token_ids']:
   for i in range(n.lineno-1,n.end_lineno):
    if 'inputs_embeds'in ls[i]:handoffs.append(dict(function=n.name,line=i+1,text=ls[i]));points.update(range(max(n.lineno-1,i-3),min(n.end_lineno,i+4)))
 rows.append(dict(path=str(p),sha256=sha,persistent_buffer_identifier_hits=hits,local_value_handoffs=handoffs,exact_use_context=[str(i+1)+': '+ls[i]for i in sorted(points)]))
p=A/'models/deepseek_v41/vl_model.py';b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha=='bb98a2e2a0e24c18aafe17229eb593937aaed2e37c1fd7cc2003c45ce872d282';ls=b.decode().splitlines();tree=ast.parse(b.decode());fs=[]
for n in ast.walk(tree):
 if isinstance(n,ast.FunctionDef)and n.name=='embed_input_ids':
  assert n.end_lineno-n.lineno<=100
  fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
rows.append(dict(path=str(p),sha256=sha,functions=fs))
p=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch/nn/functional.py');b=p.read_bytes();ls=b.decode().splitlines();tree=ast.parse(b.decode());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='embedding');points=set(range(fn.lineno-1,fn.lineno+14))
for n in ast.walk(fn):
 if isinstance(n,ast.If)and 'max_norm' in ast.unparse(n.test):points.update(range(n.lineno-1,n.end_lineno))
for i in range(fn.lineno-1,fn.end_lineno):
 if 'return torch.embedding'in ls[i]:points.update(range(i,min(fn.end_lineno,i+3)))
rows.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),function='embedding',start=fn.lineno,end=fn.end_lineno,exact_default_and_mutation_branch=[str(i+1)+': '+ls[i]for i in sorted(points)]))
print(json.dumps(dict(files=rows,scope='One namedembeddingprimitive, two knownrunnerbuffer identifier/handoff subtrees, one VLembed method and installedTorch max_norm guard;not broadmodelaudit.',no_model_or_torch_import=True)))'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_or_device_or_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
