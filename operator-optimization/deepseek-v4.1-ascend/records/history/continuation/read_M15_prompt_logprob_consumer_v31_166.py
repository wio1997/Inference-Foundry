from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=M/'M15-exact-bookkeeping-prompt-logprob-content-consumer-v31.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
rows=[]
specs=[(Path('/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'),'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c','_bookkeeping_sync','identifier'),(Path('/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py'),'87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51','_get_prompt_logprobs_dict','function')]
for p,expected,name,mode in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha==expected;text=b.decode();ls=text.splitlines();t=ast.parse(text);fs=[f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.name==name];assert len(fs)==1;f=fs[0]
 row=dict(path=str(p),sha256=sha,function=name,start=f.lineno,end=f.end_lineno)
 if mode=='function':
  assert f.end_lineno-f.lineno<150;row['source']=[str(j+1)+': '+ls[j]for j in range(f.lineno-1,f.end_lineno)]
 else:
  par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};uses=[]
  for z in ast.walk(f):
   if not isinstance(z,ast.Name)or z.id!='hidden_states':continue
   ancestors=[];q=z
   while q in par:
    q=par[q];ancestors.append(q)
    if q is f:break
   stmt=next(q for q in ancestors if isinstance(q,(ast.Assign,ast.AnnAssign,ast.Expr,ast.Return)));assert stmt.end_lineno-stmt.lineno<25
   uses.append(dict(line=z.lineno,source=[str(j+1)+': '+ls[j]for j in range(stmt.lineno-1,stmt.end_lineno)],enclosing_conditions=[dict(line=a.lineno,condition=ast.get_source_segment(text,a.test))for a in ancestors if isinstance(a,ast.If)]))
  row['hidden_state_uses']=uses;row['base_import']=[dict(module=z.module,line=z.lineno)for z in ast.walk(t)if isinstance(z,ast.ImportFrom)and any(a.name=='GPUModelRunner'for a in z.names)]
 rows.append(row)
print(json.dumps(dict(files=rows,scope='Onlyexactnormalized-hiddenbookkeepinguseandresolvedinherited_get_prompt_logprobs_dict;notallGPUrunner.',no_model_or_torch_import=True)))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
