from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
out=M/'sampler-native-greedy-and-rejection-branch-contract-v28.json';assert not out.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');V=Path('/vllm-workspace/vllm/vllm')
specs=[(V/'v1/sample/sampler.py',[('forward',97),('sample',261)]),
       (A/'sample/sampler.py',[('greedy_sample',84)]),
       (A/'sample/rejection_sampler.py',[('forward',224),('rejection_sample',472)]),
       (A/'ops/triton/reject_sample.py',[('rejection_greedy_sample_with_triton',456)]),
       (A/'worker/model_runner_v1.py',[('_sample',2980)])]
rows=[]
for p,symbols in specs:
 b=p.read_bytes();ls=b.decode().splitlines();tree=ast.parse(b.decode());fs=[]
 for name,line in symbols:
  matches=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name==name and n.lineno<=line<=n.end_lineno]
  assert len(matches)==1,(str(p),name,len(matches));n=matches[0]
  assert n.end_lineno-n.lineno<260,(str(p),name,'too large;stop')
  owners=[c.name for c in ast.walk(tree)if isinstance(c,ast.ClassDef)and c.lineno<=n.lineno<=c.end_lineno]
  fs.append(dict(name=name,owners=owners,start=n.lineno,end=n.end_lineno,profile_frame_line=line,
     source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 rows.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),functions=fs))
print(json.dumps(dict(files=rows,no_model_or_torch_import=True,scope='Nativeprofileframe-bound sampling methods only;not wholemodule/repository audit.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=45);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
