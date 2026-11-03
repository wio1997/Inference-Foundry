from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=M/'M17-exact-recovery-random-and-block-entropy-kernel-contract-v33.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend');V=Path('/vllm-workspace/vllm/vllm');root=A/'sample/rejection_sampler.py';b=root.read_bytes();assert hashlib.sha256(b).hexdigest()=='bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda';tree=ast.parse(b);imports={}
for n in ast.walk(tree):
 if not isinstance(n,ast.ImportFrom):continue
 module=n.module or''
 if n.level:module='.'.join('vllm_ascend.sample.rejection_sampler'.split('.')[:-n.level])+('.'+module if module else'')
 for a in n.names:imports[a.asname or a.name]=(module,a.name,n.lineno)
def module_path(module):
 if module.startswith('vllm_ascend.'):return A.joinpath(*module.split('.')[1:]).with_suffix('.py')
 if module.startswith('vllm.'):return V.joinpath(*module.split('.')[1:]).with_suffix('.py')
 raise AssertionError(('unexpectedmodule',module))
def read_function(p,name,from_import=None):
 blob=p.read_bytes();text=blob.decode();ls=text.splitlines();t=ast.parse(text);fs=[f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.name==name];assert len(fs)==1,(str(p),name)
 f=fs[0];assert f.end_lineno-f.lineno<310,(name,f.lineno,f.end_lineno,'namedkerneltoosized;stop')
 return dict(path=str(p),sha256=hashlib.sha256(blob).hexdigest(),name=name,start=f.lineno,end=f.end_lineno,arguments=[a.arg for a in f.args.args],resolved_import=from_import,source=[str(j+1)+': '+ls[j]for j in range(f.lineno-1,f.end_lineno)])
rows=[read_function(root,'sample_recovered_tokens')]
for name in['sample_recovered_tokens_kernel','rejection_random_sample_kernel','rejection_random_sample_block_verify_kernel']:
 local=[f for f in ast.walk(tree)if isinstance(f,ast.FunctionDef)and f.name==name]
 if local:rows.append(read_function(root,name));continue
 assert name in imports,('exactkernelimportnotresolved',name)
 module,symbol,line=imports[name];rows.append(read_function(module_path(module),symbol,dict(local_name=name,module=module,symbol=symbol,line=line)))
assert sum(len(f['source'])for f in rows)<720
print(json.dumps(dict(functions=rows,scope='Onlyexactsample_recovered_tokensandthreeexplicitprobabilitykernelcallees. Subscriptlaunchcalleeimportsresolvedwithoutmodelimports. Nofull305-linerejectionhelper/nootherkernels.',no_model_or_torch_import=True)))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
