from pathlib import Path
import json,subprocess,socket,hashlib
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
code="""from pathlib import Path
import ast,json,hashlib
files=[('/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py',None),('/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py',None)]
primary=Path('/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py');text=primary.read_text();tree=ast.parse(text)
files.append((str(primary),'AscendCommonAttentionMetadata'))
for node in tree.body:
 if isinstance(node,ast.ImportFrom):
  for a in node.names:
   if (a.asname or a.name)=='CommonAttentionMetadata':
    assert node.module.startswith('vllm.')
    files.append(('/vllm-workspace/vllm/'+node.module.replace('.','/')+'.py','CommonAttentionMetadata'))
results=[]
for path,clsname in files:
 p=Path(path);b=p.read_bytes();rec=dict(path=path,sha256=hashlib.sha256(b).hexdigest())
 if clsname:
  lines=b.decode().splitlines();tree=ast.parse(b.decode());matches=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==clsname];assert len(matches)==1,(path,clsname)
  cls=matches[0];rec.update(class_name=clsname,bases=[ast.unparse(z) for z in cls.bases],decorators=[ast.unparse(z) for z in cls.decorator_list],special_methods=[],all_method_names=[z.name for z in cls.body if isinstance(z,(ast.FunctionDef,ast.AsyncFunctionDef))])
  rec['related_imports']=[ast.unparse(z) for z in tree.body if isinstance(z,(ast.Import,ast.ImportFrom)) and ('dataclass' in ast.unparse(z) or 'CommonAttentionMetadata' in ast.unparse(z) or 'copy' in ast.unparse(z))]
  for z in cls.body:
   if isinstance(z,(ast.FunctionDef,ast.AsyncFunctionDef)) and z.name in ['replace','__init__','__post_init__','__getattribute__','__getattr__','__setattr__','seq_lens_cpu']:
    rec['special_methods'].append(dict(name=z.name,start=z.lineno,end=z.end_lineno,source=[str(i+1)+': '+lines[i] for i in range(z.lineno-1,z.end_lineno)]))
  rec['seq_lens_field_declarations']=[str(z.lineno)+': '+ast.unparse(z) for z in cls.body if isinstance(z,ast.AnnAssign) and any(k in ast.unparse(z.target) for k in ['seq_lens','query_start_loc'])]
 results.append(rec)
print(json.dumps(dict(results=results,no_target_import=True)))
"""
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
result=json.loads(r.stdout)
expected={'/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py':'3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd','/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py':'7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305'}
for rec in result['results']:
 if rec['path']in expected:assert rec['sha256']==expected[rec['path']]
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
out=P/'metadata-replace-and-source-contract-v13.json';assert not out.exists();payload=json.dumps(result,ensure_ascii=False,indent=2)+'\n';out.write_text(payload)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(payload.encode()).hexdigest()),ensure_ascii=False))
