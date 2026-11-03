from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
out=M/'F04-MC2-row-prepare-ID-alignment-and-finalize-contract-v29.json';assert not out.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe');rows=[]
specs=[(A/'moe_comm_method.py',[(None,'get_moe_comm_method'),('MoECommMethod','prepare'),('MC2CommImpl','__init__'),('MC2CommImpl','pad_and_split_input_ids')]),
       (A/'prepare_finalize.py',[('PrepareAndFinalizeWithMC2','__init__'),('PrepareAndFinalizeWithMC2','prepare'),('PrepareAndFinalizeWithMC2','finalize'),('PrepareAndFinalizeWithMC2','pad_and_split_input_ids')])]
for p,symbols in specs:
 b=p.read_bytes();ls=b.decode().splitlines();tree=ast.parse(b.decode());fs=[];missing=[]
 for cls,name in symbols:
  scope=tree if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls)
  matches=[n for n in scope.body if isinstance(n,ast.FunctionDef)and n.name==name]
  if not matches:missing.append(dict(cls=cls,name=name));continue
  assert len(matches)==1;n=matches[0];assert n.end_lineno-n.lineno<150
  fs.append(dict(class_name=cls,name=name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
 rows.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),functions=fs,missing_methods=missing))
assert sum(len(f['source'])for r in rows for f in r['functions'])<340
print(json.dumps(dict(files=rows,no_model_or_torch_import=True,scope='NamedMC2prepare/IDalignment/finalize andfactory only;noothercommunicationbackendsorsdk source audit.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=35);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
