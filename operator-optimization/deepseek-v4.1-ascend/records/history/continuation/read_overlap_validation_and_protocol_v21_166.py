from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=P/'shared-overlap-validation-call-and-protocol-v21.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
root=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
p=root/'ascend_config.py';s=p.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='d5ea2cd9a5499e5c0d7a41114005569d3c9092027e0cf03510c40dc4951ac4cd';ls=s.splitlines();tree=ast.parse(s);nodes=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)];points=set();calls=[];fs=[]
for n in nodes:
 if n.name in['get_ascend_config','init_ascend_config','from_vllm_config']and n.end_lineno-n.lineno<=90:fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
for i,l in enumerate(ls):
 if any(t in l for t in ['.derive_and_validate(', '._check_mix_placement(','.validate(', 'def _check_mix_placement']):
  pp=sorted([n for n in nodes if n.lineno<=i+1<=n.end_lineno],key=lambda n:n.end_lineno-n.lineno);calls.append(dict(line=i+1,text=l,enclosing=dict(name=pp[0].name,start=pp[0].lineno,end=pp[0].end_lineno)if pp else None));points.update(range(max(0,i-7),min(len(ls),i+9)))
config=dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=fs,call_context=[str(i+1)+': '+ls[i]for i in sorted(points)],calls=calls)
p=root/'ops/fused_moe/shared_experts.py';s=p.read_text();assert hashlib.sha256(s.encode()).hexdigest()=='471db26e9e9e4ba70c2ccd43502ad39bff6699bbd4ca191319a9b66fc20458f9';ls=s.splitlines();tree=ast.parse(s);names={'validate_consistency','_wait_for_milestone','_wait_for_routed_stage','_prepare_execution_input','_run_a8_int_mlp','_finish_shared_mlp','_select_mlp_path','wait_for_output'};fs=[]
for n in ast.walk(tree):
 if isinstance(n,ast.FunctionDef)and n.name in names:
  assert n.end_lineno-n.lineno<=100
  fs.append(dict(name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
assert {f['name']for f in fs}==names
shared=dict(path=str(p),sha256=hashlib.sha256(s.encode()).hexdigest(),functions=fs)
print(json.dumps(dict(config=config,shared_protocol=shared,no_module_import_or_device_work=True)))
'''
q=subprocess.run(['docker','exec','bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d','python3','-c',code],capture_output=True,text=True,timeout=80);assert q.returncode==0,q.stderr
v=json.loads(q.stdout);v.update(created_utc=datetime.now(timezone.utc).isoformat(),scope='Exact config validation callsites and8shared stage/numerical protocol functions;no model import/currentenv/device work.',overall_complete=False)
b=(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=v,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
