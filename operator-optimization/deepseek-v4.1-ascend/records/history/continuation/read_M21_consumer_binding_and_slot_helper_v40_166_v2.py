from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'M21-consumer-binding-and-existing-slot-helper-v40-v2.json';assert not OUT.exists()
code=r'''
from pathlib import Path
import ast,hashlib,json
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
expected={'models/deepseek_v41/indexer.py':'d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33','models/deepseek_v41/mixed_quant_attention.py':'a5d32c875617fe0e6c85061c7152328b3c20830a1cdf5a27698ee0d45db22ae3','device/device_op.py':'01ce13ea60e46f901ad69ad4c71e4be0e31cd69fbc56ebb7d898f957d5e54f67'}
files=[]
for rel,sha in expected.items():
 p=A/rel;b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha,(rel,hashlib.sha256(b).hexdigest());files.append(dict(path=str(p),sha256=sha,mode='EXACT_HASH_ONLY'))
p=A/'ops/triton/packed_cache_slot_mapping.py';b=p.read_bytes();t=ast.parse(b.decode());ls=b.decode().splitlines()
wrapper=next(f for f in t.body if isinstance(f,ast.FunctionDef)and f.name=='build_packed_cache_slot_mapping')
launches={z.func.value.id for z in ast.walk(wrapper)if isinstance(z,ast.Call)and isinstance(z.func,ast.Subscript)and isinstance(z.func.value,ast.Name)}
assert launches
functions=[]
for f in[wrapper]+[f for f in t.body if isinstance(f,ast.FunctionDef)and f.name in launches]:
 assert f.end_lineno-f.lineno<280,(f.name,f.lineno,f.end_lineno)
 s=min([f.lineno]+[z.lineno for z in f.decorator_list]);functions.append(dict(name=f.name,start=s,end=f.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(s-1,f.end_lineno)]))
assert sum(len(f['source'])for f in functions)<480
files.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),functions=functions,mode='EXACT_WRAPPER_AND_DIRECT_KERNEL'))
print(json.dumps(dict(files=files,no_model_or_torch_import=True,scope='Only threehashbindingsandnamedexistingpacked-slotwrapper/directkernel absentfromlocalfreeze;noadditionalcode/discovery.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=40);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
