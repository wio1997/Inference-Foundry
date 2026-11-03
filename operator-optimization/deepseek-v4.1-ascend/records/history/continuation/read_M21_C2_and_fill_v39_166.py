from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
OUT=M/'M21-profile-selected-C2-metadata-and-fill-contract-v39.json';assert not OUT.exists()
code=r'''
from pathlib import Path
import ast,json,hashlib
A=Path('/vllm-workspace/vllm-ascend/vllm_ascend')
specs=[(A/'attention/dsa_v41.py','3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd','C2'),(A/'worker/model_runner_v1.py','568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c','fill')]
out=[]
for p,expected,mode in specs:
 b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha==expected
 ls=b.decode().splitlines();t=ast.parse(b.decode());par={ch:z for z in ast.walk(t)for ch in ast.iter_child_nodes(z)};ff=[];ss={}
 def src(s,e):return [str(j+1)+': '+ls[j]for j in range(s-1,e)]
 def at(line):return min((f for f in ast.walk(t)if isinstance(f,ast.FunctionDef)and f.lineno<=line<=f.end_lineno),key=lambda f:f.end_lineno-f.lineno)
 def take(f,expected,limit):
  assert f.name==expected and f.end_lineno-f.lineno<limit,(f.name,f.lineno,f.end_lineno)
  ff.append(dict(name=f.name,start=f.lineno,end=f.end_lineno,source=src(f.lineno,f.end_lineno)))
 def statement(line):
  z=min((z for z in ast.walk(t)if isinstance(z,ast.stmt)and not isinstance(z,(ast.FunctionDef,ast.ClassDef,ast.If,ast.For,ast.With,ast.Try,ast.While))and z.lineno<=line<=z.end_lineno),key=lambda z:z.end_lineno-z.lineno)
  assert z.end_lineno-z.lineno<90
  q=z;guards=[]
  while q in par:
   a=par[q]
   if isinstance(a,ast.If):guards.append(dict(line=a.lineno,test=ast.unparse(a.test),arm='BODY'if q in a.body else'ORELSE'if q in a.orelse else'TEST'))
   if isinstance(a,ast.FunctionDef):break
   q=a
  ss[z.lineno]=dict(start=z.lineno,end=z.end_lineno,source=src(z.lineno,z.end_lineno),guards=guards)
 if mode=='C2':take(at(1327),'build_c2_metadata',150)
 else:
  take(at(3583),'_get_block_table_and_slot_mapping',120)
  for line in[1462,3587,3851]:statement(line)
 out.append(dict(path=str(p),sha256=sha,functions=ff,statements=list(ss.values())))
assert sum(sum(len(z['source'])for z in f['functions']+f['statements'])for f in out)<420
print(json.dumps(dict(files=out,no_model_or_torch_import=True,scope='Only 30positiveC2metadatachains3256TASK,8block-slotfillchains936TASK andworker1462fill88;first read exact producer anddirectcallguard,not SDK or wholebuilder.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=40);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
