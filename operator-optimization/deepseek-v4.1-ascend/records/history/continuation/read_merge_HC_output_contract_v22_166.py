from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket,re
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18');OUT=P/'merge-HC-output-source-contract-v22.json';assert not OUT.exists()
code=r'''from pathlib import Path
import ast,json,hashlib,re
root=Path('/vllm-workspace/vllm-ascend/vllm_ascend');rows=[]
p=root/'ops/fused_moe/fused_moe.py';s=p.read_text();sha=hashlib.sha256(s.encode()).hexdigest();assert sha=='2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4';ls=s.splitlines();tree=ast.parse(s);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='AscendMoERunner');funcs={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)};selected={'forward'};calls=[]
for n in ast.walk(funcs['forward']):
 if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute):
  calls.append(dict(line=n.lineno,target=ast.unparse(n.func)))
  if isinstance(n.func.value,ast.Name)and n.func.value.id=='self'and n.func.attr in funcs and re.search(r'scal|reduc|final|padding|zero|transform',n.func.attr):selected.add(n.func.attr)
fs=[]
for name in sorted(selected,key=lambda k:funcs[k].lineno):
 n=funcs[name];assert n.end_lineno-n.lineno<=130;fs.append(dict(class_name=cls.name,name=name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
rows.append(dict(path=str(p),sha256=sha,functions=fs,forward_calls=calls,imports=[str(i+1)+': '+l for i,l in enumerate(ls)if i<85 and 'import 'in l]))
p=root/'models/deepseek_v41/model.py';s=p.read_text();sha=hashlib.sha256(s.encode()).hexdigest();assert sha=='101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91';ls=s.splitlines();tree=ast.parse(s);fs=[]
for cls in tree.body:
 if isinstance(cls,ast.ClassDef)and cls.name=='DeepseekV41DecoderLayer':
  for n in cls.body:
   if isinstance(n,ast.FunctionDef)and n.name in['forward','hc_post']:
    assert n.end_lineno-n.lineno<=90;fs.append(dict(class_name=cls.name,name=n.name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
fs.append(dict(class_name='DeepseekV41MoE',name='forward_tail',start=300,end=328,source=[str(i+1)+': '+ls[i]for i in range(299,328)]))
rows.append(dict(path=str(p),sha256=sha,functions=fs))
print(json.dumps(dict(files=rows,no_module_import_or_device_work=True)))
'''
q=subprocess.run(['docker','exec','bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d','python3','-c',code],capture_output=True,text=True,timeout=80);assert q.returncode==0,q.stderr
v=json.loads(q.stdout);v.update(created_utc=datetime.now(timezone.utc).isoformat(),scope='Exact merge/scaling/posthook sourcefunctions andmodelconsumer tails in2pinnedfiles;no model import/sourcechange/devicework.',overall_complete=False)
b=(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(result=v,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
