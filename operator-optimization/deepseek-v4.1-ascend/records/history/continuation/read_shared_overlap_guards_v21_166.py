from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket,re
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72');OUT=P/'analysis/native-PMU-complete-v18/shared-overlap-guard-sources-v21.json';assert not OUT.exists()
env=P/'runtime/config/serve.env';b=env.read_bytes();assert hashlib.sha256(b).hexdigest()=='569c372788ba565df9a116f6cded3060e2f12caac07f8c3c304515093a2b381a'
matches=[]
for i,line in enumerate(b.decode().splitlines(),1):
 name,sep,value=line.removeprefix('export ').partition('=')
 if sep and re.search(r'overlap|shared|multi.?stream|fused_mc2|graph',name,re.I)and not re.search(r'key|secret|password|credential|auth',name,re.I):matches.append(dict(line=i,name=name,value=value))
code=r'''from pathlib import Path
import ast,json,hashlib,re
ar=Path('/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe')
specs=[('fused_moe.py','2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4','AscendMoERunner',{'__init__','_forward_impl','_prepare_shared_for_router_overlap','_prepare_router_and_milestones'}),('shared_experts.py',None,None,{'__init__','prepare_for_router_overlap','enqueue_gate_up_after_router','forward'})]
rows=[]
for file,sha,clsn,wanted in specs:
 p=ar/file;s=p.read_text();actual=hashlib.sha256(s.encode()).hexdigest()
 if sha:assert actual==sha
 ls=s.splitlines();tree=ast.parse(s);classes=[n for n in tree.body if isinstance(n,ast.ClassDef)and(clsn is None or n.name==clsn)];fs=[];loc=[]
 for cls in classes:
  funcs={n.name:n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))};selected={k for k in funcs if k in wanted}
  for name in list(selected):
   for n in ast.walk(funcs[name]):
    if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='self'and n.func.attr in funcs and re.search(r'can|should|overlap|mode|stream|enable',n.func.attr):selected.add(n.func.attr)
  for name in sorted(selected,key=lambda k:funcs[k].lineno):
   n=funcs[name];assert n.end_lineno-n.lineno<=200
   fs.append(dict(class_name=cls.name,name=name,start=n.lineno,end=n.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(n.lineno-1,n.end_lineno)]))
  loc.extend(dict(class_name=cls.name,name=n.name,start=n.lineno,end=n.end_lineno)for n in funcs.values())
 imports=[str(i+1)+': '+l for i,l in enumerate(ls)if i<85 and('import 'in l or 'envs' in l)]
 rows.append(dict(path=str(p),sha256=actual,functions=fs,function_locations=loc,imports=imports))
print(json.dumps(dict(files=rows,no_module_import_or_device_work=True)))
'''
q=subprocess.run(['docker','exec','bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d','python3','-c',code],capture_output=True,text=True,timeout=80);assert q.returncode==0,q.stderr
v=json.loads(q.stdout);v.update(created_utc=datetime.now(timezone.utc).isoformat(),saved_serve_env=dict(path=str(env),sha256=hashlib.sha256(b).hexdigest(),selected_fields=matches),scope='Two exact profile-raised MoE guard/prepare/enqueue source files andone savedenv file;no repo traversal,currentenv or model import.',overall_complete=False)
payload=(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
print(json.dumps(dict(result=v,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
