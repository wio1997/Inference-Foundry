from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,hashlib,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
M=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18')
out=M/'rejection-raw-logits-argument-consumers-v28-v2.json';assert not out.exists()
code=r'''from pathlib import Path
import ast,json,hashlib
p=Path('/vllm-workspace/vllm-ascend/vllm_ascend/sample/rejection_sampler.py');b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha=='bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda';ls=b.decode().splitlines();tree=ast.parse(b.decode());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='rejection_sample');hits=[];points=set()
for n in ast.walk(fn):
 if isinstance(n,ast.Name)and n.id=='ori_target_logits':
  points.update(range(max(fn.lineno-1,n.lineno-5),min(fn.end_lineno,n.end_lineno+5)))
  guards=[]
  for cond in ast.walk(fn):
   if isinstance(cond,ast.If)and cond.lineno<=n.lineno<=cond.end_lineno:
    branch='test'if cond.test.lineno<=n.lineno<=cond.test.end_lineno else('body'if any(st.lineno<=n.lineno<=st.end_lineno for st in cond.body)else'orelse')
    guards.append(dict(line=cond.lineno,test=ast.unparse(cond.test),branch=branch))
  hits.append(dict(line=n.lineno,context=type(n.ctx).__name__,ancestor_guards=guards))
definitions=[]
for n in ast.walk(tree):
 if isinstance(n,(ast.Assign,ast.AnnAssign))and any(isinstance(t,ast.Name)and t.id=='using_entropy_verify'and isinstance(t.ctx,ast.Store)for t in ast.walk(n)):
  assert n.end_lineno-n.lineno<16
  definitions.append(dict(start=n.lineno,end=n.end_lineno,expression=ast.unparse(n)))
  points.update(range(max(0,n.lineno-4),min(len(ls),n.end_lineno+3)))
for pos,line in enumerate(ls):
 if 'using_entropy_verify'in line:
  points.update(range(max(0,pos-2),min(len(ls),pos+3)))
for n in ast.walk(fn):
 if isinstance(n,ast.If)and n.lineno in [411,426,433,443,474,481,509,553]:
  points.update(range(n.lineno-1,min(n.end_lineno,n.lineno+3)))
assert len(points)<180
print(json.dumps(dict(path=str(p),sha256=sha,function=fn.name,start=fn.lineno,end=fn.end_lineno,argument_uses=hits,entropy_guard_definitions=definitions,guard_test_classification_correction='AloadintheIf.testisTEST,notorelse;originalv28metadataretainedhistorically.',
 source=[str(i+1)+': '+ls[i]for i in sorted(points)],no_model_or_torch_import=True,
 scope='Onlyori_target_logits loads/control ancestors plusselected samplingguardheaders in alreadyprofile-bound helper;notwholeprobabilisticbody.')))
'''
r=subprocess.run(['docker','exec','wio-vllm-v41-repro166-stage-20261030','python3','-c',code],capture_output=True,text=True,timeout=30);assert r.returncode==0,r.stderr
result=json.loads(r.stdout);result.update(created_utc=datetime.now(timezone.utc).isoformat(),no_model_import_device_service_work=True,overall_complete=False)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();out.write_bytes(b)
print(json.dumps(dict(result=result,full_result=str(out),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
