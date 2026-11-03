from pathlib import Path
from datetime import datetime, timezone
import subprocess, json, hashlib, socket

assert socket.gethostname() == 'bm-jn-zs-zone1-910C-64G-10-105'
P = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
OUT = P / 'native-PMU-complete-v18/attrInfo-exporter-field-localization-v23-v2.json'
assert not OUT.exists()
code = r'''from pathlib import Path
import subprocess, json, hashlib, ast
roots=[Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis'),Path('/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis')]
records=[]
for root in roots:
 assert root.is_dir(),str(root)
 q=subprocess.run(['grep','-R','-n','-E','--include=*.py','attrInfo|COMPUTE_TASK_INFO',str(root)],capture_output=True,text=True,timeout=20)
 assert q.returncode in (0,1),q.stderr
 lines=q.stdout.splitlines();assert len(lines)<=140,('bounded localization exceeded',len(lines))
 byfile={}
 for line in lines:
  path,lineno,text=line.split(':',2);byfile.setdefault(path,[]).append(dict(line=int(lineno),text=text))
 assert len(byfile)<=20
 for path,hits in byfile.items():
  p=Path(path);b=p.read_bytes();s=b.decode();ls=s.splitlines();tree=ast.parse(s)
  scopes=[]
  for node in ast.walk(tree):
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(node.lineno<=h['line']<=node.end_lineno for h in hits):
    scopes.append(dict(name=node.name,start=node.lineno,end=node.end_lineno,source=[str(i+1)+': '+ls[i]for i in range(node.lineno-1,node.end_lineno)] if node.end_lineno-node.lineno<=180 else None,source_omission='FUNCTION_EXCEEDS180LINES'if node.end_lineno-node.lineno>180 else None))
  points={i for h in hits for i in range(max(0,h['line']-4),min(len(ls),h['line']+3))}
  records.append(dict(path=path,sha256=hashlib.sha256(b).hexdigest(),hits=hits,exact_field_context=[str(i+1)+': '+ls[i]for i in sorted(points)],enclosing_functions=scopes))
print(json.dumps(dict(search_roots=[str(p)for p in roots],search_terms=['attrInfo','COMPUTE_TASK_INFO'],files=records,no_repository_or_whole_SDK_semantic_audit=True)))
'''
r = subprocess.run(['docker', 'exec', 'wio-vllm-v41-repro166-stage-20261030', 'python3', '-c', code], capture_output=True, text=True, timeout=55)
assert r.returncode == 0, r.stderr
result = json.loads(r.stdout)
b = (json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(), result=result, no_model_import_or_device_or_service_work=True, overall_complete=False), ensure_ascii=False, separators=(',', ':')) + '\n').encode()
OUT.write_bytes(b)
public = dict(files=[dict(path=f['path'],sha256=f['sha256'],hits=f['hits'],functions=[dict(name=q['name'],start=q['start'],end=q['end'],omitted=q['source']is None)for q in f['enclosing_functions']])for f in result['files']], full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest())
print(json.dumps(public,ensure_ascii=False))
