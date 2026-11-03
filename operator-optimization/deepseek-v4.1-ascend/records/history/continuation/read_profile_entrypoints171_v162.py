"""Read exact profiler/case entrypoints from the own container filesystem.

No docker exec, CANN/Torch import, model request, configuration edit or restart.
"""
import sys,json,base64
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,write
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
code=r'''import ast,os,json,socket,subprocess,hashlib,base64
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
cid='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae1768cfb387eb'
# The inspected migrated container has the full 64-character ID below.
cid='3be44e7b5c292de6ef7fc51085bfbbad5552acc97fae39c0ae176fe8cfb387eb'
d=json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
assert d['Id']==cid and d['State']['Running']
root=Path('/proc')/str(d['State']['Pid'])/'root'
app=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime')
specs=[('serve',app/'scripts/serve.sh',None),
 ('client',app/'tools/whole_scope166_reference_client.py',None),
 ('profiler_factory',root/'vllm-workspace/vllm-ascend/vllm_ascend/profiler/torch_npu_profiler.py',None),
 ('profiler_config',root/'vllm-workspace/vllm/vllm/config/profiler.py',None),
 ('profiler_wrapper',root/'vllm-workspace/vllm/vllm/profiler/wrapper.py',None),
 ('worker',root/'vllm-workspace/vllm-ascend/vllm_ascend/worker/worker_v1.py','profiler'),
 ('runner',root/'vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py','capture_model')]
rows=[]
for label,p,selector in specs:
 if not p.exists():rows.append({'label':label,'path':str(p),'exists':False});continue
 st=p.stat();b=p.read_bytes();after=p.stat()
 assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 assert len(b)<2*1024**2
 row={'label':label,'path':str(p),'exists':True,'source_bytes':len(b),'source_sha256':hashlib.sha256(b).hexdigest()}
 if selector:
  tree=ast.parse(b);lines=b.decode().splitlines(keepends=True);selected=[]
  for node in ast.walk(tree):
   if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and selector in node.name:
    text=''.join(lines[node.lineno-1:node.end_lineno]);assert len(text)<60000
    selected.append({'name':node.name,'start_line':node.lineno,'end_line':node.end_lineno,'text':text})
  row['selected_functions']=selected
 else:
  assert len(b)<100000;row['full_source_base64']=base64.b64encode(b).decode()
 rows.append(row)
print(json.dumps({'target':'172.16.10.171','container':cid,'container_started_at':d['State']['StartedAt'],
 'files':rows,'read_only_exact_paths':True,'runtime_or_model_calls':False,'overall_complete':False}))
'''
out=invoke('target',code,60)
cache=N/'profile-entrypoints-cache-v162';assert not cache.exists();cache.mkdir()
for row in out['files']:
 if 'full_source_base64' in row:
  p=cache/(row['label']+('.sh' if row['label']=='serve' else '.py'))
  p.write_bytes(base64.b64decode(row.pop('full_source_base64')));row['local_path']=str(p)
write(O/'profile-entrypoints-readonly-v162.json',out)
print(json.dumps({'files':[{k:v for k,v in r.items() if k not in ['selected_functions','path','local_path']} for r in out['files']],
 'selected_function_names':{r['label']:[f['name'] for f in r.get('selected_functions',[])] for r in out['files']},
 'new_device_or_model_work':False}))
