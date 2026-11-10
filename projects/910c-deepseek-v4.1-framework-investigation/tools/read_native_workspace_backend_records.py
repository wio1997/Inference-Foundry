#!/usr/bin/env python3
"""Read installed header hashes and existing DB schema; never initialize a backend."""
from pathlib import Path
from hashlib import sha256
import json
import subprocess

OUT = Path('audit/execution-search')
REMOTE = r'''
from pathlib import Path
from hashlib import sha256
import json,subprocess,sqlite3,sys
spec=json.loads(sys.argv[1]);db=Path(spec['database']);assert db.is_relative_to('/data/tiankuan/wio')
name='wio-vllm-v41-repro171-20261102'
c,=json.loads(subprocess.check_output(['docker','inspect',name]));assert c['State']['Running'] and c['Id']==spec['container_id'] and c['Image']==spec['image']
inner=r"""
from pathlib import Path
from hashlib import sha256
import json,sys
spec=json.loads(sys.argv[1]);base=Path('/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/include')
headers=[]
for x in spec['headers']:
 p=base/x['path'];row=dict(path=x['path'],installed_path=str(p),missing=not p.is_file(),reference_sha256=x['sha256'])
 if p.is_file():
  b=p.read_bytes();row.update(bytes=len(b),sha256=sha256(b).hexdigest());row['matches_reference']=row['sha256']==x['sha256']
 headers.append(row)
env=[]
for pid in spec['worker_pids']:
 p=Path('/proc')/str(pid);selected={}
 for item in (p/'environ').read_bytes().split(b'\0'):
  k,sep,v=item.partition(b'=')
  if k.decode(errors='replace') in ['PER_STREAM_QUEUE','TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING','PYTORCH_NPU_ALLOC_CONF','PYTORCH_NO_NPU_MEMORY_CACHING','MULTI_STREAM_MEMORY_REUSE']:
   selected[k.decode()]=v.decode(errors='replace')
 env.append(dict(pid=pid,initial_selected_environment=selected))
print(json.dumps(dict(headers=headers,workers=env)))
"""
d=json.loads(subprocess.check_output(['docker','exec',name,'python3','-B','-I','-S','-c',inner,json.dumps(spec)]))
assert sha256(db.read_bytes()).hexdigest()==spec['database_sha256']
with sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True) as con:
 tables=[r[0] for r in con.execute('select name from sqlite_master where type="table"')]
 schemas={t:[list(r) for r in con.execute('pragma table_info("'+t+'")')] for t in ['CANN_API','STRING_IDS','TASK'] if t in tables}
 names=[list(r) for r in con.execute("select * from STRING_IDS where value like '%alloc%' or value like '%Malloc%' or value like '%Free%' or value like '%Workspace%' or value like '%empty%' limit 250")]
 counts={t:con.execute('select count(*) from "'+t+'"').fetchone()[0] for t in tables if 'PYTORCH' in t or 'MEMORY' in t}
 api=[]
 for n,num,total,lo,hi in con.execute("select s.value,count(*),sum(a.endNs-a.startNs),min(a.endNs-a.startNs),max(a.endNs-a.startNs) from CANN_API a join STRING_IDS s on a.name=s.id where s.value like '%Workspace%' or s.value like '%Malloc%' or s.value like '%Free%' or s.value like '%Capture%' group by s.value order by s.value"):
  api.append(dict(name=n,calls=num,total_duration_ns=total,min_duration_ns=lo,max_duration_ns=hi))
 d['database']=dict(path=str(db),sha256=spec['database_sha256'],schemas=schemas,selected_string_names=names,selected_table_counts=counts,CANN_API_summary=api,scope='Rank0 whole archived DB API durations; not exclusive allocator time, not exact per-native workspace bytes or request savings.')
d.update(container_id=c['Id'],image=c['Image'],new_NPU_work=0,model_loads=0,service_requests=0,new_profile_collection=0,remote_writes=[])
print(json.dumps(d))
'''

def main():
 catalog=json.loads((OUT/'native-workspace-catalog.json').read_text())
 headers=[]
 for p in sorted((OUT/'native-workspace-upstream').rglob('*.h')):
  headers.append(dict(path=str(p.relative_to(OUT/'native-workspace-upstream')),sha256=sha256(p.read_bytes()).hexdigest()))
 service=json.loads((OUT/'indexer-package-identity.json').read_text())['worker_processes']
 spec=dict(headers=headers,worker_pids=[x['pid'] for x in service],database=catalog['database']['path'],database_sha256=catalog['database']['sha256'],container_id=catalog['container_id'],image=catalog['image'])
 r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',REMOTE,json.dumps(spec)],capture_output=True,text=True,timeout=180)
 if r.returncode:raise RuntimeError(r.stderr)
 d=json.loads(r.stdout);d['reader_sha256']=sha256(Path(__file__).read_bytes()).hexdigest()
 (OUT/'native-workspace-backend-records.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(dict(headers=d['headers'],database=d['database']),ensure_ascii=False))

if __name__=='__main__':main()
