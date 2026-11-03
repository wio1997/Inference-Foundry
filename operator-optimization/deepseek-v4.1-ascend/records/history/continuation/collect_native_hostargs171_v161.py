"""Collect bounded already-finished probe files;never submit another device attempt."""
import sys,json,base64,hashlib
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
prep=read(O/'native-hostargs-probe-preparation-v161.json')
code='BUNDLE = '+repr(prep['bundle_host'])+'\n'+r'''import json,base64,hashlib
from pathlib import Path
r=Path('/data/tiankuan/wio/migration-171-20261102/operator-evidence-v161');state=json.loads((r/'NATIVE-STATUS.json').read_text());assert state['native_attempts']==1 and (r/'NATIVE-RESULT.json').exists()
bundle=Path(BUNDLE);rows=[];total=0
for p in sorted(bundle.rglob('*')):
 assert not p.is_symlink(),'Do not follow generated symlinks'
 if not p.is_file() or p.name in ['libhost_adapter.so','probe.py','probe-before-joint-profile.py']:continue
 b=p.read_bytes();total+=len(b);assert total<=8*1024*1024
 rows.append({'relative':'bundle/'+p.relative_to(bundle).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
for name in ['APPROVAL.json','NATIVE-STATUS.json','NATIVE-RESULT.json','probe-stdout.log','probe-stderr.log','npu-before-window.txt','npu-idle-before-probe.txt']:
 p=r/name
 if p.is_file():
  b=p.read_bytes();total+=len(b);assert total<=9*1024*1024
  rows.append({'relative':'controller/'+name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64':base64.b64encode(b).decode()})
print(json.dumps({'controller_status_snapshot':state,'bytes':total,'files':rows,'collection_does_not_restart_or_repeat_the_probe':True}))
'''
d=invoke('target',code,45);folder=N/'joint-hostargs-probe-cache-v161';folder.mkdir(exist_ok=False)
index=[]
for row in d.pop('files'):
 b=base64.b64decode(row.pop('base64'));assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 p=folder/row['relative'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);row['local_path']=str(p);index.append(row)
d['files']=index;write(O/'joint-hostargs-probe-file-index-v161.json',d)
print(json.dumps({'bytes':d['bytes'],'files':[(r['relative'],r['bytes']) for r in index],'controller_state':d['controller_status_snapshot']['state'],'restore':d['controller_status_snapshot'].get('restore')},ensure_ascii=False))
