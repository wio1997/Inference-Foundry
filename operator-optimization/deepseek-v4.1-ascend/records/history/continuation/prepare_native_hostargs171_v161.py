"""Prepare only;does not start/stop containers,load CANN or allocate a device."""
import sys,json,uuid,hashlib
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke,ROOT

raw=read(N/'host-adapter-CPU166-v127c.json');cpu=json.loads(raw['envelope']['stdout'])
binary=next(r for r in cpu['binaries'] if Path(r['path']).name=='libhost_adapter.so')
live=read(ROOT/'RUNTIME-READY-171-20261102.json')
oldbin=str(Path(live['original_project_tmp_preserved'])/Path(cpu['base']).name/'libhost_adapter.so')
probe=(N/'native_hostargs_pad171_v161.py').read_text();name='wio-hostargs-pad-v161-'+uuid.uuid4().hex[:8]
source_sha=hashlib.sha256(probe.encode()).hexdigest()
prefix='PROBE = '+repr(probe)+'\nBUNDLE_NAME = '+repr(name)+'\nOLD_BINARY = '+repr(oldbin)+'\nBINARY = '+repr(binary)+'\n'
code=prefix+r'''import os,json,socket,subprocess,hashlib,shutil
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-171'
stage='117d57490a2f2f253495f7a30a438eb64784787b5d71351104b90c9008a8db2d'
d=json.loads(subprocess.check_output(['docker','inspect',stage],text=True))[0]
assert d['State']['Status']=='created' and d['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
tmp=next(m['Source'] for m in d['Mounts'] if m['Destination']=='/tmp')
root=Path(tmp)/BUNDLE_NAME;root.mkdir(mode=0o700,exist_ok=False);(root/'trace').mkdir(mode=0o700)
b=Path(OLD_BINARY).read_bytes();assert len(b)==BINARY['bytes'] and hashlib.sha256(b).hexdigest()==BINARY['sha256']
(root/'libhost_adapter.so').write_bytes(b);(root/'libhost_adapter.so').chmod(0o700)
(root/'probe.py').write_text(PROBE);(root/'probe.py').chmod(0o700)
code=compile(PROBE,'prepared-probe','exec')
p=subprocess.run(['python3','-I','-S',str(root/'probe.py')],capture_output=True,text=True,timeout=10);assert p.returncode==0,p.stderr
description=json.loads(p.stdout)
assert description['status']=='PREPARED_NOT_EXECUTED' and not (root/'probe-result.json').exists() and not (root/'APPROVED.json').exists()
result={'status':'PREPARED_AWAITING_FRESH_PROBE_AUTHORIZATION','bundle_host':str(root),'bundle_inside':'/tmp/'+BUNDLE_NAME,'stage_container_id':stage,'stage_container_name':d['Name'].lstrip('/'),'stage_container_not_started':True,'probe_source_sha256':hashlib.sha256(PROBE.encode()).hexdigest(),'adapter_binary':BINARY,'adapter_reused_from_migrated_CPU_validated_snapshot':OLD_BINARY,'default_description_checked_on_host_without_CANN_or_Torch_import':True,'plan':description,'native_attempts':0,'model_or_device_state_changed':False,'original_optimization_goal_complete':False}
dest=Path('/data/tiankuan/wio/migration-171-20261102/operator-evidence-v161');dest.mkdir(mode=0o700,exist_ok=True)
(dest/'PREPARED-PROBE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
d=invoke('target',code,40);write(O/'native-hostargs-probe-preparation-v161.json',d);print(json.dumps(d,ensure_ascii=False))
