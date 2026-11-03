import json,sys
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
H=Path('/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v171');H.mkdir(exist_ok=True)
inner=r'''import json,hashlib,re,subprocess
from pathlib import Path
root=Path('/usr/local/Ascend/ascend-toolkit/latest');names=['include/acl/acl_rt.h','include/acl/acl_base.h','include/runtime/rt.h','include/runtime/dev.h','include/runtime/base.h','include/experiment/runtime/runtime/rt.h','include/experiment/runtime/runtime/dev.h']
rows=[]
for name in names:
 p=root/name
 if not p.exists():continue
 b=p.read_bytes();lines=b.decode(errors='replace').splitlines();selected=[]
 for i,line in enumerate(lines):
  if re.search(r'Phy.*(?:Logic|Index)|Logic.*(?:Phy|User)|User.*Logic|GetDeviceInfo|GetDeviceCount|GetSocName|DevAttr|DEV_ATTR|GetDevicePhy',line):selected.append({'line':i+1,'text':'\n'.join(lines[max(0,i-2):min(len(lines),i+5)])})
 rows.append({'path':str(p),'resolved':str(p.resolve()),'sha256':hashlib.sha256(b).hexdigest(),'matches':selected[:100]})
libs=[]
for n in ['lib64/libascendcl.so','lib64/libruntime.so']:
 p=root/n
 if not p.exists():continue
 x=subprocess.run(['nm','-D',str(p)],capture_output=True,text=True,timeout=10);matches=[line for line in x.stdout.splitlines() if re.search(r'(aclrt|rt)(?:Get|Query).*(?:Phy|Logic|UserDev|DeviceInfo|DeviceCount|SocName|DevInfo)',line)]
 libs.append({'path':str(p),'resolved':str(p.resolve()),'matches':matches[:100],'rc':x.returncode})
print(json.dumps({'headers':rows,'library_symbols':libs}))'''
code='import json,subprocess\ninner='+repr(inner)+'''\nc='wio-vllm-v41-repro171-20261102'
x=json.loads(subprocess.run(['docker','inspect',c],capture_output=True,text=True,check=True).stdout)[0]
r={'container':c,'running':x['State']['Running'],'started_at':x['State']['StartedAt'],'image_id':x['Image'],'visibility_env':[v for v in x['Config']['Env'] if v.split('=',1)[0] in ['ASCEND_VISIBLE_DEVICES','ASCEND_RT_VISIBLE_DEVICES','NPU_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES']],'devices':x['HostConfig']['Devices'],'device_requests':x['HostConfig'].get('DeviceRequests')}
p=subprocess.run(['docker','exec',c,'python3','-c',inner],capture_output=True,text=True,timeout=40);r['SDK']={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-1500:]}
h=subprocess.run(['npu-smi','info','-h'],capture_output=True,text=True,timeout=10);r['npu_smi_help']={'rc':h.returncode,'text':h.stdout+h.stderr}
print(json.dumps(r))'''
r=invoke('target',code,timeout=55)
if r['SDK']['rc']==0:r['SDK']['parsed']=json.loads(r['SDK']['stdout']);r['SDK']['stdout']='ParsedSDK fields stored in parsed'
(H/'current-device-query-contract-discovery.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False))
