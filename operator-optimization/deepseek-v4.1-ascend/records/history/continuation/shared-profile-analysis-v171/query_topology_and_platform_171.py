import sys,json
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
H=Path('/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v171')
inner=r'''import json,hashlib
from pathlib import Path
root=Path('/usr/local/Ascend/ascend-toolkit/latest');rows=[]
for directory in ['data/platform_config','opp/data/platform_config','compiler/data/platform_config','opp/built-in/data/platform_config']:
 d=root/directory
 if not d.exists():continue
 for p in sorted(d.glob('*9382*')):
  if not p.is_file():continue
  b=p.read_bytes();lines=b.decode(errors='replace').splitlines();matches=[{'line':i+1,'text':line} for i,line in enumerate(lines) if any(k in line.lower() for k in ['core_num','core_cnt','core_count','freq','l2','ub_size','unified','cube','l0','l1','bandwidth','memory','aicore_num','vector_num'])]
  rows.append({'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b),'selected':matches})
print(json.dumps({'platform_configs':rows}))'''
code='import json,subprocess,hashlib,re\nfrom pathlib import Path\ninner='+repr(inner)+'''\nO=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint171');r={'commands':[],'driver_headers':[]}
for args in [['npu-smi','info','-t','topo'],['npu-smi','info','-t','sensors','-i','0','-c','0']]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=20);r['commands'].append({'argv':args,'rc':p.returncode,'text':p.stdout+p.stderr});(O/'TOPOLOGY-PLATFORM-QUERY.json').write_text(json.dumps(r,indent=2)+'\\n')
p=subprocess.run(['docker','exec','wio-vllm-v41-repro171-20261102','python3','-c',inner],capture_output=True,text=True,timeout=15);r['platform']={'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr[-1000:]}
for name in ['/usr/local/dcmi/include/dcmi_interface_api.h','/usr/local/Ascend/driver/include/dcmi_interface_api.h','/usr/local/Ascend/driver/include/ascend_hal.h']:
 p=Path(name)
 if not p.exists():continue
 b=p.read_bytes();lines=b.decode(errors='replace').splitlines();hits=[{'line':i+1,'text':chr(10).join(lines[max(0,i-2):i+8])} for i,line in enumerate(lines) if 'dcmi_get_device_frequency' in line or ('FREQ' in line and any(k in line for k in ['MAX','AICORE','AI_CORE','DCMI_FREQ']))]
 r['driver_headers'].append({'path':name,'sha256':hashlib.sha256(b).hexdigest(),'frequency_matches':hits[:60]})
(O/'TOPOLOGY-PLATFORM-QUERY.json').write_text(json.dumps(r,indent=2)+'\\n');print(json.dumps(r))'''
r=invoke('target',code,timeout=65)
if r['platform']['rc']==0:r['platform']['parsed']=json.loads(r['platform']['stdout']);r['platform']['stdout']='parsedstored'
(H/'topology-platform-frequency-query.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False))
