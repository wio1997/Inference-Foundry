import base64,json,sys,zlib,hashlib
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102');from control import invoke
H=Path(__file__).resolve().parent;root='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint168'
bundle={n:(H/n).read_text() for n in ['analyze_histories.py','offline_joint.py','post_controller.py']}
for n,s in bundle.items():compile(s,n,'exec')
payload=base64.b64encode(zlib.compress(json.dumps(bundle).encode())).decode();parts=[payload[i:i+5000] for i in range(0,len(payload),5000)]
for i,s in enumerate(parts):
 r=invoke('target',"from pathlib import Path;import json;p=Path("+repr(root)+");assert not (p/'POST-SUBMITTED.json').exists();q=p/"+repr('stage.postpart'+str(i))+";q.write_text("+repr(s)+");print(json.dumps({'part':"+str(i)+",'bytes':q.stat().st_size}))");print(json.dumps(r),flush=True)
code='ROOT='+repr(root)+'\nPARTS='+str(len(parts))+'\n'+r'''import base64,zlib,json,subprocess,hashlib
from pathlib import Path
p=Path(ROOT)
if (p/'POST-SUBMITTED.json').exists():print(json.dumps({'existing':True,**json.loads((p/'POST-SUBMITTED.json').read_text())}));raise SystemExit(0)
bundle=json.loads(zlib.decompress(base64.b64decode(''.join((p/('stage.postpart'+str(i))).read_text() for i in range(PARTS)))))
for n,s in bundle.items():compile(s,n,'exec');(p/n).write_text(s)
log=(p/'post-controller.log').open('ab');q=subprocess.Popen(['python3','-I','-S',str(p/'post_controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
h={'pid':q.pid,'start_ticks':Path('/proc/'+str(q.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],'root':str(p),'CPU_only':True,'waits_on':json.loads((p/'SUBMITTED.json').read_text()),'source_pins':[{'path':n,'sha256':hashlib.sha256(s.encode()).hexdigest()} for n,s in bundle.items()]}
(p/'POST-SUBMITTED.json').write_text(json.dumps(h,indent=2)+'\n');print(json.dumps(h))
'''
r=invoke('target',code);(H/'post-submission.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
