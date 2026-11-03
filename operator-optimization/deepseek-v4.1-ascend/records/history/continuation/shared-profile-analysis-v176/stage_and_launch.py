"""Stage only new diagnostic scripts;launch once after exact CPU preflight."""
import ast,base64,hashlib,json,sys,zlib
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/171-vllm-migration-20261102')
from control import invoke
H=Path(__file__).resolve().parent
ROOT='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/joint176'
bundle={n:(H/n).read_text() for n in ['controller.py','client.py','hook/sitecustomize.py','hook/joint_runtime.py','serve-enabled.sh']}
for n,s in bundle.items():
 if n.endswith('.py'):compile(s,n,'exec')
pinned=next(ast.literal_eval(a.value) for a in ast.parse(bundle['hook/joint_runtime.py']).body if isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINNED' for t in a.targets))
plan=json.loads((H/'PLAN.json').read_text())
plan['sources']=[{'path':n,'bytes':len(s.encode()),'sha256':hashlib.sha256(s.encode()).hexdigest()} for n,s in bundle.items()]
(H/'PLAN.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
encoded=base64.b64encode(zlib.compress(json.dumps({'bundle':bundle,'plan':plan}).encode())).decode();chunks=[encoded[i:i+5000] for i in range(0,len(encoded),5000)]
for i,chunk in enumerate(chunks):
 code='ROOT='+repr(ROOT)+'\nPART='+repr(chunk)+'\nINDEX='+str(i)+'\n'+'''from pathlib import Path
import json
p=Path(ROOT);p.mkdir(parents=True,exist_ok=True)
if (p/'SUBMITTED.json').exists():raise RuntimeError('Cohortalready submitted;observe existinghandle instead')
q=p/('stage.part'+str(INDEX));q.write_text(PART)
print(json.dumps({'part':INDEX,'bytes':q.stat().st_size}))'''
 r=invoke('target',code);print(json.dumps(r),flush=True)
code='ROOT='+repr(ROOT)+'\nPARTS='+str(len(chunks))+'\n'+r'''from pathlib import Path
import ast,base64,hashlib,json,os,subprocess,sys,zlib
p=Path(ROOT)
if (p/'SUBMITTED.json').exists():print(json.dumps({'existing':True,**json.loads((p/'SUBMITTED.json').read_text())}));raise SystemExit(0)
data=json.loads(zlib.decompress(base64.b64decode(''.join((p/('stage.part'+str(i))).read_text() for i in range(PARTS)))))
for n,s in data['bundle'].items():
 q=p/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s)
(p/'PLAN.json').write_text(json.dumps(data['plan'],indent=2)+'\n')
paths={name:'/vllm-workspace/vllm-ascend/'+name.replace('.','/')+'.py' for name in data['plan']['source_pins']}
check="import ast,hashlib,json;from pathlib import Path;checks={};pins="+repr(data['plan']['source_pins'])+";paths="+repr(paths)+"\nfor n,d in pins.items():\n p=Path(paths[n]);actual=hashlib.sha256(p.read_bytes()).hexdigest();assert actual==d,(n,actual);checks[n]=actual\nroot=Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/analysis/joint176')\nfor p in [root/'controller.py',root/'client.py',*root.glob('hook/*.py')]:compile(p.read_text(),str(p),'exec')\nprint(json.dumps({'status':'PASS_CPU_ONLY_PREFLIGHT','source_checks':checks,'scripts_compiled_inside_container':4}))"
r=subprocess.run(['docker','exec','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','wio-vllm-v41-repro171-20261102','python3','-c',check],capture_output=True,text=True,timeout=30)
receipt={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr[-2000:]}
(p/'CPU-PREFLIGHT.json').write_text(json.dumps(receipt,indent=2)+'\n');assert r.returncode==0,receipt
assert hashlib.sha256((p.parent.parent/'scripts/serve.sh').read_bytes()).hexdigest()=='c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba'
log=(p/'controller.log').open('ab');proc=subprocess.Popen(['python3','-I','-S',str(p/'controller.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
h={'pid':proc.pid,'start_ticks':Path('/proc/'+str(proc.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],'root':str(p),'plan_sha256':hashlib.sha256((p/'PLAN.json').read_bytes()).hexdigest(),'cpu_preflight':receipt}
(p/'SUBMITTED.json').write_text(json.dumps(h,indent=2)+'\n');print(json.dumps(h))
'''
r=invoke('target',code,timeout=60);(H/'submission.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps(r,ensure_ascii=False),flush=True)
