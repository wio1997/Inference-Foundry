#!/usr/bin/env python3
"""Inspect image identities and original model mount locations without reading services."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R004-images-models';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,pathlib,socket,subprocess
def run(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=30);return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
v={'hostname':socket.gethostname(),'images':run(['docker','images','--no-trunc','--format','{{.ID}}\t{{.Repository}}:{{.Tag}}']),'container_mounts':{},'models':[]}
p=run(['docker','ps','-a','--format','{{.ID}}\t{{.Names}}']);assert p['exit_code']==0,p['stderr']
paths=set(['/data/wio/GLM53/GLM-5.3-w8a8c8','/data/wio/models/GLM-5.3-w8a8c8'])
for line in p['stdout'].splitlines():
 cid,name=line.split('\t',1)
 if name.startswith('k8s_'):continue
 if not any(t in name.lower() for t in ['glm53','sglang']):continue
 x=run(['docker','inspect','--format','{{json .Mounts}}',cid]);assert x['exit_code']==0,x['stderr']
 mounts=json.loads(x['stdout']);v['container_mounts'][name]=[{k:m.get(k) for k in ['Source','Destination','RW','Type']} for m in mounts]
 for m in mounts:
  source=m.get('Source','');target=m.get('Destination','')
  if ('glm' in source.lower() and ('5.3' in source or '53' in source)) or target=='/model':paths.add(source)
paths.update(json.loads(__import__('sys').argv[1]))
for source in sorted(paths):
 p=pathlib.Path(source);config=p/'config.json'
 if not config.is_file():continue
 b=config.read_bytes();cfg=json.loads(b)
 item={'path':source,'config_sha256':hashlib.sha256(b).hexdigest(),'architectures':cfg.get('architectures'),'model_type':cfg.get('model_type'),'dtype':cfg.get('dtype'),'files':{}}
 for n in ['config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja','quant_model_description.json','quant_model_weights.safetensors.index.json']:
  f=p/n
  if f.is_file():item['files'][n]={'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
 v['models'].append(item)
print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin)
def observe(number,extra):
 pod=f'sglang-container-access-{number}-20261013'
 p=subprocess.run(['kubectl','exec','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',j['host'],json.dumps(extra)],capture_output=True,text=True,timeout=90)
 return number,{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.returncode==0 else None,'stderr':p.stderr}
number,first=observe(43,[]);v={'read_only':True,'nodes':{'43':first}}
extra=[m['path'] for m in (first.get('data') or {}).get('models',[]) if m['config_sha256']=='ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca']
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for number,data in pool.map(lambda n:observe(n,extra),[44,46]):v['nodes'][str(number)]=data
print(json.dumps(v));sys.exit(0 if all(x['exit_code']==0 for x in v['nodes'].values()) else 1)
'''
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST}),capture_output=True,text=True,timeout=210)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'services_started':False,'device_compute_executed':False,
         'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['stdout.json','stderr.log']}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode,'receipt':str(O/'receipt.json')}));sys.exit(p.returncode)
