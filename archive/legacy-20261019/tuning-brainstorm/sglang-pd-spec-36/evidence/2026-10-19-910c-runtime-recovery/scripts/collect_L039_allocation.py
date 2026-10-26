#!/usr/bin/env python3
"""Current 910C resource inventory. No service requests, state changes or device compute."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R001-allocation';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import datetime,hashlib,json,pathlib,socket,subprocess
def run(a,t=30):
 try:
  p=subprocess.run(a,capture_output=True,text=True,timeout=t);return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except subprocess.TimeoutExpired:return {'exit_code':124,'error':'timeout'}
v={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hostname':socket.gethostname(),
   'npu':run(['npu-smi','info']),
   'containers':run(['docker','ps','--format','{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Status}}']),
   'model_metadata':[],'reference_container_labels':{}}
for base in ['/data/wio/GLM53/GLM-5.3-w8a8c8','/data/wio/models/GLM-5.3-w8a8c8']:
 p=pathlib.Path(base)/'config.json';v['model_metadata'].append({'path':str(p),'exists':p.is_file(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None})
if v['containers']['exit_code']==0:
 for line in v['containers']['stdout'].splitlines():
  parts=line.split('\t')
  if len(parts)<2 or not parts[1].startswith('sglang-'):continue
  x=run(['docker','inspect','--format','{{json .Config.Labels}}',parts[0]])
  if x['exit_code']==0:
   labels=json.loads(x['stdout']) or {};v['reference_container_labels'][parts[1]]={k:val for k,val in labels.items() if any(t in k.lower() for t in ['task','owner','project','org.opencontainers.image.revision'])}
print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,datetime,json,pathlib,shlex,subprocess,sys
j=json.load(sys.stdin)
def run(a,t=30):
 try:
  p=subprocess.run(a,capture_output=True,text=True,timeout=t);return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 except subprocess.TimeoutExpired:return {'exit_code':124,'error':'timeout'}
nodes=run(['kubectl','get','nodes','-o','json']);assert nodes['exit_code']==0,nodes.get('stderr');nodes=json.loads(nodes['stdout'])['items']
pods=run(['kubectl','get','pods','-A','-o','json']);assert pods['exit_code']==0,pods.get('stderr');pods=json.loads(pods['stdout'])['items']
v={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'nodes':[],'pods':[],'host_observations':{}}
targets={f'a3-5u-atlas900rcka3-{n}' for n in range(33,49)}
for x in nodes:
 if x['metadata']['name'] not in targets:continue
 v['nodes'].append({'name':x['metadata']['name'],'ready':next((c.get('status') for c in x.get('status',{}).get('conditions',[]) if c.get('type')=='Ready'),None),
                    'unschedulable':x.get('spec',{}).get('unschedulable',False),'taints':x.get('spec',{}).get('taints',[]),
                    'accelerator_capacity':{k:val for k,val in x.get('status',{}).get('capacity',{}).items() if any(s in k.lower() for s in ['ascend','npu'])},
                    'accelerator_allocatable':{k:val for k,val in x.get('status',{}).get('allocatable',{}).items() if any(s in k.lower() for s in ['ascend','npu'])}})
access=[]
for x in pods:
 if x.get('spec',{}).get('nodeName') not in targets:continue
 meta=x['metadata'];spec=x['spec'];status=x.get('status',{})
 v['pods'].append({'namespace':meta['namespace'],'name':meta['name'],'uid':meta['uid'],'node':spec['nodeName'],'phase':status.get('phase'),
                   'resources':[{'name':c['name'],'requests':c.get('resources',{}).get('requests',{}),'limits':c.get('resources',{}).get('limits',{})} for c in spec.get('containers',[])],
                   'labels':{k:val for k,val in meta.get('labels',{}).items() if any(t in k.lower() for t in ['task','owner','project'])}})
 if meta['namespace']=='default' and meta['name'].startswith('sglang-tuning-access-') and status.get('phase')=='Running' and any(c['name']=='hostctl' for c in spec.get('containers',[])):
  access.append((meta['name'],spec['nodeName'],status.get('hostIP')))
def observe(row):
 name,node,ip=row
 p=run(['kubectl','exec','-n','default',name,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',j['host']],75)
 return node,{'route':name,'host_ip':ip,'exit_code':p['exit_code'],'data':json.loads(p['stdout']) if p['exit_code']==0 else None,'stderr':p.get('stderr'),'error':p.get('error')}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for node,data in pool.map(observe,access):v['host_observations'][node]=data
key_candidates=[pathlib.Path('/root/KeyPair-27.pem'),pathlib.Path('/root/.ssh/KeyPair-27.pem')]
key=next((p for p in key_candidates if p.is_file()),None);v['named_key_available_on_jump']=key is not None
args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ConnectionAttempts=1','-o','StrictHostKeyChecking=yes']
if key:args+=['-i',str(key)]
p=run(args+['root@172.16.12.44',shlex.join(['python3','-c',j['host']])],75)
v['direct44']={'exit_code':p['exit_code'],'data':json.loads(p['stdout']) if p['exit_code']==0 else None,'stderr':p.get('stderr'),'error':p.get('error')}
print(json.dumps(v))
'''
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST}),capture_output=True,text=True,timeout=250)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'services_started':False,'device_compute_executed':False,'inference_requests':0,
         'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['stdout.json','stderr.log']}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode,'receipt':str(O/'receipt.json')}));sys.exit(p.returncode)
