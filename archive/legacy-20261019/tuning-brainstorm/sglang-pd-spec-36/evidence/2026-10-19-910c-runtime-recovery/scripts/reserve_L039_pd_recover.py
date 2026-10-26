#!/usr/bin/env python3
"""Acquire fresh scheduler reservations only after rechecking idle candidate nodes."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R006-reservations';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,re,subprocess
p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr
hbm=list(map(int,re.findall(r'(\d+)\s*/\s*65536',p.stdout)))
assert len(hbm)==16 and max(hbm)<4096 and p.stdout.count('No running processes found in NPU')==8,'Candidate has active/unknown NPU use'
q=subprocess.run(['docker','ps','--format','{{.Names}}'],capture_output=True,text=True,timeout=25);assert q.returncode==0,q.stderr
assert not [n for n in q.stdout.splitlines() if not n.startswith('k8s_')],'Non-Kubernetes service appeared'
print(json.dumps({'npu_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'used_HBM_MiB':hbm,'no_running_npu_processes':True,'non_k8s_containers':[]}))
'''
JUMP=r'''
import datetime,json,subprocess,sys,traceback
j=json.load(sys.stdin);owner='sglang-pd-spec-36';thread='01a14fdc-45b7-7a80-a49a-ba7a50a7f6a2'
v={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':owner,'preflight':{},'created':[],'reservations':[],'device_compute_executed':False,'inference_requests':0}
def run(a,stdin=None,t=30,allow_fail=False):
 p=subprocess.run(a,input=stdin,capture_output=True,text=True,timeout=t)
 if not allow_fail:assert p.returncode==0,p.stderr
 return p
try:
 base=json.loads(run(['kubectl','get','pod','-n','default','sglang-pd4p2d-reserve-37-20261014','-o','json']).stdout)
 image=base['spec']['containers'][0]['image'];v['reservation_image']=image
 nodes=json.loads(run(['kubectl','get','nodes','-o','json']).stdout)['items'];nodes={x['metadata']['name']:x for x in nodes}
 pods=json.loads(run(['kubectl','get','pods','-A','-o','json']).stdout)['items']
 plans=[]
 for number,role in [(44,'prefill'),(46,'decode')]:
  node=f'a3-5u-atlas900rcka3-{number}';x=nodes[node]
  assert not x.get('spec',{}).get('unschedulable',False) and not x.get('spec',{}).get('taints')
  assert next(c['status'] for c in x['status']['conditions'] if c['type']=='Ready')=='True'
  assert int(x['status']['allocatable']['huawei.com/Ascend910'])==16
  for p in pods:
   if p.get('spec',{}).get('nodeName')!=node or p.get('status',{}).get('phase') in ['Failed','Succeeded']:continue
   assert all(int(c.get('resources',{}).get('requests',{}).get('huawei.com/Ascend910',0))==0 for c in p['spec']['containers']),'Existing accelerator reservation'
  access=f'sglang-container-access-{number}-20261013'
  p=run(['kubectl','exec','-n','default',access,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',j['host']],t=60)
  v['preflight'][node]=json.loads(p.stdout)
  name=f'sglang-mech36-{role}-{number}-20261019'
  assert not run(['kubectl','get','pod','-n','default',name,'--ignore-not-found','-o','json']).stdout.strip(),'Task reservation name already exists; inspect, do not replay'
  pod={'apiVersion':'v1','kind':'Pod','metadata':{'name':name,'namespace':'default','labels':{'task':owner,'owner':thread,'role':role}},
       'spec':{'nodeSelector':{'kubernetes.io/hostname':node},'priorityClassName':'sglang-mech36-nonpreempt-20261019','restartPolicy':'Never','automountServiceAccountToken':False,'enableServiceLinks':False,
               'containers':[{'name':'reservation','image':image,'imagePullPolicy':'IfNotPresent','command':['/bin/sh','-c','exec sleep 2147483647'],
                              'resources':{'requests':{'cpu':'100m','memory':'128Mi','huawei.com/Ascend910':16},'limits':{'cpu':'1','memory':'256Mi','huawei.com/Ascend910':16}},
                              'securityContext':{'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}}}]}}
  plans.append(pod)
 v['frozen_pod_manifests']=plans
 pc={'apiVersion':'scheduling.k8s.io/v1','kind':'PriorityClass','metadata':{'name':'sglang-mech36-nonpreempt-20261019','labels':{'task':owner,'owner':thread}},'value':-1000,'globalDefault':False,'preemptionPolicy':'Never','description':'Task-specific non-preempting910C experiments; no change to any default or existing workload.'}
 current=run(['kubectl','get','priorityclass',pc['metadata']['name'],'-o','json'])
 x=json.loads(current.stdout)
 assert x['metadata']['labels']['task']==owner and x['metadata']['labels']['owner']==thread
 assert x['preemptionPolicy']=='Never' and x.get('globalDefault',False) is False and x['value']==-1000
 v['verified_existing_priority_class']={'name':x['metadata']['name'],'uid':x['metadata']['uid'],'preemptionPolicy':x['preemptionPolicy'],'value':x['value'],'globalDefault':x.get('globalDefault',False)}
 for pod in plans:
  x=json.loads(run(['kubectl','create','-f','-','-o','json'],json.dumps(pod)).stdout)
  v['created'].append({'name':x['metadata']['name'],'uid':x['metadata']['uid']})
 names=[p['metadata']['name'] for p in plans]
 wait=run(['kubectl','wait','-n','default','--for=condition=Ready','--timeout=45s',*['pod/'+n for n in names]],t=50,allow_fail=True)
 v['wait']={'exit_code':wait.returncode,'stdout':wait.stdout,'stderr':wait.stderr}
 for name in names:
  x=json.loads(run(['kubectl','get','pod','-n','default',name,'-o','json']).stdout)
  assert x['metadata']['labels']['task']==owner and x['metadata']['labels']['owner']==thread
  v['reservations'].append({'name':name,'uid':x['metadata']['uid'],'node':x['spec'].get('nodeName'),'phase':x['status'].get('phase'),
                            'ready':next((c['status'] for c in x['status'].get('conditions',[]) if c['type']=='Ready'),None),
                            'resources':x['spec']['containers'][0]['resources'],
                            'container_statuses':x['status'].get('containerStatuses',[])})
 v['status']='PASS' if all(x['phase']=='Running' and x['ready']=='True' for x in v['reservations']) else 'PENDING'
except Exception as e:v['status']='ERROR';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
print(json.dumps(v));sys.exit(0 if v['status']=='PASS' else 2)
'''
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST}),capture_output=True,text=True,timeout=220)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'action':'Create two fresh non-preempting scheduler reservations for this task,16 Ascend resources each,44 P/46 D',
         'device_compute_executed':False,'inference_requests':0,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
         'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['stdout.json','stderr.log']}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode,'receipt':str(O/'receipt.json')}));sys.exit(p.returncode)
