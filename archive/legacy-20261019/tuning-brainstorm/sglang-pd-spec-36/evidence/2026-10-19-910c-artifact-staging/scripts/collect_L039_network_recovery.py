#!/usr/bin/env python3
"""Read-only node/lease reachability plus exact existing transfer handle observation."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R025-network-state';O.mkdir(parents=True,exist_ok=False)
JUMP=r'''
import concurrent.futures,json,socket,subprocess
names=['sglang-mech36-prefill-44-20261019','sglang-mech36-decode-46-20261019']
def get(a):
 p=subprocess.run(['kubectl']+a+['-o','json'],capture_output=True,text=True,timeout=25)
 if p.returncode:return {'exit_code':p.returncode,'stderr':p.stderr}
 return {'exit_code':0,'data':json.loads(p.stdout)}
v={'read_only':True,'nodes':[],'reservations':[],'kubelet_TCP_from_jump':[]}
q=get(['get','nodes','a3-5u-atlas900rcka3-44','a3-5u-atlas900rcka3-46'])
assert q['exit_code']==0,q
for x in q['data']['items']:
 v['nodes'].append({'name':x['metadata']['name'],'uid':x['metadata']['uid'],'unschedulable':x['spec'].get('unschedulable',False),'taints':x['spec'].get('taints',[]),'conditions':x['status']['conditions'],'addresses':x['status']['addresses']})
q=get(['get','pod','-n','default']+names);assert q['exit_code']==0,q
for x in q['data']['items']:
 v['reservations'].append({'name':x['metadata']['name'],'uid':x['metadata']['uid'],'task':x['metadata'].get('labels',{}).get('task'),'node':x['spec']['nodeName'],'phase':x['status']['phase'],'conditions':x['status'].get('conditions',[]),'preemptionPolicy':x['spec']['preemptionPolicy']})
def check(n):
 try:
  with socket.create_connection((f'172.16.12.{n}',10250),3):return {'node':n,'port':10250,'reachable':True}
 except OSError as e:return {'node':n,'port':10250,'reachable':False,'error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p:v['kubelet_TCP_from_jump']=list(p.map(check,[44,46]))
print(json.dumps(v))
'''
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
p=subprocess.run(cmd,capture_output=True,text=True,timeout=65)
(O/'metadata.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
q=subprocess.run(['python3','scripts/collect_L039_transfer_state.py','L039-R025'],cwd=R,capture_output=True,text=True,timeout=110)
(O/'observer-stdout.log').write_text(q.stdout);(O/'observer-stderr.log').write_text(q.stderr)
code=0 if p.returncode==q.returncode==0 else 1
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':code,'metadata_exit':p.returncode,'existing_handle_observer_exit':q.returncode,'read_only':True,'restarted_jobs':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in O.iterdir() if x.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(code)
