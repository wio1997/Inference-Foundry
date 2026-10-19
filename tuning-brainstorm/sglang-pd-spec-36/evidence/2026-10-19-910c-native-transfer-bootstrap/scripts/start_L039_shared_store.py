#!/usr/bin/env python3
"""Owned persistent CPU config store and fresh capacity facts for real model bootstrap."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R033-shared-store';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,os,pathlib,re,shlex,socket,subprocess,sys,time,traceback
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
def run(a,t=30,check=True):
 p=subprocess.run(a,capture_output=True,text=True,timeout=t)
 if check:assert p.returncode==0,p.stderr
 return p
if j['action']=='capacity':
 mem={a.split(':')[0]:int(a.split()[1])*1024 for a in pathlib.Path('/proc/meminfo').read_text().splitlines() if a.split(':')[0] in ['MemTotal','MemAvailable','SwapTotal','SwapFree']};fs=os.statvfs(base);p=run(['npu-smi','info']);q=run(['ip','-j','address']);interfaces=[]
 for interface in json.loads(q.stdout):
  for address in interface.get('addr_info',[]):
   if address.get('local')==f"172.16.12.{j['node']}":interfaces.append({'interface':interface['ifname'],'IP':address['local'],'prefixlen':address['prefixlen']})
 print(json.dumps({'node':j['node'],'CPU_affinity_count':len(os.sched_getaffinity(0)),'memory':mem,'task_fs_free_bytes':fs.f_bavail*fs.f_frsize,'NPU_no_running_processes':p.stdout.count('No running processes found in NPU')==8,'NPU_HBM_used_MiB':list(map(int,re.findall(r'(\d+)\s*/\s*65536',p.stdout))),'NPU_inventory_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'private_interfaces':interfaces}));sys.exit(0)
name='if-mech36-l039-r033-store-46';image=j['image'];root=base/'L039-R033-shared-store';v={'status':'STARTING','node':46,'store_url':'tcp://172.16.12.46:39440','name':name,'root':str(root),'model_loaded':False,'NPU_devices':False,'created':False}
try:
 assert hashlib.sha256(j['script'].encode()).hexdigest()==j['script_sha256'];assert not root.exists() and run(['docker','container','inspect',name],check=False).returncode!=0
 assert json.loads(run(['docker','image','inspect','--format','{{json .Id}}',image]).stdout)==image
 with socket.socket() as s:s.bind(('172.16.12.46',39440))
 root.mkdir(mode=0o700);(root/'store.py').write_text(j['script'])
 args=['docker','create','--name',name,'--runtime','runc','--label','task=sglang-pd-spec-36','--label','run=L039-R033','--network','host','--ipc','private','--cpus','2','--memory','1g','--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges','-e','PYTHONDONTWRITEBYTECODE=1','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','MF_CONFIG_STORE_URL='+v['store_url'],'-e','ASCEND_MF_STORE_URL='+v['store_url'],'--mount','type=bind,src='+str(root)+',dst=/work']
 for src,dst in [('/usr/local/Ascend/driver','/usr/local/Ascend/driver'),('/usr/local/Ascend/firmware','/usr/local/Ascend/firmware'),('/etc/ascend_install.info','/etc/ascend_install.info')]:args+=['--mount','type=bind,src='+src+',dst='+dst+',readonly']
 setup='set -e; if [ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then . /usr/local/Ascend/ascend-toolkit/set_env.sh; elif [ -r /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]; then . /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash; fi; exec python3 -u /work/store.py'
 args+=['--entrypoint','/bin/bash',image,'-c',setup];cid=run(args).stdout.strip();v['container_id']=cid;v['created']=True;(root/'handle.json').write_text(json.dumps({'task':'sglang-pd-spec-36','run':'L039-R033','container':name,'container_id':cid}));run(['docker','start',name])
 for _ in range(30):
  x=json.loads(run(['docker','inspect','--format','{{json .}}',name]).stdout);assert x['Id']==cid and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R033';v['state']=x['State'];assert not x['HostConfig']['Devices'] and not x['HostConfig']['Privileged']
  if (root/'state.json').is_file():
   v['store_state']=json.loads((root/'state.json').read_text())
   if v['store_state']['status'] in ['READY','FAILED']:break
  if not x['State']['Running']:break
  time.sleep(1)
 x=json.loads(run(['docker','inspect','--format','{{json .}}',name]).stdout);v['state']=x['State'];v['status']='READY' if x['State']['Running'] and v.get('store_state',{}).get('status')=='READY' else 'PENDING_RUNNING' if x['State']['Running'] else 'FAILED_EXITED';v['isolation']={'CPU_limit':2,'memory_limit_bytes':x['HostConfig']['Memory'],'devices':x['HostConfig']['Devices'],'model_mounted':False,'privileged':x['HostConfig']['Privileged']}
 logs=run(['docker','logs','--tail','120',name],check=False);v['own_logs']={'stdout':logs.stdout,'stderr':logs.stderr}
 if not x['State']['Running']:
  run(['docker','rm',name]);v['cleanup']={'container_absent':run(['docker','container','inspect',name],check=False).returncode!=0}
except BaseException as e:v['status']='ERROR';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
print(json.dumps(v));sys.exit(0 if v['status']=='READY' else 1)
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin)
for number,lease in j['leases'].items():
 p=subprocess.run(['kubectl','get','pod','-n','default',lease['name'],'-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['metadata']['uid']==lease['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['status']['phase']=='Running' and x['spec']['nodeName']==f'a3-5u-atlas900rcka3-{number}';assert next(c['status'] for c in x['status']['conditions'] if c['type']=='Ready')=='True'
def host(n,payload):
 p=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{n}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(payload),capture_output=True,text=True,timeout=100);return {'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:capacity=dict(zip(['44','46'],pool.map(lambda n:host(n,{'action':'capacity','node':n}),[44,46])))
assert all(x['exit_code']==0 for x in capacity.values()),capacity
store=host(46,{'action':'store','image':j['image'],'script':j['script'],'script_sha256':j['script_sha256']});print(json.dumps({'capacity':capacity,'store':store,'model_loaded':False,'PD_MTP_serving_executed':False}));sys.exit(store['exit_code'])
'''
script=(R/'scripts/L039_shared_store.py').read_text();p_lease=json.loads((R/'runs/L039-R027-reservations/stdout.json').read_text())['reservations'][0];d_lease=next(x for x in json.loads((R/'runs/L039-R006-reservations/stdout.json').read_text())['reservations'] if x['node'].endswith('-46'));payload={'host':HOST,'script':script,'script_sha256':hashlib.sha256(script.encode()).hexdigest(),'image':'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263','leases':{'44':{'name':p_lease['name'],'uid':p_lease['uid']},'46':{'name':d_lease['name'],'uid':d_lease['uid']}}}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=180)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);v=json.loads(p.stdout) if p.stdout else {};receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':v.get('store',{}).get('data',{}).get('status','UNKNOWN'),'model_loaded':False,'NPU_computation':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
