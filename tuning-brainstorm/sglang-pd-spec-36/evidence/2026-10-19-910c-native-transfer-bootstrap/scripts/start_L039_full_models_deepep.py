#!/usr/bin/env python3
"""Start both actual16rank PD+MTP model engines from sealed identical original clones."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R044-full-bootstrap';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,os,pathlib,shlex,shutil,socket,subprocess,sys,time,traceback
j=json.load(sys.stdin);plan=j['plan'];node=plan['node'];role=plan['role'];base=pathlib.Path('/data/wio/sglang-mech36-20261018');name=f'if-mech36-l039-r044-{role}-{node}';root=base/f'L039-R044-{role}';v={'status':'STARTING','node':node,'role':role,'name':name,'root':str(root),'model_load_attempted':False,'model_loaded':False,'PD_MTP_serving_executed':False,'created':False}
def run(a,t=30,check=True):
 p=subprocess.run(a,capture_output=True,text=True,timeout=t)
 if check:assert p.returncode==0,p.stderr
 return p
try:
 assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36';assert not root.exists() and run(['docker','container','inspect',name],check=False).returncode!=0,'Existing full engine must be inspected, not replayed'
 assert hashlib.sha256(j['bootstrap'].encode()).hexdigest()==j['bootstrap_sha256'];assert json.loads(run(['docker','image','inspect','--format','{{json .Id}}',plan['image']]).stdout)==plan['image']
 sealpath=pathlib.Path(plan['seal_path']);assert sealpath.parent==base and hashlib.sha256(sealpath.read_bytes()).hexdigest()==plan['seal_sha256'];seal=json.loads(sealpath.read_text());assert seal['node']==node and seal['model_path']==plan['model_path'] and seal['files_count']==193 and seal['weight_files']==186
 model=pathlib.Path(plan['model_path']);assert model.resolve().is_relative_to(base.resolve())
 for n,item in seal['files'].items():
  f=model/n;s=f.stat();assert not f.is_symlink() and [s.st_size,s.st_mtime_ns,s.st_ino]==item['destination_stat'],n
  if not n.endswith('.safetensors'):assert hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],n
 p=run(['npu-smi','info']);assert p.stdout.count('No running processes found in NPU')==8,'NPU use changed; no model start';v['NPU_preflight_sha256']=hashlib.sha256(p.stdout.encode()).hexdigest()
 mem=next(int(s.split()[1])*1024 for s in pathlib.Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'));assert mem>(plan['limits']['memory_GiB']+128)*1024**3,'Hostmemory headroom changed';v['host_MemAvailable_bytes']=mem
 for port in [39044 if node==44 else 39046,39098,39100+node]:
  with socket.socket() as s:s.bind((f'172.16.12.{node}',port))
 if node==46:
  store=json.loads(run(['docker','inspect','--format','{{json .}}',j['store']['name']]).stdout);assert store['Id']==j['store']['container_id'] and store['State']['Running'] and store['Config']['Labels']['task']=='sglang-pd-spec-36' and store['Config']['Labels']['run']=='L039-R033';assert json.loads((pathlib.Path(j['store']['root'])/'state.json').read_text())['status']=='READY'
 root.mkdir(mode=0o700);(root/'bootstrap.py').write_text(j['bootstrap']);(root/'plan.json').write_text(json.dumps(plan,indent=2));devices=['/dev/davinci'+str(n) for n in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']
 args=['docker','create','--name',name,'--runtime','runc','--label','task=sglang-pd-spec-36','--label','run=L039-R044','--label','role='+role,'--network','host','--ipc','private','--shm-size',str(plan['limits']['shm_GiB'])+'g','--cpus',str(plan['limits']['CPU']),'--memory',str(plan['limits']['memory_GiB'])+'g','--memory-swap',str(plan['limits']['memory_GiB'])+'g','--pids-limit',str(plan['limits']['pids']),'--ulimit','memlock=-1:-1','--cap-drop','ALL','--security-opt','no-new-privileges','--log-opt','max-size=256m','--log-opt','max-file=3']
 for k,val in plan['environment'].items():args+=['-e',k+'='+val]
 for dev in devices:args+=['--device',dev]
 mounts=[('/usr/local/Ascend/driver','/usr/local/Ascend/driver'),('/usr/local/Ascend/firmware','/usr/local/Ascend/firmware'),('/etc/ascend_install.info','/etc/ascend_install.info'),(str(model),'/model')]
 smi=shutil.which('npu-smi')
 if smi:mounts.append((str(pathlib.Path(smi).resolve()),'/usr/local/sbin/npu-smi'))
 if pathlib.Path('/etc/hccn.conf').is_file():mounts.append(('/etc/hccn.conf','/etc/hccn.conf'))
 for src,dst in mounts:args+=['--mount','type=bind,src='+src+',dst='+dst+',readonly']
 args+=['--mount','type=bind,src='+str(root)+',dst=/work']
 setup='set -e; if [ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then . /usr/local/Ascend/ascend-toolkit/set_env.sh; elif [ -r /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]; then . /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash; fi; exec python3 -u /work/bootstrap.py'
 args+=['--entrypoint','/bin/bash',plan['image'],'-c',setup];cid=run(args).stdout.strip();v['container_id']=cid;v['created']=True
 handle={'task':'sglang-pd-spec-36','run':'L039-R044','node':node,'role':role,'container':name,'container_id':cid,'model_path':str(model),'seal_sha256':plan['seal_sha256'],'argv':plan['argv'],'image':plan['image']};(root/'handle.json').write_text(json.dumps(handle,indent=2));run(['docker','start',name]);v['model_load_attempted']=True;time.sleep(4)
 x=json.loads(run(['docker','inspect','--format','{{json .}}',name]).stdout);assert x['Id']==cid and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R044';h=x['HostConfig'];assert not h['Privileged'] and h['IpcMode']=='private' and h['Memory']==plan['limits']['memory_GiB']*1024**3 and {d['PathOnHost'] for d in h['Devices']}==set(devices) and next(m for m in x['Mounts'] if m['Destination']=='/model')['RW'] is False
 v['state']=x['State'];v['status']='LAUNCHED_NOT_READY' if x['State']['Running'] else 'EXITED_DURING_STARTUP';v['plan']=plan;v['isolation']={'privileged':False,'network':h['NetworkMode'],'ipc':h['IpcMode'],'devices':devices,'CPU_limit':h['NanoCpus']/1e9,'memory_limit_bytes':h['Memory'],'shm_bytes':h['ShmSize'],'read_only_mounts':[m['Destination'] for m in x['Mounts'] if not m['RW']],'work_mount_owned':str(root)}
 if (root/'preflight.json').is_file():v['source_preflight']=json.loads((root/'preflight.json').read_text())
 logs=run(['docker','logs','--tail','100',name],t=25,check=False);v['own_initial_logs']={'stdout':logs.stdout,'stderr':logs.stderr}
except BaseException as e:v['status']='ERROR';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
print(json.dumps(v));sys.exit(0 if v['status']=='LAUNCHED_NOT_READY' else 1)
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin)
for node,lease in j['plan']['resource_leases'].items():
 p=subprocess.run(['kubectl','get','pod','-n','default',lease['name'],'-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['metadata']['uid']==lease['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['status']['phase']=='Running' and x['spec']['nodeName']==f'a3-5u-atlas900rcka3-{node}' and x['spec']['containers'][0]['resources']['requests']['huawei.com/Ascend910']=='16';assert next(c['status'] for c in x['status']['conditions'] if c['type']=='Ready')=='True'
 p=subprocess.run(['kubectl','get','node',x['spec']['nodeName'],'-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0;assert next(c['status'] for c in json.loads(p.stdout)['status']['conditions'] if c['type']=='Ready')=='True'
def start(plan):
 node=plan['node'];p=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{node}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps({'plan':plan,'bootstrap':j['bootstrap'],'bootstrap_sha256':j['bootstrap_sha256'],'store':j['store']}),capture_output=True,text=True,timeout=125);return str(node),{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(start,j['plan']['roles']))
print(json.dumps({'status':'BOTH_LAUNCHED_NOT_READY' if all(x['exit_code']==0 for x in results.values()) else 'PARTIAL_OR_FAILED_BOOTSTRAP','nodes':results,'full_model_load_attempted':True,'PD_MTP_serving_executed':False,'baseline_accepted':False,'performance':'NOT_MEASURED'}));sys.exit(0 if all(x['exit_code']==0 for x in results.values()) else 1)
'''
plan=json.loads((R/'specs/L039-full-bootstrap-plan-v3.json').read_text());bootstrap=(R/'scripts/L039_model_bootstrap.py').read_text();store=json.loads((R/'runs/L039-R033-shared-store/remote-result.json').read_text())['store']['data'];payload={'host':HOST,'plan':plan,'bootstrap':bootstrap,'bootstrap_sha256':hashlib.sha256(bootstrap.encode()).hexdigest(),'store':{k:store[k] for k in ['name','container_id','root']}}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=190)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);v=json.loads(p.stdout) if p.stdout else {};receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':v.get('status','UNKNOWN'),'PD_MTP_serving_executed':False,'baseline_accepted':False,'performance':'NOT_MEASURED','script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
