#!/usr/bin/env python3
"""Task-owned two-node native correctness probe; stdin control through existing SSH/kubectl only."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R031-paired-transfer';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,os,pathlib,shlex,socket,subprocess,sys,threading,traceback
j=json.loads(sys.stdin.readline());node=j['node'];name=f'if-mech36-l039-r031-transfer-{node}';task='sglang-pd-spec-36';image=j['plan']['image'];base=pathlib.Path('/data/wio/sglang-mech36-20261018');created=False;v={'node':node,'name':name,'status':'STARTING','model_loaded':False,'cleanup':None}
def run(a,t=30,check=True):
 p=subprocess.run(a,capture_output=True,text=True,timeout=t)
 if check:assert p.returncode==0,p.stderr
 return p
try:
 assert json.loads((base/'owner.json').read_text())['task']==task
 assert hashlib.sha256(j['probe'].encode()).hexdigest()==j['probe_sha256']
 p=run(['npu-smi','info']);assert p.stdout.count('No running processes found in NPU')==8,'NPU use changed';v['npu_preflight_sha256']=hashlib.sha256(p.stdout.encode()).hexdigest()
 assert run(['docker','container','inspect',name],check=False).returncode!=0,'Existing probe must be inspected, not replayed'
 assert json.loads(run(['docker','image','inspect','--format','{{json .Id}}',image]).stdout)==image
 if node==46:
  with socket.socket() as s:s.bind(('172.16.12.46',39431))
 assert not (base/f'L039-R031-probe-{node}-handle.json').exists(),'Prior handle exists'
 devices=['/dev/davinci0','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']
 args=['docker','create','-i','--name',name,'--runtime','runc','--label','task='+task,'--label','run=L039-R031','--network','host','--ipc','private','--shm-size','256m','--cpus','4','--memory','8g','--pids-limit','512','--cap-drop','ALL','--security-opt','no-new-privileges']
 for k,val in {'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','PYTHONDONTWRITEBYTECODE':'1','ASCEND_RT_VISIBLE_DEVICES':'0','ASCEND_PROCESS_LOG_PATH':'/tmp/task-ascend/log','TORCH_EXTENSIONS_DIR':'/tmp/task-torch-extensions','SGLANG_CACHE_DIR':'/tmp/task-sglang-cache','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}.items():args+=['-e',k+'='+val]
 for d in devices:args+=['--device',d]
 for source,target in [('/usr/local/Ascend/driver','/usr/local/Ascend/driver'),('/usr/local/Ascend/firmware','/usr/local/Ascend/firmware'),('/etc/ascend_install.info','/etc/ascend_install.info')]:args+=['--mount','type=bind,src='+source+',dst='+target+',readonly']
 program='CONFIG = '+repr({'node':node,'plan':j['plan']})+'\n'+j['probe']
 setup='set -e; if [ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then . /usr/local/Ascend/ascend-toolkit/set_env.sh; elif [ -r /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]; then . /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash; fi; exec python3 -u -c '+shlex.quote(program)
 args+=['--entrypoint','/bin/bash',image,'-c',setup];cid=run(args).stdout.strip();created=True;v['container_id']=cid
 x=json.loads(run(['docker','inspect','--format','{{json .}}',name]).stdout);h=x['HostConfig'];assert x['Id']==cid and not h['Privileged'] and h['NetworkMode']=='host' and h['IpcMode']=='private' and h['Memory']==8*1024**3 and {d['PathOnHost'] for d in h['Devices']}==set(devices) and len(x['Mounts'])==3 and all(not m['RW'] for m in x['Mounts'])
 handle={'task':task,'run':'L039-R031','node':node,'container':name,'container_id':cid}
 with (base/f'L039-R031-probe-{node}-handle.json').open('x') as f:json.dump(handle,f)
 print('L039_HOST_HANDLE='+json.dumps(handle),flush=True)
 v['isolation']={'NPU_devices':devices,'host_mounts_read_only':True,'model_mounted':False,'network':'host','ipc':'private','CPU_limit':4,'memory_limit_bytes':h['Memory']}
 p=subprocess.Popen(['docker','start','-a','-i',name],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
 def control():
  try:
   while True:
    chunk=os.read(sys.stdin.fileno(),65536)
    if not chunk:break
    view=memoryview(chunk)
    while view:view=view[os.write(p.stdin.fileno(),view):]
  except (BrokenPipeError,OSError):pass
  finally:
   try:p.stdin.close()
   except (BrokenPipeError,OSError):pass
 def relay(source,target):
  for line in source:target.write(line);target.flush()
 threads=[threading.Thread(target=control,daemon=True),threading.Thread(target=relay,args=(p.stdout,sys.stdout),daemon=True),threading.Thread(target=relay,args=(p.stderr,sys.stderr),daemon=True)]
 for t in threads:t.start()
 try:v['probe_exit_code']=p.wait(timeout=360)
 except subprocess.TimeoutExpired:
  v['watchdog_expired']=True
  labels=json.loads(run(['docker','inspect','--format','{{json .Config.Labels}}',name]).stdout);assert labels.get('task')==task and labels.get('run')=='L039-R031';run(['docker','stop','-t','3',name],t=15,check=False);v['probe_exit_code']=p.wait(timeout=20)
 for t in threads[1:]:t.join(timeout=2)
 v['status']='PASS' if v['probe_exit_code']==0 else 'FAIL'
except BaseException as e:v['status']='ERROR';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
finally:
 if created:
  q=run(['docker','inspect','--format','{{json .}}',name],check=False)
  if q.returncode==0:
   x=json.loads(q.stdout);assert x['Id']==v['container_id'] and x['Config']['Labels'].get('task')==task and x['Config']['Labels'].get('run')=='L039-R031'
   v['terminal_state']=x['State'];logs=base/f'L039-R031-runtime-logs-{node}';logs.mkdir(mode=0o700,exist_ok=False);cp=run(['docker','cp',name+':/tmp/task-ascend/log',str(logs)],check=False);v['own_runtime_logs']={'root':str(logs),'copy_exit':cp.returncode,'files':[{'path':str(f.relative_to(logs)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in logs.rglob('*') if f.is_file()]}
   run(['docker','rm','-f',name],check=False)
  v['cleanup']={'container_absent':run(['docker','container','inspect',name],check=False).returncode!=0}
print('L039_HOST_RESULT='+json.dumps(v),flush=True);sys.exit(0 if v['status']=='PASS' and v['cleanup']['container_absent'] else 1)
'''
EMERGENCY=r'''
import json,pathlib,subprocess,sys
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36';p=base/f"L039-R031-probe-{j['node']}-handle.json"
if not p.is_file():print(json.dumps({'no_handle_created':True}));sys.exit(0)
h=json.loads(p.read_text());assert h['task']=='sglang-pd-spec-36' and h['run']=='L039-R031' and h['node']==j['node'];q=subprocess.run(['docker','inspect','--format','{{json .}}',h['container']],capture_output=True,text=True,timeout=20)
if q.returncode==0:
 x=json.loads(q.stdout);assert x['Id']==h['container_id'] and x['Config']['Labels'].get('task')==h['task'] and x['Config']['Labels'].get('run')==h['run'];subprocess.run(['docker','rm','-f',h['container']],capture_output=True,text=True,timeout=30)
q=subprocess.run(['docker','container','inspect',h['container']],capture_output=True,text=True,timeout=20);print(json.dumps({'container':h['container'],'container_id':h['container_id'],'container_absent':q.returncode!=0}));sys.exit(0 if q.returncode!=0 else 1)
'''
JUMP=r'''
import json,queue,subprocess,sys,threading,time,traceback
j=json.load(sys.stdin);plan=j['plan'];Q=queue.Queue();procs={};pending={};v={'status':'STARTING','scope':plan['scope'],'events':[],'handles':{},'host_results':{},'logs':{},'cases':[],'model_loaded':False,'PD_MTP_serving_executed':False,'inter_node_transfer_verified':False,'performance':'NOT_MEASURED','emergency_cleanup':{}}
def run(a,t=30):
 p=subprocess.run(a,capture_output=True,text=True,timeout=t);assert p.returncode==0,p.stderr;return json.loads(p.stdout)
def start(role,node):
 p=subprocess.Popen(['kubectl','exec','-i','-n','default',f'sglang-container-access-{node}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1);procs[role]=(node,p);v['logs'][role]=[]
 def reader(stream,kind):
  for line in stream:Q.put((role,kind,line.rstrip('\n')))
  Q.put((role,kind,None))
 for stream,kind in [(p.stdout,'stdout'),(p.stderr,'stderr')]:threading.Thread(target=reader,args=(stream,kind),daemon=True).start()
 p.stdin.write(json.dumps({'node':node,'probe':j['probe'],'probe_sha256':j['probe_sha256'],'plan':plan})+'\n');p.stdin.flush()
def send(role,command):
 p=procs[role][1];p.stdin.write(json.dumps(command)+'\n');p.stdin.flush()
def ingest(item):
 role,kind,line=item
 if line is None:return
 if kind=='stdout' and line.startswith('L039_EVENT='):
  e=json.loads(line[len('L039_EVENT='):]);assert e['role']==role;v['events'].append(e);pending.setdefault((role,e['kind']),[]).append(e)
  if e['kind'] in ['error','cleanup_error']:raise RuntimeError(role+' '+json.dumps(e))
 elif kind=='stdout' and line.startswith('L039_HOST_HANDLE='):v['handles'][role]=json.loads(line[len('L039_HOST_HANDLE='):])
 elif kind=='stdout' and line.startswith('L039_HOST_RESULT='):
  h=json.loads(line[len('L039_HOST_RESULT='):]);v['host_results'][role]=h
  if h['status']!='PASS':raise RuntimeError('Host controller failed: '+role+' '+json.dumps(h))
 elif len(v['logs'][role])<200:v['logs'][role].append({'stream':kind,'line':line[:1600]})
def wait(role,kind,seconds):
 end=time.monotonic()+seconds
 while time.monotonic()<end:
  key=(role,kind)
  if pending.get(key):return pending[key].pop(0)
  if kind=='host_result' and role in v['host_results']:return v['host_results'][role]
  try:ingest(Q.get(timeout=min(1,max(0.01,end-time.monotonic()))))
  except queue.Empty:
   if procs[role][1].poll() is not None:raise RuntimeError(f'{role} host exited before {kind}, exit={procs[role][1].returncode}')
 raise TimeoutError(f'{role} {kind} deadline expired; do not retry transfer')
try:
 for node,lease in j['leases'].items():
  x=run(['kubectl','get','pod','-n','default',lease['name'],'-o','json']);assert x['metadata']['uid']==lease['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['status']['phase']=='Running' and x['spec']['nodeName']==f'a3-5u-atlas900rcka3-{node}' and x['spec']['containers'][0]['resources']['requests']['huawei.com/Ascend910']=='16';assert next(c['status'] for c in x['status']['conditions'] if c['type']=='Ready')=='True'
  n=run(['kubectl','get','node',x['spec']['nodeName'],'-o','json']);assert next(c['status'] for c in n['status']['conditions'] if c['type']=='Ready')=='True'
 start('D',46);dest=wait('D','ready',plan['init_deadline_s']);start('P',44);sender=wait('P','ready',plan['init_deadline_s']);assert dest['buffer_bytes']==sender['buffer_bytes']==plan['buffer_bytes'];v['descriptors']={'P':sender,'D':dest}
 for case in plan['cases']:
  send('D',{'op':'prepare','case':case['id']});armed=wait('D','armed',plan['case_deadline_s']);assert armed['case']==case['id']
  send('P',{'op':'send','case':case['id'],'peer':dest});sent=wait('P','sent',plan['case_deadline_s']);assert sent['case']==case['id'] and sent['return_code']==0
  send('D',{'op':'verify','case':case['id']});verified=wait('D','verified',plan['case_deadline_s']);assert verified['case']==case['id'] and verified['pass'];v['cases'].append({'case':case['id'],'sent':sent,'verified':verified})
 send('P',{'op':'finish'});p_done=wait('P','final',30);p_host=wait('P','host_result',45);assert p_done['status']=='PASS' and p_done['deregistered'] and p_host['cleanup']['container_absent']
 send('D',{'op':'finish'});d_done=wait('D','final',30);d_host=wait('D','host_result',45);assert d_done['status']=='PASS' and d_done['deregistered'] and d_host['cleanup']['container_absent']
 v['inter_node_transfer_verified']=True;v['status']='PASS'
except BaseException as e:v['status']='FAIL';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
finally:
 for role,(node,p) in procs.items():
  try:p.stdin.close()
  except (BrokenPipeError,OSError):pass
  try:p.wait(timeout=12)
  except subprocess.TimeoutExpired:
   q=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{node}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['emergency']],input=json.dumps({'node':node}),capture_output=True,text=True,timeout=80);v['emergency_cleanup'][role]={'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:
    p.terminate()
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
  v.setdefault('host_process_exits',{})[role]=p.poll()
 while not Q.empty():
  try:ingest(Q.get_nowait())
  except BaseException as e:v.setdefault('post_control_errors',[]).append({'type':type(e).__name__,'message':str(e)})
 if v.get('post_control_errors') or any(x is None or x!=0 for x in v.get('host_process_exits',{}).values()):v['status']='FAIL'
print(json.dumps(v));sys.exit(0 if v['status']=='PASS' else 1)
'''
plan=json.loads((R/'specs/L039-paired-transfer-plan-v1.json').read_text());probe=(R/'scripts/probe_L039_paired_transfer.py').read_text();p_lease=json.loads((R/'runs/L039-R027-reservations/stdout.json').read_text())['reservations'][0];d_lease=next(x for x in json.loads((R/'runs/L039-R006-reservations/stdout.json').read_text())['reservations'] if x['node'].endswith('-46'))
payload={'host':HOST,'emergency':EMERGENCY,'probe':probe,'probe_sha256':hashlib.sha256(probe.encode()).hexdigest(),'plan':plan,'leases':{'44':{'name':p_lease['name'],'uid':p_lease['uid']},'46':{'name':d_lease['name'],'uid':d_lease['uid']}}}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=580)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);v=json.loads(p.stdout) if p.stdout else {};receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':v.get('status','UNKNOWN'),'inter_node_transfer_verified':v.get('inter_node_transfer_verified',False),'cases_verified':len(v.get('cases',[])),'model_loaded':False,'PD_MTP_serving_executed':False,'performance':'NOT_MEASURED','leases':payload['leases'],'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
