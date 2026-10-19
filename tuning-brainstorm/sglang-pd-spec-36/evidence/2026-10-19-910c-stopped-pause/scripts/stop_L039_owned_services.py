"""User-authorized shutdown of exact task CIDs and UID-guarded reservation Pods; retain data."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];tag='L039-R093';O=R/f'runs/{tag}-shutdown';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import datetime,hashlib,json,pathlib,re,subprocess,sys,time
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36';root=base/'L039-R093-shutdown';root.mkdir(mode=0o700,exist_ok=True)
def inspect(item):
 x=json.loads(subprocess.run(['docker','inspect','--format','{{json .}}',item['container_id']],capture_output=True,text=True,check=True,timeout=15).stdout);assert x['Id']==item['container_id'] and x['Name']=='/'+item['container'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']==item['run'];return x
def proc(pid):
 p=pathlib.Path('/proc')/str(pid)
 try:
  text=(p/'stat').read_text();fields=text[text.rfind(')')+2:].split();return {'pid':pid,'state':fields[0],'start_ticks':fields[19]}
 except FileNotFoundError:return None
results=[]
for item in j['items']:
 x=inspect(item);v={'node':item['node'],'role':item['role'],'container_id':x['Id'],'container_name':x['Name'],'before':x['State'],'image':x['Image'],'operation':j['action'],'container_retained':True,'model_files_modified':False}
 if item['role']=='decode' and j['action']=='pre':
  done=json.loads((base/'L039-R092-pd-lifecycle/completion.json').read_text());assert done['actual_worker_exit_code']==0 and done['handle']['run']=='L039-R092';v['last_client_completion']=done
  p=proc(done['handle']['pid']);assert p is None or p['start_ticks']!=done['handle']['proc_start_ticks'] or p['state'] in ['Z','X'],'Native client still active; do not stop'
 if j['action']=='stop':
  processes=[]
  if x['State']['Running']:
   p=subprocess.run(['docker','top',x['Id'],'-eo','pid'],capture_output=True,text=True,timeout=15);assert p.returncode==0,p.stderr
   for line in p.stdout.splitlines()[1:]:
    if line.strip().isdigit():
     state=proc(int(line.strip()))
     if state is not None:processes.append(state)
  v['owned_processes_before']=processes
  for phase in ['before','after']:
   if phase=='after':
    if x['State']['Running']:
     p=subprocess.run(['docker','stop','--time','30',x['Id']],capture_output=True,text=True,timeout=65);v['docker_stop']={'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr};assert p.returncode==0,p.stderr
    else:v['already_stopped']=True
    x=inspect(item);v['after']=x['State'];assert not x['State']['Running'] and x['State']['Status'] in ['exited','dead','created']
   p=subprocess.run(['docker','logs','--timestamps',x['Id']],capture_output=True,timeout=25);assert p.returncode==0,p.stderr[-2000:];files={}
   for stream,data in [('stdout',p.stdout),('stderr',p.stderr)]:
    name=item['role']+'-'+phase+'.'+stream+'.log';q=root/name;assert not q.exists();q.write_bytes(data);files[stream]={'path':str(q),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
   v[phase+'_logs']=files
  deadline=time.monotonic()+12
  while True:
   active=[]
   for old in processes:
    now=proc(old['pid'])
    if now is not None and now['start_ticks']==old['start_ticks'] and now['state'] not in ['Z','X']:active.append(now)
   if not active or time.monotonic()>=deadline:break
   time.sleep(.2)
  v['original_owned_processes_still_active']=active;assert not active,'Owned process remains active after Docker stop'
 results.append(v)
p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25);inventory=p.stdout+p.stderr
out={'action':j['action'],'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host_monotonic_ns':time.monotonic_ns(),'results':results,'NPU_inventory_exit_code':p.returncode,'HBM_used_MiB':list(map(int,re.findall(r'(\d+)\s*/\s*65536',inventory))),'NPU_inventory_sha256':hashlib.sha256(inventory.encode()).hexdigest(),'scope':'No container removal, model deletion, image deletion or actions against other services.'}
if j['action']=='stop':
 q=root/(j['items'][0]['role']+'-receipt.json');assert not q.exists();q.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
'''
JUMP=r'''
import concurrent.futures,datetime,json,pathlib,subprocess,sys,tempfile,time,urllib.parse
j=json.load(sys.stdin);v={'run':'L039-R093','user_authorized':'Stop this task P/D/store, release own reservation, retain models/containers/logs','generation_requests':0,'containers_removed':0,'lease_pre':{}}
def get_pod(name):
 p=subprocess.run(['kubectl','get','pod','-n','default',name,'--ignore-not-found','-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;return json.loads(p.stdout) if p.stdout.strip() else None
for node,lease in j['leases'].items():
 x=get_pod(lease['name']);assert x is not None and x['metadata']['uid']==lease['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['spec']['nodeName']==f'a3-5u-atlas900rcka3-{node}';v['lease_pre'][node]={'name':lease['name'],'uid':x['metadata']['uid'],'phase':x['status']['phase']}
def call(node,items,action):
 p=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{node}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps({'items':items,'action':action}),capture_output=True,text=True,timeout=155)
 return {'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures={str(node):pool.submit(call,node,[x for x in j['targets'] if x['node']==node],'pre') for node in [44,46]};v['pre']={n:f.result() for n,f in futures.items()}
assert all(x['exit_code']==0 for x in v['pre'].values()),v['pre'];v['stops']=[]
for role in ['decode','prefill','store']:
 item=next(x for x in j['targets'] if x['role']==role);result=call(item['node'],[item],'stop');v['stops'].append({'role':role,**result})
 if result['exit_code']!=0:
  v['status']='FAIL_STOP_REOBSERVE_EXACT_CIDS_DO_NOT_REPLAY';print(json.dumps(v));sys.exit(1)
v['lease_release']={}
for node,lease in j['leases'].items():
 x=get_pod(lease['name']);assert x is not None and x['metadata']['uid']==lease['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36'
 # Kubernetes DeleteOptions UID precondition prevents deleting a replacement Pod.
 options={'apiVersion':'v1','kind':'DeleteOptions','gracePeriodSeconds':15,'preconditions':{'uid':lease['uid']}}
 with tempfile.TemporaryDirectory(prefix='if-mech36-L039-R093-') as d:
  q=pathlib.Path(d)/'delete-options.json';q.write_text(json.dumps(options));q.chmod(0o600)
  p=subprocess.run(['kubectl','delete','--raw','/api/v1/namespaces/default/pods/'+urllib.parse.quote(lease['name'],safe=''),'-f',str(q)],capture_output=True,text=True,timeout=35)
 result={'name':lease['name'],'expected_uid':lease['uid'],'delete_exit_code':p.returncode,'delete_stdout':p.stdout,'delete_stderr':p.stderr,'UID_precondition':True};v['lease_release'][node]=result
 if p.returncode!=0:
  v['status']='FAIL_LEASE_RELEASE_SERVICES_STOPPED';print(json.dumps(v));sys.exit(1)
 deadline=time.monotonic()+75
 while True:
  x=get_pod(lease['name'])
  if x is None or x['metadata']['uid']!=lease['uid']:break
  if time.monotonic()>=deadline:raise RuntimeError('Exact lease UID still exists after delete')
  time.sleep(1)
 result['old_UID_absent']=True;result['replacement_UID_preserved']=None if x is None else x['metadata']['uid']
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures={str(node):pool.submit(call,node,[x for x in j['targets'] if x['node']==node],'post') for node in [44,46]};v['post']={n:f.result() for n,f in futures.items()}
assert all(x['exit_code']==0 and all(not z['before']['Running'] for z in x['data']['results']) for x in v['post'].values())
v['status']='PASS_OWN_SERVICES_STOPPED_RESERVATIONS_RELEASED';v['priority_class']='Retained metadata only; no NPU reservation. Access pods and other services unchanged.';v['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(v))
'''
published=json.loads((R/'publication-L039-lifecycle-pause/published.json').read_text());assert published['remote_changed_files_exact'] and published['head']=='fb9f91a97582dc78f646abe56fbd0e3f1157fa07'
h=json.loads((R/'specs/L039-R077-targets.json').read_text());targets=[{**x,'role':'prefill' if x['node']==44 else 'decode'} for x in h['targets']]+[{**h['store'],'role':'store'}]
payload={'targets':targets,'leases':h['resource_leases'],'host':HOST}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
started=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=700)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);data=json.loads(p.stdout) if p.stdout else {}
receipt={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':data.get('status','UNKNOWN'),'publication_before_shutdown':published['head'],'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in O.iterdir() if q.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
