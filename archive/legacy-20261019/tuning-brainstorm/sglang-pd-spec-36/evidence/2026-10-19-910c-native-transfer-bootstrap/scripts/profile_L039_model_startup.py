#!/usr/bin/env python3
"""Read two5s-separated own worker CPU/IO/fault snapshots; never attach or stop processes."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R042-startup-profile';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import datetime,json,os,pathlib,subprocess,sys,time
j=json.load(sys.stdin);name=j['container'];x=json.loads(subprocess.check_output(['docker','inspect','--format','{{json .}}',name],text=True));assert x['Id']==j['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R039';v={'node':j['node'],'container':name,'container_id':x['Id'],'state':x['State'],'scope':'Startup progress only; no inference/performancebaseline or ptrace','snapshots':[],'deltas':[]}
def snapshot():
 p=subprocess.run(['docker','top',name,'-eo','pid'],capture_output=True,text=True,timeout=20);items={}
 for line in p.stdout.splitlines()[1:]:
  try:
   pid=int(line.strip());root=pathlib.Path('/proc')/str(pid);raw=(root/'stat').read_text();fields=raw[raw.rfind(')')+2:].split();io={a.split(':')[0]:int(a.split()[1]) for a in (root/'io').read_text().splitlines()};status={a.split(':')[0]:a.split(':',1)[1].strip() for a in (root/'status').read_text().splitlines() if a.split(':')[0] in ['Name','State','VmRSS','RssAnon','RssFile','RssShmem','Threads']};items[str(pid)]={'pid':pid,'state':fields[0],'starttime_ticks':int(fields[19]),'utime_ticks':int(fields[11]),'stime_ticks':int(fields[12]),'minor_faults':int(fields[7]),'major_faults':int(fields[9]),'io':io,'status':status,'wait_channel':(root/'wchan').read_text().strip()}
  except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError):continue
 return {'host_monotonic_ns':time.monotonic_ns(),'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':items,'docker_top_exit':p.returncode}
if x['State']['Running']:
 a=snapshot();time.sleep(5);b=snapshot();v['snapshots']=[a,b];elapsed=(b['host_monotonic_ns']-a['host_monotonic_ns'])/1e9;hz=os.sysconf('SC_CLK_TCK');v['same_host_elapsed_seconds']=elapsed;v['CPU_tick_Hz']=hz
 for pid,first in a['workers'].items():
  if pid not in b['workers'] or first['starttime_ticks']!=b['workers'][pid]['starttime_ticks']:continue
  last=b['workers'][pid];delta={k:last[k]-first[k] for k in ['utime_ticks','stime_ticks','minor_faults','major_faults']};delta['CPU_seconds']=(delta['utime_ticks']+delta['stime_ticks'])/hz;v['deltas'].append({'pid':int(pid),'name':last['status'].get('Name'),'state':last['state'],'wait_channel':last['wait_channel'],'delta':delta,'io_delta':{k:last['io'][k]-z for k,z in first['io'].items() if k in last['io']}})
 v['aggregate']={'same_workers':len(v['deltas']),'CPU_seconds':sum(x['delta']['CPU_seconds'] for x in v['deltas']),'minor_faults':sum(x['delta']['minor_faults'] for x in v['deltas']),'major_faults':sum(x['delta']['major_faults'] for x in v['deltas']),'read_bytes':sum(x['io_delta'].get('read_bytes',0) for x in v['deltas']),'rchar':sum(x['io_delta'].get('rchar',0) for x in v['deltas'])}
p=subprocess.run(['docker','logs','--tail','80',name],capture_output=True,text=True,timeout=20);v['own_log_tail']={'stdout':p.stdout[-30000:],'stderr':p.stderr[-30000:]};print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin)
def observe(item):
 n=item['node'];p=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{n}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(item),capture_output=True,text=True,timeout=75);return str(n),{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(observe,j['targets']))
print(json.dumps({'read_only':True,'restarted':False,'inference_requests':0,'nodes':results}));sys.exit(0 if all(x['exit_code']==0 for x in results.values()) else 1)
'''
launch=json.loads((R/'runs/L039-R039-full-bootstrap/remote-result.json').read_text());targets=[{'node':int(n),'container':x['data']['name'],'container_id':x['data']['container_id']} for n,x in launch['nodes'].items()];cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST,'targets':targets}),capture_output=True,text=True,timeout=110)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'inference_requests':0,'restarted':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
