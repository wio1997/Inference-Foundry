#!/usr/bin/env python3
"""Create one exact guarded diagnostic container on reserved910C44, preserve own logs."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1]; tag=sys.argv[1]; assert tag in ['L039-R065','L039-R066']
O=R/f'runs/{tag}-plain-exit'; O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,pathlib,shlex,shutil,socket,subprocess,sys,time,traceback
j=json.load(sys.stdin); plan=j['plan']; tag=plan['run']; base=pathlib.Path('/data/wio/sglang-mech36-20261018'); root=base/(tag+'-plain-exit'); name='if-mech36-'+tag.lower()+'-plain-exit-44'; created=False
v={'run':tag,'node':44,'name':name,'root':str(root),'status':'STARTING','model_loaded':False,'PD_MTP_serving':False,'performance':'NOT_MEASURED'}
def run(a,t=25,check=True):
 p=subprocess.run(a,capture_output=True,text=True,timeout=t)
 if check:assert p.returncode==0,p.stderr
 return p
try:
 assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
 assert not root.exists() and run(['docker','container','inspect',name],check=False).returncode!=0,'Existing probe cannot be replayed'
 assert hashlib.sha256(j['probe'].encode()).hexdigest()==plan['probe_sha256']
 assert json.loads(run(['docker','image','inspect','--format','{{json .Id}}',plan['image']]).stdout)==plan['image']
 smi=run(['npu-smi','info']).stdout; assert smi.count('No running processes found in NPU')==8,'Current device use changed'
 v['idle_device_preflight_sha256']=hashlib.sha256(smi.encode()).hexdigest()
 with socket.socket() as s:s.bind(('127.0.0.1',plan['master_port']))
 root.mkdir(mode=0o700); (root/'probe.py').write_text(j['probe']); assert hashlib.sha256(j['launcher'].encode()).hexdigest()==plan['launcher_sha256']; (root/'launch.py').write_text(j['launcher']); (root/'plan.json').write_text(json.dumps(plan,indent=2))
 devices=['/dev/davinci'+str(n) for n in range(16)]+['/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']
 args=['docker','create','--name',name,'--runtime','runc','--label','task=sglang-pd-spec-36','--label','run='+tag,'--network','host','--ipc','private','--shm-size','8g','--cpus','64','--memory','64g','--memory-swap','64g','--pids-limit','2048','--ulimit','memlock=-1:-1','--cap-drop','ALL','--security-opt','no-new-privileges']
 for d in devices:args+=['--device',d]
 for k,val in plan['environment'].items():args+=['-e',k+'='+val]
 for src,dst in [('/usr/local/Ascend/driver','/usr/local/Ascend/driver'),('/usr/local/Ascend/firmware','/usr/local/Ascend/firmware'),('/etc/ascend_install.info','/etc/ascend_install.info'),('/etc/hccn.conf','/etc/hccn.conf')]:
  if pathlib.Path(src).exists():args+=['--mount','type=bind,src='+src+',dst='+dst+',readonly']
 args+=['--mount','type=bind,src='+str(root)+',dst=/work']
 setup='set -e; if [ -r /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then . /usr/local/Ascend/ascend-toolkit/set_env.sh; elif [ -r /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash ]; then . /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash; fi; exec python3 -u /work/launch.py'
 args+=['--entrypoint','/bin/bash',plan['image'],'-c',setup]
 cid=run(args).stdout.strip(); v['container_id']=cid; created=True
 x=json.loads(run(['docker','inspect','--format','{{json .}}',name]).stdout); assert x['Id']==cid and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']==tag and not x['HostConfig']['Privileged'] and x['HostConfig']['Memory']==64*1024**3 and {d['PathOnHost'] for d in x['HostConfig']['Devices']}==set(devices)
 (root/'handle.json').write_text(json.dumps({'task':'sglang-pd-spec-36','run':tag,'container':name,'container_id':cid},indent=2))
 attached=(root/'attached.log').open('w'); proc=subprocess.Popen(['docker','start','-a',name],stdout=attached,stderr=subprocess.STDOUT,text=True); started=time.monotonic(); completed_at=None; v['process_snapshots']=[]
 while proc.poll() is None and time.monotonic()-started<240:
  results=[]
  for f in root.glob('rank-*.json'):
   try:results.append(json.loads(f.read_text()))
   except (ValueError,OSError):pass
  if len(results)==16 and all(r['status'] in ['PASS','FAIL'] for r in results):
   if completed_at is None:completed_at=time.monotonic()
   if time.monotonic()-completed_at>=35:
    top=run(['docker','top',name,'-eo','pid,ppid,stat,comm,wchan:32,args'],check=False)
    v['process_snapshots'].append({'host_monotonic_ns':time.monotonic_ns(),'seconds_since_all_rank_results':time.monotonic()-completed_at,'docker_top_exit':top.returncode,'stdout':top.stdout,'stderr':top.stderr});v['post_result_exit_timeout']=True;break
  time.sleep(2)
 v['attached_exit_code']=proc.poll()
 if proc.poll() is None:v['attached_timeout']=True
 attached.close()
 x=json.loads(run(['docker','inspect','--format','{{json .}}',name]).stdout); assert x['Id']==cid and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']==tag
 if x['State']['Running']:run(['docker','stop','-t','10',name]); v['stopped_only_own_timed_out_probe']=True
 p=run(['docker','logs',name],t=25,check=False); text=p.stdout+p.stderr; (root/'container.log').write_text(text); v['container_log_sha256']=hashlib.sha256(text.encode()).hexdigest(); v['log_tail']=text[-18000:]
 v['rank_results']={}
 for f in sorted(root.glob('rank-*.json')):v['rank_results'][f.stem]=json.loads(f.read_text())
 x=json.loads(run(['docker','inspect','--format','{{json .}}',name]).stdout); v['final_state']=x['State']; assert not x['State']['Running']
 v['status']='PASS' if len(v['rank_results'])==16 and all(r['status']=='PASS' for r in v['rank_results'].values()) and x['State']['ExitCode']==0 else 'FAIL'
 # Preserve all failed/successful probe containers and host logs; only computation is stopped.
 v['own_container_preserved']=True; v['no_active_probe_process']=True
 v['python_stacks']={f.name:f.read_text(errors='replace')[-24000:] for f in sorted(root.glob('rank-*.python-stack.log'))}
 v['launcher_result']=json.loads((root/'launcher-result.json').read_text()) if (root/'launcher-result.json').exists() else None
 v['worker_logs']={f.name:{'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'tail':f.read_text(errors='replace')[-8000:]} for f in sorted(root.glob('worker-*.log'))}
 if v['status']=='PASS':assert v['launcher_result']['status']=='PASS' and len(v['launcher_result']['children'])==16 and all(c['exit_code']==0 for c in v['launcher_result']['children'])
except BaseException as e:
 v['status']='ERROR'; v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
 if created:
  x=json.loads(run(['docker','inspect','--format','{{json .}}',name]).stdout); assert x['Id']==v['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']==tag
  if x['State']['Running']:run(['docker','stop','-t','10',name])
  v['final_state']=json.loads(run(['docker','inspect','--format','{{json .State}}',name]).stdout)
print(json.dumps(v)); sys.exit(0 if v['status']=='PASS' else 1)
'''
JUMP=r'''
import json,subprocess,sys
j=json.load(sys.stdin); lease=j['plan']['lease']; p=subprocess.run(['kubectl','get','pod','-n','default',lease['name'],'-o','json'],capture_output=True,text=True,timeout=25); assert p.returncode==0,p.stderr
x=json.loads(p.stdout); assert x['metadata']['uid']==lease['uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['spec']['nodeName']=='a3-5u-atlas900rcka3-44' and x['status']['phase']=='Running' and x['spec']['containers'][0]['resources']['requests']['huawei.com/Ascend910']=='16' and next(c['status'] for c in x['status']['conditions'] if c['type']=='Ready')=='True'
p=subprocess.run(['kubectl','get','node',x['spec']['nodeName'],'-o','json'],capture_output=True,text=True,timeout=25); assert p.returncode==0 and next(c['status'] for c in json.loads(p.stdout)['status']['conditions'] if c['type']=='Ready')=='True'
p=subprocess.run(['kubectl','exec','-i','-n','default','sglang-container-access-44-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(j),capture_output=True,text=True,timeout=325); sys.stdout.write(p.stdout); sys.stderr.write(p.stderr); sys.exit(p.returncode)
'''
plan=json.loads((R/f'specs/{tag}-registration-plan.json').read_text()); probe=(R/plan['probe_file']).read_text();launcher=(R/'scripts/L039_plain_rank_launcher_v2.py').read_text()
payload={'host':HOST,'plan':plan,'probe':probe,'launcher':launcher}; cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=370)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);v=json.loads(p.stdout) if p.stdout else {}
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':v.get('status','NOT_EXECUTED'),'remote_node':44,'NPU_compute':any(r.get('NPU_compute') for r in v.get('rank_results',{}).values()),'model_loaded':False,'PD_MTP_serving':False,'performance':'NOT_MEASURED','script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
