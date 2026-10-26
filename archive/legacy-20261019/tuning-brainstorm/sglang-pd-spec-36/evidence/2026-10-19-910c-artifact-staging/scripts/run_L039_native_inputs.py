#!/usr/bin/env python3
"""Task-owned CPU-only input check on910C node46, always cleaned after observation."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R021-inputs';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,subprocess,sys,traceback
j=json.load(sys.stdin);name='if-mech36-l039-r021-inputs-46';task='sglang-pd-spec-36';image='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263';created=False
v={'name':name,'image':image,'device_compute_executed':False,'model_loaded':False,'probe_result':None,'cleanup':None}
def run(a,stdin=None,t=30,check=True):
 p=subprocess.run(a,input=stdin,capture_output=True,text=True,timeout=t)
 if check:assert p.returncode==0,p.stderr
 return p
try:
 assert hashlib.sha256(j['probe'].encode()).hexdigest()==j['probe_sha256']
 assert run(['docker','container','inspect',name],check=False).returncode!=0,'Name already exists; inspect rather than replay'
 assert json.loads(run(['docker','image','inspect','--format','{{json .Id}}',image]).stdout)==image
 devices=[]
 model=__import__('pathlib').Path('/data/wio/sglang-mech36-20261018/L039-R009-transfer/model')
 for filename,expected in j['model_hashes'].items():assert __import__('hashlib').sha256((model/filename).read_bytes()).hexdigest()==expected,filename
 args=['docker','create','-i','--name',name,'--runtime','runc','--label','task='+task,'--label','run=L039-R021','--network','none','--ipc','private','--cpus','4','--memory','8g','--pids-limit','512','--cap-drop','ALL','--security-opt','no-new-privileges']
 for k,val in {'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','SGLANG_USE_CPU_ENGINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','ASCEND_RT_VISIBLE_DEVICES':','.join(map(str,range(16))),'ASCEND_PROCESS_LOG_PATH':'/tmp/task-ascend/log','TORCH_EXTENSIONS_DIR':'/tmp/task-torch-extensions','SGLANG_CACHE_DIR':'/tmp/task-sglang-cache','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1'}.items():args+=['-e',k+'='+val]
 for d in devices:args+=['--device',d]
 # CPU tokenizer: no driver or device mounts.
 setup='exec python3 -u -'
 for filename in j['model_hashes']:args+=['--mount','type=bind,src='+str(model/filename)+',dst=/model/'+filename+',readonly']
 args+=['--entrypoint','/bin/bash',image,'-c',setup]
 v['container_id']=run(args).stdout.strip();created=True
 x=json.loads(run(['docker','inspect','--format','{{json .}}',name]).stdout);h=x['HostConfig']
 assert x['Config']['Labels']['task']==task and x['Config']['Labels']['run']=='L039-R021'
 assert h['NetworkMode']=='none' and not h['Privileged'] and h['Memory']==8*1024**3
 assert {d['PathOnHost'] for d in h['Devices']}==set(devices)
 assert all(m['RW'] is False for m in x['Mounts']) and len(x['Mounts'])==len(j['model_hashes'])
 v['isolation']={'network':h['NetworkMode'],'privileged':h['Privileged'],'devices':devices,'host_mounts_read_only':True,'model_metadata_mounted':True,'model_weights_mounted':False,'memory_limit_bytes':h['Memory']}
 p=run(['docker','start','-a','-i',name],'FIXTURES = '+repr(j['fixtures'])+'\n'+j['probe'],t=150,check=False)
 v['probe_exit_code']=p.returncode;v['probe_stdout']=p.stdout;v['probe_stderr']=p.stderr
 lines=[s[len('L039_RESULT='):] for s in p.stdout.splitlines() if s.startswith('L039_RESULT=')]
 assert len(lines)==1,'Missing/duplicate probe receipt'
 v['probe_result']=json.loads(lines[0]);v['device_compute_executed']=v['probe_result'].get('device_compute_executed',False)
 for filename,expected in j['model_hashes'].items():assert __import__('hashlib').sha256((model/filename).read_bytes()).hexdigest()==expected,filename
 v['model_metadata_unchanged']=True
 v['status']='PASS' if p.returncode==0 and v['probe_result']['status']=='PASS' else 'FAIL'
except Exception as e:v['status']='ERROR';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
finally:
 if created:
  x=json.loads(run(['docker','inspect','--format','{{json .Config.Labels}}',name]).stdout)
  assert x['task']==task and x['run']=='L039-R021'
  p=run(['docker','rm','-f',name]);v['cleanup']={'removed':p.returncode==0,'container_absent':run(['docker','container','inspect',name],check=False).returncode!=0}
print(json.dumps(v));sys.exit(0 if v['status']=='PASS' and v['cleanup']['container_absent'] else 1)
'''
JUMP=r'''
import json,subprocess,sys
j=json.load(sys.stdin)
p=subprocess.run(['kubectl','get','pod','-n','default','sglang-mech36-decode-46-20261019','-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr
x=json.loads(p.stdout);assert x['metadata']['uid']==j['reservation_uid'] and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['spec']['nodeName']=='a3-5u-atlas900rcka3-46' and x['status']['phase']=='Running'
assert x['spec']['preemptionPolicy']=='Never' and x['spec']['containers'][0]['resources']['requests']['huawei.com/Ascend910']=='16'
p=subprocess.run(['kubectl','exec','-i','-n','default','sglang-container-access-46-20261013','-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',j['host']],input=json.dumps(j),capture_output=True,text=True,timeout=210)
sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
'''
lease=json.loads((R/'runs/L039-R006-reservations/stdout.json').read_text());assert lease['status']=='PASS'
uid=next(x['uid'] for x in lease['reservations'] if x['node']=='a3-5u-atlas900rcka3-46')
probe=(R/'scripts/probe_L039_native_inputs.py').read_text();payload={'host':HOST,'probe':probe,'probe_sha256':hashlib.sha256(probe.encode()).hexdigest(),'reservation_uid':uid,'model_hashes':{n:f['sha256'] for n,f in json.loads((R/'runs/L039-R009-transfer-launch/remote-result.json').read_text())['source']['manifest']['files'].items() if f['kind']=='metadata'}}
manifest=json.loads((R/'fixtures/M001-v1/manifest.json').read_text());payload['fixtures']=[]
for entry in manifest['fixtures']:
 path=R/entry['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256'];payload['fixtures'].append(json.loads(path.read_text()))
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=240)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
v=json.loads(p.stdout) if p.stdout else {};result=v.get('probe_result')
if result:(O/'result.json').write_text(json.dumps(result,indent=2)+'\n')
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':v.get('status','NOT_EXECUTED'),'device_compute_executed':v.get('device_compute_executed',False),'model_loaded':False,'PD_MTP_executed':False,'reservation_uid':uid,'cleanup':v.get('cleanup'),
         'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in O.iterdir() if q.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
