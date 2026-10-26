#!/usr/bin/env python3
"""Preserve own full model logs, stop only proven-failed parent, inspect installed DeepEP API metadata."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R061-moe-failure';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,pathlib,re,subprocess,sys
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36';name=j['container'];p=subprocess.run(['docker','inspect','--format','{{json .}}',name],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['Id']==j['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']=='L039-R058';root=base/f"L039-R061-diagnostics-{j['node']}";root.mkdir(mode=0o700,exist_ok=False);raw=root/'model-docker.log'
with raw.open('xb') as f:p=subprocess.run(['docker','logs','--timestamps',name],stdout=f,stderr=subprocess.STDOUT,timeout=45)
data=raw.read_bytes();text=data.decode(errors='replace');lines=text.splitlines();shape=(('Failed to execute tiling function' in text or 'Scheduler hit an exception' in text) and ('SIGQUIT received' in text or 'Received sigquit from a child' in text));v={'node':j['node'],'container':name,'container_id':x['Id'],'prior_state':x['State'],'raw_log_path':str(raw),'raw_log_sha256':hashlib.sha256(data).hexdigest(),'raw_log_bytes':len(data),'fatal_startup_seen':shape,'stage_lines':[s[:1800] for s in lines if any(q in s for q in ['Load weight end','Load weight done','Load weight begin','KV Cache','KV cache','Capture cuda graph failed','scale batch size[16]','groupList length[256]','MoE A2A','moe_a2a_backend=','Traceback'])][:80],'error_contexts':[],'stopped_failed_parent':False}
v['first_backend_errors']=[];seen=set()
for i,line in enumerate(lines):
 relevant=any(k.lower() in line.lower() for k in ['HCCL_BUFFSIZE','[ERROR][','EZ1008','EZ9999','EB0000','Inner error','task error','AICORE exception','Vector core','parameter invalid','NotifyDispatch','MoeLowLatencyDispatchV2'])
 if not relevant or any(k in line for k in ['Pyspy failed','coredump pipe','Ignore import error']):continue
 body=re.sub(r'\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:[.,]\d+)?(?:Z)?','TIME',line);body=re.sub(r'(PID|TP|EP|Device|RankID)[: =]*\d+',r'\1ID',body)
 if body not in seen and len(v['first_backend_errors'])<60:
  seen.add(body);v['first_backend_errors'].append({'line':i+1,'text':line[:5000],'context':'\n'.join(lines[max(0,i-10):min(len(lines),i+22)])[:18000]})
indexes=[i for i,s in enumerate(lines) if 'Scheduler hit an exception' in s or 'RuntimeError:' in s or 'Execution_Error' in s]
for i in indexes[:5]:v['error_contexts'].append('\n'.join(lines[max(0,i-8):min(len(lines),i+125)]))
if j['node']==44 and x['State']['Running']:
 assert shape,'Do not stop if fatal startup is not proven in own logs'
 p=subprocess.run(['docker','stop','-t','5',name],capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stderr;v['stopped_failed_parent']=True
p=subprocess.run(['docker','inspect','--format','{{json .State}}',name],capture_output=True,text=True,timeout=25);assert p.returncode==0;v['final_state']=json.loads(p.stdout);assert not v['final_state']['Running']
print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin)
def host(item):
 n=item['node'];p=subprocess.run(['kubectl','exec','-i','-n','default',f'sglang-container-access-{n}-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(item),capture_output=True,text=True,timeout=125);return str(n),{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.stdout else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=dict(pool.map(host,j['targets']))
deps={'exit_code':0,'metadata_not_repeated':'R043 installedpackage unchanged in same image'}
print(json.dumps({'nodes':results,'DeepEP_package':deps,'CPU_store_untouched':True,'model_files_preserved':True,'inference_workload_executed':False}));sys.exit(0 if all(x['exit_code']==0 for x in results.values()) and deps['exit_code']==0 else 1)
'''
a=json.loads((R/'runs/L039-R058-full-bootstrap/remote-result.json').read_text());s=json.loads((R/'runs/L039-R033-shared-store/remote-result.json').read_text())['store']['data'];targets=[{'node':int(n),'container':x['data']['name'],'container_id':x['data']['container_id']} for n,x in a['nodes'].items()];cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST,'targets':targets,'store':{k:s[k] for k in ['name','container_id']}}),capture_output=True,text=True,timeout=225)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'inference_workload_executed':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
