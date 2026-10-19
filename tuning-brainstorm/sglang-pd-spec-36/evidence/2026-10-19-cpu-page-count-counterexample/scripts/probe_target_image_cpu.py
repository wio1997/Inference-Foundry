#!/usr/bin/env python3
"""Run a frozen CPU semantic probe in a new, isolated, task-owned image container."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/L007-R004-target-image-cpu';OUT.mkdir(parents=True,exist_ok=False)
probe=ROOT/'scripts/probe_metadata_support.py'
expected=json.loads((ROOT/'runs/L007-R003-cpu-support/receipt.json').read_text())['source_hashes']
payload={'probe':probe.read_text(),'probe_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),'sources':{p:(ROOT/'sources/installed/python/sglang/srt'/p).read_text() for p in expected},'hashes':expected}
BOOTSTRAP=r'''
import hashlib,json,pathlib,runpy,sys,tempfile
j=json.load(sys.stdin)
root=pathlib.Path(tempfile.mkdtemp(prefix='if-l007-cpu-',dir='/tmp'))
script=root/'scripts/probe_metadata_support.py';script.parent.mkdir();script.write_text(j['probe'])
assert hashlib.sha256(script.read_bytes()).hexdigest()==j['probe_sha256']
source=root/'sources'
for p,b in j['sources'].items():
 assert '..' not in pathlib.PurePosixPath(p).parts and not p.startswith('/')
 q=source/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(b)
 assert hashlib.sha256(q.read_bytes()).hexdigest()==j['hashes'][p]
sys.argv=[str(script),'--source-root',str(source),'--stdout','--expected-hashes',json.dumps(j['hashes'])]
runpy.run_path(str(script),run_name='__main__')
'''
HOST=r'''
import datetime,hashlib,json,os,pathlib,subprocess,sys
j=json.load(sys.stdin);image='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
service='sglang-learning37-glm53-20261017';expected_service='f42cd0a62f5f0778c0e6576ff8177f3d8c71f0e77c03719e22aa9af7b2e7a9c8'
name='if-sglang36-l007-cpu-20261019-r004';labels={'inference-foundry.task':'sglang-pd-spec-36','inference-foundry.run':'L007-R004','inference-foundry.cpu-only':'true'}
rec={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'New CPU-only container; no model requests and no accelerator mounts. Source is hashed installed snapshot, not asserted to be image bundled source.','events':[]};cid=None
def run(args,timeout=30,stdin=None,check=True):
 p=subprocess.run(args,input=stdin,capture_output=True,text=True,timeout=timeout)
 if check and p.returncode:raise RuntimeError('Command failed: '+repr(args[:3])+' rc='+str(p.returncode)+' stderr='+p.stderr[:2000])
 return p
def inspect(ref):return json.loads(run(['docker','inspect',ref]).stdout)[0]
def own(c):
 assert c['Id']==cid and c['Image']==image and all(c['Config']['Labels'].get(k)==v for k,v in labels.items())
def service_state():
 c=inspect(service);assert c['Id']==expected_service
 return {'id':c['Id'],'image':c['Image'],'running':c['State']['Running'],'started_at':c['State']['StartedAt'],'restart_count':c['RestartCount']}
try:
 rec['service_before']=service_state();assert rec['service_before']['running']
 imageinfo=json.loads(run(['docker','image','inspect',image]).stdout)[0];assert imageinfo['Id']==image
 runtimes=json.loads(run(['docker','info','--format','{{json .Runtimes}}']).stdout);assert 'runc' in runtimes
 mem={x.split(':')[0]:int(x.split()[1]) for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')}
 rec['preflight']={'cpu_count':os.cpu_count(),'mem_available_kib':mem['MemAvailable'],'runtime':'runc','image_id':imageinfo['Id']};assert os.cpu_count()>=8 and mem['MemAvailable']>=8*1024*1024
 absent=run(['docker','container','inspect',name],check=False);assert absent.returncode!=0,'Name already exists; do not reuse'
 args=['docker','create','-i','--name',name,'--runtime','runc','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges:true','--pids-limit','64','--memory','2g','--memory-swap','2g','--cpus','1','--tmpfs','/tmp:rw,nosuid,nodev,noexec,size=64m,mode=1777']
 for k,v in labels.items():args+=['--label',k+'='+v]
 for k,v in {'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}.items():args+=['--env',k+'='+v]
 args+=['--entrypoint','python3',image,'-c',j['bootstrap']]
 p=run(args);cid=p.stdout.strip();assert len(cid)==64
 c=inspect(cid);own(c);h=c['HostConfig']
 assert not h.get('Privileged') and h['Runtime']=='runc' and h['NetworkMode']=='none' and h['ReadonlyRootfs']
 assert not h.get('Devices') and not h.get('DeviceRequests') and not h.get('Binds') and not h.get('CapAdd')
 assert h['Memory']==2*1024**3 and h['NanoCpus']==10**9 and h['PidsLimit']==64 and not h.get('PidMode')
 assert 'ALL' in h['CapDrop'] and any('no-new-privileges' in x for x in h['SecurityOpt'])
 assert set(h['Tmpfs'])=={'/tmp'} and not c.get('Mounts')
 rec['created']={'id':cid,'image':c['Image'],'labels':labels,'host_config':{k:h.get(k) for k in ['Runtime','NetworkMode','ReadonlyRootfs','Privileged','Devices','DeviceRequests','Binds','CapAdd','CapDrop','SecurityOpt','Memory','MemorySwap','NanoCpus','PidsLimit','PidMode','Tmpfs']},'create_stderr':p.stderr}
 p=run(['docker','start','-a','-i',cid],timeout=120,stdin=json.dumps(j['payload']),check=False)
 rec['probe_stdout']=p.stdout;rec['probe_stderr']=p.stderr;rec['start_exit_code']=p.returncode
 c=inspect(cid);own(c);rec['container_state']={k:c['State'].get(k) for k in ['Status','Running','ExitCode','OOMKilled','Error','StartedAt','FinishedAt']}
 assert not c['State']['Running'] and p.returncode==0 and c['State']['ExitCode']==0 and not c['State']['OOMKilled']
 result=json.loads(p.stdout);assert result['correctness']=='PASS' and result['source_hashes']==j['payload']['hashes'] and result['script_sha256']==j['payload']['probe_sha256']
 rec['correctness']='PASS'
except Exception as exc:
 rec['correctness']='ERROR';rec['error']=repr(exc)
finally:
 if cid:
  try:
   c=inspect(cid);own(c)
   if c['State']['Running']:run(['docker','stop','--time','5',cid],timeout=15);rec['events'].append('stopped own container after failure/timeout')
   run(['docker','rm',cid]);rec['cleanup']={'removed_owned_container':cid}
  except Exception as exc:rec['cleanup']={'error':repr(exc)};rec['correctness']='ERROR'
 try:
  rec['service_after']=service_state();assert rec['service_before']==rec['service_after'];rec['service_identity_and_start_unchanged']=True
 except Exception as exc:rec['service_check_error']=repr(exc);rec['correctness']='ERROR'
 rec['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();print(json.dumps(rec))
sys.exit(0 if rec['correctness']=='PASS' else 1)
'''
JUMP=r'''
import json,subprocess,sys
j=json.load(sys.stdin)
p=subprocess.run(['kubectl','exec','-i','-n','default','sglang-tuning-access-37-20261013','-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',j['host']],input=json.dumps({'bootstrap':j['bootstrap'],'payload':j['payload']}),capture_output=True,text=True,timeout=200)
sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
'''
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
p=subprocess.run(cmd,input=json.dumps({'host':HOST,'bootstrap':BOOTSTRAP,'payload':payload}),capture_output=True,text=True,timeout=230)
(OUT/'remote-result.json').write_text(p.stdout);(OUT/'stderr.log').write_text(p.stderr)
if p.returncode==0:
 remote=json.loads(p.stdout);result=json.loads(remote['probe_stdout']);(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
receipt={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'input_hashes':{'controller':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'probe':payload['probe_sha256'],'source_hashes':expected},'artifacts':{q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in OUT.iterdir() if q.is_file()}}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode,'receipt':str(OUT/'receipt.json')}));sys.exit(p.returncode)
