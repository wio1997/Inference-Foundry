#!/usr/bin/env python3
"""New owned recovery job; preserve original exited container/progress and credential isolation."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R024-resume-launch';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,os,pathlib,ssl,stat,subprocess,sys,urllib.request
j=json.load(sys.stdin);task='sglang-pd-spec-36';base=pathlib.Path('/data/wio/sglang-mech36-20261018');old=base/'L039-R009-transfer';root=base/'L039-R024-transfer-resume';name='if-mech36-transfer-dst-l039r024-44'
# The known CPU client image identity comes from the original launch contract.
image=j['client_image']
def run(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr;return p.stdout
assert json.loads((base/'owner.json').read_text())['task']==task
old_info=json.loads(run(['docker','inspect','--format','{{json .}}',j['old_container']]))
assert old_info['Id']==j['old_container_id'] and old_info['Config']['Labels']['task']==task and old_info['Config']['Labels']['run']=='L039-R009' and not old_info['State']['Running'] and old_info['State']['ExitCode']==255
assert subprocess.run(['docker','container','inspect',name],capture_output=True).returncode!=0
assert not root.exists(),'Existing recovery state must be inspected, never replayed'
assert json.loads(run(['docker','image','inspect','--format','{{json .Id}}',image]))==image
assert hashlib.sha256(j['client'].encode()).hexdigest()==j['client_sha256']
contract=json.loads((old/'contract.json').read_text());assert contract['task']==task and contract['node']==44
contract['files']={n:x for n,x in contract['files'].items() if n!='image-c5.tar'}
assert contract['files']=={n:x for n,x in j['expected_files'].items() if n!='image-c5.tar'} and len(contract['files'])==193
state_bytes=(old/'progress.json').read_bytes();previous=json.loads(state_bytes);assert previous['node']==44 and previous['status']=='COPYING'
completed={n:x for n,x in previous['completed'].items() if n!='image-c5.tar'}
for n,x in completed.items():
 assert pathlib.Path(n).name==n and n in contract['files'] and x['bytes']==contract['files'][n]['bytes'] and x['sha256']==x['source_sha256']
 p=old/'model'/n;assert stat.S_ISREG(p.lstat().st_mode) and p.stat().st_size==x['bytes']
# Machine-only use of this task's existing ephemeral mTLS client files; no material is emitted.
secrets=old/'secrets';assert hashlib.sha256((secrets/'ca.crt').read_bytes()).hexdigest()==contract['server_CA_sha256']
context=ssl.create_default_context(cafile=str(secrets/'ca.crt'));context.load_cert_chain(secrets/'client.crt',secrets/'client.key');opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=context))
with opener.open('https://172.16.12.37:39039/manifest',timeout=15) as response:remote=json.load(response)
assert all(remote['files'][n]['bytes']==x['bytes'] and remote['files'][n]['stat']==x['stat'] for n,x in contract['files'].items())
expected=sum(x['bytes'] for x in contract['files'].values());reuse=sum(x['bytes'] for x in completed.values());fs=os.statvfs(base);assert fs.f_bavail*fs.f_frsize>expected-reuse+100*1024**3
root.mkdir(mode=0o700);(root/'model').mkdir();(root/'contract.json').write_text(json.dumps(contract));(root/'client.py').write_text(j['client'])
recovery={'parent_run':'L039-R009','parent_container_id':old_info['Id'],'parent_progress_sha256':hashlib.sha256(state_bytes).hexdigest(),'completed':completed,'reuse_policy':'Hardlink only completed source/client-hash-matched files; background job rereads every reused file SHA before trusting it. Old files and partials never overwritten.'};(root/'recovery-contract.json').write_text(json.dumps(recovery))
for n in completed:os.link(old/'model'/n,root/'model'/n)
args=['docker','create','--name',name,'--runtime','runc','--label','task='+task,'--label','run=L039-R024','--network','host','--cpus','2','--memory','2g','--pids-limit','64','--cap-drop','ALL','--security-opt','no-new-privileges','-e','PYTHONDONTWRITEBYTECODE=1','--mount','type=bind,src='+str(root)+',dst=/work','--mount','type=bind,src='+str(secrets)+',dst=/work/secrets,readonly','--entrypoint','python3',image,'-u','/work/client.py']
cid=run(args).strip();run(['docker','start',name]);inspect=json.loads(run(['docker','inspect','--format','{{json .}}',name]));assert inspect['State']['Running'] and not inspect['HostConfig']['Privileged'] and not inspect['HostConfig']['Devices']
assert next(m for m in inspect['Mounts'] if m['Destination']=='/work/secrets')['RW'] is False
assert (old/'progress.json').read_bytes()==state_bytes,'Original failed progress changed'
print(json.dumps({'status':'LAUNCHED_REVALIDATING_NOT_VERIFIED','node':44,'container':name,'container_id':cid,'state':inspect['State'],'root':str(root),'model_path':str(root/'model'),'expected_files':193,'expected_bytes':expected,'reused_files_pending_revalidation':len(completed),'reused_bytes_pending_revalidation':reuse,'new_bytes_expected':expected-reuse,'parent_container':j['old_container'],'parent_container_id':j['old_container_id'],'parent_exit_code':255,'parent_progress_sha256':hashlib.sha256(state_bytes).hexdigest(),'parent_progress_unchanged':True,'private_material_emitted':False,'NPU_devices':False,'model_loaded':False,'client_image':image,'client_sha256':j['client_sha256']}))
'''
JUMP=r'''
import json,subprocess,sys
j=json.load(sys.stdin)
def get(a):
 p=subprocess.run(['kubectl']+a+['-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;return json.loads(p.stdout)
x=get(['get','pod','-n','default','sglang-mech36-prefill-44-20261019']);assert x['metadata']['uid']=='41367624-b523-4682-8be5-778f104ca373' and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['spec']['nodeName']=='a3-5u-atlas900rcka3-44' and x['status']['phase']=='Running'
x=get(['get','node','a3-5u-atlas900rcka3-44']);assert next(c for c in x['status']['conditions'] if c['type']=='Ready')['status']=='True'
p=subprocess.run(['kubectl','exec','-i','-n','default','sglang-container-access-44-20261013','-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(j),capture_output=True,text=True,timeout=100)
sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
'''
launch=json.loads((R/'runs/L039-R009-transfer-launch/remote-result.json').read_text());client=(R/'scripts/L039_transfer_resume_client.py').read_text()
payload={'host':HOST,'old_container':launch['clients']['44']['container'],'old_container_id':launch['clients']['44']['container_id'],'client':client,'client_sha256':hashlib.sha256(client.encode()).hexdigest(),'expected_files':launch['source']['manifest']['files'],'client_image':'sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=145)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
v=json.loads(p.stdout) if p.stdout else {};receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':v.get('status','UNKNOWN'),'credentials_in_result':False,'model_files_on_Mac':False,'transfer_complete':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['remote-result.json','stderr.log']}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
