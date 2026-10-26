#!/usr/bin/env python3
"""Read-only graph source closure; no HTTP or model requests."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L015-R002-kernel-source';O.mkdir(parents=True,exist_ok=False);request=json.loads((R/'specs/L015-kernel-request.json').read_text())
INNER=r'''

import hashlib,importlib.metadata,json,pathlib,subprocess,sys
j=json.loads(sys.argv[1]);sgl=pathlib.Path('/sgl-workspace/sglang');head=subprocess.check_output(['git','-C',str(sgl),'rev-parse','HEAD'],text=True).strip();assert head==j['expected_commit']
roots=[]
for entry in sys.path:
 p=pathlib.Path(entry or '.').resolve()
 if p not in roots:roots.append(p)
v={'source_commit':head,'files':{},'read_only':True,'generation_requests':0,'selection_basis':'sys.path file scan without importing package; candidate file identity only, not actual loaded native operator','distribution_versions':{}}
for dist in importlib.metadata.distributions():
 name=dist.metadata.get('Name','')
 if name.lower().replace('-','_')=='sgl_kernel_npu':v['distribution_versions'][name]=dist.version
for rel in j['paths']:
 assert rel.startswith('sgl_kernel_npu/') and '..' not in pathlib.PurePosixPath(rel).parts and rel.endswith('.py')
 candidates=[]
 for root in roots:
  p=root/rel
  if p.is_file() and p.resolve() not in candidates:candidates.append(p.resolve())
 if not candidates:continue
 p=candidates[0];b=p.read_bytes();git={};q=subprocess.run(['git','-C',str(p.parent),'rev-parse','--show-toplevel'],capture_output=True,text=True,timeout=15)
 if q.returncode==0:
  repo=pathlib.Path(q.stdout.strip());revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();relative=str(p.relative_to(repo));show=subprocess.run(['git','-C',str(repo),'show',revision+':'+relative],capture_output=True,timeout=15);git={'root':str(repo),'commit':revision,'relative_path':relative,'matches_git_head':show.returncode==0 and show.stdout==b}
 v['files'][rel]={'sha256':hashlib.sha256(b).hexdigest(),'path_candidate':str(p),'all_path_candidates':[str(x) for x in candidates],'git':git,'content':b.decode()}
v['probe_status']='SOURCE_FOUND' if 'sgl_kernel_npu/sample/verify_tree_greedy.py' in v['files'] else 'UNAVAILABLE_IN_PYTHON_PATH_SCAN';v['missing_files']=sorted(set(j['paths'])-v['files'].keys());print(json.dumps(v))

'''
NODE=r'''
import json,subprocess,sys,socket
j=json.load(sys.stdin);name='sglang-learning37-glm53-20261017'
cid=subprocess.check_output(['docker','inspect','--format','{{.Id}}',name],text=True,timeout=15).strip();assert cid==j['expected_container_id'],'Observed container identity drift'
p=subprocess.run(['docker','exec',name,'python3','-c',j['inner'],json.dumps(j)],capture_output=True,text=True,timeout=90)
assert p.returncode==0,p.stderr
v=json.loads(p.stdout);v['observed_hostname']=socket.gethostname();v['container_id']=cid;print(json.dumps(v))
'''
JUMP=r'''
import json,subprocess,sys
j=json.load(sys.stdin);pod='sglang-tuning-access-37-20261013'
p=subprocess.run(['kubectl','get','pod','-n','default',pod,'-o','json'],capture_output=True,text=True,timeout=20);assert p.returncode==0,p.stderr
v=json.loads(p.stdout);assert v['status']['hostIP']=='172.16.12.37' and v['status']['phase']=='Running'
p=subprocess.run(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',j['node']],input=json.dumps(j),capture_output=True,text=True,timeout=110)
sys.stdout.write(p.stdout);sys.stderr.write(p.stderr);sys.exit(p.returncode)
'''
cmd=['ssh','-S','/tmp/glm53-active-20261018.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
request.update({'inner':INNER,'node':NODE});started=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(request),capture_output=True,text=True,timeout=130)
(O/'source-and-config.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
if p.returncode==0:
 for n,f in json.loads(p.stdout)['files'].items():
  target=R/'sources/installed-kernel-npu'/n;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==f['sha256'],'Snapshot drift: '+n
  else:target.write_text(f['content'])
v={'started_utc':started,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':cmd,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'request_sha256':hashlib.sha256((R/'specs/L015-kernel-request.json').read_bytes()).hexdigest(),'exit_code':p.returncode,'read_only':True,'generation_requests':0,'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['source-and-config.json','stderr.log']}};(O/'receipt.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode}));sys.exit(p.returncode)
