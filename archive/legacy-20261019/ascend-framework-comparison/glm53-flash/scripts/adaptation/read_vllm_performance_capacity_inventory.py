# SPDX-License-Identifier: Apache-2.0
"""Read-only Flash A-line inventory. No exec into containers, model calls,
resource mutations, provider/env reads or changes to another project's work.
"""
import argparse,concurrent.futures,datetime,json,pathlib,shlex,subprocess
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
ssh=['ssh','-S','/Users/wio/work/sglang-glm53-pd/ssh-control','-p','60011','-o','BatchMode=yes','-o','ConnectTimeout=15','root@61.241.77.34'];pods={14:'zhixing-dockerctl-25xz2',15:'zhixing-dockerctl-42pbf'}
code='''import pathlib,json,subprocess,socket,hashlib,time
node=NODE;expected={14:'A6-15U-Atlas900RCKA3-14',15:'A6-23U-Atlas900RCKA3-15'}[node];assert socket.gethostname()==expected
image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'
def run(args,t=20):
 q=subprocess.run(args,capture_output=True,text=True,timeout=t);return {'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
out={'project_id':'glm53-flash','workstream':'A_vllm_performance','node':node,'hostname':expected,'monotonic_ns':time.monotonic_ns(),'resource_mutations':0,'model_requests':0}
out['npu']=run(['npu-smi','info']);out['idle_cards']=out['npu']['stdout'].count('No running processes found') if out['npu']['rc']==0 else None
out['image']=run(['docker','image','inspect',image,'--format','{{.Id}}']);out['pinned_image_present']=out['image']['rc']==0 and out['image']['stdout'].strip()==image
q=run(['docker','ps','-a','--filter','label=project_id=glm53-flash','--format','{{.ID}}']);assert q['rc']==0;ids=q['stdout'].split();assert len(ids)<128;out['own_containers']=[]
for cid in ids:
 q=run(['docker','inspect',cid]);assert q['rc']==0;v=json.loads(q['stdout'])[0];labels=v['Config']['Labels'];assert labels['project_id']=='glm53-flash'
 out['own_containers'].append({'name':v['Name'].lstrip('/'),'image':v['Image'],'run_id':labels.get('run_id'),'running':v['State']['Running'],'exit_code':v['State']['ExitCode']})
model=pathlib.Path('/data/wio/models/GLM-5.3-Flash-w8a8');out['model_metadata']={}
for name in ['config.json','quant_model_description.json','tokenizer_config.json']:
 p=model/name;out['model_metadata'][name]={'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None,'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None}
if model.is_dir():
 files=[p for p in model.rglob('*') if p.is_file()];out['model_recursive_files']=len(files);out['model_recursive_bytes']=sum(p.stat().st_size for p in files);out['nested_assets']=[{'path':str(p.relative_to(model)),'bytes':p.stat().st_size} for p in files if p.parent!=model]
out['memory']={}
for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
 if line.split(':',1)[0] in ('MemTotal','MemAvailable','HugePages_Total','HugePages_Free','Hugepagesize','Hugetlb'):out['memory'][line.split(':',1)[0]]=line.split(':',1)[1].strip()
out['required_host_paths']={p:pathlib.Path(p).exists() for p in ['/etc/hccn.conf','/dev/shm','/dev/davinci0','/dev/davinci15']};out['listening_ports']=run(['ss','-H','-ltn']);out['loadavg']=pathlib.Path('/proc/loadavg').read_text().strip();print(json.dumps(out))
'''
def one(node):
 q=subprocess.run(ssh+[shlex.join(['kubectl','get','pod','-n','default',pods[node],'-o','jsonpath={.status.hostIP}'])],capture_output=True,text=True,timeout=25);assert q.returncode==0 and q.stdout.strip()=='172.16.12.'+str(node)
 cmd=['kubectl','exec','-i','-n','default',pods[node],'-c','hostctl','--','nsenter','-t1','-m','--','python3','-'];q=subprocess.run(ssh+[shlex.join(cmd)],input=code.replace('NODE',str(node)),capture_output=True,text=True,timeout=60);assert q.returncode==0,q.stderr[-1500:];v=json.loads(q.stdout);(a.output/f'node{node}.json').write_text(json.dumps(v,indent=2)+'\n');return node,v
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:rows=dict(ex.map(one,pods))
out={'project_id':'glm53-flash','workstream':'A_vllm_performance','as_of':datetime.datetime.now(datetime.timezone.utc).isoformat(),'resource_mutations':0,'actual_model_requests':0,'nodes':{str(n):{'idle_cards':v['idle_cards'],'pinned_image_present':v['pinned_image_present'],'own_running_containers':[c for c in v['own_containers'] if c['running']],'model_metadata':v['model_metadata'],'model_recursive_files':v.get('model_recursive_files'),'model_recursive_bytes':v.get('model_recursive_bytes'),'nested_assets':v.get('nested_assets'),'required_host_paths':v['required_host_paths'],'loadavg':v['loadavg'],'host_memory':v['memory']} for n,v in rows.items()},'formal_requests':0,'baseline':None,'resource_window_available':all(v['idle_cards']==8 and v['pinned_image_present'] and not any(c['running'] for c in v['own_containers']) for v in rows.values())};(a.output/'summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
