#!/usr/bin/env python3
"""Observe exact launched containers/progress without reading credentials or restarting jobs."""
import datetime,hashlib,json,pathlib,re,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];tag=sys.argv[1];assert re.fullmatch(r'L039-R\d{3}',tag);O=R/'runs'/(tag+'-transfer-verification');O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import datetime,hashlib,json,pathlib,subprocess,sys,time
j=json.load(sys.stdin);root=pathlib.Path(j['root']);v={'node':j['node'],'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monotonic_ns':time.monotonic_ns(),'root_exists':root.is_dir()}
assert json.loads((root.parent/'owner.json').read_text())['task']=='sglang-pd-spec-36'
p=subprocess.run(['docker','inspect','--format','{{json .}}',j['container']],capture_output=True,text=True,timeout=25)
v['container_inspect_exit']=p.returncode
if p.returncode==0:
 x=json.loads(p.stdout);assert x['Id']==j['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']==j['run']
 v['container_id']=x['Id'];v['container']=j['container'];v['state']=x['State']
else:v['container_error']=p.stderr
if j['node']==37:
 p=root/'server-state.json'
 if p.is_file():
  s=json.loads(p.read_text());v['progress']={'status':s['status'],'bytes_sent':s['bytes_sent'],'active_requests':s['active_requests'],'files_with_source_SHA':sum(x['sha256'] is not None for x in s['files'].values()),'file_count':len(s['files']),'errors':s['errors']}
else:
 p=root/'progress.json'
 if p.is_file():
  s=json.loads(p.read_text());done=s.get('completed',{});v['progress']={k:val for k,val in s.items() if k not in ['completed','inflight']};v['progress']['completed_files']=len(done);v['progress']['inflight']=s.get('inflight',{});v['progress']['all_completed_hashes_match']=all(x['sha256']==x['source_sha256'] for x in done.values());v['progress']['completed_manifest_sha256']=hashlib.sha256(json.dumps(done,sort_keys=True).encode()).hexdigest()
  v['progress']['verified_completed_bytes']=sum(x['bytes'] for x in done.values())
  v['progress']['metadata_verified']={n:x for n,x in done.items() if not n.endswith('.safetensors') and n!='image-c5.tar'}
  v['progress']['image_transfer_verified']='image-c5.tar' in done and done['image-c5.tar']['sha256']==done['image-c5.tar']['source_sha256']
  if s['status']=='VERIFIED':
   expected=j['expected_files'];assert set(done)==set(expected),'Final filename set differs from initial contract'
   assert all(x['sha256']==x['source_sha256'] and x['bytes']==expected[n]['bytes'] for n,x in done.items()),'Final hash/size contract failed'
   v['progress']['completed_manifest']=done;v['progress']['final_contract_matches']=True
   v['progress']['model_shards']=sum(n.endswith('.safetensors') for n in done)
 p=root/'image-c5.tar'
 if p.is_file():v['image_archive_bytes']=p.stat().st_size
print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin)
def observe(item):
 number=int(item['node']);pod='sglang-tuning-access-37-20261013' if number==37 else f'sglang-container-access-{number}-20261013'
 p=subprocess.run(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(item),capture_output=True,text=True,timeout=50)
 return str(number),{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.returncode==0 else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=dict(pool.map(observe,j['targets']))
p=subprocess.run(['kubectl','get','nodes','a3-5u-atlas900rcka3-44','a3-5u-atlas900rcka3-46','-o','json'],capture_output=True,text=True,timeout=20)
node_conditions=[]
if p.returncode==0:
 for x in json.loads(p.stdout)['items']:node_conditions.append({'node':x['metadata']['name'],'uid':x['metadata']['uid'],'ready':next(c for c in x['status']['conditions'] if c['type']=='Ready'),'taints':x['spec'].get('taints',[])})
print(json.dumps({'read_only':True,'restarted_jobs':False,'nodes':results,'node_conditions':node_conditions,'node_query_exit':p.returncode}));sys.exit(0 if all(x['exit_code']==0 for x in results.values()) else 1)
'''
launch=json.loads((R/'runs/L039-R009-transfer-launch/remote-result.json').read_text());recovery=json.loads((R/'runs/L039-R028-resume-launch/remote-result.json').read_text());assert recovery['status']=='LAUNCHED_REVALIDATING_NOT_VERIFIED'
oldroot='/data/wio/sglang-mech36-20261018/L039-R009-transfer'
targets=[{'node':37,'container':launch['source']['container'],'container_id':launch['source']['container_id'],'run':'L039-R009','root':oldroot},{'node':46,'container':launch['clients']['46']['container'],'container_id':launch['clients']['46']['container_id'],'run':'L039-R009','root':oldroot},{'node':44,'container':recovery['container'],'container_id':recovery['container_id'],'run':'L039-R028','root':recovery['root']}]
for item in targets:
 if item['node']!=37:item['expected_files']={n:x for n,x in launch['source']['manifest']['files'].items() if n!='image-c5.tar'}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST,'targets':targets}),capture_output=True,text=True,timeout=90)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'restarted_jobs':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['stdout.json','stderr.log']}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode,'receipt':str(O/'receipt.json')}));sys.exit(p.returncode)
