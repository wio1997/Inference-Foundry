#!/usr/bin/env python3
"""Read exact model storage identities and image/disk sizes; never read weight payloads."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R008-storage-capacity';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import collections,hashlib,json,os,pathlib,socket,struct,subprocess,sys
number=int(sys.argv[1]);image='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'
def run(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=30);return {'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def mount(path):
 x=run(['findmnt','-J','-T',str(path),'-o','SOURCE,TARGET,FSTYPE']);return json.loads(x['stdout']) if x['exit_code']==0 else x
v={'hostname':socket.gethostname(),'node':number,'read_only':True,'disks':{},'models':[]}
for path in ['/var/lib/docker','/data/wio']:
 p=pathlib.Path(path);stat=os.statvfs(p);v['disks'][path]={'total_bytes':stat.f_blocks*stat.f_frsize,'available_bytes':stat.f_bavail*stat.f_frsize,'realpath':str(p.resolve()),'mount':mount(p)}
p=run(['docker','image','inspect','--format','{{json .}}',image]);v['image']={'exit_code':p['exit_code']}
if p['exit_code']==0:
 x=json.loads(p['stdout']);v['image'].update({k:x.get(k) for k in ['Id','Size','Architecture','RepoDigests']})
else:v['image']['stderr']=p['stderr']
roots=['/data/wio/GLM53/GLM-5.3-w8a8c8'] if number==37 else ['/data/wio/glm53-w8a8-local']
for name in roots:
 root=pathlib.Path(name);assert root.is_dir(),name
 item={'path':name,'realpath':str(root.resolve()),'mount':mount(root),'metadata':{},'weight_payload_read':False}
 for name in ['config.json','quant_model_description.json','quant_model_weights.safetensors.index.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']:
  p=root/name
  if not p.is_file():item['metadata'][name]={'missing':True};continue
  b=p.read_bytes();entry={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
  if name=='quant_model_description.json':
   q=json.loads(b);assert isinstance(q,dict)
   entry['top_level_count']=len(q);entry['first_keys']=list(q)[:8]
   entry['scalar_value_histogram']=dict(collections.Counter(str(v) if isinstance(v,(str,int,float,bool)) and len(str(v))<80 else type(v).__name__ for v in q.values()))
   entry['cache_quant_entries']=[{'key':k,'value':v if isinstance(v,(str,int,float,bool)) else type(v).__name__} for k,v in q.items() if any(s in k.lower() for s in ['cache','faquant','fa_quant','kvquant','kv_quant'])][:32]
  elif name.endswith('safetensors.index.json'):
   q=json.loads(b);wm=q.get('weight_map',{});names=sorted(set(wm.values()));entry['index_metadata']=q.get('metadata');entry['tensor_count']=len(wm);entry['shard_count']=len(names);entry['shards']=[]
   for shard in names:
    p=(root/shard).resolve();assert p.is_relative_to(root.resolve());assert p.is_file(),shard
    entry['shards'].append({'name':shard,'bytes':p.stat().st_size})
   entry['shard_total_bytes']=sum(x['bytes'] for x in entry['shards'])
   # Only the format header, not tensor data, of one indexed shard.
   first=root/names[0]
   with first.open('rb') as f:
    length=struct.unpack('<Q',f.read(8))[0];assert length<=32*1024**2
    header=f.read(length);assert len(header)==length
   h=json.loads(header);entry['first_shard_header']={'file':names[0],'bytes':length,'sha256':hashlib.sha256(header).hexdigest(),'tensors':{k:{a:b for a,b in val.items() if a in ['dtype','shape']} for k,val in list(h.items())[:8] if k!='__metadata__'}}
  item['metadata'][name]=entry
 v['models'].append(item)
print(json.dumps(v))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys
j=json.load(sys.stdin)
for name,uid,node in [('sglang-mech36-prefill-44-20261019','41367624-b523-4682-8be5-778f104ca373',44),('sglang-mech36-decode-46-20261019','bae63432-b774-44d1-98c4-e9b81c09a377',46)]:
 p=subprocess.run(['kubectl','get','pod','-n','default',name,'-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr
 x=json.loads(p.stdout);assert x['metadata']['uid']==uid and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['status']['phase']=='Running' and x['spec']['nodeName']==f'a3-5u-atlas900rcka3-{node}'
def observe(number):
 pod='sglang-tuning-access-37-20261013' if number==37 else f'sglang-container-access-{number}-20261013'
 p=subprocess.run(['kubectl','exec','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-c',j['host'],str(number)],capture_output=True,text=True,timeout=150)
 return str(number),{'exit_code':p.returncode,'data':json.loads(p.stdout) if p.returncode==0 else None,'stderr':p.stderr}
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=dict(pool.map(observe,[37,44,46]))
print(json.dumps({'read_only':True,'reservations_revalidated':True,'nodes':results}));sys.exit(0 if all(x['exit_code']==0 for x in results.values()) else 1)
'''
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps({'host':HOST}),capture_output=True,text=True,timeout=210)
(O/'stdout.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'read_only':True,'weight_payload_read':False,'device_compute_executed':False,
         'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['stdout.json','stderr.log']}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'exit_code':p.returncode,'receipt':str(O/'receipt.json')}));sys.exit(p.returncode)
