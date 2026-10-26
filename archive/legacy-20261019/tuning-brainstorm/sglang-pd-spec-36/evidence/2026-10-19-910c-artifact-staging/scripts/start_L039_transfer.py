#!/usr/bin/env python3
"""Start owned, authenticated private-network transfers; keep credentials out of results/prompts."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R009-transfer-launch';O.mkdir(parents=True,exist_ok=False)
SOURCE=r'''
import datetime,hashlib,json,os,pathlib,socket,subprocess,sys,time
j=json.load(sys.stdin);task='sglang-pd-spec-36';base=pathlib.Path('/data/wio/sglang-mech36-20261018');root=base/'L039-R009-transfer';model=pathlib.Path('/data/wio/GLM53/GLM-5.3-w8a8c8');image='sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263';name='if-mech36-transfer-src-l039r009-37'
def run(a,t=60):
 p=subprocess.run(a,capture_output=True,text=True,timeout=t);assert p.returncode==0,p.stderr;return p.stdout
def claim():
 if base.exists():assert json.loads((base/'owner.json').read_text())['task']==task
 else:base.mkdir(mode=0o700);(base/'owner.json').write_text(json.dumps({'task':task}))
 assert not root.exists(),'Transfer already prepared; inspect existing state instead of replay'
 root.mkdir(mode=0o700)
assert hashlib.sha256(j['server'].encode()).hexdigest()==j['server_sha256']
assert hashlib.sha256((model/'quant_model_description.json').read_bytes()).hexdigest()=='ddff020caf4158859060a932e90c52339f6cf3274694ba3bb9f7bfc153ae2378'
index_bytes=(model/'quant_model_weights.safetensors.index.json').read_bytes();assert hashlib.sha256(index_bytes).hexdigest()=='db3d87f661c373c7d8a906622f8c0f7ab8bf1e4a0e791696e216724286e350b2'
for filename,expected in j['expected_metadata'].items():assert hashlib.sha256((model/filename).read_bytes()).hexdigest()==expected,filename
idx=json.loads(index_bytes);weights=sorted(set(idx['weight_map'].values()));assert len(weights)==186
size=sum((model/n).stat().st_size for n in weights);fs=os.statvfs(base.parent);assert fs.f_bavail*fs.f_frsize>50*1024**3
x=json.loads(run(['docker','image','inspect','--format','{{json .}}',image]));assert x['Id']==image
with socket.socket() as s:s.bind(('172.16.12.37',39039))
claim();secrets=root/'secrets';secrets.mkdir(mode=0o700);(secrets/'newcerts').mkdir();(secrets/'index.txt').write_text('');(secrets/'serial').write_text('01\n');(secrets/'index.txt.attr').write_text('unique_subject = no\n')
config="""[ca]\ndefault_ca=task_ca\n[task_ca]\ndir=SECRETS\ndatabase=$dir/index.txt\nnew_certs_dir=$dir/newcerts\ncertificate=$dir/ca.crt\nprivate_key=$dir/ca.key\nserial=$dir/serial\ndefault_md=sha256\ndefault_days=3\npolicy=task_policy\nunique_subject=no\n[task_policy]\ncommonName=supplied\n[ca_ext]\nbasicConstraints=critical,CA:TRUE,pathlen:0\nkeyUsage=critical,keyCertSign,cRLSign\n[server_ext]\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=IP:172.16.12.37\n[client_ext]\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=clientAuth\n""".replace('SECRETS',str(secrets));(secrets/'openssl.cnf').write_text(config)
def req(prefix,cn,initial_ca=False):
 args=['openssl','req','-newkey','rsa:2048','-nodes','-keyout',str(secrets/(prefix+'.key')),'-out',str(secrets/(prefix+('.crt' if initial_ca else '.csr'))),'-subj','/CN='+cn]
 if initial_ca:args+=['-x509','-days','3']
 run(args);os.chmod(secrets/(prefix+'.key'),0o600)
req('ca','mech36-transfer-ca',True)
run(['openssl','req','-new','-key',str(secrets/'ca.key'),'-out',str(secrets/'ca.csr'),'-subj','/CN=mech36-transfer-ca'])
now=datetime.datetime.now(datetime.timezone.utc);start=(now-datetime.timedelta(days=1)).strftime('%y%m%d%H%M%SZ');end=(now+datetime.timedelta(days=3)).strftime('%y%m%d%H%M%SZ')
def sign(prefix,extensions,selfsign=False):
 args=['openssl','ca','-batch','-notext','-config',str(secrets/'openssl.cnf'),'-in',str(secrets/(prefix+'.csr')),'-out',str(secrets/(prefix+'.signed.crt')),'-startdate',start,'-enddate',end,'-extensions',extensions]
 if selfsign:args+=['-selfsign']
 run(args);os.replace(secrets/(prefix+'.signed.crt'),secrets/(prefix+'.crt'))
sign('ca','ca_ext',True)
req('server','mech36-transfer-source37');sign('server','server_ext')
for n in [44,46]:req('client'+str(n),'mech36-node'+str(n));sign('client'+str(n),'client_ext')
# No persistent cluster auth is changed. These keys belong only to this isolated transfer.
(root/'server.py').write_text(j['server']);assert hashlib.sha256((root/'server.py').read_bytes()).hexdigest()==j['server_sha256']
run(['docker','save','--output',str(root/'image-c5.tar'),image],t=300)
files={};names=['config.json','quant_model_description.json','quant_model_weights.safetensors.index.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','generation_config.json','special_tokens_map.json']
for n in names+weights:
 p=(model/n).resolve();assert p.is_relative_to(model.resolve())
 if not p.is_file():assert n in ['generation_config.json','special_tokens_map.json'];continue
 s=p.stat();files[n]={'bytes':s.st_size,'stat':[s.st_size,s.st_mtime_ns,s.st_ino],'sha256':hashlib.sha256(p.read_bytes()).hexdigest() if n not in weights else None,'kind':'weight' if n in weights else 'metadata'}
for filename,expected in j['expected_metadata'].items():assert files[filename]['sha256']==expected,filename
p=root/'image-c5.tar';s=p.stat();digest=hashlib.sha256()
with p.open('rb') as f:
 for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
files['image-c5.tar']={'bytes':s.st_size,'stat':[s.st_size,s.st_mtime_ns,s.st_ino],'sha256':digest.hexdigest(),'kind':'image'}
manifest={'task':task,'source_root':str(model),'source_index_sha256':hashlib.sha256(index_bytes).hexdigest(),'image':image,'files':files,'weight_files':len(weights),'weight_bytes':size,'aggregate_rate_limit_bytes_per_second':512*1024**2,'server_script_sha256':j['server_sha256']};(root/'initial-manifest.json').write_text(json.dumps(manifest))
assert subprocess.run(['docker','container','inspect',name],capture_output=True).returncode!=0
args=['docker','create','--name',name,'--runtime','runc','--label','task='+task,'--label','run=L039-R009','--network','host','--cpus','2','--memory','1g','--pids-limit','64','--cap-drop','ALL','--security-opt','no-new-privileges','-e','PYTHONDONTWRITEBYTECODE=1','--mount','type=bind,src='+str(model)+',dst=/model,readonly','--mount','type=bind,src='+str(root)+',dst=/work','--entrypoint','python3',image,'-u','/work/server.py']
cid=run(args).strip();run(['docker','start',name]);x=json.loads(run(['docker','inspect','--format','{{json .}}',name]));assert x['State']['Running'] and not x['HostConfig']['Privileged'] and not x['HostConfig']['Devices'];assert next(m for m in x['Mounts'] if m['Destination']=='/model')['RW'] is False
credentials={str(n):{'ca':(secrets/'ca.crt').read_text(),'cert':(secrets/('client'+str(n)+'.crt')).read_text(),'key':(secrets/('client'+str(n)+'.key')).read_text()} for n in [44,46]}
print(json.dumps({'root':str(root),'container':name,'container_id':cid,'state':x['State'],'manifest':manifest,'certificate_authority_sha256':hashlib.sha256((secrets/'ca.crt').read_bytes()).hexdigest(),'credentials':credentials,'device_compute_executed':False,'source_model_read_only':True}))
'''
CLIENT=r'''
import hashlib,json,os,pathlib,subprocess,sys,time,urllib.request,ssl
j=json.load(sys.stdin);node=j['node'];task='sglang-pd-spec-36';base=pathlib.Path('/data/wio/sglang-mech36-20261018');root=base/'L039-R009-transfer';name=f'if-mech36-transfer-dst-l039r009-{node}';image='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
assert hashlib.sha256(j['client'].encode()).hexdigest()==j['client_sha256']
if base.exists():assert json.loads((base/'owner.json').read_text())['task']==task
else:base.mkdir(mode=0o700);(base/'owner.json').write_text(json.dumps({'task':task}))
assert not root.exists(),'Transfer destination already exists; inspect rather than replay'
total=sum(v['bytes'] for v in j['manifest']['files'].values() if node==44 or v['kind']!='image');stat=os.statvfs(base);assert stat.f_bavail*stat.f_frsize>total+100*1024**3
root.mkdir(mode=0o700);secrets=root/'secrets';secrets.mkdir(mode=0o700)
for key,filename in [('ca','ca.crt'),('cert','client.crt'),('key','client.key')]:
 p=secrets/filename;p.write_text(j['credentials'][key]);os.chmod(p,0o600)
files={n:v for n,v in j['manifest']['files'].items() if node==44 or v['kind']!='image'}
contract={'task':task,'node':node,'files':files,'source_index_sha256':j['manifest']['source_index_sha256'],'server_CA_sha256':j['manifest_CA_sha256']};(root/'contract.json').write_text(json.dumps(contract));(root/'client.py').write_text(j['client']);assert hashlib.sha256((root/'client.py').read_bytes()).hexdigest()==j['client_sha256']
context=ssl.create_default_context(cafile=str(secrets/'ca.crt'));context.load_cert_chain(secrets/'client.crt',secrets/'client.key')
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=context))
last=None
for attempt in range(15):
 try:
  with opener.open('https://172.16.12.37:39039/manifest',timeout=5) as response:remote=json.load(response)
  assert set(files).issubset(remote['files']);last=None;break
 except Exception as e:last=type(e).__name__+': '+str(e);time.sleep(0.5)
assert last is None,last
def run(a):
 p=subprocess.run(a,capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr;return p.stdout
assert json.loads(run(['docker','image','inspect','--format','{{json .Id}}',image]))==image
assert subprocess.run(['docker','container','inspect',name],capture_output=True).returncode!=0
cid=run(['docker','create','--name',name,'--runtime','runc','--label','task='+task,'--label','run=L039-R009','--network','host','--cpus','2','--memory','2g','--pids-limit','64','--cap-drop','ALL','--security-opt','no-new-privileges','-e','PYTHONDONTWRITEBYTECODE=1','--mount','type=bind,src='+str(root)+',dst=/work','--entrypoint','python3',image,'-u','/work/client.py']).strip();run(['docker','start',name]);x=json.loads(run(['docker','inspect','--format','{{json .}}',name]));assert not x['HostConfig']['Privileged'] and not x['HostConfig']['Devices']
print(json.dumps({'node':node,'root':str(root),'container':name,'container_id':cid,'state':x['State'],'files':len(files),'expected_bytes':total,'client_script_sha256':j['client_sha256'],'manifest_CA_sha256':j['manifest_CA_sha256'],'tls_mutual_authentication_verified':True,'device_compute_executed':False,'model_loaded':False}))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys,traceback
j=json.load(sys.stdin);v={'status':'STARTING','source':None,'clients':{},'credentials_in_result':False,'Mac_received_model_bytes':False,'device_compute_executed':False}
def host(number,code,payload,timeout):
 pod='sglang-tuning-access-37-20261013' if number==37 else f'sglang-container-access-{number}-20261013'
 p=subprocess.run(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',code],input=json.dumps(payload),capture_output=True,text=True,timeout=timeout)
 assert p.returncode==0,p.stderr
 return json.loads(p.stdout)
try:
 for name,uid,node in [('sglang-mech36-prefill-44-20261019','41367624-b523-4682-8be5-778f104ca373',44),('sglang-mech36-decode-46-20261019','bae63432-b774-44d1-98c4-e9b81c09a377',46)]:
  p=subprocess.run(['kubectl','get','pod','-n','default',name,'-o','json'],capture_output=True,text=True,timeout=25);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['metadata']['uid']==uid and x['metadata']['labels']['task']=='sglang-pd-spec-36' and x['status']['phase']=='Running' and x['spec']['nodeName']==f'a3-5u-atlas900rcka3-{node}'
 source=host(37,j['source_code'],{'server':j['server'],'server_sha256':j['server_sha256'],'expected_metadata':j['expected_metadata']},480)
 credentials=source.pop('credentials');v['source']=source
 def start(number):
  result=host(number,j['client_code'],{'node':number,'client':j['client'],'client_sha256':j['client_sha256'],'credentials':credentials[str(number)],'manifest':source['manifest'],'manifest_CA_sha256':source['certificate_authority_sha256']},120)
  return str(number),result
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(start,n):n for n in [44,46]};errors=[]
  for future in concurrent.futures.as_completed(futures):
   try:
    number,result=future.result();v['clients'][number]=result
   except Exception as e:errors.append({'node':futures[future],'type':type(e).__name__,'message':str(e)})
  v['client_setup_errors']=errors
  assert not errors,'One or more client setup errors; inspect partial owned jobs'
 v['status']='LAUNCHED_NOT_VERIFIED'
except Exception as e:v['status']='ERROR';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
print(json.dumps(v));sys.exit(0 if v['status']=='LAUNCHED_NOT_VERIFIED' else 1)
'''
storage=json.loads((R/'runs/L039-R008-storage-capacity/stdout.json').read_text());expected_metadata={k:v['sha256'] for k,v in storage['nodes']['37']['data']['models'][0]['metadata'].items()}
server=(R/'scripts/L039_transfer_server.py').read_text();client=(R/'scripts/L039_transfer_client.py').read_text();payload={'source_code':SOURCE,'client_code':CLIENT,'server':server,'server_sha256':hashlib.sha256(server.encode()).hexdigest(),'client':client,'client_sha256':hashlib.sha256(client.encode()).hexdigest(),'expected_metadata':expected_metadata}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])]
start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=750)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr)
v=json.loads(p.stdout) if p.stdout else {};receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':v.get('status','UNKNOWN'),'credentials_in_result':False,'model_files_on_Mac':False,'transfer_complete':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['remote-result.json','stderr.log']}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
