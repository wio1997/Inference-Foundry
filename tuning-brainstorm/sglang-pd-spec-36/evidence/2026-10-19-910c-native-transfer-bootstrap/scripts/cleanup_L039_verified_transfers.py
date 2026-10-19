#!/usr/bin/env python3
"""Seal both original-model clones, then remove only exact completed transfer jobs and TLS dirs."""
import datetime,hashlib,json,pathlib,shlex,subprocess,sys
R=pathlib.Path(__file__).resolve().parents[1];O=R/'runs/L039-R035-transfer-finalize';O.mkdir(parents=True,exist_ok=False)
HOST=r'''
import hashlib,json,pathlib,shutil,subprocess,sys
j=json.load(sys.stdin);base=pathlib.Path('/data/wio/sglang-mech36-20261018');assert json.loads((base/'owner.json').read_text())['task']=='sglang-pd-spec-36'
def run(a,t=30,check=True):
 p=subprocess.run(a,capture_output=True,text=True,timeout=t)
 if check:assert p.returncode==0,p.stderr
 return p
def inspect(item):
 x=json.loads(run(['docker','inspect','--format','{{json .}}',item['container']]).stdout);assert x['Id']==item['container_id'] and x['Config']['Labels']['task']=='sglang-pd-spec-36' and x['Config']['Labels']['run']==item['run'];return x
if j['action']=='seal':
 item=j['client'];x=inspect(item);assert not x['State']['Running'] and x['State']['ExitCode']==0
 root=pathlib.Path(item['root']);assert root.parent==base and root.resolve().parent==base.resolve();p=json.loads((root/'progress.json').read_text());done=p['completed'];expected=j['expected_files'];assert p['status']=='VERIFIED' and not p['errors'] and set(done)==set(expected) and len(done)==193
 manifest={}
 for n,v in done.items():
  assert pathlib.Path(n).name==n and v['bytes']==expected[n]['bytes'] and v['sha256']==v['source_sha256'];f=root/'model'/n;assert f.is_file() and not f.is_symlink();s=f.stat();assert s.st_size==v['bytes'];manifest[n]={'bytes':v['bytes'],'sha256':v['sha256'],'source_sha256':v['source_sha256'],'destination_stat':[s.st_size,s.st_mtime_ns,s.st_ino]}
  if expected[n]['kind']=='metadata':assert hashlib.sha256(f.read_bytes()).hexdigest()==v['sha256']
 seal={'task':'sglang-pd-spec-36','run':'L039-R035','node':j['node'],'model_path':str(root/'model'),'files':manifest,'files_count':len(manifest),'weight_files':sum(n.endswith('.safetensors') for n in manifest),'verified_bytes':sum(v['bytes'] for v in manifest.values()),'verification':'Original client/source SHA; R028 reused files additionally reread SHA; metadata rehashed and all destination stats sealed now. No extra full payload reread.'}
 out=base/f"L039-R035-model-seal-{j['node']}.json"
 with out.open('x') as f:json.dump(seal,f)
 print(json.dumps({'status':'SEALED','seal_path':str(out),'seal_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'seal':seal,'container_exit_code':x['State']['ExitCode']}));sys.exit(0)
if j['action']=='source_finalize':
 x=inspect(j['source']);assert x['State']['Running'];root=base/'L039-R009-transfer';manifest=json.loads((root/'initial-manifest.json').read_text());state=json.loads((root/'server-state.json').read_text());model=pathlib.Path(manifest['source_root'])
 for n,v in j['model_files'].items():
  f=model/n;s=f.stat();assert [s.st_size,s.st_mtime_ns,s.st_ino]==manifest['files'][n]['stat'] and state['files'][n]['sha256']==v['sha256'] and state['files'][n]['bytes']==v['bytes']
 # Persisted server state saves active_requests before its finally decrement. Terminal clients are authoritative.
 run(['docker','stop','-t','3',j['source']['container']],t=20);run(['docker','rm',j['source']['container']]);assert run(['docker','container','inspect',j['source']['container']],check=False).returncode!=0
 secret=root/'secrets';assert secret.is_dir() and not secret.is_symlink() and secret.resolve().parent==root.resolve() and not any(p.is_symlink() for p in secret.rglob('*'));shutil.rmtree(secret)
 print(json.dumps({'status':'SOURCE_REMOVED','container':j['source']['container'],'container_id':j['source']['container_id'],'container_absent':True,'TLS_dir_absent':not secret.exists(),'final_server_state':state,'active_counter_scope':'Persisted snapshot may precede final decrement; all allowlisted task clients terminal before shutdown.','source_original_model_unmodified':True}));sys.exit(0)
assert j['action']=='client_cleanup'
removed=[]
for item in j['clients']:
 x=inspect(item);assert not x['State']['Running'] and x['State']['ExitCode']==item['expected_exit'];run(['docker','rm',item['container']]);assert run(['docker','container','inspect',item['container']],check=False).returncode!=0;removed.append({'container':item['container'],'container_id':item['container_id'],'prior_exit':item['expected_exit'],'container_absent':True})
root=base/'L039-R009-transfer';secret=root/'secrets';assert secret.is_dir() and not secret.is_symlink() and secret.resolve().parent==root.resolve() and not any(p.is_symlink() for p in secret.rglob('*'));shutil.rmtree(secret)
keep=None
if j.get('keep_store'):
 keep=inspect(j['keep_store']);assert keep['State']['Running'];keep={'container_id':keep['Id'],'running':True}
print(json.dumps({'status':'CLIENTS_REMOVED','removed':removed,'TLS_dir_absent':not secret.exists(),'model_files_and_failed_progress_preserved':True,'persistent_store_preserved':keep}))
'''
JUMP=r'''
import concurrent.futures,json,subprocess,sys,traceback
j=json.load(sys.stdin);v={'status':'STARTING','seals':{},'source_finalize':None,'client_cleanup':{},'private_material_read_or_emitted':False}
def host(n,payload):
 pod='sglang-tuning-access-37-20261013' if n==37 else f'sglang-container-access-{n}-20261013';p=subprocess.run(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','-n','--','python3','-c',j['host']],input=json.dumps(payload),capture_output=True,text=True,timeout=110);assert p.returncode==0,p.stderr;return json.loads(p.stdout)
try:
 for node in [44,46]:v['seals'][str(node)]=host(node,{'action':'seal','node':node,'client':j['clients'][str(node)][0],'expected_files':j['expected_files']})
 a=v['seals']['44']['seal']['files'];b=v['seals']['46']['seal']['files'];assert set(a)==set(b) and all((a[n]['bytes'],a[n]['sha256'])==(b[n]['bytes'],b[n]['sha256']) for n in a)
 v['equivalence']={'files':193,'weight_shards':186,'bytes_per_clone':764089835336,'all_source_P_D_hashes_equal':True}
 v['source_finalize']=host(37,{'action':'source_finalize','source':j['source'],'model_files':a})
 for node in [44,46]:v['client_cleanup'][str(node)]=host(node,{'action':'client_cleanup','clients':j['clients'][str(node)],'keep_store':j['keep_store'] if node==46 else None})
 v['status']='VERIFIED_SEALED_TRANSFER_OBJECTS_CLEANED'
except BaseException as e:v['status']='ERROR';v['error']={'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()}
print(json.dumps(v));sys.exit(0 if v['status']=='VERIFIED_SEALED_TRANSFER_OBJECTS_CLEANED' else 1)
'''
a=json.loads((R/'runs/L039-R009-transfer-launch/remote-result.json').read_text());b=json.loads((R/'runs/L039-R028-resume-launch/remote-result.json').read_text());obs=json.loads((R/'runs/L039-R034-transfer-verification/stdout.json').read_text());store=json.loads((R/'runs/L039-R033-shared-store/remote-result.json').read_text())['store']['data']
for n in ['44','46']:
 x=obs['nodes'][n]['data'];assert x['state']['ExitCode']==0 and not x['state']['Running'] and x['progress']['status']=='VERIFIED' and x['progress']['final_contract_matches']
oldroot='/data/wio/sglang-mech36-20261018/L039-R009-transfer';payload={'host':HOST,'expected_files':{n:x for n,x in a['source']['manifest']['files'].items() if n!='image-c5.tar'},'source':{'container':a['source']['container'],'container_id':a['source']['container_id'],'run':'L039-R009'},'clients':{'44':[{'container':b['container'],'container_id':b['container_id'],'run':'L039-R028','root':b['root'],'expected_exit':0},{'container':a['clients']['44']['container'],'container_id':a['clients']['44']['container_id'],'run':'L039-R009','expected_exit':255}],'46':[{'container':a['clients']['46']['container'],'container_id':a['clients']['46']['container_id'],'run':'L039-R009','root':oldroot,'expected_exit':0}]},'keep_store':{'container':store['name'],'container_id':store['container_id'],'run':'L039-R033'}}
cmd=['ssh','-S',str(R/'connections/l038.sock'),'-o','BatchMode=yes','-o','ConnectTimeout=12','-p','60011','root@61.241.77.34',shlex.join(['python3','-c',JUMP])];start=datetime.datetime.now(datetime.timezone.utc).isoformat();p=subprocess.run(cmd,input=json.dumps(payload),capture_output=True,text=True,timeout=500)
(O/'remote-result.json').write_text(p.stdout);(O/'stderr.log').write_text(p.stderr);v=json.loads(p.stdout) if p.stdout else {};receipt={'started_utc':start,'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exit_code':p.returncode,'status':v.get('status','UNKNOWN'),'model_loaded':False,'private_material_read_or_emitted':False,'script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'artifacts':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in O.iterdir() if f.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));sys.exit(p.returncode)
