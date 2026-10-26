"""Revalidate completed hardlinks after interrupted parent transfer, then copy missing files to a new task directory."""
import concurrent.futures,hashlib,json,os,pathlib,ssl,threading,time,traceback,urllib.parse,urllib.request
ROOT=pathlib.Path('/work');MODEL=ROOT/'model';assert MODEL.is_dir();contract=json.loads((ROOT/'contract.json').read_text());node=contract['node'];lock=threading.Lock();stopping=threading.Event()
context=ssl.create_default_context(cafile=str(ROOT/'secrets/ca.crt'));context.load_cert_chain(ROOT/'secrets/client.crt',ROOT/'secrets/client.key');context.minimum_version=ssl.TLSVersion.TLSv1_2
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),urllib.request.HTTPSHandler(context=context))
base='https://172.16.12.37:39039';state={'status':'STARTING','node':node,'started_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'bytes_received':0,'reused_bytes_verified':0,'reused_files_verified':0,'completed':{},'inflight':{},'errors':[],'model_loaded':False,'device_compute_executed':False}
def write():
 with lock:
  p=ROOT/'progress.json.tmp';p.write_text(json.dumps(state));os.replace(p,ROOT/'progress.json')
def observer():
 while not stopping.wait(2):write()
def fetch_manifest():
 with opener.open(base+'/manifest',timeout=30) as r:return json.load(r)
def download(name):
 item=contract['files'][name];target=(ROOT/'image-c5.tar') if name=='image-c5.tar' else MODEL/name
 assert not target.exists() and not target.with_suffix(target.suffix+'.part').exists(),'Do not overwrite transfer state'
 temporary=target.with_suffix(target.suffix+'.part');digest=hashlib.sha256();count=0
 with lock:state['inflight'][name]={'bytes':0,'expected':item['bytes']}
 with opener.open(base+'/file/'+urllib.parse.quote(name,safe=''),timeout=120) as response,temporary.open('xb') as f:
  assert int(response.headers['Content-Length'])==item['bytes']
  while True:
   if stopping.is_set():raise RuntimeError('Transfer stopped after another failure')
   block=response.read(4*1024**2)
   if not block:break
   f.write(block);digest.update(block);count+=len(block)
   with lock:state['bytes_received']+=len(block);state['inflight'][name]['bytes']=count
  f.flush();os.fsync(f.fileno())
 assert count==item['bytes'],'Truncated transfer'
 sha=digest.hexdigest();source=None
 for attempt in range(10):
  source=fetch_manifest()['files'][name]['sha256']
  if source is not None:break
  time.sleep(0.2)
 assert source==sha and item.get('sha256') in [None,sha],'Source/client SHA256 mismatch'
 os.rename(temporary,target)
 with lock:state['completed'][name]={'bytes':count,'sha256':sha,'source_sha256':source};state['inflight'].pop(name,None)
 write()
try:
 actual=fetch_manifest();assert all(name in actual['files'] and actual['files'][name]['bytes']==item['bytes'] and actual['files'][name]['stat']==item['stat'] for name,item in contract['files'].items())
 state['status']='REVALIDATING_EXISTING';state['expected_bytes']=sum(i['bytes'] for i in contract['files'].values());write()
 recovery=json.loads((ROOT/'recovery-contract.json').read_text());state['parent_run']='L039-R009';state['reused_files_expected']=len(recovery['completed'])
 for name,old in recovery['completed'].items():
  item=contract['files'][name];target=MODEL/name;assert target.is_file() and not target.is_symlink() and target.stat().st_size==item['bytes']
  digest=hashlib.sha256()
  with target.open('rb') as f:
   for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
  sha=digest.hexdigest();source=actual['files'][name]['sha256'];assert source is not None and sha==source==old['sha256']==old['source_sha256'],'Reused file hash mismatch: '+name
  state['completed'][name]={'bytes':item['bytes'],'sha256':sha,'source_sha256':source,'origin':'R009_completed_and_R028_disk_revalidated'};state['reused_bytes_verified']+=item['bytes'];state['reused_files_verified']+=1;write()
 state['status']='COPYING_MISSING';write();threading.Thread(target=observer,daemon=True).start()
 # Metadata first;44 also receives the immutable image export before weight copying.
 initial=[n for n in contract['files'] if not n.endswith('.safetensors') and n not in state['completed']]
 for name in initial:download(name)
 weights=[n for n in contract['files'] if n.endswith('.safetensors') and n not in state['completed']]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(download,n) for n in weights]
  try:
   for future in concurrent.futures.as_completed(futures):future.result()
  except Exception:
   stopping.set()
   for future in futures:future.cancel()
   raise
 assert len(state['completed'])==len(contract['files'])
 state['status']='VERIFIED';state['finished_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();state['all_stream_hashes_match']=True
except Exception as e:
 state['status']='FAILED';state['errors'].append({'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
finally:
 stopping.set();write()
print(json.dumps({'status':state['status'],'node':node,'completed_files':len(state['completed']),'bytes_received':state['bytes_received']}));raise SystemExit(0 if state['status']=='VERIFIED' else 1)
