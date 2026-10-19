from pathlib import Path
import gzip,hashlib,json,re,shutil,socket,subprocess,urllib.request
host=socket.gethostname();node=32 if host=='B4-33U-Atlas900RCKA3-32' else 33;assert host in ['B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33']
roots={32:'/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live1-20261018/node32',33:'/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live1-20261018/node33'};cids={32:'e812d51531c504f8c7a036671499ad4b8fcf45ca3741a43dd3ef22a846a10db7',33:'e202811aa3535d277385bb4a401ada4cf0c6fcfec5536a3adae2b6724d53210f'};uids={32:'1f257348-8296-4fa5-8bc9-1791e1553b29',33:'e43c003b-ee2a-4f8b-a551-302c6017066a'};root=Path(roots[node]);cid=cids[node];plan=json.loads((root/'plan.json').read_text());launch=json.loads((root/'launch.json').read_text());context=json.loads((root/'context.json').read_text());assert launch['container_id']==cid and context['client']['lease']['uid']==plan['reservation']['expected_uid']==uids[node]
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['Id']==cid and d['Image']==plan['image']['id'] and d['Name']=='/'+plan['container']['name'];metrics=[]
if d['State']['Running']:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,29481 if node==32 else 29482),timeout=5) as f:s=f.read().decode()
 metrics=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(metrics)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in metrics)
assert not (root/'sfa-replay-order-probe/enable').exists()
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False);selected=[]
for p in root.rglob('*'):
 if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.jsonl','.bin','.py']:
  rel=p.relative_to(root)
  if rel.parts[0]=='profile' or 'rank0-raw' in rel.parts[0] or p.name=='trace_view.json':continue
  selected.append(p)
size=sum(p.stat().st_size for p in selected);assert size<256*1024**2,size
for p in selected:
 q=out/'model-root'/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
for patch in plan['runtime_patches']:
 p=Path(patch['source']);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==patch['sha256'];q=out/'overlay-sources'/p.name;q.parent.mkdir(exist_ok=True);q.write_bytes(b)
items=[]
if node==32:
 pcid='120fe694738b968313ea3a4006bf8f50517937708251afc9cd1b837416a70ee1';pl=json.loads((root/'official-proxy-v1/launch.json').read_text());pd=json.loads(subprocess.check_output(['docker','inspect',pcid]))[0];assert pl['CID']==pcid and pd['Image']==d['Image'] and not pd['HostConfig'].get('Devices') and pd['Name']=='/'+pl['argv'][pl['argv'].index('--name')+1]
 if pd['State']['Running']:
  with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck',timeout=5) as f:assert json.load(f)['request_num']==0
 items.append(('official-proxy',pcid,pd))
items.append(('model',cid,d));result=[]
for name,ident,before in items:
 q=subprocess.run(['docker','stop','-t','30',ident],capture_output=True,text=True,timeout=45);after=json.loads(subprocess.check_output(['docker','inspect',ident]))[0];assert q.returncode==0 and not after['State']['Running'];n=0
 with gzip.open(out/(name+'-console.txt.gz'),'wb') as f:
  proc=subprocess.Popen(['docker','logs','--timestamps',ident],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  for b in iter(lambda:proc.stdout.read(1048576),b''):n+=len(b);assert n<512*1024**2;f.write(b)
  assert proc.wait(timeout=15)==0
 result.append({'name':name,'CID':ident,'before':before['State'],'after':after['State'],'console_bytes':n})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'node':node,'stopped':result,'snapshot_files':len(selected),'snapshot_bytes':size,'idle_metrics':metrics,'holder_kept':True,'scope':'Only exact task-owned finite model and CPU proxy, already-exited windows preserved; profile raw/export separately SHA backed up. No other jobs stopped.'})+'\nGLM53_RESULT_JSON_END')
