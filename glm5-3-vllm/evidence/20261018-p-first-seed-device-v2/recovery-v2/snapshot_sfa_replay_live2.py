from pathlib import Path
import gzip,hashlib,json,re,shutil,socket,subprocess,urllib.request
host=socket.gethostname();node=32 if host=='B4-33U-Atlas900RCKA3-32' else 33;assert host in ['B4-33U-Atlas900RCKA3-32','B5-5U-Atlas900RCKA3-33']
roots={32:'/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live2-20261018/node32',33:'/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live2-20261018/node33'};cids={32:'d9762493f4e60c450b06d964bf6c7b9d5fbcfd5261e44086a86f2031ff7ec33f',33:'05e98c8425d8bb0db0c30425ae054f9e8ee7b747914a18defaa2f18d8aee8f2a'};uids={32:'fefe6a4e-e48f-4043-b180-e5afc0d162fb',33:'82fcb8d0-44cb-4666-8ed3-1a329f446023'};root=Path(roots[node]);cid=cids[node];plan=json.loads((root/'plan.json').read_text());launch=json.loads((root/'launch.json').read_text());context=json.loads((root/'context.json').read_text());assert launch['container_id']==cid and context['client']['lease']['uid']==plan['reservation']['expected_uid']==uids[node]
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
 pcid='8c8eac8d5d10bfdc993223fd0f2e4cad98091352f000656ffc8532e4e1362860';pl=json.loads((root/'official-proxy-v1/launch.json').read_text());pd=json.loads(subprocess.check_output(['docker','inspect',pcid]))[0];assert pl['CID']==pcid and pd['Image']==d['Image'] and not pd['HostConfig'].get('Devices') and pd['Name']=='/'+pl['argv'][pl['argv'].index('--name')+1]
 if pd['State']['Running']:
  with urllib.request.urlopen('http://172.16.12.32:29880/healthcheck',timeout=5) as f:assert json.load(f)['request_num']==0
 items.append(('official-proxy',pcid,pd))
items.append(('model',cid,d));result=[]
for name,ident,before in items:
 after=json.loads(subprocess.check_output(['docker','inspect',ident]))[0];assert after['State']['Running'];n=0
 with gzip.open(out/(name+'-console.txt.gz'),'wb') as f:
  proc=subprocess.Popen(['docker','logs','--timestamps',ident],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
  for b in iter(lambda:proc.stdout.read(1048576),b''):n+=len(b);assert n<512*1024**2;f.write(b)
  assert proc.wait(timeout=15)==0
 result.append({'name':name,'CID':ident,'before':before['State'],'after':after['State'],'console_bytes':n})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'node':node,'snapshotted_and_kept_running':result,'snapshot_files':len(selected),'snapshot_bytes':size,'idle_metrics':metrics,'holder_kept':True,'scope':'Snapshot exact task-owned running model and CPU proxy, no stop or reload; all models kept running for continued research.'})+'\nGLM53_RESULT_JSON_END')
