from pathlib import Path
import gzip,hashlib,json,socket,subprocess,urllib.request
assert socket.gethostname()=='B4-33U-Atlas900RCKA3-32'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/p-bucket-control-v1-20261019/node32');cid='ccf21ea1b27e7fdbe4c45d4a862f9dc86a9ca4df38d937f55cdf7a6212465758';proxy='5c38dc4d5ea985fccddf7305173f49e91372f6ebcd29bce5ff36fe3aeb76426c';p=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert d['Id']==cid and d['Name']=='/'+p['container']['name'] and d['Image']==p['image']['id']
for n,port in [(32,29481),(33,29482)]:
 with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:s=f.read().decode()
 z=[v for v in s.splitlines() if v.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))];assert len(z)>=3 and all(float(v.rsplit(' ',1)[1])==0 for v in z)
out=Path(__file__).resolve().parent/'artifacts';out.mkdir(exist_ok=False)
for file in root.rglob('*'):
 if file.is_file() and not file.is_symlink() and file.suffix in ['.json','.jsonl','.py']:
  q=out/'model-root'/file.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(file.read_bytes())
for patch in p['runtime_patches']:
 file=Path(patch['source']);b=file.read_bytes();assert hashlib.sha256(b).hexdigest()==patch['sha256'];q=out/'overlay-sources'/file.name;q.parent.mkdir(exist_ok=True);q.write_bytes(b)
rows=[];capture=[]
for label,ident in [('model',cid),('official-proxy',proxy)]:
 d=json.loads(subprocess.check_output(['docker','inspect',ident]))[0]
 if label=='official-proxy':assert not d['HostConfig'].get('Devices') and json.loads((root/'official-proxy-v1/launch.json').read_text())['CID']==ident
 q=subprocess.run(['docker','logs','--timestamps',ident],capture_output=True,timeout=30);assert q.returncode==0;b=q.stdout+b'\n'+q.stderr
 with gzip.open(out/(label+'-console.txt.gz'),'wb') as f:f.write(b)
 if label=='model':capture=[s for s in b.decode(errors='replace').splitlines() if any(z in s for z in ['Capturing','capturing','graph capturing','Graph capturing'])][-16:]
 rows.append({'kind':label,'CID':ident,'state':d['State'],'console_bytes':len(b),'console_sha256':hashlib.sha256(b).hexdigest()})
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'snapshots':rows,'capture_log_tail':capture,'P_D_idle':True,'scope':'Snapshot and original console only, no stop/restart/profile. Current temporary P native bucket configuration retained under original finite watchdog; D native unchanged.'})+'\nGLM53_RESULT_JSON_END')
