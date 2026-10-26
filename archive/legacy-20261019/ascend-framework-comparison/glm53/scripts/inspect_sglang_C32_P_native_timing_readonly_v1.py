"""Read-only P timing-source and immutable native-log evidence; no inference."""
from pathlib import Path
import hashlib,importlib.util,json,re,shlex,concurrent.futures
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',p);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
HOST=r'''import socket,subprocess,json,pathlib,hashlib,gzip,re,collections
assert socket.gethostname()==EXPECTED
p=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert p.returncode==0;d=json.loads(p.stdout)[0];assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running'] and d['Config']['Labels']['run_id']=='20261018T-L10-sglang-C32-target-graph-fullmesh-v3'
source_code="import pathlib,json,hashlib;root=pathlib.Path('/sgl-workspace/sglang/python/sglang/srt');paths=['observability/req_time_stats.py','disaggregation/prefill.py'];out={};"+"\nfor rel in paths:\n p=root/rel;raw=p.read_bytes();assert len(raw)<512*1024;out[rel]={'source':raw.decode(),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}\nprint(json.dumps(out))\n"
p=subprocess.run(['docker','exec',CID,'python3','-S','-c',source_code],capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr
source=json.loads(p.stdout)
for rel,item in source.items():assert item['sha256']==SOURCE_SHAS[rel]
out={'project_id':'glm53','node':NODE,'container_id':CID,'model_requests':0,'SDK_init_calls':0,'source':source,'logs':{}}
for stream,item in LOGS.items():
 path=pathlib.Path(item['archive']);assert str(path).startswith('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-target-graph-post-native-v1/') and path.is_file()
 packed=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):packed.update(block)
 assert packed.hexdigest()==item['archive_sha256']
 digest=hashlib.sha256();total=0;timing=[];batches=[];other_time_count=0
 with gzip.open(path,'rb') as f:
  for raw in f:
   total+=len(raw);assert total<=8*1024**3;digest.update(raw);line=raw.decode(errors='replace').rstrip('\r\n')
   if 'queue_duration=' in line or 'bootstrap_duration=' in line:
    other_time_count+=1
    if NAMESPACE in line:timing.append(line)
   if 'Prefill batch.' in line:batches.append(line)
   assert len(timing)<=10000 and len(batches)<=100000
 assert total==item['bytes'] and digest.hexdigest()==item['sha256']
 out['logs'][stream]={'archive':str(path),'full_archive_verified_bytes':total,'full_archive_sha256':digest.hexdigest(),'all_time_stats_lines':other_time_count,'current_formal_request_time_stats_lines':timing,'all_lifecycle_Prefill_batch_lines':batches,'scope':'Immutable completed-model-log capture. Timing rows require exact formalnamespace; batch lines cover entire runtime, not claimed as formal-only.'}
print(json.dumps(out))
'''
def render(node):
 settings=json.loads((ROOT/'config/C32-target-graph-runtime-v3/formal-C32-target-graph-v1.json').read_text());item=settings['runtime_containers'][str(node)];wire=json.loads((ROOT/f'runs/20261018T-L10-sglang-target-graph-post-native-v1/node-{node}.json').read_text());o=json.loads(wire['stdout']);assert wire['rc']==0 and o['container_id']==item['container_id']
 vals={'EXPECTED':b.NODES[node][1],'NODE':node,'CID':item['container_id'],'NAME':item['name'],'IMAGE':settings['model_image'],'NAMESPACE':settings['namespace'],'LOGS':{k:{f:v[f] for f in ['archive','archive_sha256','sha256','bytes']} for k,v in o['logs'].items()},'SOURCE_SHAS':{'observability/req_time_stats.py':'bbdbc5655e0f297c5bc3e7fa054dd851fcea973257b51e091a3d3b1c2458f9d7','disaggregation/prefill.py':'3f63b9c170966346ccd51eeb161a889e81f79370b3f261e563d4d824bf1a6fe9'}}
 code=re.sub(r'\b(?:'+'|'.join(vals)+r')\b',lambda m:repr(vals[m[0]]),HOST);return code
def inspect(node):
 pod=b.NODES[node][0];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']==f'172.16.12.{node}'
 return b.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],render(node),timeout=600)
def main():
 cert=json.loads((ROOT/'results/sglang-C32-L10-target-graph-v1-measurement-v1.json').read_text());assert cert['measurement_valid'] is True
 run=ROOT/'runs/20261018T-L11-P-native-timing-readonly-v1';run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write('once')
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'nodes':{}}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(inspect,n):n for n in [34,35]}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f]
    try:o=f.result()
    except Exception as error:o={'error':repr(error)}
    (run/f'node-{n}.json').write_text(json.dumps(o,indent=2)+'\n');receipt['nodes'][str(n)]=o.get('rc',o.get('error'));print(n,receipt['nodes'][str(n)],flush=True)
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
