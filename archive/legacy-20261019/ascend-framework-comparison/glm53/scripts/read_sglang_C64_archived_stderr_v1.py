"""Read exact already-archived small stderr files; no live model/SDK calls."""
import ast,concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261017T2014-sglang-C64-archived-stderr-review-v1'
def load(name,path,sha):
 p=ROOT/path;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('inventory','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
codec=load('codec','scripts/glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
HOST="""import pathlib,json,hashlib,gzip,io,socket
assert socket.gethostname()==HOSTNAME
source=pathlib.Path(SOURCE);assert str(source).startswith('/data/wio/inference-foundry/glm53/runs/20261017T2002-sglang-L04-C64-native-v2/') and source.name=='model.stderr.gz' and source.is_file() and not source.is_symlink()
packed=source.open('rb').read(8*1024**2+1);assert len(packed)<=8*1024**2 and hashlib.sha256(packed).hexdigest()==PACKED_SHA
with gzip.GzipFile(fileobj=io.BytesIO(packed)) as file:raw=file.read(16*1024**2+1)
assert len(raw)==RAW_BYTES<=16*1024**2
archive=pathlib.Path(OUTDIR);assert not archive.exists();archive.mkdir(parents=True)
text=raw.decode(errors='replace');out={'project_id':'glm53','node':NODE,'model_requests':0,'SDK_init_calls':0,'source_archive':str(source),'source_archive_sha256':PACKED_SHA,'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'utf8_lossless':text.encode()==raw,'archived_stderr_text':text}
"""
def render(node,item):
 values={'HOSTNAME':base.NODES[node][1],'SOURCE':item['archive'],'PACKED_SHA':item['archive_sha256'],'RAW_BYTES':item['raw_bytes'],'NODE':node,'OUTDIR':'/data/wio/inference-foundry/glm53/runs/'+RID};code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),HOST)+codec.ENCODER;ast.parse(code);return code

def invoke(node,item):
 pod,_=base.NODES[node];mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 if mapped['rc']!=0 or mapped['stdout']!='172.16.12.'+str(node):return mapped|{'mapping_failed':True}
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);wire=base.invoke([command],render(node,item),timeout=120)
 return codec.decode(wire) if wire['rc']==0 else wire

def main():
 p=ROOT/'config/diagnostic-runs-v1/sglang-C64-failed-stderr-archives-v1.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='eae1ad9c61d2957ed236b0238048ea8ca528d98d8b197977e76f5bd5ffb8aa26';manifest=json.loads(p.read_text());assert set(manifest['files'])=={'34','35','36','41'}
 prior=json.loads((ROOT/'runs/20261017T2002-sglang-L04-C64-native-v2/receipt.json').read_text());assert len(prior['nodes'])==5 and all(v==0 for v in prior['nodes'].values())
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'nodes':{}}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   fs={pool.submit(invoke,int(n),item):n for n,item in manifest['files'].items()}
   for f in concurrent.futures.as_completed(fs):
    n=fs[f]
    try:r=f.result()
    except Exception as error:r={'error':repr(error)}
    (run/f'node-{n}.json').write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][n]=r.get('rc',r.get('error'));print(n,receipt['nodes'][n],flush=True)
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
