"""Exact current SGLang cache/bootstrap source read; zero model/SDK calls."""
import ast,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261017T2018-sglang-C64-native-cache-source-v1'
def load(name,file,sha):
 p=ROOT/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('inventory','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
codec=load('codec','scripts/glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
FILES=['srt/disaggregation/prefill.py','srt/disaggregation/decode.py','srt/disaggregation/base/conn.py','srt/disaggregation/ascend/conn.py','srt/environ.py','srt/managers/cache_controller.py','srt/mem_cache/storage/npu_memcache/npu_memcache_store.py','srt/mem_cache/unified_radix_cache.py','srt/mem_cache/buffer_mode/pipeline.py','srt/mem_cache/unified_cache/unified_tree_core.py','srt/mem_cache/unified_cache/storage_attachment.py']
PROBE=r"""import pathlib,json,hashlib,sys
root=pathlib.Path('/sgl-workspace/sglang/python/sglang');entries=[];total=0
for name in FILES:
 path=root/name;assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
 raw=path.read_bytes();total+=len(raw);assert len(raw)<=2*1024**2 and total<=4*1024**2
 entries.append({'relative':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'source':raw.decode()})
assert not any(x.startswith(('torch','sglang','memcache_hybrid','memfabric_hybrid')) for x in sys.modules)
print(json.dumps({'files':entries,'model_requests':0,'SDK_init_calls':0,'framework_imported':False}))
"""
HOST=r"""import pathlib,subprocess,json,hashlib,gzip,socket
assert socket.gethostname()==HOSTNAME
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0
info=json.loads(r.stdout)[0];assert info['Id']==CID and info['Name']=='/'+NAME and info['Image']==IMAGE and info['State']['Running'] and info['Config']['Labels']['project_id']=='glm53' and info['Config']['Labels']['run_id']=='20261017T1954-sglang-metadata-v1'
status=json.loads(pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1954-sglang-metadata-v1/supervisor-status.json').read_text());assert status['phase']=='MODEL_RUNNING' and 'model_exit_code' not in status
r=subprocess.run(['docker','exec','-i',CID,'env','PYTHONDONTWRITEBYTECODE=1','python3','-'],input=PROBE,capture_output=True,text=True,timeout=90);assert r.returncode==0,r.stderr
out=json.loads(r.stdout);out.update({'project_id':'glm53','node':34,'container_id':CID,'image':IMAGE})
archive=pathlib.Path(OUTDIR);assert not archive.exists();archive.mkdir(parents=True)
"""
def replace(code,values):
 code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),code);ast.parse(code);return code
def render(runtime):
 item=runtime['runtime_containers']['34'];return replace(HOST,{'HOSTNAME':base.NODES[34][1],'CID':item['container_id'],'NAME':item['name'],'IMAGE':runtime['model_image'],'PROBE':replace(PROBE,{'FILES':FILES}),'OUTDIR':'/data/wio/inference-foundry/glm53/runs/'+RID})+codec.ENCODER
def main():
 p=ROOT/'config/optimization-overlays/routing-metadata-v1/runtime.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='3818f8b619234767815380604f727af854a57e68ed17ff8aa4819ef636698856';runtime=json.loads(p.read_text())
 for run in ['20261017T2002-sglang-L04-C64-native-v2','20261017T2014-sglang-C64-archived-stderr-review-v1']:
  prior=json.loads((ROOT/'runs'/run/'receipt.json').read_text());assert all(v==0 for v in prior['nodes'].values())
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'nodes':{}}
 try:
  pod,_=base.NODES[34];mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.34'
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);wire=base.invoke([command],render(runtime),timeout=150);out=codec.decode(wire) if wire['rc']==0 else wire
  (run/'node-34.json').write_text(json.dumps(out,indent=2)+'\n');receipt['nodes']['34']=out['rc'];print('SOURCE_READ_RC='+str(out['rc']),flush=True)
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
