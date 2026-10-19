#!/usr/bin/env python3
"""Query known actual model prefix keys using metadata-only native SDK; no NPU."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('b',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs/20261017T0730-model-cache-keys-v1'
CID='d6894b55dc4fc874aa6187bb4723b8dec43cdb21bb321b7702d114c6757d142e'
QUERY=r'''
import hashlib,pickle,os,json,pathlib,time
assert len(IDS)==74400
salt='glm53-long-functional-20261017-v1-warm'
seed=os.environ.get('PYTHONHASHSEED','vllm-none-hash')
def h(x):return hashlib.sha256(pickle.dumps(x,protocol=pickle.HIGHEST_PROTOCOL)).digest()
parent=h(seed);keys=[]
for start in range(0,len(IDS)-127,128):
 parent=h((parent,tuple(IDS[start:start+128]),(salt,) if start==0 else None))
 keys.append('GLM-5.3-w8a8@dcp:0@head_or_tp_rank:0@pp_rank:0@group:0@cache_role:kv@cache_family:default@'+parent.hex())
assert len(keys)==581
import memcache_hybrid as mc
assert str(mc.__file__).startswith('/opt/glm53/sdk13/')
store=mc.DistributedObjectStore();out={'project_id':'glm53','metadata_only':True,'init_bm':False,'namespace':salt,'block_size':128,'logical_prefix_tokens':74400,'queried_complete_tokens':74368,'known_key_count':len(keys),'hash_seed':seed,'pickle_protocol':pickle.HIGHEST_PROTOCOL,'sdk_path':mc.__file__,'sdk_version':pathlib.Path(mc.__file__).with_name('VERSION').read_text(),'keys':[],'dispatch_monotonic_ns':time.monotonic_ns()}
rc=store.init(0,init_bm=False);out['init_rc']=rc;assert rc==0
try:
 infos=store.batch_get_key_info(keys);assert len(infos)==len(keys)
 for i,(key,info) in enumerate(zip(keys,infos)):
  row={'block_index':i,'key':key}
  if info is not None:row.update(bytes=info.size(),locations=info.loc_list(),types=info.type_list(),gvas=info.gva_list())
  else:row['missing']=True
  out['keys'].append(row)
 out['complete_positive_keys']=sum(x.get('bytes',0)>0 and any(v>0 for v in x.get('gvas',[])) for x in out['keys'])
 out['physical_object_bytes']=sum(x.get('bytes',0) for x in out['keys'])
finally:out['close_rc']=store.close();out['terminal_monotonic_ns']=time.monotonic_ns();print('GLM53_MODEL_CACHE_KEYS='+json.dumps(out),flush=True)
assert out['close_rc']==0
'''
HOST=r'''
import socket,subprocess,json,pathlib,hashlib,sys
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=15);assert r.returncode==0
q=json.loads(r.stdout)[0];assert q['Id']==CID and q['Name']=='/glm53-vllm-correctness-v2-20261017-43' and q['State']['Running']
assert q['Image']=='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83' and q['Config']['Labels']['project_id']=='glm53'
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T0615-formal-inputs-v3/artifacts/C16')
assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()=='dff4e3943c14f8d12c1964f0aad9ed419441e6b94274f40507d2fdd34f4e3ab9'
record=json.loads((root/'prefix-00.json').read_text());ids=record['input_ids']
import struct
assert hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest()=='b49ef8d1a93d8229930061a30ae5b1fe327fc3249a35e42c699fed0043ddbbba'
code='IDS='+repr(ids)+'\n'+QUERY
bootstrap='export PYTHONPATH=/opt/glm53/sdk13:${PYTHONPATH:-}; export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; export PYTHONDONTWRITEBYTECODE=1; exec python3 -'
r=subprocess.run(['docker','exec','-i',CID,'bash','-c',bootstrap],input=code,capture_output=True,text=True,timeout=90)
print(r.stdout,end='');print(r.stderr,file=sys.stderr,end='');sys.exit(r.returncode)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 pod,host=b.NODES[43];m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.43'
 code='CID='+repr(CID)+'\nQUERY='+repr(QUERY)+'\n'+HOST
 cmd=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 r=b.invoke([cmd],code,timeout=130);(RUN/'node-43.json').write_text(json.dumps(r,indent=2)+'\n')
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'rc':r['rc'],'model_requests':0,'init_bm':False},indent=2)+'\n');print('model cache query',r['rc'],flush=True)
if __name__=='__main__':main()
