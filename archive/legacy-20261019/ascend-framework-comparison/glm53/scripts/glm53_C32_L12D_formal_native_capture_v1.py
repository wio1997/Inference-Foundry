"""Mixed-root read-only native evidence for new P plus retained L10 D/Meta."""
import ast,concurrent.futures,hashlib,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(name,file,sha):
 p=ROOT/'scripts'/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def plans(capture_id):
 assert capture_id.startswith('20261019T-L12C-sglang-') and Path(capture_id).name==capture_id
 old=load('old_capture','glm53_sglang_C32_target_graph_native_capture_v3.py','8cad4c626dd24b028d35bdea17863f047d150ac9e15e7e1486669fe5ad668e03')
 identity=load('mixed_identity','glm53_C32_L12C_mixed_model_identity_v1.py','ff5a92342a3a385256b619e9c004294b881634678326d706eca026413c5e6a98')
 cap=json.loads((ROOT/'results/sglang-C32-L12C-selected-P-pool-capacity-v1.json').read_text());assert cap['pass'];settings=json.loads((ROOT/'config/C32-target-graph-runtime-v3/formal-C32-target-graph-v1.json').read_text())
 text=old.HOST.replace("root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v3');",'root=pathlib.Path(NODE_REMOTE_ROOT);')
 assert text!=old.HOST
 stream=old.STREAM.replace("if 'GLM53_C32_PACKED_DISPATCH_' in line", "if 'GLM53_C32_NATIVE_MC_MAPPING_REUSED=' in line or 'GLM53_C32_NATIVE_HICACHE_' in line or 'GLM53_NATIVE_HICACHE_FATAL' in line or 'GLM53_C32_PACKED_DISPATCH_' in line")
 text+='''
out['native_optional_extension_worker_maps']=[]
if NATIVE_P:
 sdk_pids={event['pid'] for item in out['persistent_audit'].values() for event in item['rows'] if event['event']=='initialize_complete'}
 for row in rows:
  if row['namespace_pid'] not in sdk_pids:continue
  raw=(pathlib.Path('/proc')/str(row['host_pid'])/'maps').read_bytes();assert len(raw)<=8*1024**2
  paths=sorted({line.split(maxsplit=5)[5].strip() for line in raw.decode().splitlines() if 'libmf_hybm_accoffload.so' in line})
  assert paths==['/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_accoffload.so']
  out['native_optional_extension_worker_maps'].append({'namespace_pid':row['namespace_pid'],'host_pid':row['host_pid'],'loaded_extension_paths':paths,'process_maps_sha256':hashlib.sha256(raw).hexdigest()})
 assert len(sdk_pids)==16 and len(out['native_optional_extension_worker_maps'])==16
'''
 codes={}
 for n,item in cap['providers'].items():
  node=int(n);values={'CID':item['container_id'],'NAME':item['name'],'IMAGE':settings['model_image'],'NODE':node,'IDENTITY':identity.render(node,cap),'IDENTITY_ENV':{},'STREAM':stream,'SDK_SOURCE':old.SDK_SOURCE,'PORT':30500 if node in [34,35] else 30600,'META':cap['meta_container_id'],'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+capture_id,'FORMAL_RID':'20261019T-L12D-sglang-formal-C32-native-HiCache-v1','CONCURRENCY':32,'NEED_FORMAL':True,'NODE_REMOTE_ROOT':item['remote_root'],'NATIVE_P':node in [34,35] and cap['selected_variant']=='candidate'}
  codes[node]=old.render(text,values);ast.parse(codes[node])
 return old,codes

def invoke_with_raw_failure(old,node,code,archive):
 import shlex
 pod,host=old.base.NODES[node];mapped=old.base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 local=ROOT/'runs'/Path(archive).name
 (local/('node-'+str(node)+'-mapping-wire.json')).write_text(json.dumps(mapped,indent=2)+'\n');assert mapped['rc']==0 and mapped['stdout']=='172.16.12.'+str(node)
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+code+old.codec.ENCODER;ast.parse(code)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 wire=old.base.invoke([command],code,timeout=900);(local/('node-'+str(node)+'-raw-wire.json')).write_text(json.dumps(wire,indent=2)+'\n')
 if wire['rc']!=0:return wire
 return old.codec.decode(wire)

def execute(capture_id,review_path):
 old,codes=plans(capture_id);review=json.loads((ROOT/review_path).read_text());assert review['host_codes_sha256']=={str(n):hashlib.sha256(c.encode()).hexdigest() for n,c in codes.items()}
 startup=json.loads((ROOT/'runs/20261019T-L12C-selected-P-models-and-observe-v1/receipt.json').read_text());assert startup['startup_pass']
 rpc=json.loads((ROOT/'runs/20261019T-L12D-sglang-formal-C32-native-HiCache-v1/controller-rpc-state.json').read_text());assert rpc['done_received'] and rpc['connection_rc'] is not None
 run=ROOT/'runs'/capture_id;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'max_SSH':2,'nodes':{}}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   jobs={pool.submit(invoke_with_raw_failure,old,n,code,'/data/wio/inference-foundry/glm53/runs/'+capture_id):n for n,code in codes.items()}
   for f in concurrent.futures.as_completed(jobs):
    n=jobs[f]
    try:r=f.result()
    except Exception as e:r={'error':repr(e)}
    (run/('node-'+str(n)+'.json')).write_text(json.dumps(r,indent=2)+'\n');receipt['nodes'][str(n)]=r.get('rc',r.get('error'));print(n,receipt['nodes'][str(n)],flush=True)
  assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values());receipt['pass']=True
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 return receipt
