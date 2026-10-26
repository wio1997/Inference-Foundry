"""Current owned vLLM D source/process/idle evidence, no new profile or inference."""
import base64
import gzip
import hashlib
import importlib.util
import json
import os
import shlex
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RID='20261019T-L19-vllm-D-profile-readiness-v1'
REVIEW='results/C32-L19-vllm-profile-readiness-review-v1.json'
SOCKET='/Users/wio/work/sglang-glm53-pd/ssh-control-l18-20261019'
PROBE=r'''import pathlib,json,hashlib,gzip,base64,sys,urllib.request
roots=['/vllm-workspace/vllm','/vllm-workspace/vllm-ascend']
files={roots[0]:['vllm/config/profiler.py','vllm/profiler/wrapper.py','vllm/v1/worker/gpu_worker.py','vllm/entrypoints/serve/profile/api_router.py'],roots[1]:['vllm_ascend/profiler/torch_npu_profiler.py','vllm_ascend/worker/worker_v1.py','vllm_ascend/spec_decode/llm_base_proposer.py','vllm_ascend/compilation/acl_graph.py']}
out={'project_id':'glm53','files':[],'missing':[],'worker_processes':[],'model_requests':0,'SDK_init_calls':0,'new_profiles':0};total=0
for root,names in files.items():
 for name in names:
  p=pathlib.Path(root)/name
  if not p.is_file():out['missing'].append(str(p));continue
  assert not p.is_symlink() and p.resolve().is_relative_to(pathlib.Path(root).resolve())
  raw=p.read_bytes();total+=len(raw);assert len(raw)<=2*1024**2 and total<=6*1024**2
  out['files'].append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source':raw.decode()})
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes()
  if not any(s in cmd for s in [b'VLLM::',b'VllmWorker',b'VllmEngineCore',b'vllm.entrypoints.openai.api_server']):continue
  env=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
  out['worker_processes'].append({'pid':int(p.name),'cmdline':cmd.decode().split('\0')[:100],'selected_environment':{k:env.get(k.encode(),b'').decode() for k in ['TASK_QUEUE_ENABLE','ACL_OP_INIT_MODE','ASCEND_LAUNCH_BLOCKING','HCCL_BUFFSIZE']}})
 except (OSError,UnicodeDecodeError):continue
assert len(out['worker_processes'])<=64
try:
 with urllib.request.urlopen('http://127.0.0.1:__PORT__/metrics',timeout=8) as response:
  raw=response.read(4*1024**2+1);assert len(raw)<=4*1024**2
  out['metrics']={'status':response.status,'body':raw.decode(),'sha256':hashlib.sha256(raw).hexdigest()}
except Exception as error:out['metrics']={'error':repr(error)}
profile=pathlib.Path('/run/glm53/profile')
traces=sorted(profile.glob('*/ASCEND_PROFILER_OUTPUT/trace_view.json'));assert len(traces)<=128
out['existing_profile_traces']=[{'relative':str(p.relative_to(profile)),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in traces]
assert not any(k.startswith(('torch','vllm','sglang','memcache_hybrid','memfabric_hybrid')) for k in sys.modules)
raw=json.dumps(out).encode();print(json.dumps({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gzip_base64':base64.b64encode(gzip.compress(raw,mtime=0)).decode()}))
'''


def load(name,relative,expected):
    p=ROOT/relative;assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def plans():
    helper=load('l19_identity','scripts/inspect_glm53_C32_indexer_partition_installed_sources_v1.py','2f9cac36b2776d7eb5ce00397218dd4c019e7ad964d0f0fe64f4c40daf1e9ac7')
    inv=load('l19_inventory','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
    inv.SSH=list(inv.SSH);inv.SSH[2]=SOCKET
    reference=json.loads((ROOT/'results/vllm-formal-C32-v2-native-v2.json').read_bytes());assert reference['pass']
    out={}
    for node in [43,45]:
        cfg=json.loads((ROOT/f'config/vllm-correctness-v10/node-{node}.json').read_bytes())
        probe=PROBE.replace('__PORT__',str(cfg['api_port']))
        code=helper.render(helper.HOST,{'HOSTNAME':inv.NODES[node][1],'CID':reference['runtime_CIDs'][str(node)],'NAME':f'glm53-vllm-shared-v4-20261017-{node}','IMAGE':cfg['image_id'],'SHARED_RID':'20261017T1766-vllm-shared-pool-v4','PROBE':probe,'NODE':node})
        out[node]={'code':code,'sha256':hashlib.sha256(code.encode()).hexdigest()}
    return inv,out


def main():
    inv,plan=plans();review=json.loads((ROOT/REVIEW).read_bytes())
    assert review['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert review['host_codes_sha256']=={str(n):v['sha256'] for n,v in plan.items()}
    assert Path(SOCKET).is_socket()
    run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()))
    receipt={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'new_profiles':0,'remote_mutations':False,'max_SSH':1,'nodes':{},'pass':False}
    try:
        for node,item in plan.items():
            pod=inv.NODES[node][0]
            mapping=inv.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=20)
            (run/f'mapping-{node}.json').write_text(json.dumps(mapping)+'\n');assert mapping['rc']==0 and mapping['stdout']=='172.16.12.'+str(node)
            command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
            wire=inv.invoke([command],item['code'],timeout=180);(run/f'node-{node}-wire.json').write_text(json.dumps(wire)+'\n')
            receipt['nodes'][str(node)]={'rc':wire['rc']};assert wire['rc']==0
            doc=json.loads(wire['stdout']);probe=doc['source_probe'];assert probe['rc']==0,probe['stderr']
            packed=json.loads(probe['stdout']);raw=gzip.decompress(base64.b64decode(packed['gzip_base64'],validate=True));assert len(raw)==packed['bytes'] and hashlib.sha256(raw).hexdigest()==packed['sha256']
            (run/f'node-{node}-source.json').write_bytes(raw)
            obj=json.loads(raw);dest=run/'source'/str(node);dest.mkdir(parents=True)
            for f in obj['files']:
                data=f['source'].encode();assert hashlib.sha256(data).hexdigest()==f['sha256'];(dest/f['path'].strip('/').replace('/','__')).write_bytes(data)
            receipt['nodes'][str(node)].update(source_bytes=len(raw),source_sha256=packed['sha256'],processes=len(obj['worker_processes']),existing_traces=len(obj['existing_profile_traces']))
        receipt['pass']=True
    finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
