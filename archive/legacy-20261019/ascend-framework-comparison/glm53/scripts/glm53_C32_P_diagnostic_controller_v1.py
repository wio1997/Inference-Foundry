"""One CPU43 diagnostic child with native two-step P profile; ZCode only.

No primary benchmark changes. Sequential native prechecks then one SSH holds the
CPU child; profiling uses HTTP inside that child, so global SSH occupancy <=1.
"""
import ast,base64,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(name,relative,digest):
    p=ROOT/relative;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('inventory','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
identity=load('identity','scripts/glm53_C32_target_graph_model_identity_v4.py','67277de6f7b7f167997b53a53a411b30e663630daf56957312fb6b609abebd91')
codec=load('codec','scripts/glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')

PRECHECK=r'''
import urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(URL+'/metrics',timeout=10) as response:raw=response.read(8*1024**2+1);assert response.status==200 and len(raw)<=8*1024**2
gauges=[float(x.rsplit(' ',1)[1]) for x in raw.decode().splitlines() if x.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))]
assert gauges and all(x==0 for x in gauges)
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert npu.returncode==0;rows=[]
for line in npu.stdout.split('Process id',1)[-1].splitlines():
 cols=[x.strip() for x in line.split('|')]
 if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():
  proc=pathlib.Path('/proc')/cols[2];assert CID in (proc/'cgroup').read_text();rows.append(int(cols[2]))
assert len(rows)==18
parent=pathlib.Path('/proc')/str(model_identity['host_pid'])
env=dict(x.split(b'=',1) for x in (parent/'environ').read_bytes().split(b'\0') if b'=' in x)
profile_env={k:env.get(k.encode(),b'').decode() for k in ['SGLANG_PROFILE_V2','SGLANG_TORCH_PROFILER_DIR']}
assert profile_env['SGLANG_PROFILE_V2'] in ['', '0', 'false', 'False']
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v3')
assert not (root/'profile'/DIAG_RID).exists()
import shutil
if NODE in [34,35]:assert shutil.disk_usage(root).free>=24*1024**3,'Preserve originals; insufficient room for bounded all32P profile'
ptracer_files=[]
if NODE in [34,35]:
 reader="import pathlib,json,hashlib;out=[]\nfor p in sorted(pathlib.Path('/var/log/memfabric_hybrid').glob('ptracer_*.dat')):\n raw=p.read_bytes();assert len(raw)<=8*1024**2;out.append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()})\nassert len(out)<=64;print(json.dumps(out))"
 snap=subprocess.run(['docker','exec',CID,'python3','-S','-c',reader],capture_output=True,text=True,timeout=30);assert snap.returncode==0;ptracer_files=json.loads(snap.stdout)
if NODE in [34,35]:
 source="from pathlib import Path;import hashlib;print(hashlib.sha256(Path('/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/profiler_manager.py').read_bytes()).hexdigest())"
 p=subprocess.run(['docker','exec',CID,'python3','-c',source],capture_output=True,text=True,timeout=30)
 assert p.returncode==0 and p.stdout.strip()=='a962ab8283b468d6c3f10e53527613ea79569490cfdef340f3c22ffbee6bdcc3'
print(json.dumps({'project_id':'glm53','node':NODE,'container_id':CID,'model_identity':model_identity,'idle_gauges':gauges,'metrics_body':raw.decode(),'metrics_sha256':hashlib.sha256(raw).hexdigest(),'native_ptracer_files':ptracer_files,'owned_NPU_contexts':len(rows),'loadavg':pathlib.Path('/proc/loadavg').read_text(),'meminfo':pathlib.Path('/proc/meminfo').read_text(),'profile_environment':profile_env,'model_requests':0,'SDK_init_calls':0}))
'''

CPU=r'''import pathlib,subprocess,json,hashlib,base64,gzip,socket
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
root=pathlib.Path(CPU_ROOT);assert not root.exists();root.mkdir(parents=True)
archive=root/'native-export';archive.mkdir()
image='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
p=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20);assert p.returncode==0 and p.stdout.strip()==image
for name,item in FILES.items():
 assert pathlib.Path(name).name==name
 raw=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==item['sha256'];(root/name).write_bytes(raw)
prior=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-formal-C32-target-graph-v1/output/formal-result.json')
assert hashlib.sha256(prior.read_bytes()).hexdigest()==CLIENT['prior_report_sha256']
dataset=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T2056-sglang-C32-fifth-suffix-inputs-v1/artifacts')
assert hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()==CLIENT['input_manifest_sha256']
raw=(json.dumps(CLIENT,indent=2)+'\n').encode();(root/'client-settings.json').write_bytes(raw);settings_sha=hashlib.sha256(raw).hexdigest()
output=root/'output';output.mkdir()
args=['docker','run','--rm','--pull=never','--network=host','--pids-limit=512','--memory=16g','--cpus=4','--label','project_id=glm53','--label','run_id='+RID,'--name','glm53-C32-P-two-step-diag-v1-20261018-43',
 '-v',str(root)+':/run/glm53','-v',str(dataset)+':/diagnostic-inputs:ro','-v',str(prior)+':/prior-report.json:ro','-v','/data/wio/aisbench-offline-43-20261014:/work:ro','--workdir','/run/glm53']
environment={'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','PYTHONPATH':'/run/glm53:/work/benchmark','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','CUDA_VISIBLE_DEVICES':'','ASCEND_RT_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':'/run/glm53/hf','XDG_CACHE_HOME':'/run/glm53/cache','MPLCONFIGDIR':'/run/glm53/mpl','TORCH_HOME':'/run/glm53/torch','NLTK_DATA':'/work/nltk_data','TIKTOKEN_CACHE_DIR':'/work/tiktoken_cache'}
for k,v in environment.items():args+=['-e',k+'='+v]
args+=['--entrypoint','/work/venv/bin/python',image,'/run/glm53/glm53_C32_P_diagnostic_client_v1.py','--settings','/run/glm53/client-settings.json','--sha256',settings_sha]
out={'project_id':'glm53','run_id':RID,'diagnostic_only':True,'performance_certificate_allowed':False,'SDK_init_calls':0}
with (root/'client-stdout.txt').open('x') as stdout,(root/'client-stderr.txt').open('x') as stderr:
 try:
  p=subprocess.run(args,stdout=stdout,stderr=stderr,timeout=1800);out['client_rc']=p.returncode
 except subprocess.TimeoutExpired:
  out['host_wait_timeout']=True;out['client_rc']=None
  # Preserve possibly live child/profile; do not kill an unverified model or retry.
for name in ['client-stdout.txt','client-stderr.txt']:
 raw=(root/name).read_bytes();assert len(raw)<=32*1024**2;out[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'tail':raw[-24000:].decode(errors='replace')}
report=output/'diagnostic-result.json'
if report.is_file():
 raw=report.read_bytes();assert len(raw)<=64*1024**2;out['report']={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
progress=output/'P-profile-control-progress.json'
if progress.is_file():out['profile_control']=json.loads(progress.read_text())
'''

def render(source,values):
    pattern=r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b'
    code=re.sub(pattern,lambda m:repr(values[m[0]]),source);ast.parse(code);return code

def invoke(node,code,timeout):
    pod,hostname=base.NODES[node]
    mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
    assert mapped['rc']==0 and mapped['stdout']==f'172.16.12.{node}'
    code='import socket\nassert socket.gethostname()=='+repr(hostname)+'\n'+code;ast.parse(code)
    command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
    return base.invoke([command],code,timeout=timeout)

def execute(relative,digest):
    path=ROOT/relative;assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    settings=json.loads(path.read_text());assert settings['project_id']=='glm53' and settings['concurrency']==32 and settings['diagnostic_only'] and not settings['performance_certificate_allowed']
    assert settings['run_id']=='20261018T-L11-sglang-C32-P-two-step-profile-v1'
    for relative,digest in settings['prerequisite_sha256'].items():assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
    cert=json.loads((ROOT/'results/sglang-C32-L10-target-graph-v1-measurement-v1.json').read_text());assert cert['measurement_valid'] and cert['formal_requests']==128 and cert['native_postrun_audit_required'] is False
    post=json.loads((ROOT/'runs/20261018T-L10-sglang-target-graph-post-native-v1/receipt.json').read_text());assert all(x==0 for x in post['nodes'].values())
    run=ROOT/'runs'/settings['run_id'];run.mkdir(parents=True,exist_ok=True)
    with (run/'execution-once.lock').open('x') as file:file.write(str(__import__('os').getpid()))
    receipt={'project_id':'glm53','run_id':settings['run_id'],'diagnostic_only':True,'prechecks':{},'client_started':False,'SDK_init_calls':0}
    try:
        for ns,item in settings['runtime_containers'].items():
            node=int(ns);code=render(identity.HOST+PRECHECK,{'CID':item['container_id'],'NAME':item['name'],'IMAGE':settings['model_image'],'NODE':node,'URL':f'http://172.16.12.{node}:' + ('30500' if node in [34,35] else '30600'),'DIAG_RID':settings['run_id']})
            result=invoke(node,code,120);(run/f'precheck-{node}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['prechecks'][ns]=result['rc'];assert result['rc']==0
        payload={}
        for name,relative in settings['files'].items():
            raw=(ROOT/relative).read_bytes();assert hashlib.sha256(raw).hexdigest()==settings['files_sha256'][name]
            payload[name]={'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()}
        code=render(CPU,{'CPU_ROOT':'/data/wio/inference-foundry/glm53/runs/'+settings['run_id'],'RID':settings['run_id'],'FILES':payload,'CLIENT':settings['client_settings']})+codec.ENCODER
        receipt['client_started']=True
        wire=invoke(43,code,1900)
        if wire['rc']==0:wire=codec.decode(wire)
        (run/'node-43.json').write_text(json.dumps(wire,indent=2)+'\n');receipt['node43_rc']=wire['rc']
        if wire['rc']==0:
            actual=json.loads(wire['stdout']);receipt['client_rc']=actual.get('client_rc');receipt['host_wait_timeout']=actual.get('host_wait_timeout',False)
            if 'report' in actual:
                raw=actual['report']['text'].encode();assert len(raw)==actual['report']['bytes'] and hashlib.sha256(raw).hexdigest()==actual['report']['sha256']
                (run/'diagnostic-result.remote.json').write_bytes(raw);report=json.loads(raw);receipt['backend_HTTP_attempts']=report['backend_HTTP_attempts'];receipt['phase']=report['phase']
    except BaseException as error:
        receipt['error']=repr(error);raise
    finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt));return receipt
