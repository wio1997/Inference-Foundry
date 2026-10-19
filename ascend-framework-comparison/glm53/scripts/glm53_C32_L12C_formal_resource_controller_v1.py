"""Paired-stage orchestration: matching vLLM certificate + native SG qualification.

Only an independently reviewed per-run ZCode launcher may call execute(). All
native correctness certificates, CIDs, paths and SHA pins are caller-frozen.
"""
import ast,base64,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(name,path,sha):
    path=Path(path);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def render(template,values):
    pattern=r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b'
    result=re.sub(pattern,lambda m:repr(values[m[0]]),template);ast.parse(result);return result

def encoded(files):
    out={}
    for name,path in files.items():
        assert Path(name).name==name
        raw=Path(path).read_bytes();out[name]={'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()}
    return out

CPU_STAGE=r'''import pathlib,subprocess,json,base64,hashlib,socket
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
image='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7'
r=subprocess.run(['docker','image','inspect',image,'--format','{{.Id}}'],capture_output=True,text=True,timeout=20);assert r.returncode==0 and r.stdout.strip()==image
assets=pathlib.Path('/data/wio/aisbench-offline-43-20261014/benchmark/ais_bench/benchmark')
for item in AIS_IDENTITY['files']:assert hashlib.sha256((assets/item['path']).read_bytes()).hexdigest()==item['sha256']
data=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T0615-formal-inputs-v3/artifacts')/('C'+str(CONCURRENCY));assert hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()==MANIFEST_SHA
root=pathlib.Path(CPU_ROOT);assert not root.exists(),'No formal host-stage overwrite/replay';root.mkdir(parents=True)
for name,item in FILES.items():
 assert pathlib.Path(name).name==name
 raw=base64.b64decode(item['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==item['sha256'];(root/name).write_bytes(raw)
raw=json.dumps(SETTINGS,indent=2).encode()+b'\n';(root/'host-settings.json').write_bytes(raw)
print(json.dumps({'project_id':'glm53','node':43,'root':str(root),'settings_sha256':hashlib.sha256(raw).hexdigest(),'files_sha256':{k:v['sha256'] for k,v in FILES.items()},'model_requests':0,'SDK_init_calls':0}))
'''

def execute(settings):
    # No defaults for run identity or qualification; a launcher must freeze each.
    assert settings['framework']=='sglang' and settings['concurrency']==32
    assert settings['project_id']=='glm53' and settings['framework'] in ['vllm','sglang'] and settings['concurrency'] in [16,32,64]
    native_config=ROOT/settings['native_client_config_path'];assert hashlib.sha256(native_config.read_bytes()).hexdigest()==settings['native_client_config_sha256']
    native={line.split('=',1)[0].strip():line.split('=',1)[1].strip() for line in native_config.read_text().splitlines() if '=' in line and not line.lstrip().startswith('#')}
    for key,field in [('meta_service_url','ock.mmc.meta_service_url'),('config_store_url','ock.mmc.local_service.config_store_url'),('max_dram_size','ock.mmc.local_service.max.dram.size')]:assert settings[key]==native[field],('Formal gate must use frozen native runtime config',key)
    assert native['ock.mmc.local_service.protocol']=='device_sdma' and native['ock.mmc.local_service.world_size']=='128' and native['ock.mmc.local_service.dram.size']=='0GB'
    run=ROOT/'runs'/settings['run_id'];assert run.is_dir()
    assert settings['namespace'].startswith('glm53-formal-') and settings['run_id'] in settings['namespace'] and settings['namespace'].endswith('-C'+str(settings['concurrency']))
    base=load('inventory',ROOT/'scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
    rpc=load('formal_rpc',ROOT/'scripts/glm53_formal_rpc_v2.py','da3b46240572916e9b6433da4937ce33433a452912d2279e937df77fd57d8163')
    gate=load('formal_gate',ROOT/'scripts/glm53_formal_gate_remote_v1.py',settings['gate_renderer_sha256'])
    transfer=load('formal_transfer',ROOT/settings['prefix_transfer_helper'],settings['prefix_transfer_sha256'])
    identity=load('native_model_identity',ROOT/settings['model_identity_helper'],settings['model_identity_sha256'])
    certificate=ROOT/settings['native_qualification_path'];assert hashlib.sha256(certificate.read_bytes()).hexdigest()==settings['native_qualification_sha256']
    qualification=json.loads(certificate.read_text())
    assert qualification['pass'] is True and qualification['project_id']=='glm53'
    assert qualification['framework']==settings['framework'] and qualification['shared_run_id']==settings['shared_run_id']
    assert qualification['runtime_CIDs']=={n:item['container_id'] for n,item in settings['runtime_containers'].items()}
    assert settings['framework']=='sglang' and settings['execution_policy_revision']=='paired-stage-v1'
    required=[('vllm',settings['concurrency'])]+[('sglang',c) for c in [16,32,64] if c<settings['concurrency']]
    supplied=settings.get('predecessor_formal_certificates',{})
    assert set(supplied)=={fw+'-C'+str(c) for fw,c in required},'Only matching vLLM stage plus lower SG stages required'
    for prior_fw,prior_C in required:
        item=supplied[prior_fw+'-C'+str(prior_C)];path=ROOT/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        previous=json.loads(path.read_text());assert previous['project_id']=='glm53' and previous['framework']==prior_fw and previous['concurrency']==prior_C and previous['measurement_valid'] is True

    files={name:ROOT/path for name,path in settings['client_files'].items()}
    for name,path in files.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==settings['client_files_sha256'][name]
    gate_files={name:ROOT/path for name,path in settings['gate_files'].items()}
    for name,path in gate_files.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==settings['gate_files_sha256'][name]
    C=settings['concurrency'];fw=settings['framework'];rid=settings['run_id'];namespace=settings['namespace'];Pnode=30 if fw=='vllm' else 34
    cpu_root='/data/wio/inference-foundry/glm53/runs/'+rid
    def invoke(node,code,timeout=120):
        pod,hostname=base.NODES[node];mapping=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapping['rc']==0 and mapping['stdout']==f'172.16.12.{node}'
        code='import socket\nassert socket.gethostname()=='+repr(hostname)+'\n'+code;ast.parse(code)
        return base.invoke([shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])],code,timeout=timeout)
    def save(name,value):(run/name).write_text(json.dumps(value,indent=2)+'\n');return value
    precheck_codes={}
    receipt={'project_id':'glm53','framework':fw,'run_id':rid,'concurrency':C,'namespace':namespace,'model_precheck':{},'phase':'PRECHECK','native_baseline_acceptance':'PENDING_POST_RUN_NATIVE_AUDIT'}
    try:
        for n,item in settings['runtime_containers'].items():
            node=int(n)
            code=identity.render(node,{'providers':settings['runtime_containers'],'selected_variant':'candidate'})+r'''
import urllib.request
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(URL,timeout=10) as response:assert response.status==200;raw=response.read(8388609);assert len(raw)<=8388608
lines=raw.decode().splitlines();gauges=[float(l.rsplit(' ',1)[1]) for l in lines if l.startswith(GAUGE_PREFIXES)];assert gauges and all(x==0 for x in gauges),'Existing workload active; no formal warmup dispatch'
import pathlib
npu=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert npu.returncode==0
owned=[]
for line in npu.stdout.split('Process id',1)[-1].splitlines():
 cols=[v.strip() for v in line.split('|')]
 if len(cols)>=5 and len(cols[1].split())==2 and all(v.isdigit() for v in cols[1].split()) and cols[2].isdigit():
  pid=int(cols[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text(),'Foreign NPU process: record condition and do not benchmark'
  owned.append(pid)
assert owned
print(json.dumps({'project_id':'glm53','node':NODE,'model_identity':model_identity,'idle_native_gauges':gauges,'loadavg':pathlib.Path('/proc/loadavg').read_text(),'meminfo':pathlib.Path('/proc/meminfo').read_text(),'netdev':pathlib.Path('/proc/net/dev').read_text(),'owned_NPU_host_PIDs':owned,'native_NPU_snapshot':npu.stdout,'model_requests':0}))
'''
            roleP=node in ([30,31] if fw=='vllm' else [34,35]);port=(30300 if roleP else 30400) if fw=='vllm' else (30500 if roleP else 30600)
            values={'CID':item['container_id'],'NAME':item['name'],'IMAGE':settings['model_image'],'NODE':node,'URL':f'http://172.16.12.{node}:{port}/metrics','GAUGE_PREFIXES':('vllm:num_requests_running{','vllm:num_requests_waiting{') if fw=='vllm' else ('sglang:num_running_reqs{','sglang:num_queue_reqs{')}
            precheck_codes[node]=render(code,values)
            result=save(f'precheck-{node}.json',invoke(node,precheck_codes[node]));receipt['model_precheck'][n]=result['rc'];assert result['rc']==0
        receipt['phase']='STAGE'
        if settings.get('prefix_data_staged_receipt'):
            existing=ROOT/settings['prefix_data_staged_receipt'];assert hashlib.sha256(existing.read_bytes()).hexdigest()==settings['prefix_data_staged_receipt_sha256']
        else:
            for node in [43,Pnode]:
                pod=base.NODES[node][0];mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']==f'172.16.12.{node}'
            command,destination=transfer.render(fw,C,settings['input_manifest_sha256'],rid,base.NODES[43][0],base.NODES[Pnode][0])
            result=save('prefix-transfer.json',base.invoke([command],timeout=180));assert result['rc']==0;details=json.loads(result['stdout']);assert details['manifest_sha256']==settings['input_manifest_sha256'] and len(details['files'])==C+1
        gate_root='/run/glm53/formal-gates/'+rid+'/control'
        gate_settings={'framework':fw,'expected_sdk_root':settings['expected_sdk_root'],'output':gate_root,'dataset':'/run/glm53/formal-gates/'+rid+'/data','meta_service_url':settings['meta_service_url'],'config_store_url':settings['config_store_url'],'world_size':128,'protocol':'device_sdma','max_dram_size':settings['max_dram_size'],'device_id':0}
        gate_payload=encoded(gate_files);raw=json.dumps(gate_settings,indent=2).encode()+b'\n';gate_payload['gate-settings.json']={'base64':base64.b64encode(raw).decode(),'sha256':hashlib.sha256(raw).hexdigest()}
        pitem=settings['runtime_containers'][str(Pnode)]
        gate_values={'CID':pitem['container_id'],'NAME':pitem['name'],'IMAGE':settings['model_image'],'SHARED_RID':settings['shared_run_id'],'SHARED_ROOT':settings['shared_remote_root'],'RID':rid,'FRAMEWORK':fw,'NODE':Pnode,'MANIFEST_SHA':settings['input_manifest_sha256'],'FILES':gate_payload}
        result=save('gate-stage.json',invoke(Pnode,gate.render(gate.STAGE,gate_values)));assert result['rc']==0
        cpu_payload=encoded(files)
        host={'project_id':'glm53','framework':fw,'concurrency':C,'namespace':namespace,'run_id':rid,'root':cpu_root,'input_manifest_sha256':settings['input_manifest_sha256'],'files_sha256':{k:v['sha256'] for k,v in cpu_payload.items()},'container_name':settings['client_container_name'],'client_settings':{'framework':fw,'concurrency':C,'namespace':namespace,'run_id':rid,'dataset':'/formal-inputs','output':'/run/glm53/output'}}
        ais_path=ROOT/'runs/20261017T0610-aisbench-assets-v1/identity-summary.json';assert hashlib.sha256(ais_path.read_bytes()).hexdigest()=='7cf76483c63972186dc4c589fe04e4d8c49d8fa8eb020aab49fc4a10b955bb8f'
        result=save('CPU-stage.json',invoke(43,render(CPU_STAGE,{'AIS_IDENTITY':json.loads(ais_path.read_text()),'CONCURRENCY':C,'MANIFEST_SHA':settings['input_manifest_sha256'],'CPU_ROOT':cpu_root,'FILES':cpu_payload,'SETTINGS':host})));assert result['rc']==0;host_details=json.loads(result['stdout'])
        # Hold one SSH only; the gate callback may use at most one additional SSH.
        mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(base.NODES[43][0])+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.43'
        command=base.SSH+[shlex.join(['kubectl','exec','-i','-n','default',base.NODES[43][0],'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-u',cpu_root+'/glm53_formal_host_controller_v2.py','--settings',cpu_root+'/host-settings.json','--sha256',host_details['settings_sha256']])]
        def on_gate(request):
            # Recheck the actual background and owned runtime after all real
            # serial warmups, immediately before readback and the formal clock.
            for node,code in precheck_codes.items():
                checked=save(f'preformal-context-{node}.json',invoke(node,code));assert checked['rc']==0
            values=gate_values|{'REQUEST':request,'NAMESPACE':namespace,'CONCURRENCY':C,'SDK_PREFIX':'/opt/glm53/sdk13:' if fw=='vllm' else '', 'LIB_PREFIX':'export LD_LIBRARY_PATH=/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:${LD_LIBRARY_PATH:-}; ' if fw=='vllm' else ''}
            result=save('cache-gate-native.json',invoke(Pnode,gate.render(gate.INVOKE,values),280))
            payload=json.loads(result['stdout']);assert result['rc']==0 and payload['query_rc']==0;report=payload['report'];assert hashlib.sha256((json.dumps(report,indent=2)+'\n').encode()).hexdigest()==payload['report_sha256']
            return report
        receipt['phase']='REAL_WARMUP_AND_FORMAL_CLIENT';save('resource-controller-status.json',receipt)
        result=rpc.rpc(command,run,{'project_id':'glm53','framework':fw,'namespace':namespace,'concurrency':C,'manifest_sha256':settings['input_manifest_sha256']},on_gate);receipt['RPC']=result;receipt['phase']='CLIENT_COMPLETE_NATIVE_POST_AUDIT_REQUIRED'
    except BaseException as error:receipt['error']=repr(error);raise
    finally:save('resource-controller-receipt.json',receipt)
    return receipt
