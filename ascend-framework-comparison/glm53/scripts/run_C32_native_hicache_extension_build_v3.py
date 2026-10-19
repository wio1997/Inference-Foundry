"""Exact-once CPU-only private SDK optional-extension build via ZCode."""
import ast,base64,hashlib,importlib.util,json,os,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261018T-L12-native-HiCache-extension-build-v3'
HOST=r'''
import pathlib,subprocess,socket,json,hashlib,base64,shutil
assert socket.gethostname()==HOSTNAME
p=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert p.returncode==0;d=json.loads(p.stdout)[0]
assert d['Id']==CID and d['Image']==IMAGE and d['State']['Running'] and d['Config']['Labels']['run_id']==SHARED
def owned():
 p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=30);assert p.returncode==0;rows=[]
 for line in p.stdout.split('Process id',1)[-1].splitlines():
  cols=[x.strip() for x in line.split('|')]
  if len(cols)>=5 and len(cols[1].split())==2 and all(x.isdigit() for x in cols[1].split()) and cols[2].isdigit():
   proc=pathlib.Path('/proc')/cols[2];assert CID in (proc/'cgroup').read_text();rows.append(int(cols[2]))
 return sorted(rows)
before=owned();assert len(before)==18
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs')/RID;assert not root.exists();assert shutil.disk_usage(root.parent).free>=4*1024**3;root.mkdir()
raw=base64.b64decode(BUILD_SOURCE,validate=True);assert hashlib.sha256(raw).hexdigest()==BUILD_SHA;(root/'build.py').write_bytes(raw)
args=['docker','run','--rm','--pull=never','--network=none','--read-only','--pids-limit=2048','--memory=16g','--cpus=4','--tmpfs','/tmp:rw,size=4g','--name','glm53-C32-native-HiCache-extension-build-v3-20261018-34','--label','project_id=glm53','--label','run_id='+RID,'-v',str(root)+':/run/glm53','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro']
env={'PYTHONDONTWRITEBYTECODE':'1','ASCEND_VISIBLE_DEVICES':'','ASCEND_RT_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','XDG_CACHE_HOME':'/run/glm53/cache','TORCH_HOME':'/run/glm53/torch','MPLCONFIGDIR':'/run/glm53/mpl'}
for k,v in env.items():args+=['-e',k+'='+v]
args+=['--entrypoint','/usr/bin/timeout',IMAGE,'--signal=TERM','--kill-after=15s','900s','python3','/run/glm53/build.py']
p=subprocess.run(args,capture_output=True,timeout=950);(root/'container.stdout').write_bytes(p.stdout);(root/'container.stderr').write_bytes(p.stderr)
out={'project_id':'glm53','container_rc':p.returncode,'argv':args,'model_requests':0,'SDK_init_calls':0,'system_SDK_changed':False,'owned_context_PIDs_before':before,'owned_context_PIDs_after':owned(),'full_stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'full_stderr_sha256':hashlib.sha256(p.stderr).hexdigest(),'stdout':p.stdout.decode(errors='replace'),'stderr':p.stderr.decode(errors='replace'),'private_remote_root':str(root)}
out['original_NPU_contexts_unchanged']=out['owned_context_PIDs_before']==out['owned_context_PIDs_after']
assert len(p.stdout)+len(p.stderr)<=8*1024**2
if (root/'build-result.json').is_file():out['build_result']=json.loads((root/'build-result.json').read_text())
out['pass']=p.returncode==0 and out.get('build_result',{}).get('pass') is True and out['original_NPU_contexts_unchanged']
print(json.dumps(out))
'''
def plan():
    settings=json.loads((ROOT/'config/C32-target-graph-runtime-v3/P-two-step-diagnostic-v2.json').read_text());item=settings['runtime_containers']['34'];source=(ROOT/'scripts/build_C32_native_hicache_extension_cpu_v3.py').read_bytes()
    assert hashlib.sha256(source).hexdigest()=='6f7b9004300ea304c25046adf4971a8b49ac88ca22952160dda854e168cad4f1'
    values={'HOSTNAME':'B5-15U-Atlas900RCKA3-34','CID':item['container_id'],'IMAGE':settings['model_image'],'SHARED':settings['shared_run_id'],'RID':RID,'BUILD_SOURCE':base64.b64encode(source).decode(),'BUILD_SHA':hashlib.sha256(source).hexdigest()}
    code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),HOST);ast.parse(code);return code,values
def main():
    prior=json.loads((ROOT/'runs/20261018T-L12-C32-native-HiCache-NPU-fixture-v1/node-34.json').read_text());prior=json.loads(prior['stdout']);assert prior['fixture_rc']==1 and not prior['pass'] and prior['original_NPU_contexts_unchanged'] and 'file too short' in prior['stdout']
    path=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(path.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777';spec=importlib.util.spec_from_file_location('inventory',path);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        code,values=plan();(run/'reviewed-host-code.py').write_text(code)
        pod=b.NODES[34][0];mapped=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.34'
        cmd=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);result=b.invoke([cmd],code,timeout=1020);(run/'node-34.json').write_text(json.dumps(result,indent=2)+'\n');receipt['node_rc']=result['rc']
        if result['rc']==0:
            obj=json.loads(result['stdout']);receipt.update(pass_=obj['pass'],container_rc=obj['container_rc'],original_NPU_contexts_unchanged=obj['original_NPU_contexts_unchanged']);receipt['pass']=receipt.pop('pass_')
    finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
