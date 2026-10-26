"""Recover an already-linked optional extension; CPU loader audit, no rebuild."""
import ast,base64,gzip,hashlib,importlib.util,json,os,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261018T-L12-native-HiCache-built-artifact-recovery-v1'
PROBE=r'''
import pathlib,subprocess,json,os,importlib.util,sys
p=pathlib.Path('/run/glm53/extension/libmf_hybm_accoffload.so');out={}
for name,args in [('readelf',['readelf','-d',str(p)]),('symbols',['nm','-D','--defined-only',str(p)]),('ldd_original',['ldd',str(p)])]:
 r=subprocess.run(args,capture_output=True,text=True,timeout=30);out[name]={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
home=pathlib.Path(ASCEND_HOME);assert str(home).startswith('/usr/local/Ascend/') and home.is_dir()
dirs=[]
for name in ['torch','torch_npu']:
 spec=importlib.util.find_spec(name);assert spec and spec.origin;dirs.append(str(pathlib.Path(spec.origin).parent/'lib'))
dirs += [str(home/'lib64'),str(home/'aarch64-linux/lib64'),'/usr/local/Ascend/driver/lib64/driver','/usr/local/Ascend/driver/lib64/common']
assert all('devlib' not in d for d in dirs)
env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(dirs)+':'+env.get('LD_LIBRARY_PATH','')
r=subprocess.run(['ldd',str(p)],env=env,capture_output=True,text=True,timeout=30);out['ldd_native_library_directories']={'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH']}
assert not pathlib.Path('/dev/davinci_manager').exists() and not list(pathlib.Path('/dev').glob('davinci[0-9]*'))
assert not any(n.startswith(('torch','memfabric_hybrid','memcache_hybrid')) for n in sys.modules)
out['model_requests']=out['SDK_init_calls']=0;out['NPU_devices_exposed']=False
out['pass']=out['readelf']['rc']==out['symbols']['rc']==r.returncode==0 and all(name in out['symbols']['stdout'].split() for name in ['AccOffloadSparseCopy','AccOffloadGroupPackCopy','AccOffloadKvExchange','AccOffloadEntryGather']) and 'not found' not in r.stdout
print(json.dumps(out))
'''
HOST=r'''
import pathlib,json,subprocess,socket,hashlib,base64,gzip
assert socket.gethostname()==HOSTNAME
p=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert p.returncode==0;d=json.loads(p.stdout)[0]
assert d['Id']==CID and d['Image']==IMAGE and d['State']['Running'] and d['Config']['Labels']['run_id']==SHARED
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L12-native-HiCache-extension-build-v3')
result=json.loads((root/'build-result.json').read_text());assert result['stages'] and all(x['rc']==0 for x in result['stages']) and result['SDK_package_files_unchanged'] and result['measured_ABI_flag']=='-D_GLIBCXX_USE_CXX11_ABI=1'
lib=root/'extension/libmf_hybm_accoffload.so';raw=lib.read_bytes();assert 4096<len(raw)<=64*1024**2 and raw[:4]==b'\x7fELF' and raw[18:20]==(183).to_bytes(2,'little')
ascend=[arg.split('=',1)[1] for arg in result['stages'][0]['actual_argv'] if arg.startswith('-DASCEND_HOME_PATH=')];assert len(ascend)==1
probe=PROBE.replace('ASCEND_HOME',repr(ascend[0]))
args=['docker','run','--rm','--pull=never','--read-only','--network=none','--cpus=2','--memory=4g','--pids-limit=256','--tmpfs','/tmp:rw,size=1g','--label','project_id=glm53','--label','run_id='+RID,'--name','glm53-C32-native-HiCache-loader-audit-v1-20261018-34','-v',str(root)+':/run/glm53:ro','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','-e','PYTHONDONTWRITEBYTECODE=1','-e','ASCEND_VISIBLE_DEVICES=','-e','ASCEND_RT_VISIBLE_DEVICES=','--entrypoint','python3',IMAGE,'-c',probe]
p=subprocess.run(args,capture_output=True,text=True,timeout=150);out={'project_id':'glm53','node':34,'container_id':CID,'CPU_probe_rc':p.returncode,'CPU_probe_stdout':p.stdout,'CPU_probe_stderr':p.stderr,'library':{'remote_path':str(lib),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'gzip_base64':base64.b64encode(gzip.compress(raw,mtime=0)).decode()},'original_build_result_sha256':hashlib.sha256((root/'build-result.json').read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'recompiled':False,'existing_files_modified':False}
assert lib.read_bytes()==raw
if p.returncode==0:out['CPU_loader_audit']=json.loads(p.stdout)
print(json.dumps(out))
'''
def plan():
    cfg=json.loads((ROOT/'config/C32-target-graph-runtime-v3/P-two-step-diagnostic-v2.json').read_text());item=cfg['runtime_containers']['34']
    values={'HOSTNAME':'B5-15U-Atlas900RCKA3-34','CID':item['container_id'],'IMAGE':cfg['model_image'],'SHARED':cfg['shared_run_id'],'PROBE':PROBE,'RID':RID}
    code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),HOST);ast.parse(code);return code,values
def main():
    prior=json.loads(json.loads((ROOT/'runs/20261018T-L12-native-HiCache-extension-build-v3/node-34.json').read_text())['stdout']);assert prior['build_result']['stages'] and all(x['rc']==0 for x in prior['build_result']['stages']) and prior['original_NPU_contexts_unchanged'] and prior['build_result']['SDK_package_files_unchanged']
    p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777';spec=importlib.util.spec_from_file_location('inventory',p);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    run=ROOT/'runs'/RID;run.mkdir(exist_ok=False);(run/'execution-once.lock').write_text(str(os.getpid()));receipt={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'recompiled':False,'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        code,values=plan();(run/'reviewed-host-code.py').write_text(code);pod=b.NODES[34][0];mapped=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.34'
        cmd=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);wire=b.invoke([cmd],code,timeout=190);(run/'node-34.json').write_text(json.dumps(wire,indent=2)+'\n');receipt['node_rc']=wire['rc']
        if wire['rc']==0:
            obj=json.loads(wire['stdout']);blob=obj['library'];raw=gzip.decompress(base64.b64decode(blob['gzip_base64'],validate=True));assert len(raw)==blob['bytes'] and hashlib.sha256(raw).hexdigest()==blob['sha256']
            artifacts=run/'artifacts';artifacts.mkdir();(artifacts/'libmf_hybm_accoffload.so').write_bytes(raw)
            receipt.update(library_bytes=len(raw),library_sha256=blob['sha256'],CPU_probe_rc=obj['CPU_probe_rc'],CPU_loader_audit_pass=obj.get('CPU_loader_audit',{}).get('pass',False))
    finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
