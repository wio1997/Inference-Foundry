"""Read current installed profiler/API source without importing SDK/frameworks."""
import ast,base64,gzip,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1];RUN=ROOT/'runs/20261017T1858-sglang-decode-cache-source-v1'
def load(name,file,sha):
 path=P.with_name(file);assert hashlib.sha256(path.read_bytes()).hexdigest()==sha;spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
b=load('inventory','inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
codec=load('codec','glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
PROBE="import hashlib,importlib.util,json,pathlib,sys\nassert FRAMEWORK=='sglang' and not any(x.startswith(('memcache_hybrid','memfabric_hybrid')) for x in sys.modules)\nspec=importlib.util.find_spec('sglang');assert spec.origin;root=pathlib.Path(spec.origin).resolve().parent\npaths={root/'srt/server_args.py'}\nfor path in (root/'srt').rglob('*.py'):\n if path.is_symlink() or path.stat().st_size>2*1024**2 or 'disaggregation' in path.relative_to(root).parts:continue\n text=path.read_text()\n if 'disaggregation_decode_enable_radix_cache' in text:paths.add(path)\nassert len(paths)<=24;entries=[];total=0\nfor path in sorted(paths):\n assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root)\n raw=path.read_bytes();total+=len(raw);assert len(raw)<=2*1024**2 and total<=8*1024**2\n entries.append({'path':str(path),'relative':str(path.relative_to(root)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source':raw.decode()})\nassert not any(x.startswith(('memcache_hybrid','memfabric_hybrid')) for x in sys.modules)\nprint(json.dumps({'project_id':'glm53','framework':FRAMEWORK,'packages':{'sglang':{'origin':spec.origin,'root':str(root),'files':entries}},'source_bytes':total,'SDK_imported':False,'SDK_init_calls':0,'model_requests':0,'profiling_started':False}))\n"
HOST=r'''import pathlib,subprocess,json,hashlib,gzip,base64
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0;d=json.loads(r.stdout)[0];assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running'] and d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED_RID
r=subprocess.run(['docker','exec','-i',CID,'env','PYTHONDONTWRITEBYTECODE=1','python3','-'],input=PROBE,capture_output=True,text=True,timeout=120);assert r.returncode==0,r.stderr
out=json.loads(r.stdout);out['node']=NODE;out['container_id']=CID;out['model_requests']=0;out['SDK_init_calls']=0;out['profiling_started']=False
shared=pathlib.Path('/data/wio/inference-foundry/glm53/runs')/SHARED_RID;state=json.loads((shared/'supervisor-status.json').read_text());assert state['phase']=='MODEL_RUNNING' and state['model_started'] and 'model_exit_code' not in state
rows=subprocess.run(['docker','top',CID,'-eo','pid'],capture_output=True,text=True,timeout=20);assert rows.returncode==0;parents=[]
for line in rows.stdout.splitlines()[1:]:
 if not line.strip().isdigit():continue
 pid=int(line.strip());proc=pathlib.Path('/proc')/str(pid)
 try:namespace=int(next(x.split()[-1] for x in (proc/'status').read_text().splitlines() if x.startswith('NSpid:')))
 except (OSError,StopIteration):continue
 if namespace==state['model_pid']:
  assert CID in (proc/'cgroup').read_text();environment=dict(item.split(b'=',1) for item in (proc/'environ').read_bytes().split(b'\0') if b'=' in item)
  parents.append({'host_pid':pid,'namespace_pid':namespace,'profile_only_environment':{name:environment.get(name.encode(),b'').decode() for name in ['SGLANG_PROFILE_V2','SGLANG_TORCH_PROFILER_DIR','VLLM_TORCH_PROFILER_DIR']}})
assert len(parents)==1;out['actual_model_profile_environment']=parents[0]

archive=pathlib.Path(ARCHIVE);assert not archive.exists();archive.mkdir(parents=True)
'''
def render(code,values):
 pattern=r'\b(?:'+'|'.join(re.escape(k) for k in values)+r')\b';code=re.sub(pattern,lambda match:repr(values[match[0]]),code);ast.parse(code);return code

def capture(framework,item):
 node=36;pod,host=b.NODES[node];mapped=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']==f'172.16.12.{node}'
 values={'CID':item['container_id'],'NAME':item['name'],'IMAGE':'sha256:'+('090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83' if framework=='vllm' else 'c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263'),'SHARED_RID':'20261017T1766-vllm-shared-pool-v4' if framework=='vllm' else '20261017T1720-sglang-shared-pool-v2','NODE':node,'PROBE':render(PROBE,{'FRAMEWORK':framework}),'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/20261017T1858-sglang-decode-cache-source-v1'}
 code='import socket\nassert socket.gethostname()=='+repr(host)+'\n'+render(HOST+codec.ENCODER,values)
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);response=b.invoke([command],code,timeout=180);return codec.decode(response)
def main():
 current=json.loads((ROOT/'runs/20261017T1870-vllm-formal-C32-v2/cache-gate-native.json').read_text());assert current['rc']==0 and json.loads(current['stdout'])['report']['pass'],'Wait forC32callbackcomplete before sourceSSH'
 gate=json.loads((ROOT/'runs/20261017T1860-vllm-formal-C16-v2/cache-gate-native.json').read_text());assert gate['rc']==0 and json.loads(gate['stdout'])['report']['pass'],'Wait until current C16 gate callback finished; no extra SSH during callback'
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as file:file.write(str(__import__('os').getpid()))
 state=json.loads((ROOT/'STATE.json').read_text());receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'nodes':{},'SDK_init_calls':0,'model_requests':0,'profiling_started':False}
 try:
  for framework,item in [('sglang',state['runtime_sglang_containers']['36'])]:
   response=capture(framework,item);(RUN/(framework+'.json')).write_text(json.dumps(response,indent=2)+'\n');receipt['nodes'][framework]=response['rc'];print(framework,response['rc'],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
