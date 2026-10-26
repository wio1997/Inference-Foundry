"""Bounded separate-Meta SDK check inside owned container; preserves live models."""
import ast, hashlib, importlib.util, json, re, shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RID='20261017T2020-sglang-memcache-LRU-native-v2'
CORE='scripts/glm53_memcache_LRU_native_fixture_v2.py'
CORE_SHA='e14ca45465bb3e4e9190497cc8cdf9157656f270b2484d63eaa2b47fb049cb04'
def load(name,path,sha):
 p=ROOT/path;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('inventory','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
codec=load('codec','scripts/glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
INNER=r'''import pathlib,os,subprocess,signal,json,time
root=pathlib.Path('/run/glm53/diagnostics/20261017T2020-memcache-LRU-native-v2');assert root.is_dir()
environment=os.environ.copy();environment['GLM53_LRU_SESSION']='20261017T2020-sglang-memcache-LRU-native-v2';environment.pop('MMC_LOCAL_CONFIG_PATH',None);environment.pop('MMC_META_CONFIG_PATH',None)
log=(root/'fixture-console.txt').open('wb')
child=subprocess.Popen(['python3',str(root/'fixture.py')],env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
result={'group_leader_namespace_pid':child.pid,'model_requests':0}
def members():
 found=[]
 for proc in pathlib.Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
   if int(fields[2])!=child.pid or fields[0]=='Z':continue
   assert int(fields[3])==child.pid
   environment_items=(proc/'environ').read_bytes().split(b'\0')
   assert b'GLM53_LRU_SESSION=20261017T2020-sglang-memcache-LRU-native-v2' in environment_items
   found.append(int(proc.name))
  except (FileNotFoundError,ProcessLookupError):continue
 return found
try:
 try:result['fixture_exit_code']=child.wait(timeout=220)
 except subprocess.TimeoutExpired:result['fixture_timeout']=True
finally:
 remaining=members();result['remaining_session_before_cleanup']=remaining
 if remaining:
  os.killpg(child.pid,signal.SIGTERM)
  deadline=time.monotonic()+8
  while members() and time.monotonic()<deadline:time.sleep(.2)
  if members():os.killpg(child.pid,signal.SIGKILL);result['owned_session_required_kill']=True
  try:child.wait(timeout=5)
  except subprocess.TimeoutExpired:pass
 log.close();result['remaining_session_after_cleanup']=members();assert not result['remaining_session_after_cleanup']
 (root/'session-receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
'''
HOST=r'''import pathlib,subprocess,json,gzip,hashlib,socket
assert socket.gethostname()==HOSTNAME
def cmd(args,timeout=30):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);return {'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
r=cmd(['docker','inspect',CID]);assert r['rc']==0;info=json.loads(r['stdout'])[0]
assert info['Id']==CID and info['Image']==IMAGE and info['State']['Running'] and info['Config']['Labels']['project_id']=='glm53' and info['Config']['Labels']['run_id']=='20261017T1954-sglang-metadata-v1'
shared=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1954-sglang-metadata-v1')
state=json.loads((shared/'supervisor-status.json').read_text());assert state['phase']=='MODEL_RUNNING' and 'model_exit_code' not in state
for port in [29880,29881,29882]:
 sock=socket.socket();sock.bind(('0.0.0.0',port));sock.close()
archive=pathlib.Path(OUTDIR);assert not archive.exists();archive.mkdir(parents=True)
root=shared/'diagnostics/20261017T2020-memcache-LRU-native-v2';assert not root.exists();root.mkdir(parents=True)
(root/'fixture.py').write_text(SOURCE);assert hashlib.sha256((root/'fixture.py').read_bytes()).hexdigest()==SOURCE_SHA
def owned_npu():
 r=cmd(['npu-smi','info']);assert r['rc']==0;rows=[]
 for line in r['stdout'].split('Process id',1)[-1].splitlines():
  cols=[x.strip() for x in line.split('|')]
  if len(cols)>=5 and len(cols[1].split())==2 and all(v.isdigit() for v in cols[1].split()) and cols[2].isdigit():
   pid=int(cols[2]);assert CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();rows.append(pid)
 assert len(rows)==18;return sorted(rows)
before=owned_npu()
process=subprocess.run(['docker','exec','-i',CID,'env','PYTHONDONTWRITEBYTECODE=1','python3','-'],input=INNER,capture_output=True,text=True,timeout=260)
out={'project_id':'glm53','model_requests':0,'experiment':RID,'execution':{'rc':process.returncode,'stdout':process.stdout[-8000:],'stderr':process.stderr[-8000:]},'NPU_process_pids_before':before,'NPU_process_pids_after':owned_npu(),'production_Meta_contacted':False,'separate_Meta_endpoints':['127.0.0.1:29880','127.0.0.1:29881','127.0.0.1:29882']}
assert out['NPU_process_pids_before']==out['NPU_process_pids_after']
out['case_results']=[]
for mode in ['control','exist_touch']:
 path=root/mode/'result.json'
 if path.is_file():out['case_results'].append(json.loads(path.read_text()))
for name in ['result.json','session-receipt.json']:
 path=root/name
 if path.is_file():out[name]=json.loads(path.read_text())
out['private_logs']=[]
for path in root.rglob('*.txt'):
 raw=path.open('rb').read(16*1024**2+1);assert len(raw)<=16*1024**2
 dest=archive/(str(path.relative_to(root)).replace('/','__')+'.gz');dest.write_bytes(gzip.compress(raw,mtime=0));out['private_logs'].append({'source':str(path),'archive':str(dest),'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'archive_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
 if path.name=='fixture-console.txt':out['fixture_console_tail']=raw[-12000:].decode(errors='replace')
after=cmd(['docker','inspect',CID]);assert after['rc']==0;now=json.loads(after['stdout'])[0];assert now['State']['Running'] and now['State']['StartedAt']==info['State']['StartedAt'];out['production_container_unchanged']=True
'''
def render(source):
 values={'HOSTNAME':base.NODES[34][1],'CID':'dc23a47a7050f1ad3fb958846fe7bec789a87ec7775f4e715536a6602a6c68d3','IMAGE':'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263','OUTDIR':'/data/wio/inference-foundry/glm53/runs/'+RID,'SOURCE':source,'SOURCE_SHA':CORE_SHA,'INNER':INNER,'RID':RID}
 code=re.sub(r'\b(?:'+'|'.join(values)+r')\b',lambda m:repr(values[m[0]]),HOST)+codec.ENCODER;ast.parse(code);ast.parse(INNER);return code
def main():
 source=(ROOT/CORE).read_text();assert hashlib.sha256(source.encode()).hexdigest()==CORE_SHA;ast.parse(source)
 prior=json.loads((ROOT/'runs/20261017T2019-sglang-memcache-LRU-CPU-v1/receipt.json').read_text());assert prior['nodes']=={'34':0}
 run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'core_sha256':CORE_SHA,'model_requests':0,'nodes':{}}
 try:
  pod,_=base.NODES[34];mapped=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapped['rc']==0 and mapped['stdout']=='172.16.12.34'
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);wire=base.invoke([command],render(source),timeout=320);result=codec.decode(wire) if wire['rc']==0 else wire
  (run/'node-34.json').write_text(json.dumps(result,indent=2)+'\n');receipt['nodes']['34']=result['rc'];print('NATIVE_LRU_CAPTURE_RC='+str(result['rc']),flush=True)
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
