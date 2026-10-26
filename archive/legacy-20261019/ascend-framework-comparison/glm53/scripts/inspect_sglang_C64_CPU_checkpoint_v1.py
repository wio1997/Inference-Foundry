"""One bounded CPU43 client checkpoint; no model/SDK requests or changes."""
import ast,hashlib,importlib.util,json,re,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RID='20261017T2012-sglang-C64-CPU-checkpoint-v1'
def load(name,file,sha):
 p=ROOT/file;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
base=load('inventory','scripts/inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
codec=load('codec','scripts/glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
HOST=r"""import pathlib,subprocess,json,hashlib,gzip,time,collections,socket
assert socket.gethostname()=='B7-23U-Atlas900RCKA3-43'
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T2000-sglang-formal-C64-L04-v2');assert root.is_dir()
archive=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T2012-sglang-C64-CPU-checkpoint-v1');assert not archive.exists();archive.mkdir(parents=True)
out={'project_id':'glm53','model_requests':0,'SDK_init_calls':0,'model_API_requests':0,'node':43,'monotonic_ns':time.monotonic_ns(),'files':[]}
for folder in [root,root/'output']:
 paths=sorted(folder.iterdir()) if folder.is_dir() else [];assert len(paths)<=128
 for p in paths:
  if p.is_file():st=p.stat();out['files'].append({'path':str(p.relative_to(root)),'bytes':st.st_size,'same_host_mtime_age_seconds':(time.time_ns()-st.st_mtime_ns)/1e9})
status=root/'controller-status.json'
if status.is_file():
 raw=status.read_bytes();assert len(raw)<=65536;out['host_controller_status']=json.loads(raw)
fmt='{"Id":{{json .Id}},"Name":{{json .Name}},"Image":{{json .Image}},"State":{{json .State}},"Labels":{{json .Config.Labels}}}'
p=subprocess.run(['docker','inspect',CNAME,'--format',fmt],capture_output=True,text=True,timeout=20);out['client_docker_inspect_rc']=p.returncode
if p.returncode==0:
 info=json.loads(p.stdout);assert info['Name']=='/'+CNAME and info['Image']=='sha256:008d2ebcd28a18f4dfbc855d23d763a2bdc63c4ce82c6b7b761580ad6d15d0a7' and info['Labels']['project_id']=='glm53' and info['Labels']['run_id']=='20261017T2000-sglang-formal-C64-L04-v2';assert (root/'client.cid').read_text().strip()==info['Id'];out['CPU_client']=info
 if info['State']['Running']:
  pid=info['State']['Pid'];proc=pathlib.Path('/proc')/str(pid);assert info['Id'] in (proc/'cgroup').read_text()
  def sample():
   raw=(proc/'stat').read_text();fields=raw.rsplit(')',1)[1].split();return {'monotonic_ns':time.monotonic_ns(),'state':fields[0],'user_ticks':int(fields[11]),'system_ticks':int(fields[12])}
  try:out['CPU_process_samples']=[sample()];time.sleep(1);out['CPU_process_samples'].append(sample())
  except FileNotFoundError:out['CPU_process_exited_during_sample']=True
else:out['client_inspect_stderr']=p.stderr[:4096]
log=root/'client-stdout.txt'
if log.is_file():
 size=log.stat().st_size;assert size<=64*1024**2;raw=log.open('rb').read(size);rows=[];non_event_lines=0
 for line in raw.splitlines(keepends=True):
  if not line.endswith(b'\n'):continue
  if not line.startswith(b'GLM53_SGLANG_GATEWAY_EVENT='):non_event_lines+=1;continue
  row=json.loads(line.split(b'=',1)[1]);assert row.get('namespace',NAMESPACE)==NAMESPACE;rows.append(row)
 counts=collections.Counter(('warm' if row.get('input_tokens')==74400 else 'formal' if row.get('input_tokens')==80000 else 'unknown',row.get('status')) for row in rows)
 out['client_gateway_terminal_counts_diagnostic_only']=[{'phase':phase,'status':status,'count':count} for (phase,status),count in sorted(counts.items())];out['complete_gateway_event_rows']=len(rows);out['stdout_bytes_read']=len(raw);out['stdout_complete_last_line']=not raw or raw.endswith(b'\n');out['non_event_stdout_lines']=non_event_lines
 if rows:out['same_host_seconds_since_last_gateway_terminal']=(time.monotonic_ns()-max(row['gateway_terminal_monotonic_ns'] for row in rows))/1e9
result=root/'output/formal-result.json'
if result.is_file():
 raw=result.read_bytes();assert len(raw)<=64*1024**2;r=json.loads(raw);out['completed_client_report']={k:r.get(k) for k in ['project_id','framework','concurrency','phase','baseline_accepted','backend_HTTP_attempts','summary']}
"""
def render(settings):
 values={'CNAME':settings['client_container_name'],'NAMESPACE':settings['namespace']};code=re.sub(r'\b(?:CNAME|NAMESPACE)\b',lambda m:repr(values[m[0]]),HOST)+codec.ENCODER;ast.parse(code);return code

def main():
 p=ROOT/'config/optimization-overlays/routing-metadata-v1/formal-C64-v2.json';assert hashlib.sha256(p.read_bytes()).hexdigest()=='a714ac75525bdd00812d757dcb8233481298d94d2748d3e914f3f6ed34c25761';settings=json.loads(p.read_text());client=ROOT/'runs'/settings['run_id'];run=ROOT/'runs'/RID;run.mkdir(exist_ok=True)
 with (run/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'model_requests':0,'SDK_init_calls':0,'nodes':{}}
 try:
  rpc_path=client/'controller-rpc-state.json'
  if rpc_path.exists():
   rpc=json.loads(rpc_path.read_text())
   if rpc['done_received'] and rpc['connection_rc'] is not None:receipt['skipped_client_already_complete']=True;return
  gate=json.loads((client/'cache-gate-native.json').read_text());assert gate['rc']==json.loads(gate['stdout'])['query_rc']==0
  for node in [34,35,36,41]:assert json.loads((client/f'preformal-context-{node}.json').read_text())['rc']==0
  pod,hostname=base.NODES[43];mapping=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert mapping['rc']==0 and mapping['stdout']=='172.16.12.43'
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-']);wire=base.invoke([command],render(settings),timeout=180)
  result=codec.decode(wire) if wire['rc']==0 else wire
  (run/'node-43.json').write_text(json.dumps(result,indent=2)+'\n');receipt['nodes']['43']=result['rc'];print('CPU43_CHECKPOINT_RC='+str(result['rc']),flush=True)
 except Exception as error:receipt['error']=repr(error);raise
 finally:(run/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
