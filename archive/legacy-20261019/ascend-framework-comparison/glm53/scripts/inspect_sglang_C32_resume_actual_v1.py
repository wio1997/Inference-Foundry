"""Post-SSH-recovery owned runtime/source/health/idle inventory; no SDK/model/NPU operations."""
import ast,concurrent.futures,hashlib,importlib.util,json,re,shlex
from pathlib import Path
P=Path(__file__).resolve();ROOT=P.parents[1];RID='20261018T1100-sglang-C32-resume-actual-v1';RUN=ROOT/'runs'/RID
def load(name,file,sha):
 p=P.with_name(file);assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('inventory','inventory_readonly.py','e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777')
codec=load('codec','glm53_native_output_transport_v1.py','ed265b69437cc93a9fe8afdc1ac87f151e8168e9c82ff28a5d374f14d0c622d5')
RUNTIME_PATH=ROOT/'config/C32-only-L04-restore/runtime.json';assert hashlib.sha256(RUNTIME_PATH.read_bytes()).hexdigest()=='1100bc9f5c4a938becd0233c00d513e2e1a3384adc95ea18f257f30f215fd526'
RUNTIME=json.loads(RUNTIME_PATH.read_text())
MANIFEST=ROOT/'config/sglang-buffer-dedup-rollback-v1-stage-v1.json';assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest()=='5ceb6e48c0948e671197e5409b3ad37c8f75b5088bf274fb40cb1fc2d860e189'
BUNDLE=json.loads(MANIFEST.read_text())['bundles'][0]
HOST=r'''import pathlib,subprocess,json,hashlib,urllib.request,socket,time,gzip,base64
assert socket.gethostname()==HOSTNAME
def command(args,timeout=30):
 r=subprocess.run(args,capture_output=True,text=True,timeout=timeout);assert r.returncode==0,(args,r.returncode,r.stderr);return r.stdout
d=json.loads(command(['docker','inspect',CID]))[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running'] and not d['HostConfig']['Privileged']
assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']==SHARED_RID
root=pathlib.Path('/data/wio/inference-foundry/glm53/runs')/SHARED_RID
supervisor=json.loads((root/'supervisor-status.json').read_text());provider=json.loads((root/'provider-status.json').read_text())
assert supervisor['phase']=='MODEL_RUNNING' and supervisor['model_started'] and 'model_exit_code' not in supervisor and provider['phase']=='READY'
for name,item in FILES.items():assert hashlib.sha256((pathlib.Path(STAGED)/name).read_bytes()).hexdigest()==item['sha256']
top=command(['docker','top',CID,'-eo','pid,args']);parents=[];container_pids=[]
for line in top.splitlines()[1:]:
 fields=line.split(None,1)
 if len(fields)!=2 or not fields[0].isdigit():continue
 pid=int(fields[0]);container_pids.append(pid);proc=pathlib.Path('/proc')/str(pid)
 try:namespace=int(next(x.split()[-1] for x in (proc/'status').read_text().splitlines() if x.startswith('NSpid:')))
 except (OSError,StopIteration):continue
 if namespace==supervisor['model_pid']:
  assert CID in (proc/'cgroup').read_text();environment=dict(item.split(b'=',1) for item in (proc/'environ').read_bytes().split(b'\0') if b'=' in item)
  selected={key:environment.get(key.encode(),b'').decode() for key in ['DEEP_USE_MODE','PYTHONPATH','ASCEND_MF_TRANSFER_PROTOCOL','SGLANG_MOONCAKE_SEND_AUX_TCP','PYTHONHASHSEED','MMC_LOCAL_CONFIG_PATH','ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE','GLM53_DEEPEP_METADATA_BATCH','GLM53_NODE','GLM53_BUFFER_DEDUP_VALIDATE','GLM53_C32_OPS_AUDIT','DEEPEP_HCCL_BUFFSIZE']}
  assert selected['DEEP_USE_MODE']=='alltoall' and selected['ASCEND_MF_TRANSFER_PROTOCOL']=='sdma' and selected['SGLANG_MOONCAKE_SEND_AUX_TCP']=='0' and selected['PYTHONHASHSEED']=='0' and selected['ASCEND_ENABLE_USE_FABRIC_MEM']==selected['ACL_OP_INIT_MODE']=='1'
  assert selected['GLM53_BUFFER_DEDUP_VALIDATE']==selected['GLM53_C32_OPS_AUDIT']==selected['DEEPEP_HCCL_BUFFSIZE']==''
  assert selected['GLM53_DEEPEP_METADATA_BATCH']==('1' if NODE in [36,41] else '')
  args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
  config=json.loads((pathlib.Path(STAGED)/('config/node-'+str(NODE)+'.json')).read_text());assert args[1:]==['-m','sglang.launch_server',*config['launch_args']],args
  parents.append({'host_pid':pid,'namespace_pid':namespace,'selected_environment':selected,'exact_frozen_launch_args_match':True})
assert len(parents)==1
npu=command(['npu-smi','info']);npu_rows=[]
for line in npu.split('Process id',1)[-1].splitlines():
 columns=[v.strip() for v in line.split('|')]
 if len(columns)>=5 and len(columns[1].split())==2 and all(v.isdigit() for v in columns[1].split()) and columns[2].isdigit():
  pid=int(columns[2]);assert pid in container_pids and CID in (pathlib.Path('/proc')/str(pid)/'cgroup').read_text();npu_rows.append(pid)
assert len(npu_rows)==18,('Unexpected NPU contexts; no mutations allowed',npu_rows)
out={'project_id':'glm53','node':NODE,'container_id':CID,'image':IMAGE,'shared_run_id':SHARED_RID,'supervisor':supervisor,'provider':provider,'actual_model_parent':parents[0],'NPU_owned_process_count':len(npu_rows),'npu_sha256':hashlib.sha256(npu.encode()).hexdigest(),'all26_staged_files_unchanged':True,'loadavg':pathlib.Path('/proc/loadavg').read_text(),'MemAvailable_bytes':int(next(x.split()[1] for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024,'model_requests':0,'SDK_init_calls':0,'service_mutations':0,'endpoints':{}}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));port=30500 if NODE in [34,35] else 30600
for endpoint in ['/health','/metrics']+(['/get_server_info'] if NODE in [34,36] else []):
 with opener.open('http://172.16.12.'+str(NODE)+':'+str(port)+endpoint,timeout=15) as response:raw=response.read(8*1024**2+1);assert response.status==200 and len(raw)<=8*1024**2
 entry={'status':200,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 if endpoint=='/metrics':
  gauges=[line for line in raw.decode().splitlines() if line.startswith(('sglang:num_running_reqs{','sglang:num_queue_reqs{'))];assert gauges and all(float(x.rsplit(' ',1)[1])==0 for x in gauges);entry['idle_gauges']=gauges
 if endpoint=='/get_server_info':
  data=json.loads(raw);entry['server_info']=data
  assert data['deepep_mode']=='normal' and data['speculative_num_steps']==5 and data['speculative_num_draft_tokens']==6 and data['max_total_num_tokens']==(254848 if NODE==34 else 181632)
 out['endpoints'][endpoint]=entry
if NODE==34:
 meta=json.loads(command(['docker','inspect',META]))[0];assert meta['Id']==META and meta['State']['Running'] and meta['Config']['Labels']['project_id']=='glm53' and meta['Config']['Labels']['run_id']==SHARED_RID
 out['meta']={'container_id':META,'running':True}
out['owned_running_containers']=command(['docker','ps','--filter','label=project_id=glm53','--no-trunc','--format','{{.ID}} {{.Names}} {{.Status}}'])
out['client_or_controller_locks']=[{'name':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in root.glob('*.once') if p.is_file()]
archive=pathlib.Path(ARCHIVE);assert not archive.exists();archive.mkdir(parents=True)
'''
def render(node):
 item=RUNTIME['runtime_containers'][str(node)]
 values={'HOSTNAME':base.NODES[node][1],'CID':item['container_id'],'NAME':item['name'],'IMAGE':RUNTIME['model_image'],'SHARED_RID':RUNTIME['shared_run_id'],'META':RUNTIME['meta_container_id'],'NODE':node,'FILES':BUNDLE['files'],'STAGED':BUNDLE['remote'],'ARCHIVE':'/data/wio/inference-foundry/glm53/runs/'+RID}
 pattern=r'\b(?:'+'|'.join(values)+r')\b';text=re.sub(pattern,lambda m:repr(values[m[0]]),HOST+codec.ENCODER);ast.parse(text);return text
def capture(node):
 pod,_=base.NODES[node];mapping=base.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30)
 (RUN/f'mapping-{node}.json').write_text(json.dumps(mapping,indent=2)+'\n');assert mapping['rc']==0 and mapping['stdout']==f'172.16.12.{node}'
 command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
 return codec.decode(base.invoke([command],render(node),timeout=150))
def main():
 recovery=Path('/Users/wio/work/sglang-glm53-33-41-deploy/ssh-control-recovery-20261018/RESULT.json');r=json.loads(recovery.read_text());assert r['phase']=='CONTROL_RESTORED_IDENTITY_VERIFIED' and r['hostname']=='A3-5U-Atlas900RCKA3-1' and r['model_requests']==r['remote_mutations']==0
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 receipt={'project_id':'glm53','launcher_sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'recovery_receipt_sha256':hashlib.sha256(recovery.read_bytes()).hexdigest(),'nodes':{},'model_requests':0,'SDK_init_calls':0,'service_mutations':0}
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
   pending={pool.submit(capture,n):n for n in [34,35,36,41]}
   for f in concurrent.futures.as_completed(pending):
    n=pending[f]
    try:result=f.result()
    except Exception as error:result={'error':repr(error)}
    (RUN/f'node-{n}.json').write_text(json.dumps(result,indent=2)+'\n');receipt['nodes'][str(n)]=result.get('rc',result.get('error'));print(n,receipt['nodes'][str(n)],flush=True)
 finally:(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 assert len(receipt['nodes'])==4 and all(x==0 for x in receipt['nodes'].values())
 print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
