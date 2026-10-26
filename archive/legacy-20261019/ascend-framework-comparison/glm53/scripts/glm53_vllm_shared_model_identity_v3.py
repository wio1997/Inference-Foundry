"""Host-side guard for the real model child environment in vLLM1766 containers."""
HOST = r'''import subprocess,json,pathlib,hashlib
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0
d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running']
assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261017T1766-vllm-shared-pool-v4'
shared_root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261017T1766-vllm-shared-pool-v4')
supervisor=json.loads((shared_root/'supervisor-status.json').read_text());provider=json.loads((shared_root/'provider-status.json').read_text())
assert supervisor['phase']=='MODEL_RUNNING' and supervisor['model_started'] and 'model_exit_code' not in supervisor
assert provider['phase']=='READY'
top=subprocess.run(['docker','top',CID,'-eo','pid,args'],capture_output=True,text=True,timeout=20);assert top.returncode==0
matches=[]
for line in top.stdout.splitlines()[1:]:
 fields=line.split(None,1)
 if len(fields)!=2 or not fields[0].isdigit():continue
 pid=int(fields[0]);proc=pathlib.Path('/proc')/str(pid)
 try:nspid=next(row.split()[1:] for row in (proc/'status').read_text().splitlines() if row.startswith('NSpid:'))
 except (FileNotFoundError,StopIteration):continue
 if int(nspid[-1])==supervisor['model_pid']:
  assert CID in (proc/'cgroup').read_text()
  env=dict(item.split(b'=',1) for item in (proc/'environ').read_bytes().split(b'\0') if b'=' in item)
  selected={key:env.get(key.encode(),b'').decode() for key in ['PYTHONPATH','LD_LIBRARY_PATH','VLLM_HOST_IP','PYTHONHASHSEED','MMC_LOCAL_CONFIG_PATH','ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE']}
  assert selected['PYTHONHASHSEED']=='0' and selected['ASCEND_ENABLE_USE_FABRIC_MEM']=='1' and selected['ACL_OP_INIT_MODE']=='1'
  assert selected['PYTHONPATH'].startswith('/opt/glm53/sdk13:/opt/glm53/source/python:') and 'site-packages' in selected['PYTHONPATH']
  assert selected['MMC_LOCAL_CONFIG_PATH']=='/opt/glm53/config/local.conf'
  assert selected['VLLM_HOST_IP']=='172.16.12.'+str(NODE)
  assert selected['LD_LIBRARY_PATH'].startswith('/opt/glm53/sdk13/memfabric_hybrid/lib:/opt/glm53/sdk13/memcache_hybrid/lib:')
  matches.append({'host_pid':pid,'namespace_pid':supervisor['model_pid'],'selected_environment':selected})
assert len(matches)==1,('Cannot identify exact live native model parent',top.stdout,supervisor)
model_identity=matches[0]
'''
