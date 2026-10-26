"""Host-side guard for the real model child environment in C32L04restored2029 metadata containers."""
HOST = r'''import subprocess,json,pathlib,hashlib
r=subprocess.run(['docker','inspect',CID],capture_output=True,text=True,timeout=20);assert r.returncode==0
d=json.loads(r.stdout)[0]
assert d['Id']==CID and d['Name']=='/'+NAME and d['Image']==IMAGE and d['State']['Running']
assert d['Config']['Labels']['project_id']=='glm53' and d['Config']['Labels']['run_id']=='20261018T-L10-sglang-C32-target-graph-fullmesh-v3'
shared_root=pathlib.Path('/data/wio/inference-foundry/glm53/runs/20261018T-L10-sglang-C32-target-graph-fullmesh-v3')
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
  selected={key:env.get(key.encode(),b'').decode() for key in ['DEEP_USE_MODE','PYTHONPATH','ASCEND_MF_TRANSFER_PROTOCOL','SGLANG_MOONCAKE_SEND_AUX_TCP','PYTHONHASHSEED','MMC_LOCAL_CONFIG_PATH','ASCEND_ENABLE_USE_FABRIC_MEM','ACL_OP_INIT_MODE','GLM53_DEEPEP_METADATA_BATCH','GLM53_NODE','GLM53_BUFFER_DEDUP_VALIDATE','GLM53_PACKED_DISPATCH','HCCL_BUFFSIZE','DEEPEP_HCCL_BUFFSIZE','TASK_QUEUE_ENABLE','SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK','GLM53_C32_TARGET_GRAPH_ALIGN','GLM53_C32_OPS_AUDIT']}
  assert not selected['GLM53_PACKED_DISPATCH'] and selected['HCCL_BUFFSIZE']==('256' if NODE in [36,41] else '')
  assert selected['DEEPEP_HCCL_BUFFSIZE']=='' and selected['GLM53_C32_OPS_AUDIT']==''
  assert selected['TASK_QUEUE_ENABLE']==('0' if NODE in [36,41] else '')
  assert selected['SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK']==('32' if NODE in [36,41] else '')
  assert selected['GLM53_C32_TARGET_GRAPH_ALIGN']==('1' if NODE in [36,41] else '')
  assert selected['DEEP_USE_MODE']==('ops' if NODE in [36,41] else 'alltoall') and selected['ASCEND_MF_TRANSFER_PROTOCOL']=='sdma' and selected['SGLANG_MOONCAKE_SEND_AUX_TCP']=='0'
  assert selected['PYTHONHASHSEED']=='0' and selected['ASCEND_ENABLE_USE_FABRIC_MEM']=='1' and selected['ACL_OP_INIT_MODE']=='1'
  assert selected['PYTHONPATH'].startswith('/opt/glm53/source/python:') and 'site-packages' in selected['PYTHONPATH']
  assert selected['MMC_LOCAL_CONFIG_PATH']=='/opt/glm53/source/config/local.conf'
  node=provider['configuration']['node']
  assert selected['GLM53_DEEPEP_METADATA_BATCH']==('1' if node in [36,41] else '')
  assert selected['GLM53_NODE']==(str(node) if node in [36,41] else '')
  assert selected['GLM53_BUFFER_DEDUP_VALIDATE']==''
  argv=[x for x in (proc/'cmdline').read_bytes().decode().split('\0') if x]
  def option(name):assert name in argv;return argv[argv.index(name)+1]
  assert option('--model-path')=='/data/wio/models/GLM-5.3-w8a8' and option('--quantization')=='modelslim' and option('--dtype')==option('--kv-cache-dtype')=='bfloat16'
  assert option('--tp-size')==option('--ep-size')=='32' and option('--dp-size')==('16' if NODE in [36,41] else '4')
  assert option('--speculative-num-steps')=='5' and option('--speculative-num-draft-tokens')=='6' and option('--max-running-requests')==('64' if NODE in [36,41] else '32')
  assert '--disable-overlap-schedule' in argv and '--disable-prefill-cuda-graph' in argv
  if NODE in [36,41]:
   assert option('--deepep-mode')=='low_latency' and option('--cuda-graph-backend-decode')=='full' and option('--cuda-graph-max-bs-decode')=='4' and '--disable-decode-cuda-graph' not in argv
  else:assert option('--deepep-mode')=='normal' and '--disable-decode-cuda-graph' in argv
  matches.append({'host_pid':pid,'namespace_pid':supervisor['model_pid'],'selected_environment':selected,'actual_model_argv':argv})
assert len(matches)==1,('Cannot identify exact live native model parent',top.stdout,supervisor)
model_identity=matches[0]
'''
