# SPDX-License-Identifier: Apache-2.0
import pathlib,json,subprocess,os,socket,time,hashlib,base64,traceback
node=__NODE__;assert node in (14,15) and socket.gethostname()=={14:'A6-15U-Atlas900RCKA3-14',15:'A6-23U-Atlas900RCKA3-15'}[node]
R=pathlib.Path('/data/wio/inference-foundry/glm53-flash/runs/20261018-r157-native-smem-runtime-alias-cache-sdma');R.mkdir(parents=True,exist_ok=False);os.close(os.open(R/'once.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600));files=__FILES__
name='if-glm53-flash-r157-cache-sdma-'+str(node);image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83';out={'project_id':'glm53-flash','node':node,'image':image,'generation_requests':0};created=False;meta_created=False;meta_name='if-glm53-flash-r157-cpu-meta'
def run(args,t=30):
 q=subprocess.run(args,capture_output=True,text=True,timeout=t);return {'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
def save():(R/'host-status.json').write_text(json.dumps(out,indent=2))
try:
 out['pre_npu']=run(['npu-smi','info']);assert out['pre_npu']['rc']==0 and out['pre_npu']['stdout'].count('No running processes found')==8
 out['image_check']=run(['docker','image','inspect',image,'--format','{{.Id}}']);assert out['image_check']['stdout'].strip()==image
 for port in ([19048,19052,19053,19054,18500] if node==14 else []):
  with socket.socket() as sock:sock.bind(('172.16.12.'+str(node),port))
 for filename,entry in files.items():
  p=pathlib.PurePosixPath(filename);assert not p.is_absolute() and '..' not in p.parts;b=base64.b64decode(entry['data']);assert hashlib.sha256(b).hexdigest()==entry['sha256'];(R/filename).write_bytes(b)
 if node==14:
  meta_args=['docker','run','-d','--pull=never','--name',meta_name,'--label','project_id=glm53-flash','--label','run_id=r157','--label','component=cpu-meta','--network=host','--ipc=private','--shm-size=2g','--cpus=4','--memory=16g','--pids-limit=256','--ulimit','core=0:0','--log-opt','max-size=64m','--log-opt','max-file=2','-v',str(R)+':/run-evidence','-e','MMC_META_CONFIG_PATH=/run-evidence/mmc-meta.conf','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','HF_HUB_OFFLINE=1','--entrypoint','python3',image,'/run-evidence/meta_child.py'];out['CPU_meta_command']=meta_args;out['CPU_meta_launch']=run(meta_args,60);assert out['CPU_meta_launch']['rc']==0;meta_created=True;save();deadline=time.monotonic()+45
  while True:
   q=run(['docker','inspect',meta_name]);assert q['rc']==0;v=json.loads(q['stdout'])[0];assert v['State']['Running'],('CPU_meta_exited',v['State']['ExitCode']);pid=v['State']['Pid'];inodes=set()
   for f in pathlib.Path('/proc/'+str(pid)+'/fd').iterdir():
    try:target=os.readlink(f)
    except FileNotFoundError:continue
    if target.startswith('socket:['):inodes.add(target[8:-1])
   listeners={}
   for table in ['/proc/net/tcp','/proc/net/tcp6']:
    for line in pathlib.Path(table).read_text().splitlines()[1:]:
     f=line.split();port=int(f[1].split(':')[-1],16)
     if port in [19052,19053,19054] and f[3]=='0A' and f[9] in inodes:listeners[str(port)]=f[9]
   if set(listeners)=={'19052','19053','19054'}:
    (R/'meta-ready.json').write_text(json.dumps({'project_id':'glm53-flash','node':14,'CPU_meta_no_NPU_devices':True,'host_pid':pid,'passive_owned_LISTEN':listeners,'TCP_probe_connections':0}));out['CPU_meta_passive_ready']=True;break
   if time.monotonic()>deadline:raise TimeoutError('CPU_meta_owned_LISTEN')
   time.sleep(.5)
 args=['docker','run','-d','--pull=never','--name',name,'--label','project_id=glm53-flash','--label','run_id=r157','--log-opt','max-size=64m','--log-opt','max-file=2','--network=host','--ipc=host','--ulimit','memlock=-1:-1','--ulimit','core=0:0','--memory=16g','--cpus=4','--pids-limit=512','-v',str(R)+':/run-evidence']
 for f,target in {'flash_native_sdk_bindings.py':'flash_native_sdk_bindings.py','vllm_flash_sdma_performance_engine.py':'vllm_flash_sdma_performance_engine.py','sdma_payload_hash.py':'sdma_payload_hash.py'}.items():args+=['-v',str(R/f)+':/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/utils/'+target+':ro']
 for k,v in {'PYTORCH_NPU_ALLOC_CONF':'expandable_segments:True','FLASH_COMPONENT_NODE':str(node),'ASCEND_RT_VISIBLE_DEVICES':'0','FLASH_SDMA_VERIFY_BYTES':'1','ASCEND_MF_STORE_URL':'tcp://172.16.12.14:19048','MMC_LOCAL_CONFIG_PATH':'/run-evidence/mmc-local.conf','MMC_META_CONFIG_PATH':'/run-evidence/mmc-meta.conf','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','PYTHONHASHSEED':'0'}.items():args+=['-e',k+'='+v]
 for p in ['/usr/local/Ascend/driver','/usr/local/dcmi','/etc/ascend_install.info','/etc/hccn.conf']:
  if pathlib.Path(p).exists():args+=['-v',p+':'+p+':ro']
 for p in ['/dev/davinci0','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:assert pathlib.Path(p).exists();args+=['--device',p]
 args+=['--entrypoint','/bin/bash',image,'-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh && exec python3 /run-evidence/supervisor.py'];out['command']=args;out['launch']=run(args,60);assert out['launch']['rc']==0;created=True;save();out['wait']=run(['docker','wait',name],400)
except BaseException as exc:out.update(error=repr(exc),traceback=traceback.format_exc())
finally:
 if created:
  try:
   q=run(['docker','inspect',name]);x=json.loads(q['stdout'])[0];assert x['Image']==image and x['Config']['Labels'].get('project_id')=='glm53-flash' and x['Config']['Labels'].get('run_id')=='r157';out['cleanup_stop']=run(['docker','stop','--time','20',name],35);out['post_inspect']=run(['docker','inspect',name,'--format','{{.State.Running}} {{.State.ExitCode}}'])
  except BaseException as exc:out['cleanup_error']=repr(exc)
 if meta_created:
  try:
   q=run(['docker','inspect',meta_name]);v=json.loads(q['stdout'])[0];assert v['Image']==image and v['Config']['Labels'].get('project_id')=='glm53-flash' and v['Config']['Labels'].get('run_id')=='r157' and v['Config']['Labels'].get('component')=='cpu-meta';out['CPU_meta_stop']=run(['docker','stop','--time','25',meta_name],35);out['CPU_meta_post_inspect']=run(['docker','inspect',meta_name,'--format','{{.State.Running}} {{.State.ExitCode}}']);out['CPU_meta_logs']=run(['docker','logs',meta_name]);
   if (R/'meta-child-result.json').exists():out['CPU_meta_result']=json.loads((R/'meta-child-result.json').read_text())
  except BaseException as exc:out['CPU_meta_cleanup_error']=repr(exc)
 for _ in range(12):
  out['post_npu']=run(['npu-smi','info'])
  if out['post_npu']['stdout'].count('No running processes found')==8:break
  time.sleep(5)
 for filename in ['worker-result.json','supervisor-result.json']:
  if (R/filename).is_file():out[filename]=json.loads((R/filename).read_text())
 out['transport_events']=[]
 for p in sorted((R/'transport-events').glob('*.jsonl')):
  assert p.stat().st_size<4*1024*1024;out['transport_events'].extend(json.loads(l) for l in p.read_text().splitlines())
 save();print(json.dumps(out))
