import pathlib,json,subprocess,os,socket,base64,hashlib,traceback,time
assert socket.gethostname()=='A6-23U-Atlas900RCKA3-15'
R=pathlib.Path('/data/wio/inference-foundry/glm53-flash/runs/20261018-r152-native-meta-direct-bm-binding-cpu');R.mkdir(parents=True,exist_ok=False);os.close(os.open(R/'once.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600))
files=__FILES__;mounts=__MOUNTS__;name='if-glm53-flash-r152-memcache-capability';image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83';out={'project_id':'glm53-flash','node':15,'image':image,'generation_requests':0};created=False
def run(args,t=30):
 q=subprocess.run(args,capture_output=True,text=True,timeout=t);return {'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
def save():
 p=R/'host-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(out));t.replace(p)
try:
 out['pre_npu']=run(['npu-smi','info']);assert out['pre_npu']['rc']==0 and out['pre_npu']['stdout'].count('No running processes found')==8
 for port in [19062,19063,19064]:
  with socket.socket() as sock:sock.bind(('172.16.12.15',port))
 out['image_check']=run(['docker','image','inspect',image,'--format','{{.Id}}']);assert out['image_check']['stdout'].strip()==image
 for filename,entry in files.items():
  p=pathlib.PurePosixPath(filename);assert not p.is_absolute() and '..' not in p.parts;b=base64.b64decode(entry['data']);assert hashlib.sha256(b).hexdigest()==entry['sha256'];(R/filename).write_bytes(b)
 args=['docker','run','-d','--log-opt','max-size=64m','--log-opt','max-file=2','--pull=never','--name',name,'--label','project_id=glm53-flash','--label','run_id=r152','--network=host','--ipc=private','--shm-size=2g','--ulimit','memlock=-1:-1','--cpus=4','--memory=16g','--pids-limit=256','--ulimit','core=0:0','-v',str(R)+':/run-evidence']
 for k,v in {'MMC_META_CONFIG_PATH':'/run-evidence/mmc-meta.conf','TORCH_DEVICE_BACKEND_AUTOLOAD':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'1','SGLANG_OPT_USE_TILELANG_MHC_PRE':'0','SGLANG_OPT_USE_TILELANG_MHC_POST':'0'}.items():args+=['-e',k+'='+v]
 # CPUmetadata intentionallyexposesnoNPU/driverdevicesormounts.
 for f,target in mounts.items():args+=['-v',str(R/f)+':/sgl-workspace/sglang/python/'+target+':ro']
 args+=['--entrypoint','python3',image,'/run-evidence/probe.py'];out['command']=args;out['launch']=run(args,60);assert out['launch']['rc']==0;created=True;save();out['wait']=run(['docker','wait',name],120)
except Exception as e:out.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if created:
  try:
   q=run(['docker','inspect',name]);assert q['rc']==0;x=json.loads(q['stdout'])[0];assert x['Image']==image and x['Config']['Labels'].get('project_id')=='glm53-flash' and x['Config']['Labels'].get('run_id')=='r152';out['cleanup_stop']=run(['docker','stop','--time','20',name],35);out['post_inspect']=run(['docker','inspect',name,'--format','{{.State.Running}} {{.State.ExitCode}}'])
  except Exception as e:out['cleanup_error']=repr(e)
 for i in range(12):
  out['post_npu']=run(['npu-smi','info'])
  if out['post_npu']['stdout'].count('No running processes found')==8:break
  time.sleep(5)
 save()
 try:
  if created:out['logs']=run(['docker','logs','--timestamps',name],30)
  if (R/'result.json').exists():out['result']=json.loads((R/'result.json').read_text())
 except Exception as e:out['collection_error']=repr(e)
 save();print(json.dumps(out))
