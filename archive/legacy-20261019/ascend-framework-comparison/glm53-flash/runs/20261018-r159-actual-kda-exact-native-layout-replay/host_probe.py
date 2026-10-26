import pathlib,json,subprocess,os,socket,base64,hashlib,traceback,time
assert socket.gethostname()=='A6-15U-Atlas900RCKA3-14'
R=pathlib.Path('/data/wio/inference-foundry/glm53-flash/runs/20261018-r159-actual-kda-exact-native-layout-replay');R.mkdir(parents=True,exist_ok=False);os.close(os.open(R/'once.lock',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600))
files=__FILES__;mounts=__MOUNTS__;name='if-glm53-flash-r159-triton-fp32-chunk';image='sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83';out={'project_id':'glm53-flash','node':14,'image':image,'generation_requests':0};created=False
def run(args,t=30):
 q=subprocess.run(args,capture_output=True,text=True,timeout=t);return {'rc':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
def save():
 p=R/'host-status.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(out));t.replace(p)
try:
 out['pre_npu']=run(['npu-smi','info']);assert out['pre_npu']['rc']==0 and out['pre_npu']['stdout'].count('No running processes found')==8
 out['image_check']=run(['docker','image','inspect',image,'--format','{{.Id}}']);assert out['image_check']['stdout'].strip()==image
 for filename,entry in files.items():
  p=pathlib.PurePosixPath(filename);assert not p.is_absolute() and '..' not in p.parts;b=base64.b64decode(entry['data']);assert hashlib.sha256(b).hexdigest()==entry['sha256'];(R/filename).write_bytes(b)
 private=pathlib.Path('/data/wio/inference-foundry/glm53-flash/runs/20261018-r156-actual-kda-cpu-replay-inputs/private-replay-inputs.pt');assert hashlib.sha256(private.read_bytes()).hexdigest()=='40447d146f74a022aabda64430d2e30ce0ad15c5e2b0d80e6d86494c297a7975'
 args=['docker','run','-d','--log-opt','max-size=64m','--log-opt','max-file=2','--pull=never','--name',name,'--label','project_id=glm53-flash','--label','run_id=r159','--network=none','--ipc=private','--shm-size=2g','--ulimit','memlock=-1:-1','--cpus=4','--memory=16g','--pids-limit=256','-v',str(R)+':/run-evidence','-v',str(private)+':/run-evidence/private-replay-inputs.pt:ro']
 for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','NUMEXPR_NUM_THREADS':'1','SGLANG_OPT_USE_TILELANG_MHC_PRE':'0','SGLANG_OPT_USE_TILELANG_MHC_POST':'0'}.items():args+=['-e',k+'='+v]
 for p in ['/usr/local/Ascend/driver','/usr/local/dcmi','/etc/ascend_install.info']:
  if pathlib.Path(p).exists():args+=['-v',p+':'+p+':ro']
 for p in ['/dev/davinci0','/dev/davinci_manager','/dev/devmm_svm','/dev/hisi_hdc']:assert pathlib.Path(p).exists();args+=['--device',p]
 for f,target in mounts.items():args+=['-v',str(R/f)+':/vllm-workspace/vllm-ascend/'+target+':ro']
 args+=['--entrypoint','/bin/bash',image,'-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh && exec python3 /run-evidence/probe.py'];out['command']=args;out['launch']=run(args,60);assert out['launch']['rc']==0;created=True;save();out['wait']=run(['docker','wait',name],480)
except Exception as e:out.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if created:
  try:
   q=run(['docker','inspect',name]);assert q['rc']==0;x=json.loads(q['stdout'])[0];assert x['Image']==image and x['Config']['Labels'].get('project_id')=='glm53-flash' and x['Config']['Labels'].get('run_id')=='r159';out['cleanup_stop']=run(['docker','stop','--time','20',name],35);out['post_inspect']=run(['docker','inspect',name,'--format','{{.State.Running}} {{.State.ExitCode}}'])
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
