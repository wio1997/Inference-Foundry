#!/usr/bin/env python3
"""CPU-only empty MetaService lifecycle check in both fixed images."""
import hashlib,importlib.util,json,shlex
from pathlib import Path
P=Path(__file__).resolve();prior=P.parent/'inventory_readonly.py'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777'
s=importlib.util.spec_from_file_location('base',prior);b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
RUN=b.ROOT/'runs'/'20261017T0345-meta-lifecycle'
CODE=r'''
import json,pathlib,os,sys,subprocess,threading,socket,time,shutil,importlib.metadata as md
from urllib.request import urlopen
cfg=pathlib.Path('/tmp/glm53-meta.conf');cfg.write_text('ock.mmc.meta_service_url = tcp://127.0.0.1:38500\nock.mmc.meta_service.config_store_url = tcp://127.0.0.1:38501\nock.mmc.meta_service.metrics_url = http://127.0.0.1:38502\nock.mmc.log_level = info\n')
os.environ['MMC_META_CONFIG_PATH']=str(cfg)
out={'version':md.version('memcache_hybrid'),'gdb':shutil.which('gdb'),'no_clients':True,'devices_exposed':False}
p=subprocess.Popen([sys.executable,'-c','from memcache_hybrid import MetaService; import sys; sys.exit(MetaService.main())'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
def forward():
 for line in p.stdout: print('GLM53_META_LOG='+line.rstrip('\n'),flush=True)
t=threading.Thread(target=forward,daemon=True);t.start()
try:
 deadline=time.monotonic()+35
 while True:
  if p.poll() is not None:raise RuntimeError('Meta exited before readiness: %s'%p.returncode)
  try:
   with socket.create_connection(('127.0.0.1',38500),timeout=1):break
  except OSError:
   if time.monotonic()>deadline:raise TimeoutError('Meta readiness')
   time.sleep(.1)
 out['ready']=True
 try:
  with urlopen('http://127.0.0.1:38502/metrics',timeout=3) as response:out['metrics']=response.read(200000).decode(errors='replace')
 except Exception as e:out['metrics_error']=str(e)
 out['maps']=sorted(set(line.split()[-1] for line in pathlib.Path('/proc/%s/maps'%p.pid).read_text().splitlines() if '.so' in line and any(x in line.lower() for x in ['libmf_','boundscheck','spdlog','_pymmc','libbio','libhcom'])))
 time.sleep(2) # empty-service quiescence, no NPU or model timing
 p.terminate();out['rc']=p.wait(timeout=10);t.join(timeout=2)
except Exception as e:
 out['error']=repr(e)
finally:
 if p.poll() is None:p.kill();p.wait(timeout=5);out['forced_cleanup']=True
 print('GLM53_META_LIFECYCLE='+json.dumps(out),flush=True)
'''
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 with (RUN/'execution-once.lock').open('x') as f:f.write(str(__import__('os').getpid()))
 for n,image in [(30,'sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83'),(34,'sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263')]:
  pod,hostname=b.NODES[n]
  m=b.invoke(['kubectl get pod -n default '+shlex.quote(pod)+" -o jsonpath='{.status.hostIP}'"],timeout=30);assert m['rc']==0 and m['stdout']=='172.16.12.'+str(n)
  args=['docker','run','--rm','--pull=never','--network=none','--pids-limit=512','--memory=3g','--cpus=2','--label','project_id=glm53','--name','glm53-meta-lifecycle-20261017-'+str(n),'-e','OMP_NUM_THREADS=1','-e','MKL_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-v','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro','--entrypoint','timeout',image,'70','python3','-c',CODE]
  hc='import subprocess,socket,sys\nassert socket.gethostname()=='+repr(hostname)+'\nr=subprocess.run('+repr(args)+',capture_output=True,text=True,timeout=90)\nprint(r.stdout,end="");print(r.stderr,file=sys.stderr,end="");sys.exit(r.returncode)\n'
  command=shlex.join(['kubectl','exec','-i','-n','default',pod,'-c','hostctl','--','nsenter','-t','1','-m','--','python3','-'])
  r=b.invoke([command],hc,timeout=120);(RUN/('node-%s.json'%n)).write_text(json.dumps(r,indent=2));print(n,r['rc'],flush=True)
 (RUN/'receipt.json').write_text(json.dumps({'project_id':'glm53','sha256':hashlib.sha256(P.read_bytes()).hexdigest(),'model_requests':0,'npu_compute_requests':0},indent=2))
if __name__=='__main__':main()
