#!/usr/bin/env python3
"""One bounded isolated process; all remote new files stay in the owned mount."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess

REMOTE=r'''
import hashlib,json,pathlib,subprocess,sys
source=sys.argv[1];attempt=sys.argv[2];assert attempt.isalnum();name='wio-vllm-v41-repro171-20261102'
i,=json.loads(subprocess.check_output(['docker','inspect',name]));assert i['State']['Running'] and i['Image']=='sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b'
folder='fusion-pinned-h2d-generation-20261109-'+attempt
host=pathlib.Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime')/folder
cont=pathlib.Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash')/folder
assert not host.exists();host.mkdir()
for d in ['logs','cache','tmp','triton','extensions','inductor']:(host/d).mkdir()
(host/'probe.py').write_text(source)
env_reader="""import pathlib,json
keys=['TASK_QUEUE_ENABLE','ASCEND_LAUNCH_BLOCKING','PYTORCH_NPU_ALLOC_CONF']
rows=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdecimal():continue
 try:
  cmd=(p/'cmdline').read_bytes()
  if not cmd.split(b'\\0')[0].lower().startswith(b'vllm::worker'):continue
  env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\\0') if b'=' in x)
  rows.append({k:env.get(k.encode(),b'').decode(errors='replace') for k in keys})
 except (OSError,ValueError):continue
print(json.dumps(rows))
"""
er=subprocess.run(['docker','exec',name,'python3','-I','-S','-c',env_reader],capture_output=True,text=True);assert er.returncode==0,er.stderr
worker_env=json.loads(er.stdout);unique={json.dumps(v,sort_keys=True) for v in worker_env}
assert len(unique)==1,('missing/inconsistent exact VLLM worker queue env',worker_env)
queue_env=worker_env[0]
def health():return subprocess.run(['curl','--noproxy','*','--max-time','5','-sS','-o','/dev/null','-w','%{http_code}','http://127.0.0.1:30075/health'],capture_output=True,text=True).stdout
before=health();assert before=='200',before
env={'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','ASCEND_GLOBAL_LOG_LEVEL':'3','ASCEND_PROCESS_LOG_PATH':str(cont/'logs'),'ASCEND_CACHE_PATH':str(cont/'cache'),'TMPDIR':str(cont/'tmp'),'TRITON_CACHE_DIR':str(cont/'triton'),'TORCH_EXTENSIONS_DIR':str(cont/'extensions'),'TORCHINDUCTOR_CACHE_DIR':str(cont/'inductor')}
for k,v in queue_env.items():
 if v:env[k]=v
cmd=['docker','exec','-w',str(cont)]
for k,v in env.items():cmd+=['-e',k+'='+v]
bootstrap="import os,sys,json,runpy; d=json.loads(sys.argv[1]); [(os.environ.__setitem__(k,v) if v else os.environ.pop(k,None)) for k,v in d.items()]; runpy.run_path('probe.py',run_name='__main__')"
cmd+=[name,'timeout','--kill-after=5s','110s','python','-c',bootstrap,json.dumps(queue_env)]
r=subprocess.run(cmd,capture_output=True,text=True)
(host/'stdout.log').write_text(r.stdout);(host/'stderr.log').write_text(r.stderr)
result=json.loads((host/'result.json').read_bytes()) if (host/'result.json').exists() else None
out=dict(host_dir=str(host),container_dir=str(cont),container_id=i['Id'],image=i['Image'],source_sha256=hashlib.sha256(source.encode()).hexdigest(),exit_code=r.returncode,health_before=before,health_after=health(),scoped_worker_queue_env=queue_env,matching_workers=len(worker_env),stdout=r.stdout,stderr=r.stderr,result=result,new_NPU_processes=1,inference_requests=0,new_profile_collection=False)
(host/'evidence.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
'''

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',required=True);args=ap.parse_args();assert args.attempt.isalnum()
    src=Path('tools/probe_pinned_h2d_generation.py').read_text()
    p=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-c',REMOTE,src,args.attempt],capture_output=True,text=True)
    out=Path('audit/execution-search/native-pinned-h2d-'+args.attempt);out.mkdir(exist_ok=False)
    d=json.loads(p.stdout) if p.returncode==0 else dict(ssh_exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    d['runner_stderr']=p.stderr;d['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'evidence.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');(out/'probe.py').write_text(src)
    print(json.dumps(dict(path=str(out),exit_code=d.get('exit_code'),health_before=d.get('health_before'),health_after=d.get('health_after'),result=d.get('result')),ensure_ascii=False))
