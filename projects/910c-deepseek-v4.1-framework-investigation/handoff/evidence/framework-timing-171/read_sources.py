import json,subprocess,pathlib,hashlib
HERE=pathlib.Path(__file__).resolve().parent
paths={
'multiproc_executor.py':'/vllm-workspace/vllm/vllm/v1/executor/multiproc_executor.py',
'shm_broadcast.py':'/vllm-workspace/vllm/vllm/distributed/device_communicators/shm_broadcast.py',
'ascend-multiproc_executor.py':'/vllm-workspace/vllm-ascend/vllm_ascend/executor/multiproc_executor.py',
'core.py':'/vllm-workspace/vllm/vllm/v1/engine/core.py',
'scheduler.py':'/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py',
'gpu_model_runner.py':'/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py',
'dsa_v41.py':'/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py',
'device_metadata.py':'/vllm-workspace/vllm-ascend/vllm_ascend/worker/device_metadata.py',
}
remote='''import subprocess,tarfile,io,json,hashlib,sys
out=[]
for name,path in json.loads(sys.argv[1]).items():
 r=subprocess.run(['docker','cp','wio-vllm-v41-repro171-20261102:'+path,'-'],capture_output=True)
 if r.returncode:out.append({'name':name,'path':path,'missing':True});continue
 tf=tarfile.open(fileobj=io.BytesIO(r.stdout));m=next(x for x in tf if x.isfile());b=tf.extractfile(m).read();out.append({'name':name,'path':path,'sha256':hashlib.sha256(b).hexdigest(),'text':b.decode()})
print(json.dumps(out))
'''
r=subprocess.run(['/Users/wio/.local/bin/baai-ssh','171','--','python3','-B','-I','-S','-c',remote,json.dumps(paths)],capture_output=True,timeout=90,check=True)
rows=json.loads(r.stdout)
for row in rows:
 if 'text' in row:(HERE/'source'/row['name']).write_text(row.pop('text'))
(HERE/'source-map.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
