"""Read two exact operator-choice helper files via Docker archive on verified original host."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,W,read,write

def main():
    assert not(O/'REVIEW-CHECKPOINT-v105.json').exists()
    assert not(N/'operator-selector-source166-v105.json').exists()
    control=N/'ssh166-v105-xinyufeng.sock';assert control.exists()
    model='/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py'
    paths=[model,'/vllm-workspace/vllm-ascend/vllm_ascend/attention/utils.py',
        '/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/common_cp.py']
    code='''import os,socket,json,subprocess,hashlib,tarfile,io,base64,time
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
cap=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/profile-capabilities-index-v12-v2.json')
assert hashlib.sha256(cap.read_bytes()).hexdigest()=='79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095'
container='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'
paths=PATHS;rows=[];start=time.monotonic()
for path in paths:
 assert time.monotonic()-start<50
 p=subprocess.run(['docker','cp',container+':'+path,'-'],capture_output=True,timeout=18)
 if p.returncode:
  rows.append(dict(path=path,status='READ_ERROR',returncode=p.returncode,stderr=p.stderr.decode(errors='replace')[-700:]));break
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as t:
  fs=[m for m in t.getmembers()if m.isfile()];assert len(fs)==1 and fs[0].size<512*1024
  b=t.extractfile(fs[0]).read();mtime=fs[0].mtime
 h=hashlib.sha256(b).hexdigest()
 if path==paths[0]:assert h=='568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c'
 rows.append(dict(path=path,status='READ',bytes=len(b),sha256=h,archive_mtime=mtime,
  historical_control_match=path==paths[0],source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(hostname=socket.gethostname(),uid=os.geteuid(),container_id=container,files=rows,
 readonly_archive_copy=True,container_process_started=False,model_Torch_NPU_or_capture=False,
 service_health_process_or_configuration_queries=False)))
'''.replace('PATHS',repr(paths))
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(control),'-o','ControlMaster=no','-o','ProxyCommand=false',
        '-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224',
        'xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=60)
    result=None
    if p.returncode==0:
        for line in p.stdout.splitlines():
            try:
                row=json.loads(line)
                if isinstance(row,dict)and row.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=row
            except ValueError:pass
    write(N/'operator-selector-source166-v105.json',dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,
        result=result,stderr=p.stderr[-2000:],no_model_device_capture_or_container_process=True))
    assert result is not None,(p.returncode,p.stderr[-1000:])
    assert len(result['files'])==3 and all(r['status']=='READ'for r in result['files'])
    outputs=[]
    for r in result['files']:
        b=base64.b64decode(r['source_b64']);assert hashlib.sha256(b).hexdigest()==r['sha256']
        q=N/'operator-selector-source-v105'/Path(r['path']).relative_to('/vllm-workspace')
        q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();q.write_bytes(b)
        outputs.append(dict({k:v for k,v in r.items()if k!='source_b64'},local_path=str(q)))
    write(O/'operator-selector-source-index-v105.json',dict(files=outputs,hostname=result['hostname'],container_id=result['container_id'],
        historical_runner_control_matched=True,new_helper_files=2,historical_loaded_helper_objects_attested=False,
        no_model_Torch_NPU_capture_service_or_container_process=True,overall_complete=False))
    print(json.dumps(dict(status='READONLY_SOURCE_RETRIEVED',files=[dict(path=r['path'],bytes=r['bytes'],sha256=r['sha256'])for r in outputs],new_helper_files=2,no_model_or_device_work=True)))

if __name__=='__main__':main()
