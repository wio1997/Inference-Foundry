"""Read exact installed SDK/graph-wrapper source without library/model initialization."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    assert not(O/'REVIEW-CHECKPOINT-v107.json').exists()
    receipt_path=N/'sdk-operand-source166-v107.json';assert not receipt_path.exists()
    control=N/'ssh166-v107-xinyufeng.sock';assert control.exists()
    paths=['/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py',
        '/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h',
        '/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_base.h',
        '/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/npu/graphs.py']
    hashes={paths[0]:'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c',
        paths[1]:'62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7'}
    code='''import os,socket,json,subprocess,hashlib,tarfile,io,base64,time
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
cap=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/profile-capabilities-index-v12-v2.json')
assert hashlib.sha256(cap.read_bytes()).hexdigest()=='79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095'
container='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d';paths=PATHS;controls=CONTROLS;rows=[];start=time.monotonic()
for path in paths:
 assert time.monotonic()-start<65
 p=subprocess.run(['docker','cp','-L',container+':'+path,'-'],capture_output=True,timeout=18)
 if p.returncode:
  rows.append(dict(path=path,status='READ_ERROR',returncode=p.returncode,stderr=p.stderr.decode(errors='replace')[-700:]));continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as t:
  fs=[m for m in t.getmembers()if m.isfile()];assert len(fs)==1 and fs[0].size<2*1024*1024
  b=t.extractfile(fs[0]).read();mtime=fs[0].mtime
 h=hashlib.sha256(b).hexdigest()
 if path in controls:assert h==controls[path]
 rows.append(dict(path=path,status='READ',bytes=len(b),sha256=h,archive_mtime=mtime,
  historical_control_match=path in controls,source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(hostname=socket.gethostname(),uid=os.geteuid(),container_id=container,files=rows,
 readonly_archive_copy=True,container_process_started=False,libraries_loaded=False,
 model_Torch_NPU_or_capture=False,service_health_process_or_configuration_queries=False)))
'''.replace('PATHS',repr(paths)).replace('CONTROLS',repr(hashes))
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(control),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','LogLevel=ERROR',
        '-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=75);result=None
    if p.returncode==0:
        for line in p.stdout.splitlines():
            try:
                v=json.loads(line)
                if isinstance(v,dict)and v.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=v
            except ValueError:pass
    write(receipt_path,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,stderr=p.stderr[-2000:],
        no_model_Torch_NPU_capture_or_library_loading=True))
    assert result is not None,(p.returncode,p.stderr[-1000:])
    local=[]
    for r in result['files']:
        if r['status']!='READ':local.append(r);continue
        b=base64.b64decode(r['source_b64']);assert hashlib.sha256(b).hexdigest()==r['sha256']
        q=N/'sdk-operand-source-v107'/Path(r['path']).relative_to('/')
        q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();q.write_bytes(b)
        local.append(dict({k:v for k,v in r.items()if k!='source_b64'},local_path=str(q)))
    write(O/'sdk-operand-source-index-v107.json',dict(files=local,container_id=result['container_id'],hostname=result['hostname'],
        controls_matched=sum(r.get('historical_control_match',False)for r in local)==2,
        no_libraries_loaded_model_device_or_container_process=True,overall_complete=False))
    print(json.dumps(dict(status='READONLY_SOURCE_FINISHED',files=[{k:r[k]for k in ['path','status']}for r in local],
        SDK_control_matched=next((r.get('historical_control_match')for r in local if r['path']==paths[1]),False),no_libraries_or_device_work=True)))

if __name__=='__main__':main()
