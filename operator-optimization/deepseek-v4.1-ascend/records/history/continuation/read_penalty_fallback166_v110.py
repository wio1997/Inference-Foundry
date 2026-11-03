"""Read one missing operator utility and two cached caller controls;no container process."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,W,read,write

def main():
    receipt=N/'penalty-fallback-source166-v110.json';assert not receipt.exists()
    root='/vllm-workspace/vllm/vllm/'
    paths=[root+'model_executor/layers/utils.py',root+'v1/sample/sampler.py',root+'v1/sample/ops/penalties.py']
    frozen=W/'frozen-source-snapshot/vllm/vllm'
    controls={root+rel:hashlib.sha256((frozen/rel).read_bytes()).hexdigest()for rel in ['v1/sample/sampler.py','v1/sample/ops/penalties.py']}
    code='''import socket,os,subprocess,json,tarfile,io,hashlib,base64
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'and os.geteuid()==0
cap=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/profile-capabilities-index-v12-v2.json')
assert hashlib.sha256(cap.read_bytes()).hexdigest()=='79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095'
cid='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d';rows=[];controls=CONTROLS
for path in PATHS:
 p=subprocess.run(['docker','cp','-L',cid+':'+path,'-'],capture_output=True,timeout=15)
 if p.returncode:rows.append(dict(path=path,status='READ_ERROR',returncode=p.returncode,stderr=p.stderr.decode(errors='replace')[-600:]));continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*')as t:
  fs=[x for x in t.getmembers()if x.isfile()];assert len(fs)==1 and fs[0].size<2*1024*1024;b=t.extractfile(fs[0]).read()
 h=hashlib.sha256(b).hexdigest()
 if path in controls:assert h==controls[path]
 rows.append(dict(path=path,status='READ',sha256=h,bytes=len(b),caller_control=path in controls,source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(hostname=socket.gethostname(),container=cid,files=rows,read_only_archive_copy=True,container_process_started=False,NPU_or_libraries_loaded=False)))
'''.replace('CONTROLS',repr(controls)).replace('PATHS',repr(paths))
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(N/'ssh166-penalty-v110.sock'),'-o','ControlMaster=no','-o','ProxyCommand=false',
        '-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=60);result=None
    for line in p.stdout.splitlines():
        try:
            d=json.loads(line)
            if isinstance(d,dict)and d.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=d
        except ValueError:pass
    write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,stderr=p.stderr[-3000:]))
    assert result is not None,(p.returncode,p.stderr[-1000:]);rows=[]
    for row in result['files']:
        r={k:v for k,v in row.items()if k!='source_b64'}
        if row['status']=='READ':
            raw=base64.b64decode(row['source_b64']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
            out=N/'penalty-source-v110'/Path(row['path']).relative_to('/');out.parent.mkdir(exist_ok=True,parents=True);assert not out.exists();out.write_bytes(raw);r['local_path']=str(out)
        rows.append(r)
    write(O/'penalty-fallback-source-index-v110.json',dict(files=rows,new_utility_files=1,caller_controls=2,
        no_container_process_libraries_or_device_work=True,overall_complete=False))
    print(json.dumps(dict(status='READ_ONLY_SOURCE_COMPLETE',files=[dict(path=r['path'],status=r['status'])for r in rows],new_utility_files=1)))

if __name__=='__main__':main()
