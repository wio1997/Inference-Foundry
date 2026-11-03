"""Follow exactly the A3 HC kernel includes, without traversing the framework."""
from pathlib import Path
import hashlib,base64,json
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,read,write
import mirror_saved_profile_core_v53 as transport
from read_original_source_archive_v53 import CONTAINER
transport.CONTROL=N/'ssh166-v56-xinyufeng.sock'

def main():
    assert not (N/'HC-dependencies-source166-v58.json').exists()
    root='/vllm-workspace/vllm-ascend/csrc/moe/'
    paths=[root+p for p in ['hc_pre/op_kernel/hc_pre_base.h','hc_pre/op_kernel/hc_pre_cube_compute.h',
      'hc_post/op_kernel/hc_post_d_split.h','hc_pre/op_host/hc_pre_tiling.cpp']]
    code='''import socket,os,subprocess,tarfile,io,json,hashlib,base64
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
rows=[]
for path in %r:
 p=subprocess.run(['docker','cp','-L',%r+':'+path,'-'],capture_output=True,timeout=20)
 if p.returncode:
  rows.append(dict(path=path,status='UNAVAILABLE_AT_EXACT_PATH',stderr=p.stderr.decode(errors='replace')[-400:]));continue
 with tarfile.open(fileobj=io.BytesIO(p.stdout),mode='r:*') as tar:
  fs=[x for x in tar.getmembers() if x.isfile()];assert len(fs)==1 and fs[0].size<2*1024*1024
  b=tar.extractfile(fs[0]).read()
 rows.append(dict(path=path,status='READ',sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),data=base64.b64encode(b).decode()))
print(json.dumps(dict(files=rows,container_process_started=False)))
'''%(paths,CONTAINER)
    p=transport.run_readonly_python(code,timeout=110)
    rec=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,stdout=p.stdout.decode(errors='replace'),stderr=p.stderr.decode(errors='replace'),no_model_device_work=True)
    write(N/'HC-dependencies-source166-v58.json',rec);assert p.returncode==0,rec['stderr'][-500:]
    d=read(O/'HC-operator-inputs-v58.json')
    for r in json.loads(rec['stdout'])['files']:
      if r['status']!='READ':d['files'].append(r);continue
      b=base64.b64decode(r['data']);assert hashlib.sha256(b).hexdigest()==r['sha256']
      dest=N/'HC-operator-source-v58'/Path(r['path']).relative_to('/')
      dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
      d['files'].append(dict({k:v for k,v in r.items() if k!='data'},local_path=str(dest)))
    write(O/'HC-operator-inputs-v58.json',d)
    print(json.dumps([dict(path=r['path'],status=r['status']) for r in d['files'][-len(paths):]]))

if __name__=='__main__':main()
