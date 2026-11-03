"""Locate only installed Tritonlaunch/profiler source paths; no package import/device work."""
from datetime import datetime,timezone
import json,shlex,subprocess
from finalize_origin_coverage_v9 import N,write

def main():
    out=N/'sampling-profile-paths166-v117.json';assert not out.exists()
    code=r'''import socket,os,json,subprocess
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
cid='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'
script=r"""import sysconfig,sys,json,hashlib
from pathlib import Path
h=Path('/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h');assert hashlib.sha256(h.read_bytes()).hexdigest()=='62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7'
base=Path(sysconfig.get_path('purelib'));root=base/'triton/backends/ascend'
rows=[]
if root.is_dir():
 for p in root.iterdir():
  if p.is_file() and p.suffix in ['.py','.c','.cpp','.h','.cc']:rows.append(dict(path=str(p),bytes=p.stat().st_size))
versions=[]
for p in base.glob('*triton*dist-info/METADATA'):
 b=p.read_bytes();lines=b.decode(errors='replace').splitlines();versions.append(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),fields=[s for s in lines if s.startswith(('Name:','Version:'))]))
print(json.dumps(dict(python=sys.version,package_root=str(base),backend_root=str(root),backend_exists=root.is_dir(),files=rows,metadata=versions,no_Torch_triton_or_NPU_library_import=True)))
"""
p=subprocess.run(['docker','exec',cid,'python3','-I','-S','-c',script],capture_output=True,text=True,timeout=15)
print(json.dumps(dict(hostname=socket.gethostname(),container=cid,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
'''
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(N/'ssh166-sampling-v117.sock'),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=30);result=None
    for line in p.stdout.splitlines():
        try:
            d=json.loads(line)
            if isinstance(d,dict)and d.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=d
        except ValueError:pass
    write(out,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,stderr=p.stderr[-3000:]))
    assert result is not None and result['returncode']==0
    print(result['stdout'])

if __name__=='__main__':main()
