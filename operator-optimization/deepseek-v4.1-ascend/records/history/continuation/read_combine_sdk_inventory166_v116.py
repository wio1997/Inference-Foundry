"""One bounded CPU-only SDK filename inventory for Cast/mask operator headers."""
from datetime import datetime,timezone
import json,shlex,subprocess
from finalize_origin_coverage_v9 import N,write

def main():
    out=N/'combine-sdk-inventory166-v116.json';assert not out.exists()
    inventory=r'''import os,json,time,hashlib
from pathlib import Path
control=Path('/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h')
assert hashlib.sha256(control.read_bytes()).hexdigest()=='62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7'
root=Path('/usr/local/Ascend/ascend-toolkit/latest').resolve();start=time.monotonic();dirs=0;files=0;matches=[];status='DONE'
names={'kernel_operator_vec_vconv_intf.h','kernel_operator_vec_vconv_impl.h','kernel_operator_common_intf.h','kernel_operator_common_impl.h'}
for path,children,leaves in os.walk(root,followlinks=False):
 dirs+=1;files+=len(leaves)
 if dirs>18000 or files>200000 or time.monotonic()-start>12:status='BOUNDED_STOP';break
 for name in leaves:
  if name in names:matches.append(str(Path(path)/name))
 if len(matches)>100:status='MATCH_CAP_STOP';break
print(json.dumps(dict(root=str(root),status=status,directories=dirs,files=files,matches=matches,elapsed_seconds=time.monotonic()-start,filename_only=True,model_Torch_NPU_or_library_load=False)))
'''
    code=r'''import socket,os,json,subprocess
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
cid='bd6ff19d7328271dc0b82f20414129fca1dc9e8eefa472f259eabcfb3e48f46d'
p=subprocess.run(['docker','exec',cid,'python3','-I','-S','-c',INVENTORY],capture_output=True,text=True,timeout=20)
print(json.dumps(dict(hostname=socket.gethostname(),container=cid,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)))
'''.replace('INVENTORY',repr(inventory))
    args=['/usr/bin/ssh','-F','/dev/null','-S',str(N/'ssh166-combine-v116.sock'),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
    p=subprocess.run(args,capture_output=True,text=True,timeout=35);result=None
    for line in p.stdout.splitlines():
        try:
            d=json.loads(line)
            if isinstance(d,dict)and d.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=d
        except ValueError:pass
    write(out,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,stderr=p.stderr[-3000:]))
    assert result is not None and result['returncode']==0
    found=json.loads(result['stdout']);print(json.dumps(found))

if __name__=='__main__':main()
