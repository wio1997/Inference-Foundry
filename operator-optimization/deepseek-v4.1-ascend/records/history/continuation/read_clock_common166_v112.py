"""One supported common-info query; no accelerator compute or configuration calls."""
from datetime import datetime,timezone
import json,shlex,subprocess
from finalize_origin_coverage_v9 import N,write
pout=N/'clock-common166-v112.json';assert not pout.exists()
code=r'''import socket,os,json,subprocess,hashlib
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
header=Path('/usr/local/Ascend/driver/include/dcmi_interface_api.h');assert hashlib.sha256(header.read_bytes()).hexdigest()=='a7be9d5fad53b3278fc043a9c2fc6147fc93ac12aebb2c73e5ad368a975d6203'
args=['/usr/local/sbin/npu-smi','info','-t','common','-i','0','-c','0']
p=subprocess.run(args,capture_output=True,text=True,timeout=25);assert len(p.stdout)+len(p.stderr)<100000
print(json.dumps(dict(hostname=socket.gethostname(),argv=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,new_model_tensor_capture_benchmark_or_config_change=False)))
'''
a=['/usr/bin/ssh','-F','/dev/null','-S',str(N/'ssh166-clock-v112.sock'),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
p=subprocess.run(a,capture_output=True,text=True,timeout=40);result=None
for line in p.stdout.splitlines():
 try:
  d=json.loads(line)
  if isinstance(d,dict)and d.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=d
 except ValueError:pass
write(pout,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,stderr=p.stderr[-3000:]))
assert result is not None
print(json.dumps(result))
