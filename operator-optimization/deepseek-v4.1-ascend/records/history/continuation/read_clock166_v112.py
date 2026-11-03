"""Read one card's clock metadata and exact installed interface headers; no compute probe."""
from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,shlex,subprocess
from finalize_origin_coverage_v9 import N,O,write

def main():
 receipt=N/'clock-source166-v112.json';assert not receipt.exists()
 code=r'''import socket,os,json,subprocess,hashlib,base64,shutil
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
cap=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/profile-capabilities-index-v12-v2.json')
assert hashlib.sha256(cap.read_bytes()).hexdigest()=='79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095'
exe=shutil.which('npu-smi');assert exe
real=Path(exe).resolve();binary=dict(path=exe,resolved=str(real),sha256=hashlib.sha256(real.read_bytes()).hexdigest())
commands=[]
for args in [[exe,'info','-t','freq','-i','0','-c','0'],[exe,'info','-h']]:
 p=subprocess.run(args,capture_output=True,text=True,timeout=25)
 assert len(p.stdout)+len(p.stderr)<100000
 commands.append(dict(argv=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
files=[]
for name in ['/usr/local/dcmi/include/dcmi_interface_api.h','/usr/local/Ascend/driver/include/dcmi_interface_api.h','/usr/local/Ascend/driver/include/ascend_hal.h']:
 p=Path(name)
 if not p.is_file(): files.append(dict(path=name,status='ABSENT'));continue
 b=p.read_bytes();assert len(b)<3*1024*1024
 files.append(dict(path=name,resolved=str(p.resolve()),status='READ',bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),source_b64=base64.b64encode(b).decode()))
print(json.dumps(dict(hostname=socket.gethostname(),binary=binary,commands=commands,files=files,read_only=True,new_model_or_tensor_or_capture_or_benchmark=False,admin_tool_library_loading_not_excluded=True)))
'''
 args=['/usr/bin/ssh','-F','/dev/null','-S',str(N/'ssh166-clock-v112.sock'),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224','xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn','python3 -I -S -c '+shlex.quote(code)]
 p=subprocess.run(args,capture_output=True,text=True,timeout=80);result=None
 for line in p.stdout.splitlines():
  try:
   d=json.loads(line)
   if isinstance(d,dict) and d.get('hostname')=='bm-jn-zs-zone1-910C-64G-10-105':result=d
  except ValueError:pass
 write(receipt,dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,result=result,stderr=p.stderr[-3000:]))
 assert result is not None,(p.returncode,p.stderr[-1000:])
 rows=[]
 for item in result['files']:
  row={k:v for k,v in item.items() if k!='source_b64'}
  if item['status']=='READ':
   b=base64.b64decode(item['source_b64']);assert hashlib.sha256(b).hexdigest()==item['sha256']
   q=N/'clock-source-v112'/Path(item['path']).relative_to('/');q.parent.mkdir(exist_ok=True,parents=True);assert not q.exists();q.write_bytes(b);row['local_path']=str(q)
  rows.append(row)
 write(O/'clock-source-index-v112.json',dict(files=rows,binary=result['binary'],commands=result['commands'],new_device_compute=False,overall_complete=False))
 print(json.dumps(dict(status='READ_ONLY_CLOCK_METADATA',commands=result['commands'],files=rows),ensure_ascii=False))

if __name__=='__main__':main()
