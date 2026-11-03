"""Read-only inventory of original archived profiler files over verified166master."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,shlex,json,hashlib
from finalize_origin_coverage_v9 import N,O,read,write

CONTROL=N/'ssh166-v155-xinyufeng.sock'
ACCOUNT='xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn'
BASE='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72'


def ssh_args(code):
 assert CONTROL.exists(),'Verified fresh155master unavailable; no fallback account/asset.'
 return ['/usr/bin/ssh','-F','/dev/null','-S',str(CONTROL),'-o','ControlMaster=no','-o','ProxyCommand=false','-o','BatchMode=yes','-o','LogLevel=ERROR','-T','-p','2224',ACCOUNT,'python3 -I -S -c '+shlex.quote(code)]


def run_read(code,timeout=55):
 return subprocess.run(ssh_args(code),capture_output=True,timeout=timeout)


def main():
 receipt=N/'raw-profile-inventory-receipt-v155.json';output=O/'raw-profile-inventory-v155.json';assert not receipt.exists()and not output.exists();ident=read(N/'original166-xinyufeng-recheck-v155.json');assert ident['application_status']=='REMOTE_ROOT_IDENTITY_RETURNED'and ident['remote_result']['hostname']=='bm-jn-zs-zone1-910C-64G-10-105'and ident['remote_result']['euid']==0
 for z in ident['remote_result']['saved_profile_paths']:
  local=N/'original-profile-cache-v53'/str(Path(z['path']).relative_to(BASE));assert local.stat().st_size==z['bytes']and hashlib.sha256(local.read_bytes()).hexdigest()==z['sha256']
 code='''import os,socket,json,time,stat
from pathlib import Path
from collections import Counter
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
base=Path(%r);root=base/'runtime'/'official-traces';assert root.is_dir();start=time.monotonic();files=[];errors=[];stopped=None
for parent,dirs,names in os.walk(root,followlinks=False):
 dirs.sort();names.sort()
 if time.monotonic()-start>35 or len(files)>=50000:stopped='deadline_or_filecap';break
 for name in names:
  p=Path(parent)/name
  try:
   s=p.lstat();files.append(dict(relative=str(p.relative_to(base)),bytes=s.st_size,mtime_ns=s.st_mtime_ns,is_symlink=stat.S_ISLNK(s.st_mode)))
  except OSError as e:errors.append(dict(relative=str(p.relative_to(base)),error=type(e).__name__))
  if len(files)>=50000:stopped='filecap';break
 if stopped:break
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),original_profile_root=str(base),inventory_root=str(root),files=files,counts_by_suffix=dict(Counter(Path(x['relative']).suffix for x in files)),counts_by_top_profile=dict(Counter(Path(x['relative']).parts[2]if len(Path(x['relative']).parts)>2 else''for x in files)),top_level_entries=[dict(name=p.name,is_dir=p.is_dir(),is_symlink=p.is_symlink())for p in sorted(base.iterdir())],errors=errors,stopped=stopped,complete_within_inventory_root=stopped is None and not errors,total_files=len(files),total_bytes=sum(x['bytes']for x in files),elapsed_seconds=time.monotonic()-start,operation='Archived file metadata only; no NPU/model/CANN/Torch/import/query,process attach,service request,source alteration or new profiling.')))
'''%BASE
 p=run_read(code);parsed=None
 try:parsed=json.loads(p.stdout)
 except (json.JSONDecodeError,UnicodeDecodeError):pass
 rec=dict(observed_utc=datetime.now(timezone.utc).isoformat(),returncode=p.returncode,operation='Read-only original archived runtime/official-traces file metadata inventory',stderr=p.stderr.decode(errors='replace')[-1500:],stdout_bytes=len(p.stdout),parsed=parsed is not None,remote_or_device_work='SSH/read-only Python standard library file metadata;no device/context/model/service/profiling/source writes',new_device_work=False,overall_complete=False)
 if parsed is not None:write(output,parsed);rec['result_path']=str(output);rec['result_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
 else:rec['stdout_tail']=p.stdout.decode(errors='replace')[-1200:]
 write(receipt,rec);assert p.returncode==0 and parsed is not None,rec;assert parsed['hostname']=='bm-jn-zs-zone1-910C-64G-10-105'and parsed['euid']==0
 print(json.dumps(dict(status='PASS',files=parsed['total_files'],bytes=parsed['total_bytes'],complete=parsed['complete_within_inventory_root'],suffixes=parsed['counts_by_suffix'],stopped=parsed['stopped'],receipt=str(receipt))))


if __name__=='__main__':main()
