"""Copy existing pinned profile files over the verified166master;never collect a new workload."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shlex
import socket
import subprocess

from finalize_origin_coverage_v9 import N, O, read, write
from finalize_Engram_whole_frontier_v30 import packet

ORIGINAL = Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
CACHE = N / 'original-profile-cache-v53'
CONTROL = N / 'ssh166-v53-xinyufeng.sock'
ACCOUNT = 'xinyufeng@root@10.55.0.166@bastion.aiops.baai.ac.cn'


def ssh_args(command):
    assert CONTROL.exists(), 'Verified166master unavailable;do not silently use old credentials or another asset.'
    return ['/usr/bin/ssh', '-F', '/dev/null', '-S', str(CONTROL), '-o', 'ControlMaster=no',
            '-o', 'ProxyCommand=false', '-o', 'BatchMode=yes', '-o', 'LogLevel=ERROR',
            '-T', '-p', '2224', ACCOUNT, command]


def run_readonly_python(code, timeout=170):
    return subprocess.run(ssh_args('python3 -I -S -c ' + shlex.quote(code)),
                          capture_output=True, timeout=timeout)


def fetch_pinned(relative, expected, cap=40*1024*1024):
    relative = Path(relative);assert not relative.is_absolute() and '..' not in relative.parts
    dest = CACHE / relative;dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        b = dest.read_bytes();assert hashlib.sha256(b).hexdigest() == expected
        return dict(relative=str(relative), local_path=str(dest), sha256=expected, bytes=len(b), cache_reused=True)
    code = '''import os,socket,sys
from pathlib import Path
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105' and os.geteuid()==0
p=Path(%s);before=p.stat();assert before.st_size<=%d
with p.open('rb')as f:
 while True:
  block=f.read(1024*1024)
  if not block:break
  sys.stdout.buffer.write(block)
after=p.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
''' % (repr(str(ORIGINAL / relative)), cap)
    temp = dest.with_suffix(dest.suffix + '.partial')
    with temp.open('wb') as output:
        p = subprocess.run(ssh_args('python3 -I -S -c ' + shlex.quote(code)), stdout=output,
                           stderr=subprocess.PIPE, timeout=170)
    if p.returncode != 0:
        raise RuntimeError(dict(relative=str(relative), ssh_returncode=p.returncode, stderr=p.stderr.decode(errors='replace')[-1000:]))
    b = temp.read_bytes();assert hashlib.sha256(b).hexdigest() == expected, 'Dataset bytes mismatch;SSHrc0aloneisnotaccepted.'
    temp.replace(dest)
    return dict(relative=str(relative), local_path=str(dest), sha256=expected, bytes=len(b), cache_reused=False)


def main():
    identity = read(N / 'original166-xinyufeng-recheck-v53.json')
    assert identity['application_status'] == 'REMOTE_ROOT_IDENTITY_RETURNED'
    assert identity['remote_result']['hostname'] == 'bm-jn-zs-zone1-910C-64G-10-105'
    refs = [
        ('analysis/profile-capabilities-index-v12-v2.json', packet('host-profile-inventory166-v12-v2')['result_sha256']),
        ('analysis/native-execution-identity-v14/index.json', packet('native-execution-identity166-v14')['result_sha256']),
        ('analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json', 'a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991'),
        ('analysis/native-PMU-complete-v18/index.json', packet('complete-PMU166-v18')['result_sha256']),
    ]
    rows=[]
    for relative, sha in refs:
        row=fetch_pinned(relative, sha);rows.append(row)
        print(json.dumps(dict(cached=relative,bytes=row['bytes'],sha256=sha)),flush=True)
    write(O / 'original-profile-core-cache-index-v53.json', dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), original_root=str(ORIGINAL), cache_root=str(CACHE),
        files=rows, all_four_match_historical_hashes=True,
        all_original_profile_files_copied=False, source='Existing166savedfilesonly;no recollection or local material upload.',
        no_model_import_device_service_health_process_work=True, overall_complete=False))


if __name__ == '__main__':
    main()
