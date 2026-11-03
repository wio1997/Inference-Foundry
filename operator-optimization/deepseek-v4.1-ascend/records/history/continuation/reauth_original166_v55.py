"""User-requested xinyufeng166recheck;provided credential only through hidden stdin/anonymousFD."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import shlex
import subprocess
import sys
import termios

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'original166-xinyufeng-recheck-v55.json'
CONTROL = ROOT / 'ssh166-v55-xinyufeng.sock'
assert not OUT.exists(), 'Refuse duplicate recheck receipt'
assert not CONTROL.exists(), 'Fresh dedicated control path required'
assert sys.stdin.isatty(), 'Echo-disabled TTY required'
original = termios.tcgetattr(sys.stdin.fileno());hidden = original.copy();hidden[3] &= ~termios.ECHO
termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, hidden)
print('READY_FOR_EPHEMERAL_CREDENTIAL_INPUT_ECHO_DISABLED', flush=True)
password = None
try:
    payload = json.loads(sys.stdin.readline());password = payload.pop('password')
    assert not payload and isinstance(password, str) and password and '\n' not in password
    base = '/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72'
    code = '''import json,socket,os,hashlib
from pathlib import Path
root=Path(%s)
rels=['analysis/profile-capabilities-index-v12-v2.json','analysis/native-execution-identity-v14/index.json','analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json','analysis/native-PMU-complete-v18/index.json']
rows=[]
for rel in rels:
 p=root/rel;size=p.stat().st_size if p.is_file()else None
 digest=hashlib.sha256(p.read_bytes()).hexdigest()if size is not None and size<16*1024*1024 else None
 rows.append(dict(path=str(p),exists=p.exists(),bytes=size,sha256=digest,hash_read_capped_at_16MiB=True))
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),saved_capture_root=str(root),saved_profile_paths=rows,no_model_torch_device_service_health_process_work=True)))
''' % repr(base)
    command = 'python3 -I -S -c ' + shlex.quote(code)
    account = 'xinyufeng@root@10.55.0.166'
    argv = ['/opt/homebrew/bin/sshpass', '-d', 'FD', '/usr/bin/ssh', '-F', '/dev/null', '-T',
        '-o', 'ControlMaster=yes', '-o', 'ControlPath=' + str(CONTROL), '-o', 'ControlPersist=600',
        '-o', 'PubkeyAuthentication=no', '-o', 'PreferredAuthentications=password',
        '-o', 'NumberOfPasswordPrompts=1', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(ROOT / 'known_hosts-fanyichao'),
        '-o', 'LogLevel=ERROR', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
        '-p', '2224', account + '@bastion.aiops.baai.ac.cn', command]
    rfd, wfd = os.pipe();argv[2] = str(rfd)
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, pass_fds=(rfd,), text=True)
    os.close(rfd);os.write(wfd, (password + '\n').encode());os.close(wfd)
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=50)
    except subprocess.TimeoutExpired:
        timed_out = True;proc.kill();stdout, stderr = proc.communicate()
        stderr += '\nBounded50second timeout;only this owned local client was terminated.'
    stdout = stdout.replace(password, '<redacted>');stderr = stderr.replace(password, '<redacted>')
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and 'hostname' in parsed and parsed.get('euid') == 0:
        status = 'REMOTE_ROOT_IDENTITY_RETURNED'
    elif 'match asset failed' in stdout or '未发现匹配的资产' in stdout:
        status = 'BASTION_ASSET_NOT_MATCHED'
    else:
        status = 'REMOTE_IDENTITY_NOT_VERIFIED'
    result = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
        user_steering='Continueauthorizedsaved-fileanalysisafterownedmuxexpiry;exact10.55.0.166withpreviouslyprovidedxinyufengcredential.',
        account=account, bastion='bastion.aiops.baai.ac.cn', port=2224,
        returncode=proc.returncode, application_status=status, timed_out=timed_out,
        stdout=stdout, stderr=stderr, remote_result=parsed, control_socket=str(CONTROL),
        fresh_dedicated_control_path=True, password_persisted=False,
        no_model_torch_device_service_health_process_work=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(receipt=str(OUT), returncode=proc.returncode, application_status=status,
        remote_result=parsed, non_json_stdout=stdout[-1500:] if parsed is None else None,
        stderr=stderr[-500:]), ensure_ascii=False), flush=True)
finally:
    password = None
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, original)
