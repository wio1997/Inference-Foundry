"""One authorized base-account login to inspect the asset menu;no automatic menu action."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import subprocess
import sys
import termios

root = Path(__file__).resolve().parent
assert sys.stdin.isatty()
original = termios.tcgetattr(sys.stdin.fileno());hidden = original.copy();hidden[3] &= ~termios.ECHO
termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, hidden)
print('READY_FOR_EPHEMERAL_CREDENTIAL_INPUT_ECHO_DISABLED', flush=True)
password = None
try:
    payload = json.loads(sys.stdin.readline());password = payload.pop('password')
    assert not payload and isinstance(password, str) and password and '\n' not in password
    argv = ['/opt/homebrew/bin/sshpass', '-d', 'FD', '/usr/bin/ssh', '-F', '/dev/null', '-tt',
        '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'PubkeyAuthentication=no',
        '-o', 'PreferredAuthentications=password', '-o', 'NumberOfPasswordPrompts=1',
        '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'UserKnownHostsFile=' + str(root / 'known_hosts-fanyichao'),
        '-o', 'LogLevel=ERROR', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2',
        '-p', '2224', 'fanyichao@bastion.aiops.baai.ac.cn']
    rfd, wfd = os.pipe();argv[2] = str(rfd)
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, original)
    child = subprocess.Popen(argv, pass_fds=(rfd,))
    os.close(rfd);os.write(wfd, (password + '\n').encode());os.close(wfd);password = None
    status = child.wait()
    print(json.dumps(dict(ended_utc=datetime.now(timezone.utc).isoformat(), base_account='fanyichao',
        returncode=status, purpose='Read/search authorized asset menuonly;noautomaticassetselectionorpermissionchange.',
        password_persisted=False, remote_model_device_service_health_process_work=False), ensure_ascii=False), flush=True)
finally:
    password = None
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, original)
