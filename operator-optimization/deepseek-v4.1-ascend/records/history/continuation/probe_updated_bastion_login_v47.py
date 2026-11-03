"""Bounded read-only authorized login probes; password supplied through echo-disabled stdin and pipe FD."""
import json,os,select,shlex,subprocess,sys,termios
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parent
if not sys.stdin.isatty():raise RuntimeError('Require echo-disabled TTY input for ephemeral credential.')
attrs=termios.tcgetattr(sys.stdin.fileno());hidden=attrs.copy();hidden[3]&=~termios.ECHO
termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,hidden)
print('READY_FOR_EPHEMERAL_CREDENTIAL_INPUT_ECHO_DISABLED',flush=True)
try:
    payload=json.loads(sys.stdin.readline());password=payload.pop('password')
    assert not payload and isinstance(password,str)and password and '\n'not in password
    base='/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72'
    remote_code='''import json,socket,os
from pathlib import Path
root=Path(%s)
paths=[root,root/'analysis/native-execution-identity-v14/index.json',root/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json',root/'analysis/native-PMU-complete-v18/index.json']
print(json.dumps(dict(hostname=socket.gethostname(),euid=os.geteuid(),saved_profile_paths=[dict(path=str(p),exists=p.exists(),bytes=p.stat().st_size if p.is_file()else None)for p in paths],no_model_device_service_health_work=True)))
'''%repr(base)
    remote_command='python3 -c '+shlex.quote(remote_code)
    if sys.argv[1:]==['original-via-known105']:
        hosts=[('166-via105','fanyichao@root@172.16.10.105')]
    else:
        assert not sys.argv[1:]
        hosts=[('193','fanyichao@root@172.16.10.193'),('166','fanyichao@root@10.55.0.166')]
    for label,account in hosts:
        out=ROOT/('updated-login-'+label+'-v47.json');assert not out.exists(),'Refuse duplicate login probe receipt'
        control=ROOT/('ssh'+label+'-fanyichao.sock');assert len(str(control).encode())<100
        argv=['/opt/homebrew/bin/sshpass','-d','FD_PLACEHOLDER','/usr/bin/ssh','-F','/dev/null','-T',
            '-o','ControlMaster=auto','-o','ControlPath='+str(control),'-o','ControlPersist=600',
            '-o','PubkeyAuthentication=no','-o','PreferredAuthentications=password','-o','NumberOfPasswordPrompts=1',
            '-o','ConnectTimeout=15','-o','ConnectionAttempts=1','-o','StrictHostKeyChecking=accept-new',
            '-o','UserKnownHostsFile='+str(ROOT/'known_hosts-fanyichao'),'-o','LogLevel=ERROR',
            '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-p','2224',account+'@bastion.aiops.baai.ac.cn',remote_command]
        rfd,wfd=os.pipe();argv[2]=str(rfd)
        proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,pass_fds=(rfd,),text=True)
        os.close(rfd);os.write(wfd,(password+'\n').encode());os.close(wfd)
        try:stdout,stderr=proc.communicate(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill();stdout,stderr=proc.communicate();stderr+='\nBounded45second probe timeout;only this client process terminated.'
        result=dict(observed_utc=datetime.now(timezone.utc).isoformat(),asset=label,account=account,bastion='bastion.aiops.baai.ac.cn',port=2224,
            returncode=proc.returncode,stdout=stdout.replace(password,'<redacted>'),stderr=stderr.replace(password,'<redacted>'),
            control_socket=str(control),password_persisted=False,new_model_device_service_health_work=False)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(dict(asset=label,returncode=proc.returncode,receipt=str(out),stdout=result['stdout'][-1800:],stderr=result['stderr'][-350:]),ensure_ascii=False),flush=True)
    password=None
finally:
    termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,attrs)
