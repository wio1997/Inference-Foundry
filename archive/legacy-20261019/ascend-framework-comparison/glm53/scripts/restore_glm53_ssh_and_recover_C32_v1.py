"""Restore user-authorized SSH master with an in-memory one-shot secret, recover only."""
from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,os,pty,select,signal,socket,subprocess,termios,time
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/20261018T-L09-ssh-restore-and-recover-v1'
BROKER='/var/folders/6v/cmq6k1kn4v17s64j4zvjfr5c0000gn/T/glm53-ssh-auth-ujqh2kq1/once.sock'
CONTROL='/Users/wio/work/sglang-glm53-pd/ssh-control'
CHILD='scripts/inspect_sglang_C32_L09_interrupted_control_recovery_v2.py'
CHILD_SHA='5f71f518fe62a5ae8667f37e1c54ab2193e0708e43d47376905ed36ddb0119aa'
def main():
 p=ROOT/CHILD;assert hashlib.sha256(p.read_bytes()).hexdigest()==CHILD_SHA;ast.parse(p.read_text())
 assert not (ROOT/'runs/20261018T-L09-interrupted-control-recovery-v2').exists(),'No child replay'
 assert not Path(CONTROL).exists(),'Existing control socket: stop and audit; do not replace'
 RUN.mkdir(exist_ok=False);(RUN/'execution-once.lock').write_text(str(os.getpid()))
 receipt={'project_id':'glm53','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'password_stored':False,'model_requests':0,'remote_mutations':False,'child_sha256':CHILD_SHA};secret=b'';buf=b'';pid=None
 try:
  with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as s:
   s.settimeout(10);s.connect(BROKER);s.sendall(b'glm53-ssh-control-restore-v1')
   while not secret.endswith(b'\n'):
    part=s.recv(1024);assert part and len(secret)+len(part)<=1024;secret+=part
  secret=secret.rstrip(b'\r\n');assert secret
  args=['ssh','-M','-N','-f','-S',CONTROL,'-p','60011','-o','ControlPersist=3600','-o','BatchMode=no','-o','PreferredAuthentications=password,keyboard-interactive','-o','PubkeyAuthentication=no','-o','NumberOfPasswordPrompts=1','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','root@61.241.77.34']
  pid,fd=pty.fork()
  if pid==0:os.execvp(args[0],args)
  deadline=time.monotonic()+75;sent=False;status=None
  while time.monotonic()<deadline:
   readable,_,_=select.select([fd],[],[],0.2)
   if readable:
    try:chunk=os.read(fd,8192)
    except OSError:chunk=b''
    buf+=chunk;assert len(buf)<=131072,'Unexpected SSH output size'
    if b'assword:' in buf.lower() and not sent:
     attr=termios.tcgetattr(fd);attr[3]&=~termios.ECHO;termios.tcsetattr(fd,termios.TCSANOW,attr);os.write(fd,secret+b'\n');sent=True
   done,st=os.waitpid(pid,os.WNOHANG)
   if done:status=os.waitstatus_to_exitcode(st);break
  if status is None:
   os.kill(pid,signal.SIGTERM);os.waitpid(pid,0);raise TimeoutError('SSH authentication deadline; no retry')
  os.close(fd);receipt['ssh_auth_rc']=status;receipt['password_prompt_answered']=sent;receipt['ssh_output_redacted']=buf.replace(secret,b'[REDACTED]').decode(errors='replace');secret=b''
  assert status==0,'SSH authentication failed; no retry'
  c=subprocess.run(['ssh','-S',CONTROL,'-O','check','-p','60011','root@61.241.77.34'],capture_output=True,text=True,timeout=15);receipt['master_check']={'rc':c.returncode,'stdout':c.stdout,'stderr':c.stderr};assert c.returncode==0
  p=ROOT/'scripts/inventory_readonly.py';assert hashlib.sha256(p.read_bytes()).hexdigest()=='e95abea94f79ba05c770e17314b1cf53ad43284c7d6bb1b41afd7de3394a7777';spec=importlib.util.spec_from_file_location('inventory',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  c=m.invoke(['hostname'],timeout=30);receipt['source_identity']=c;assert c['rc']==0 and c['stdout'].strip()=='A3-5U-Atlas900RCKA3-1'
  receipt['ssh_restored']=True;(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print('SSH_RESTORED_IDENTITY_VERIFIED',flush=True)
  spec=importlib.util.spec_from_file_location('recovery',ROOT/CHILD);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main();receipt['recovery_child_returned']=True
 except BaseException as e:
  receipt['error']=type(e).__name__+': '+str(e).replace(secret.decode(errors='replace'),'[REDACTED]') if secret else repr(e);raise
 finally:
  secret=b'';(RUN/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':main()
