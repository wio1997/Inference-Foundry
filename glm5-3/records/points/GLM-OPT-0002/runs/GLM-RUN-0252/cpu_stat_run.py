"""One owned profiling-OFF PD fixture; no source edits/reload or NPU profile."""
import importlib.util,json,subprocess,sys,shlex,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'GLM-RUN-0251/pd_gather_compare.py'
s=importlib.util.spec_from_file_location('cmp',OLD);x=importlib.util.module_from_spec(s);s.loader.exec_module(x)
x.ROOT=ROOT;x.m.ROOT=ROOT;x.PLAN=dict(x.PLAN,run_id='GLM-RUN-0252')
ROLES=json.loads((ROOT/'resident_roles.json').read_text())
def guard(host):
 root=ROLES[host];assert x.m.live(root)
 members=x.m.tree(root['pid'],x.m.catalogue());pids,raw=x.m.device_owners();assert len(pids)==16 and pids.issubset({z['pid'] for z in members})
 x.m.assert_idle(x.PLAN['ports'][host]);status,_=x.fetch('http://127.0.0.1:%d/health'%x.PLAN['ports'][host]);assert status==200
 if host=='167':
  assert x.source_digest()==x.PLAN['original_sha256'];assert (ROOT.parent/'GLM-RUN-0251/gather_mode.bin').read_bytes()==b'\x00'
 return dict(root=root,worker_pids=sorted(pids),worker_cmdline={str(p):Path('/proc/%d/cmdline'%p).read_bytes().replace(b'\x00',b' ').decode(errors='replace') for p in pids},health=status,idle=True)
def remote_guard():
 cmd=shlex.join(['/usr/bin/python3',str(Path(__file__).resolve()),'guard','167'])
 p=subprocess.run(['ssh','-o','BatchMode=yes','root@172.16.10.167',cmd],capture_output=True,text=True,timeout=30);assert p.returncode==0,p.stderr;return json.loads(p.stdout)
def workflow():
 owner=json.loads(Path('/data/tiankuan/wio/glm52-pd/controller-owner.json').read_text());assert owner['run_id']=='GLM-RUN-0252' and owner['status']=='running' and x.m.live(owner['owner'])
 before=dict(P=guard('166'),D=remote_guard());x.m.write('guards_before.json',before)
 command=shlex.join(['/usr/bin/python3',str(ROOT/'observe_cpu_clock.py'),str(ROOT),json.dumps(before['D']['worker_pids'])])
 p=subprocess.Popen(['ssh','-o','BatchMode=yes','root@172.16.10.167',command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(ROOT/'observer.stderr.log').open('wb'),text=True)
 started=False
 try:
  ready=json.loads(p.stdout.readline());assert ready['ready'];x.m.write('observer_ready.json',ready)
  fetch=x.fetch
  def marked_fetch(url,*args,**kwargs):
   nonlocal started
   response=fetch(url,*args,**kwargs)
   if url.endswith(':9081/v1/chat/completions'):
    p.stdin.write('go\n');p.stdin.flush();ack=json.loads(p.stdout.readline());assert ack['go'];x.m.write('observer_go.json',ack);started=True
   return response
  x.fetch=marked_fetch
  result=x.request('cpuclock')
  assert started
  p.stdin.write('done\n');p.stdin.flush();done=json.loads(p.stdout.readline());assert done['done'];x.m.write('observer_done.json',done)
  assert p.wait(timeout=15)==0
  x.m.write('final_status.json',dict(status='completed',request=result,observer=done,performance_claim=False,profiler_active=False))
 finally:
  if p.poll() is None:
   if p.stdin:p.stdin.close()
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=10)
 x.m.write('guards_after.json',dict(P=guard('166'),D=remote_guard()))
if __name__=='__main__':
 if sys.argv[1]=='guard':print(json.dumps(guard(sys.argv[2])))
 else:workflow()
