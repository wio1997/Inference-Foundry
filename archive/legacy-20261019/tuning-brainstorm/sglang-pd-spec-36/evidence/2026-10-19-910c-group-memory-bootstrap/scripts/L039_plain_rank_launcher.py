#!/usr/bin/env python3
"""Own-container16rank launcher: stdlib parent, actual child codes, bounded cleanup."""
import json,os,pathlib,subprocess,sys,time
root=pathlib.Path('/work'); children=[];files=[];start=time.monotonic();v={'parent_imported_torch':'torch' in sys.modules,'world':16,'status':'STARTING','children':[]};assert not v['parent_imported_torch']
def save():
 v['elapsed_seconds']=time.monotonic()-start
 v['children']=[{'rank':rank,'pid':p.pid,'exit_code':p.poll()} for rank,p in children]
 (root/'launcher-result.json').write_text(json.dumps(v,indent=2)+'\n')
try:
 for rank in range(16):
  env=os.environ.copy();env.update({'MASTER_ADDR':'127.0.0.1','MASTER_PORT':'39550','WORLD_SIZE':'16','LOCAL_WORLD_SIZE':'16','RANK':str(rank),'LOCAL_RANK':str(rank),'GROUP_RANK':'0'})
  f=(root/f'worker-{rank:02}.log').open('w');files.append(f)
  p=subprocess.Popen([sys.executable,'-u','/work/probe.py'],env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append((rank,p));save()
 while time.monotonic()-start<210 and any(p.poll() is None for _,p in children):
  if any(p.poll() not in (None,0) for _,p in children):break
  time.sleep(.5)
 v['status']='PASS' if len(children)==16 and all(p.poll()==0 for _,p in children) else 'FAIL'
finally:
 for _,p in children:
  if p.poll() is None:p.terminate()
 limit=time.monotonic()+8
 while time.monotonic()<limit and any(p.poll() is None for _,p in children):time.sleep(.2)
 for _,p in children:
  if p.poll() is None:p.kill()
  p.wait(timeout=8)
 for f in files:f.close()
 save()
print('L039_LAUNCHER_RESULT='+json.dumps(v),flush=True)
sys.exit(0 if v['status']=='PASS' and all(p.poll()==0 for _,p in children) else 1)
