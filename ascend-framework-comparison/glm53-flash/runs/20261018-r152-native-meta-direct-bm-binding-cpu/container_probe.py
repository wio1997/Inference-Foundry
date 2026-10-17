import pathlib,json,subprocess,os,signal,time,socket,sys,traceback
R=pathlib.Path('/run-evidence');out={'project_id':'glm53-flash','workstream':'A','passed':False,'scope':'OneCPU-nativeMetaService lifecycle withdirectinstalled_pymmcMetaAPI/noMFpackagepreloads/noTorch/noCANN/passiveLISTEN; componentonly, noactualcache/PD/model/performance','generation_requests':0,'formal_requests':0};p=None
try:
 with (R/'meta-native.log').open('wb') as log:
  p=subprocess.Popen([sys.executable,str(R/'meta_child.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);out['child_pid']=p.pid;deadline=time.monotonic()+45
  while True:
   assert p.poll() is None,('NativeMeta exitedbeforelistening',p.returncode)
   # Never send empty TCP connections to custom protocol ports. Read only
   # this owned child's fd socket inodes and the matching LISTEN rows.
   inodes=set()
   for f in pathlib.Path('/proc/'+str(p.pid)+'/fd').iterdir():
    try:target=os.readlink(f)
    except FileNotFoundError:continue
    if target.startswith('socket:['):inodes.add(target[8:-1])
   listeners={}
   for table in ['/proc/net/tcp','/proc/net/tcp6']:
    for line in pathlib.Path(table).read_text().splitlines()[1:]:
     fields=line.split();port=int(fields[1].split(':')[-1],16)
     if port in [19062,19063,19064] and fields[3]=='0A' and fields[9] in inodes:listeners[str(port)]={'state':'LISTEN','inode':fields[9],'owned_child_pid':p.pid,'local_address_hex':fields[1]}
   if set(listeners)=={'19062','19063','19064'}:out['passive_owned_LISTEN']=listeners;out['readiness_TCP_client_connections']=0;break
   if time.monotonic()>deadline:raise TimeoutError('OwnMeta portsready')
   time.sleep(.5)
  out['ports_ready']=[19062,19063,19064];os.killpg(p.pid,signal.SIGTERM);out['controlled_native_SIGTERM_sent']=True
  try:out['native_terminal_rc']=p.wait(timeout=25)
  except subprocess.TimeoutExpired:out['native_timed_out']=True;os.killpg(p.pid,signal.SIGKILL);out['native_terminal_rc']=p.wait(timeout=5)
 if (R/'child-result.json').exists():out['child']=json.loads((R/'child-result.json').read_text())
 out['native_log']=(R/'meta-native.log').read_text(errors='replace')[-131072:]
 out['passed']=out.get('native_terminal_rc')==0 and out.get('child',{}).get('native_main_rc')==0 and not any(x in out['native_log'] for x in ['corrupted size','double free','LATE_ABORT_STACK'])
except BaseException as exc:out.update(error=repr(exc),traceback=traceback.format_exc())
finally:
 if p is not None and p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
 if (R/'meta-native.log').exists():out['native_log']=(R/'meta-native.log').read_text(errors='replace')[-131072:]
 if (R/'child-status.json').exists():out['child_status']=json.loads((R/'child-status.json').read_text())
 (R/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
