from pathlib import Path
import ast,base64,hashlib,json
s=Path('router_ready_controller_g553.py').read_text().replace('router_health_ready_g553','router_health_ready_g560')
old=" if 'sglang_router.launch_router' in a and '--port' in a and a[a.index('--port')+1]=='18080':rows.append({'pid':int(p.parent.name),'argv':a})"
new=""" if a==['sglang::router']:
  inodes=set()
  for table in ['tcp','tcp6']:
   for line in Path('/proc/net',table).read_text().splitlines()[1:]:
    fields=line.split()
    if int(fields[1].rsplit(':',1)[1],16)==18080 and fields[3]=='0A':inodes.add(fields[9])
  owned=[]
  for fd in p.parent.joinpath('fd').iterdir():
   try:target=os.readlink(fd)
   except OSError:continue
   if target.startswith('socket:[') and target[8:-1] in inodes:owned.append(target[8:-1])
  if owned:rows.append({'pid':int(p.parent.name),'argv':a,'listen_port':18080,'socket_inodes':owned})"""
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("assert rows[0]['argv'][-1]=='--disable-retries';assert '--health-check-endpoint' not in rows[0]['argv']","assert rows[0]['argv']==['sglang::router'] and rows[0]['listen_port']==18080")
s=s.replace("assert rows[0]['argv'][-2:]==['--health-check-endpoint','/ready']","assert rows[0]['argv']==['sglang::router'] and rows[0]['listen_port']==18080")
s=s.replace("rows=namespace_router();assert len(rows)==1;save('old_router_namespace.json',rows)","rows=namespace_router();save('old_router_namespace.json',rows);assert len(rows)==1")
p=Path('router_ready_controller_g560.py');p.write_text(s);ast.parse(s);raw=p.read_bytes();proof=Path('G546_parent_256K_first_cold_failure.json').read_bytes()
start=Path('start_native_ready_router_g553.sh').read_text();ls=start.splitlines();ls=[('raw=base64.b64decode('+repr(base64.b64encode(raw).decode())+')') if l.startswith('raw=base64.b64decode(') else ('expected='+repr(hashlib.sha256(raw).hexdigest())) if l.startswith('expected=') else l.replace('router_health_ready_g553','router_health_ready_g560').replace('G553_','G560_') for l in ls]
Path('start_native_ready_router_g560.sh').write_text('\n'.join(ls)+'\n')
s=Path('observe_native_ready_router_g555.sh').read_text().replace('router_health_ready_g553','router_health_ready_g560').replace('G555_','G561_');Path('observe_native_ready_router_g561.sh').write_text(s)
print('Exactrouteridentity viafixedlauncher setproctitle + actual18080listensocket FD; no cmdlineguess or broadkill')
