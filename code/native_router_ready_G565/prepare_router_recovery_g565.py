from pathlib import Path
import ast,base64,hashlib
s=Path('router_ready_controller_g560.py').read_text().replace('router_health_ready_g560','router_health_ready_g565')
start=s.index(" oldcmd=Path('/proc/958636/cmdline')")
end=s.index(" PORTLOCK=Path('<COORD_ROOT>/<ROLE_NODE>-port18080.lock')",start)
replacement=''' prior=ROOT/'router_health_ready_g560'
 previous=json.loads((prior/'state.json').read_text());assert previous['phase']=='failed_requires_review' and previous['pid']==1027953
 assert "oldproc=Path('/proc/958636/cmdline')" in (prior/'controller.log').read_text()
 oldrows=json.loads((prior/'old_router_namespace.json').read_text());assert len(oldrows)==1 and oldrows[0]['pid']==20823 and oldrows[0]['listen_port']==18080
 proc=Path('/proc/1027953/cmdline');assert not proc.exists() or not proc.read_bytes() or str(prior/'controller.py').encode() not in proc.read_bytes()
 oldproc=Path('/proc/958636/cmdline');deadline=time.monotonic()+15
 while oldproc.exists() and oldproc.read_bytes():
  assert time.monotonic()<deadline,'Old owned dockerexec still active; do not launch';time.sleep(.5)
 assert not namespace_router(),'An actual listen-socket owner remains; preserve it'
 with socket.socket() as sock:sock.bind(('<HOST>',18080))
 for base in ['http://<HOST>:19083','http://<HOST>:19084']:
  assert http(base+'/ready')['status']==200
 save('recovery_previous_state.json',previous);save('old_router_namespace.json',oldrows);save('old_router_handle.json',old)
 oldargv=old['argv'];assert oldargv[:3]==['flock','--nonblock','<COORD_ROOT>/<ROLE_NODE>-port18080.lock']
 state('verified_exact_old_router_stopped_and_port_free_before_new_start')
'''
s=s[:start]+replacement+s[end:];p=Path('router_ready_controller_g565.py');p.write_text(s);ast.parse(s);raw=p.read_bytes();start=Path('start_native_ready_router_g560.sh').read_text();ls=start.splitlines();ls=[('raw=base64.b64decode('+repr(base64.b64encode(raw).decode())+')') if l.startswith('raw=base64.b64decode(') else ('expected='+repr(hashlib.sha256(raw).hexdigest())) if l.startswith('expected=') else l.replace('router_health_ready_g560','router_health_ready_g565').replace('G560_','G565_') for l in ls];Path('start_native_ready_router_g565.sh').write_text('\n'.join(ls)+'\n')
s=Path('observe_native_ready_router_g561.sh').read_text().replace('router_health_ready_g560','router_health_ready_g565').replace('G561_','G566_');Path('observe_native_ready_router_g566.sh').write_text(s)
print('Prepared forwardrecovery requiringactualoldroutergone/portfree; previousasync-exitconfirmationfailurepreserved; modeluntouched')
