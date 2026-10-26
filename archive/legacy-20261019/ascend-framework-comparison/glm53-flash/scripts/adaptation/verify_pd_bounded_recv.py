# SPDX-License-Identifier: Apache-2.0
"""CPU native receive selector and unchanged-method audit, zero sockets/NPU."""
import argparse,ast,hashlib,json,pathlib,types

p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--candidate',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
original=a.source.read_text();candidate=a.candidate.read_text();left=ast.parse(original);right=ast.parse(candidate)
def methods(t):return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.FunctionDef)}
old,new=methods(left),methods(right);assert set(old)==set(new) and [k for k in old if old[k]!=new[k]]==['_make_worker_recv']
native=next(n for n in ast.walk(left) if isinstance(n,ast.FunctionDef) and n.name=='_make_worker_recv')
fn=next(n for n in ast.walk(right) if isinstance(n,ast.FunctionDef) and n.name=='_make_worker_recv')
assert ast.dump(native.args,include_attributes=False)==ast.dump(fn.args,include_attributes=False)
# Remove only local imports, inject stand-ins for the event sink and environment.
class Strip(ast.NodeTransformer):
 def visit_Import(self,node):return None
 def visit_ImportFrom(self,node):return None
fn=Strip().visit(fn);fn.returns=None
for arg in fn.args.args:arg.annotation=None
compiled=compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'actual-native-bounded-recv-CPU-fixture','exec')
cases=[]
for flag,role in [('0',False),('1',False),('0',True),('1',True)]:
 trace=[];events=[]
 class Poller:
  def register(self,sock,event):trace.append(('register',event))
  def poll(self,timeout):trace.append(('poll',timeout));return [] if len([v for v in trace if v[0]=='poll'])==1 else [(1,1)]
 socket=types.SimpleNamespace(recv_multipart=lambda:trace.append(('recv',)) or [b'actual-native-message'])
 scope={'os':types.SimpleNamespace(environ={'FLASH_PD_GRACEFUL_RECV':flag}),'zmq':types.SimpleNamespace(Poller=Poller,POLLIN=1),'append':lambda kind,**fields:events.append({'kind':kind,**fields}),'hashlib':hashlib,'pathlib':pathlib,'__file__':str(a.candidate)}
 exec(compiled,scope);manager=types.SimpleNamespace(server_args=types.SimpleNamespace(enable_pd_role_switch=role),attn_tp_rank=3)
 recv=scope['_make_worker_recv'](manager,socket)
 if flag=='0' and not role:assert recv is socket.recv_multipart and trace==[] and events==[]
 else:
  assert recv() is None and recv()==[b'actual-native-message'] and [v for v in trace if v[0]=='poll']==[('poll',500),('poll',500)]
  assert len(events)==int(flag=='1')
  if events:assert events[0]['native_timeout_ms']==500 and events[0]['role_switch']==role and events[0]['device']==3
 cases.append({'explicit_Flash_flag':flag,'native_role_switch':role,'passed':True})
report={'project_id':'glm53-flash','passed':True,'original_source_sha256':hashlib.sha256(original.encode()).hexdigest(),'candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),'only_changed_method':'_make_worker_recv','native_default_poll_timeout_ms_unchanged':500,'native_poller_message_or_None_contract_unchanged':True,'all_other_method_AST_unchanged':True,'cases':cases,'actual_socket_NPU_model_calls':0,'scope':'CPU extracted actual selector with stand-ins only; real TP16 native stop/worker join/ZMQ close qualification pending.'}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
