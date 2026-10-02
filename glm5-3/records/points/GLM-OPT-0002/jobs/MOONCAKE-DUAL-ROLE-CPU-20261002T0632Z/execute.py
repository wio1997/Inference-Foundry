from pathlib import Path
import ast,json,subprocess,hashlib,copy,threading,time
j=Path(__file__).parent
f="/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py";raw=subprocess.check_output(["docker","exec","glm52-single","cat",f],timeout=30);assert hashlib.sha256(raw).hexdigest()=="f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533";(j/"native_mooncake.py").write_bytes(raw);tree=ast.parse(raw)
classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
def method(cls,name):return next(n for n in classes[cls].body if isinstance(n,ast.FunctionDef)and n.name==name)
reg=method("MooncakeConnectorWorker","register_kv_caches");ix=next(i for i,n in enumerate(reg.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="ready_event"for t in n.targets))
tail=ast.FunctionDef(name="register_threads",args=ast.arguments(posonlyargs=[],args=[ast.arg(arg="self"),ast.arg(arg="metadata")],kwonlyargs=[],kw_defaults=[],defaults=[]),body=copy.deepcopy(reg.body[ix:]),decorator_list=[],returns=None,type_comment=None)
finished=copy.deepcopy(method("MooncakeConnectorWorker","get_finished"));finished.returns=None
for a in finished.args.args:a.annotation=None
class Stub:
 def __getattr__(self,k):return None
class ThreadStub:
 def __init__(self,kind,args):self.kind=kind;self.ready=args[7 if kind=="send"else 11]
 def start(self):self.ready.set()
 def is_alive(self):return True
 def get_and_clear_finished_requests(self):return {self.kind+"_done"}
ns=dict(threading=threading,time=time,logger=Stub(),KVCacheSendingThread=lambda *a:ThreadStub("send",a),KVCacheRecvingThread=lambda *a:ThreadStub("recv",a))
exec(compile(ast.fix_missing_locations(ast.Module(body=[tail,finished],type_ignores=[])),f+"#native-control-tail","exec"),ns)
rows=[]
for role in["kv_producer","kv_consumer","kv_both"]:
 o=Stub();o.kv_role=role;o.tp_rank=1;o.kv_send_thread=None;o.kv_recv_thread=None
 ns["register_threads"](o,None);done=ns["get_finished"](o)
 row=dict(role=role,send_constructed=o.kv_send_thread is not None,recv_constructed=o.kv_recv_thread is not None,done_sending=sorted(done[0]),done_recving=sorted(done[1]));rows.append(row)
assert rows[0]["send_constructed"]and not rows[0]["recv_constructed"]and rows[0]["done_sending"]==["send_done"]
assert not rows[1]["send_constructed"]and rows[1]["recv_constructed"]and rows[1]["done_recving"]==["recv_done"]
assert not rows[2]["send_constructed"]and rows[2]["recv_constructed"]and not rows[2]["done_sending"]and not rows[2]["done_recving"]
send=method("KVCacheSendingThread","run");recv=method("KVCacheRecvingThread","run");remote=method("KVCacheRecvingThread","_get_remote_socket")
remote_calls=[n for n in ast.walk(remote)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="make_zmq_socket"];assert len(remote_calls)==1 and any(k.arg=="bind"and isinstance(k.value,ast.Constant)and k.value.value is False for k in remote_calls[0].keywords)
assert not any(isinstance(n,ast.Attribute)and n.attr=="bind"for n in ast.walk(recv))
ctx=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="zmq_ctx")
snips=dict(send_run=ast.get_source_segment(raw.decode(),send),recv_run=ast.get_source_segment(raw.decode(),recv),recv_remote_socket=ast.get_source_segment(raw.decode(),remote),zmq_context=ast.get_source_segment(raw.decode(),ctx))
limits=["Native registration tail/get_finished executed against no-socket fake thread readiness/completion; no SDK/NPU/live socket/KV/cache proof","Sender binds one ROUTER perworker, receiver connects REQ bindFalse; no duplicated local bind in examinednative code, not OS namespace/runtime certificate","kv_both is not functional via config alone; requires role predicates/dualreadiness/finished and invalid-block handling, plus exactnative scheduler/transfer/lifetime E2E","Native start_load_kv adds requests to both present trackers; dual-role candidate must prove terminal draining/allshard lifetime, no production leak assertion","No runtime/vendor/operator modification or candidate deployment in this Job"]
out=dict(native_path=f,native_sha256=hashlib.sha256(raw).hexdigest(),registration_completion_rows=rows,source_socket_contract=snips,signals=0,NPU_workers=0,models=0,inference=0,sockets=0,limits=limits)
(j/"reduction.json").write_text(json.dumps(out,indent=2)+"\n");b=(j/"reduction.json").read_bytes()
(j/"result.json").write_text(json.dumps(dict(schema_version=1,job_id=j.name,status="completed",summary="Native Mooncake kv_both CPU control tail creates receive only and returns no completion; receiver connects bindFalse, sender only localROUTER; no candidate/GPU/socket",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="exacthashnative ASTcontrol/source socket paths")],unknowns=limits,decision_request=None,next_check_at=None),indent=2)+"\n");print(json.dumps(dict(rows=rows,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())))

