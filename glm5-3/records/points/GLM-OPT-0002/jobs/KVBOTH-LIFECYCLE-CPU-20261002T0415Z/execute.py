from pathlib import Path
import ast,json,hashlib,types
p=Path("/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py");src=p.read_text();sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha=="f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533"
t=ast.parse(src);cls=next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=="MooncakeConnectorWorker")
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name in["get_finished","get_block_ids_with_load_errors"]]
ns={};mod=ast.Module(body=methods,type_ignores=[]);exec(compile(mod,str(p),"exec"),ns)
class Thread:
 def __init__(self,label):self.label=label;self.calls=[]
 def get_and_clear_finished_requests(self):self.calls.append("finished");return{self.label}
 def get_and_clear_invalid_block_ids(self):self.calls.append("invalid");return{17}
cases=[]
for role in["kv_producer","kv_consumer","kv_both"]:
 send=Thread("send");recv=Thread("recv");o=types.SimpleNamespace(kv_role=role,kv_send_thread=send,kv_recv_thread=recv,tp_rank=1)
 finished=ns["get_finished"](o);errors=ns["get_block_ids_with_load_errors"](o)
 cases.append(dict(role=role,finished=[sorted(x)for x in finished],load_errors=sorted(errors),send_calls=send.calls,recv_calls=recv.calls))
assert cases[0]["finished"]==[["send"],[]]and cases[1]["finished"]==[[],["recv"]]and cases[2]["finished"]==[[],[]]
assert cases[1]["load_errors"]==[17]and cases[2]["load_errors"]==[] and not cases[2]["send_calls"]and not cases[2]["recv_calls"]
reg=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=="register_kv_caches")
branch=next(n for n in ast.walk(reg)if isinstance(n,ast.If)and ast.unparse(n.test)=='self.kv_role == "kv_producer"')
assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="KVCacheSendingThread"for n in ast.walk(ast.Module(body=branch.body,type_ignores=[])))
assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=="KVCacheRecvingThread"for n in ast.walk(ast.Module(body=branch.orelse,type_ignores=[])))
j=Path(__file__).parent;lines=src.splitlines();excerpt="\n".join(lines[branch.lineno-1:branch.end_lineno])+"\n\n"+ "\n".join("\n".join(lines[n.lineno-1:n.end_lineno])for n in methods)+"\n";(j/"native_role_excerpt.py").write_text(excerpt)
out=dict(native_path=str(p),native_sha256=sha,CPU_cases=cases,register_thread_branch="producer -> send only; everyotherrole inclboth -> recv only",classification="native kv_both accepted config is insufficient for actual bidirectional thread/completion lifecycle in installed Ascend connector",limits=["CPU exactAST methods withmockthreadqueues/noNPU/import/model/request/signal","Source controlflow only, no actual kv_both worker startup/transfer/weight sharing/runtime fit proof","No native source patch or operator change","Does not classify alternative connector or dualweight memory range as impossible"])
def ref(f):
 b=f.read_bytes();return dict(path=str(f),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
(j/"reduction.json").write_text(json.dumps(out,indent=2)+"\n")
(j/"result.json").write_text(json.dumps(dict(schema_version=1,job_id=j.name,status="completed",summary="Installed Ascend Mooncake exactCPU role lifecycle proof: kv_both startsrecv-only and returnsneitherfinishedset/no loaderrors; config-only both insufficient, noNPU",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="exact nativeAST get_finished/load_errors 3role controls andregistrationbranch",**ref(j/"reduction.json")),dict(id="source",locator="installed source guarded registerthread andfinished role branches",**ref(j/"native_role_excerpt.py"))],unknowns=out["limits"],decision_request=None,next_check_at=None),indent=2)+"\n")
print(json.dumps(out))

