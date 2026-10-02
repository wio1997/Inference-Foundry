from pathlib import Path
import ast,json,hashlib
j=Path(__file__).parent;f=Path("/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py");src=f.read_text();sha=hashlib.sha256(f.read_bytes()).hexdigest();assert sha=="f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533";t=ast.parse(src)
c=next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=="KVCacheRecvingThread");fn=next(n for n in c.body if isinstance(n,ast.FunctionDef)and n.name=="_handle_request")
def calls(n,name):return sorted(x.lineno for x in ast.walk(n)if isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr==name)
done=calls(fn,"update_done_task_count");ack=calls(fn,"_send_done_recv_signal");assert len(done)==len(ack)==1 and done[0]<ack[0]
send=next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=="KVCacheSendingThread");loop=next(n for n in send.body if isinstance(n,ast.FunctionDef)and n.name=="run_busy_loop")
marks=calls(loop,"update_done_task_count");ack_lines=[n.lineno for n in ast.walk(loop)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=="send_multipart"and any(isinstance(x,ast.Constant)and x.value==b"ACK"for x in ast.walk(n))]
assert marks and ack_lines and max(marks)<min(ack_lines)
sf=Path("/vllm-workspace/vllm/vllm/v1/core/sched/scheduler.py");ss=sf.read_text();st=ast.parse(ss);finish=next(n for n in ast.walk(st)if isinstance(n,ast.FunctionDef)and n.name=="finish_requests")
assert any(isinstance(n,ast.If)and any(isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr=="is_finished"for x in ast.walk(n.test))and any(isinstance(x,ast.Continue)for x in n.body)for n in ast.walk(finish))
parts=[]
for path,text,node in[(f,src,fn),(f,src,loop),(sf,ss,finish)]:
 lines=text.splitlines();parts.append("# native "+str(path)+" lines "+str(node.lineno)+"-"+str(node.end_lineno)+"\n"+"\n".join(lines[node.lineno-1:node.end_lineno]))
(j/"native_lifetime_excerpt.py").write_text("\n\n".join(parts)+"\n")
out=dict(source=dict(path=str(f),sha256=sha),scheduler_source=dict(path=str(sf),sha256=hashlib.sha256(sf.read_bytes()).hexdigest()),D_receive_finished_line=done[0],D_send_P_ack_line=ack[0],P_tracker_finished_lines=marks,P_send_ACK_lines=ack_lines,already_finished_abort_skipped=True,classification="Native D finished-receive publication precedes remote P ACK; consumer first output is insufficient as allPshardsfree proof",limits=["Exact installed AST/source order only; no real timing/overlap/GPU/requests/signals","Producer delayedfree native timeout/ACK lifecycle distinct from HTTPleases","Alreadyfinished Phelper cannot be reclaimed by merely aborting sameID throughfinish_requests; no native API call attempted","No claim actual52 has leak/ACK delay/race/capacityloss","Public PD controller must handle request cancellation and producer leases withactualnativeevidence"])
(j/"reduction.json").write_text(json.dumps(out,indent=2)+"\n")
def ref(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
(j/"result.json").write_text(json.dumps(dict(schema_version=1,job_id=j.name,status="completed",summary="Native source control-order proof: D receive completion beforeP ACK; P ACK aftertrackerfinish; finishedrequestabort skipped. No actualGPUtiming/leak/52claim",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",locator="guardednativeAST callordering andfinishedrequestabort branch",**ref(j/"reduction.json")),dict(id="source",locator="native thread/scheduler lifetime excerpts",**ref(j/"native_lifetime_excerpt.py"))],unknowns=out["limits"],decision_request=None,next_check_at=None),indent=2)+"\n")
print(json.dumps(out))

