from pathlib import Path
import json,subprocess,hashlib,sys,shlex
sys.path.insert(0,"/data/tiankuan/wio/Inference-Foundry/glm5-3/runtime")
from phase_runner import atomic_json,utc
j=Path(__file__).parent;r=j.parents[1]/"runs/GLM-RUN-0056";owners=json.loads((r/"adopted_model_identities.json").read_text());rows={}
code='''from pathlib import Path
import ast,json,hashlib,zmq,msgspec,sys
f=Path("/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py");raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()=="f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533"
t=ast.parse(raw);msg=next(ast.literal_eval(n.value)for n in t.body if isinstance(n,ast.Assign)and any(isinstance(z,ast.Name)and z.id=="GET_META_MSG"for z in n.targets));assert msg==b"get_meta_msg"
ctx=zmq.Context();sock=ctx.socket(zmq.REQ);sock.setsockopt(zmq.RCVTIMEO,10000);sock.setsockopt(zmq.SNDTIMEO,10000);sock.connect(sys.argv[1]);sock.send(msgspec.msgpack.encode((msg,)));data=sock.recv();Path(sys.argv[2]).write_bytes(data);meta=msgspec.msgpack.decode(data);sock.close(linger=0);ctx.term()
# Addresses remain in server-only original bytes, omitted from reviewable metadata.
meta.pop("kv_caches_base_addr");print(json.dumps(meta))
'''
(j/"get_meta_cpu.py").write_text(code)
for key in ["P0","P1"]:
 o=owners[key];node=o["host"];port=62000+16*o["rank"];rawfile=j/(key+".agent.bin")
 # Script is visible through the existing shared task mount on166; run both readonly sockets from166.
 args=["docker","exec","glm52-single","python3",str(j/"get_meta_cpu.py"),"tcp://172.16.10."+node+":"+str(port),str(rawfile)]
 t=subprocess.run(args,capture_output=True,timeout=30);(j/(key+".stdout")).write_bytes(t.stdout);(j/(key+".stderr")).write_bytes(t.stderr);t.check_returncode();meta=json.loads(t.stdout);assert meta["engine_id"]=="glm56-P-DP"+str(o["rank"])and meta["num_blocks"]==41 and meta["handshake_port"]==port
 (j/(key+".metadata.json")).write_text(json.dumps(meta,indent=2)+"\n")
 rows[key]=meta
assert rows["P0"]["block_size"]==rows["P1"]["block_size"]
out=dict(at=utc(),metadata=rows,protocol="Actualnative GET_META_MSG only, no DONE_RECVING/ACK/request manipulation",new_inference=0,signals=0,models=0,limits=["Nativeworker handshake metadata describes cache geometry and registeredbuffers, not exact scheduler BlockPool occupancy/liveallocation","Originalrawmetadata binary pointers serveronly; curatedmetadata excludesbaseaddresses","NPUresident nativeP blockcount41 vs nativeBlockPool reservesnullblock is admissionhypothesis; exactnative KVmanager CPU allocation next beforecandidate","GET_META doesnottransferKV/finish/free helper requests; noE2E/capacity credit"])
atomic_json(j/"reduction.json",out);b=(j/"reduction.json").read_bytes();atomic_json(j/"result.json",dict(schema_version=1,job_id=j.name,status="completed",summary="ActualbothP GET_META readonly:41nativeblocks/cachegeometry preserved; rawbinary serveronly/pointeromit; no inference/transfer/finish/signals",execution=dict(inner_exit_code=0,acceptance="passed",processes=[]),findings=[],evidence=[dict(id="reduction",path=str(j/"reduction.json"),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),locator="nativeGET_META_MSG P0/P1 directREQ observedregisteredgeometry")],unknowns=out["limits"],decision_request=None,next_check_at=None));print(json.dumps(dict(blocks={k:x["num_blocks"]for k,x in rows.items()},block_size=rows["P0"]["block_size"],group_specs=rows["P0"]["kv_group2layeridx"])))

