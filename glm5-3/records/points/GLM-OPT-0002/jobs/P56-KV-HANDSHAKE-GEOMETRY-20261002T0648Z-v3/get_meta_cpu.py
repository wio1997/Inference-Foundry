from pathlib import Path
import ast,json,hashlib,zmq,msgspec,sys
f=Path("/vllm-workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py");raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()=="f30572890014d0244bf31d8e15dc5d93fb84147df37e0b2c1d1b01cbba6dd533"
t=ast.parse(raw);msg=next(ast.literal_eval(n.value)for n in t.body if isinstance(n,ast.Assign)and any(isinstance(z,ast.Name)and z.id=="GET_META_MSG"for z in n.targets));assert msg==b"get_meta_msg"
ctx=zmq.Context();sock=ctx.socket(zmq.REQ);sock.setsockopt(zmq.RCVTIMEO,10000);sock.setsockopt(zmq.SNDTIMEO,10000);sock.connect(sys.argv[1]);sock.send(msgspec.msgpack.encode((msg,)));data=sock.recv();Path(sys.argv[2]).write_bytes(data);meta=msgspec.msgpack.decode(data);sock.close(linger=0);ctx.term()
# Addresses remain in server-only original bytes, omitted from reviewable metadata.
meta.pop("kv_caches_base_addr");print(json.dumps(meta))
