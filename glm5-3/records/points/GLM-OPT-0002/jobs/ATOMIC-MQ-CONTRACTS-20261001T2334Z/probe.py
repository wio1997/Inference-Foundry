import socket,threading,concurrent.futures,json,hashlib
from pathlib import Path
import zmq
from vllm.distributed.device_communicators import shm_broadcast as m
from atomic_mq_bind import install,EXPECTED_NATIVE_SHA
guard=socket.socket();guard.bind(("127.0.0.1",0));guard.listen(1);occupied=guard.getsockname()[1]
old=m.get_open_port;m.get_open_port=lambda:occupied
try:
 m.MessageQueue(1,0,connect_ip="127.0.0.1")
 raise AssertionError("original native constructor should reject forced occupied port")
except zmq.ZMQError as e:assert e.errno==zmq.EADDRINUSE
install();install()
barrier=threading.Barrier(32)
def pair(i):
 writer=m.MessageQueue(1,0,connect_ip="127.0.0.1")
 writer.remote_socket.setsockopt(zmq.RCVTIMEO,10000)
 endpoint=writer.export_handle().remote_subscribe_addr
 assert int(endpoint.rsplit(":",1)[1])>0 and not endpoint.endswith(":"+str(occupied))
 barrier.wait(timeout=20)
 received=[];errors=[]
 def read():
  try:
   reader=m.MessageQueue.create_from_handle(writer.export_handle(),0);reader.remote_socket.setsockopt(zmq.RCVTIMEO,10000)
   reader.wait_until_ready();received.append(reader.dequeue(timeout=10));reader.remote_socket.close(linger=0)
  except BaseException as e:errors.append(repr(e))
 t=threading.Thread(target=read);t.start();writer.wait_until_ready()
 payload={"id":i,"native_wire":"preserved","data":bytes(range(256))*512}
 writer.enqueue(payload);t.join(12);assert not t.is_alive()and not errors and received==[payload],errors
 writer.remote_socket.close(linger=0)
 return endpoint
with concurrent.futures.ThreadPoolExecutor(max_workers=32)as pool:endpoints=list(pool.map(pair,range(32)))
assert len(set(endpoints))==32
m.get_open_port=old;guard.close()
assert hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()==EXPECTED_NATIVE_SHA
print(json.dumps({"cpu_only":True,"model_calls":0,"weights_loaded":0,"forced_native_EADDRINUSE_reproduced":True,"atomic_native_pairs":32,"distinct_live_endpoints":32,"native_queue_payload_roundtrips":32,"native_file_unchanged":True,"endpoints":endpoints}))
