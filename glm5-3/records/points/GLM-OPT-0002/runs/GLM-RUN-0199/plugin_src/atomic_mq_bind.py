"""Task-scoped atomic native MessageQueue TCP binding; no vendor file or operator edits."""
import hashlib,inspect,textwrap,os,json
from pathlib import Path
EXPECTED_NATIVE_SHA="a78bbfb8f750812049646a1224ccf1f12ca8e49f8765481cefb3ead93dff82a3"
def install():
 from vllm.distributed.device_communicators import shm_broadcast as m
 if getattr(m.MessageQueue.__init__,"_glm_atomic_bind",False):return
 assert hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()==EXPECTED_NATIVE_SHA,"native MessageQueue source changed"
 source=textwrap.dedent(inspect.getsource(m.MessageQueue.__init__))
 changes={
 "        remote_subscribe_port = get_open_port()\n":"",
 '        socket_addr = f"tcp://{connect_ip}:{remote_subscribe_port}"\n':'        socket_addr = f"tcp://{connect_ip}:0"\n',
 '        self.remote_socket.bind(socket_addr)\n':'        self.remote_socket.bind(socket_addr)\n        remote_subscribe_port = int(self.remote_socket.getsockopt(_glm_LAST_ENDPOINT).decode().rsplit(":", 1)[1])\n'}
 for old,new in changes.items():
  assert source.count(old)==1,(old,source.count(old))
  source=source.replace(old,new)
 import zmq
 ns=dict(m.__dict__);ns["_glm_LAST_ENDPOINT"]=zmq.LAST_ENDPOINT
 exec(compile(source,__file__,"exec"),ns)
 fn=ns["__init__"];fn._glm_atomic_bind=True
 m.MessageQueue.__init__=fn
 print("GLM_ATOMIC_MQ_BIND_INSTALLED "+json.dumps({"pid":os.getpid(),"native_sha256":EXPECTED_NATIVE_SHA,"control_only":True}),flush=True)
