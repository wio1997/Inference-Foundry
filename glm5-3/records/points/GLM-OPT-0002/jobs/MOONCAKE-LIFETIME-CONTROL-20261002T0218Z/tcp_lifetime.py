import json,ctypes,gc
from mooncake.engine import TransferEngine
e=TransferEngine();r=e.initialize("127.0.0.1","P2PHANDSHAKE","tcp","");print(json.dumps({"initialize_ret":r}),flush=True);assert r==0
b=(ctypes.c_ubyte*8192)();r=e.register_memory(ctypes.addressof(b),ctypes.sizeof(b));print(json.dumps({"register_memory_ret":r,"rpc_port":e.get_rpc_port()}),flush=True);assert r==0
r=e.unregister_memory(ctypes.addressof(b));print(json.dumps({"unregister_ret":r}),flush=True);assert r==0
del e;gc.collect();print("explicit_engine_release_complete",flush=True)
