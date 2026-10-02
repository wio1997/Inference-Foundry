import json,acl
r=acl.init();print(json.dumps({"acl_init":r}),flush=True);assert r==0
from mooncake.engine import TransferEngine
print("mooncake_import_complete",flush=True)
r=acl.finalize();print(json.dumps({"acl_finalize":r}),flush=True);assert r==0
