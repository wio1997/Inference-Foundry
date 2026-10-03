"""V12 composes unchanged V11 lease/state/wire paths with bounded native CPU memo."""
import argparse
import json
import os
from pathlib import Path

import httpx
import response_affinity_gateway_v11 as v11
from native_chat_token_memo import NativeChatTokenMemoTransport

def create_app(config=None, transport=None, policy=None, groups=None,
               fault_state_path=None, response_owner_state_path=None,
               pd_config=None, pd_native_plans=None, shape_config=None):
    replicas=config if config is not None else json.loads(os.environ["GLM_REPLICAS"])
    native_groups=groups if groups is not None else json.loads(os.environ.get("GLM_EXECUTION_GROUPS","[]"))
    pd=pd_config if pd_config is not None else json.loads(os.environ.get("GLM_PD_PRODUCERS","{}"))
    memo=None
    if not pd:
        epochs={}
        for replica in replicas:
            matches=[g for g in native_groups if replica["id"] in g["members"]]
            if len(matches)!=1 or not isinstance(matches[0].get("epoch"),str) or not matches[0]["epoch"]:
                continue
            origin=replica["url"].rstrip("/")
            if origin in epochs and epochs[origin]!=matches[0]["epoch"]:
                raise ValueError("one native URL cannot carry distinct token cache epochs")
            epochs[origin]=matches[0]["epoch"]
        memo=NativeChatTokenMemoTransport(
            transport if transport is not None else httpx.AsyncHTTPTransport(retries=0,trust_env=False),
            epochs, max_bytes=int(os.environ.get("GLM_TOKEN_MEMO_MAX_BYTES","33554432")),
            max_pending=int(os.environ.get("GLM_TOKEN_MEMO_MAX_PENDING","128")),
            audit_dir=os.environ.get("GLM_TOKEN_MEMO_AUDIT_DIR"),
            trace_path=os.environ.get("GLM_TOKEN_MEMO_TRACE_PATH"),
            stats_path=os.environ.get("GLM_TOKEN_MEMO_STATS_PATH"))
        transport=memo
    app=v11.create_app(config=replicas,transport=transport,policy=policy,groups=native_groups,
        fault_state_path=fault_state_path,response_owner_state_path=response_owner_state_path,
        pd_config=pd,pd_native_plans=pd_native_plans,shape_config=shape_config)
    app.state.native_chat_token_memo=memo
    return app

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8002);args=parser.parse_args()
    import uvicorn
    uvicorn.run("response_affinity_gateway_v12:create_app",host=args.host,port=args.port,
                workers=1,factory=True,app_dir=str(Path(__file__).parent))
