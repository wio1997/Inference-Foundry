#!/usr/bin/env python3
"""Reversible first-prefill MoE custom-op ABI probe."""
import argparse,hashlib,json
from pathlib import Path
BASE=Path("/data/wio/vllm_ascend_26/framework")
FILES={"moe":BASE/"vllm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py",
       "runner":BASE/"vllm-ascend/vllm_ascend/worker/model_runner_v1.py"}
BKP=Path("/tmp/loop055_run212_backup")
MOE=r"""
# EXTREME_LOOP055_RUN212_MOE
_extreme_run212_orig_moe = _moe_forward_shared
def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import builtins as _bi, hashlib as _hashlib, json as _json, os as _os
    from pathlib import Path as _Path
    state=getattr(_bi,"_extreme_run212_state",None)
    if state is None or state.get("captured"):
        return _extreme_run212_orig_moe(hidden_states,router_logits,shared_experts_input,input_ids,layer_name,hidden_dim_unpadded)
    ctx=get_forward_context()
    idx=int(getattr(ctx,"moe_layer_index",-1))
    name=_resolve_layer_name(layer_name)
    actual=ctx.all_moe_layers[idx] if name=="from_forward_context" else name
    layer=ctx.no_compile_layers[actual]
    def tensor(t):
        if t is None:return None
        q={"shape":list(t.shape),"dtype":str(t.dtype),"stride":list(t.stride()),"address":int(t.data_ptr())}
        if t.numel()<=1000000:
            q["sha256"]=_hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        return q
    before={"layer_name":str(actual),"encoded_layer_name":str(name),"moe_layer_index_before":idx,
            "hidden_states":tensor(hidden_states),"router_logits":tensor(router_logits),
            "shared_experts_input":tensor(shared_experts_input),"input_ids":tensor(input_ids),
            "hidden_dim_unpadded":int(hidden_dim_unpadded),
            "dynamic_eplb":bool(getattr(layer,"dynamic_eplb",False)),
            "multi_stage":bool(getattr(layer,"multi_stage",False)),
            "multistream_overlap_shared_expert":bool(getattr(layer,"multistream_overlap_shared_expert",False)),
            "quant_type":str(getattr(layer,"quant_type",None)),
            "lora_context_present":getattr(getattr(layer,"routed_experts",None),"_ascend_moe_lora_context",None) is not None,
            "dp_metadata_type":type(getattr(ctx,"dp_metadata",None)).__name__}
    try:
        from vllm_ascend.ops.fused_moe.fused_moe import _EXTRA_CTX
        before["moe_comm_type"]=str(getattr(_EXTRA_CTX,"moe_comm_type",None))
        before["flash_comm_v1_enabled"]=bool(getattr(_EXTRA_CTX,"flash_comm_v1_enabled",False))
        before["eplb_heat_collection_status"]=bool(getattr(_EXTRA_CTX,"eplb_heat_collection_status",False))
    except Exception as exc:
        before["extra_ctx_error"]=type(exc).__name__+":"+str(exc)[:120]
    result=_extreme_run212_orig_moe(hidden_states,router_logits,shared_experts_input,input_ids,layer_name,hidden_dim_unpadded)
    before["moe_layer_index_after"]=int(getattr(ctx,"moe_layer_index",-1))
    before["outputs"]=[tensor(v) for v in result]
    state["captured"]=True
    out=_Path(state["out"])
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(_json.dumps(before,separators=(",",":"))+"\n")
    return result

"""
RUNNER=r"""
# EXTREME_LOOP055_RUN212_RUNNER
if os.getenv("EXTREME_RUN212_OUT_DIR"):
    _extreme_run212_orig_forward=NPUModelRunner._model_forward
    def _extreme_run212_forward(self,num_tokens_padded,*args,**kwargs):
        import builtins as _bi
        from pathlib import Path as _Path
        tagfile=os.getenv("EXTREME_RUN212_TAG_FILE")
        if not tagfile or not os.path.exists(tagfile) or int(num_tokens_padded)!=88:
            return _extreme_run212_orig_forward(self,num_tokens_padded,*args,**kwargs)
        ctx=get_forward_context()
        if str(getattr(ctx,"cudagraph_runtime_mode",""))!="NONE":
            return _extreme_run212_orig_forward(self,num_tokens_padded,*args,**kwargs)
        tag=_Path(tagfile).read_text().strip()
        rank=int(get_tp_group().rank_in_group)
        out=_Path(os.environ["EXTREME_RUN212_OUT_DIR"])/f"{tag}_rank{rank}.json"
        if out.exists():
            return _extreme_run212_orig_forward(self,num_tokens_padded,*args,**kwargs)
        state={"out":str(out),"captured":False}
        _bi._extreme_run212_state=state
        try:
            return _extreme_run212_orig_forward(self,num_tokens_padded,*args,**kwargs)
        finally:
            _bi._extreme_run212_state=None
    NPUModelRunner._model_forward=_extreme_run212_forward
"""
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("action",choices=["install","restore"]);p.add_argument("--record",required=True);a=p.parse_args()
 result={"action":a.action,"files":{}}
 if a.action=="install":
  assert not BKP.exists();BKP.mkdir()
  for key,path in FILES.items():
   before=path.read_bytes();assert b"EXTREME_LOOP055_RUN212" not in before
   (BKP/key).write_bytes(before)
   if key=="moe":
    s=before.decode();needle="direct_register_custom_op(\n    op_name=\"moe_forward_shared\","
    assert s.count(needle)==1
    after=s.replace(needle,MOE+needle,1).encode()
   else:after=before+RUNNER.encode()
   compile(after,str(path),"exec");path.write_bytes(after)
   result["files"][key]={"original_sha256":sha(before),"patched_sha256":sha(after)}
 else:
  assert BKP.exists()
  for key,path in FILES.items():
   before=path.read_bytes();orig=(BKP/key).read_bytes()
   assert b"EXTREME_LOOP055_RUN212" in before
   path.write_bytes(orig)
   result["files"][key]={"patched_sha256":sha(before),"restored_sha256":sha(orig)}
  for f in BKP.iterdir():f.unlink()
  BKP.rmdir()
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result))
if __name__=="__main__":main()
