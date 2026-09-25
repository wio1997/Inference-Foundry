#!/usr/bin/env python3
"""Reversible single-layer prefill MoE graph serving substitution experiment."""
import argparse,hashlib,json
from pathlib import Path
from loop055_run212_patch import FILES,RUNNER as BASE_RUNNER
BKP=Path("/tmp/loop055_run218_backup")
MOE=r"""
# EXTREME_LOOP055_RUN218_MOE
_extreme_run218_orig_moe = _moe_forward_shared
def _moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_experts_input: torch.Tensor | None,
    input_ids: torch.Tensor | None,
    layer_name: _layer_name_type,
    hidden_dim_unpadded: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import builtins as _bi, json as _json, time as _time, hashlib as _hashlib
    from pathlib import Path as _Path
    from vllm.distributed import get_tp_group as _get_tp_group
    state=getattr(_bi,"_extreme_run218_state",None)
    if state is None or state.get("captured"):
        return _extreme_run218_orig_moe(hidden_states,router_logits,shared_experts_input,input_ids,layer_name,hidden_dim_unpadded)
    ctx=get_forward_context()
    name=_resolve_layer_name(layer_name)
    rank=int(_get_tp_group().rank_in_group)
    tag=state["tag"]
    outpath=_Path(state["out"])
    row={"rank":rank,"tag":tag,"layer_name":str(name),"shape":list(hidden_states.shape),"dtype":str(hidden_states.dtype)}
    if name!="model.layers.0.mlp.experts" or tuple(hidden_states.shape)!=(11,4096) or router_logits.data_ptr()!=hidden_states.data_ptr() or shared_experts_input is None or shared_experts_input.data_ptr()!=hidden_states.data_ptr() or input_ids is not None:
        row["status"]="signature_mismatch"
        outpath.parent.mkdir(parents=True,exist_ok=True)
        with (outpath.parent/f"{tag}_rank{rank}_mismatch.jsonl").open("a") as f:
            f.write(_json.dumps(row)+"\n")
        state["captured"]=True
        return _extreme_run218_orig_moe(hidden_states,router_logits,shared_experts_input,input_ids,layer_name,hidden_dim_unpadded)
    if tag=="C":
        graph_state=getattr(_bi,"_extreme_run218_graph",None)
        if graph_state is None or graph_state["layer"]!=str(name):
            raise RuntimeError("missing A graph for C")
        t0=_time.perf_counter()
        graph_state["input"].copy_(hidden_states)
        graph_state["graph"].replay()
        row["submit_wall_ms"]=(_time.perf_counter()-t0)*1000
        row["status"]="substituted"
        state["captured"]=True
        outpath.parent.mkdir(parents=True,exist_ok=True)
        outpath.write_text(_json.dumps(row,separators=(",",":"))+"\n")
        return graph_state["output"]
    if tag=="D":
        t0=_time.perf_counter()
        eager=_extreme_run218_orig_moe(hidden_states,router_logits,shared_experts_input,input_ids,layer_name,hidden_dim_unpadded)
        row["submit_wall_ms"]=(_time.perf_counter()-t0)*1000
        row["status"]="eager_control"
        state["captured"]=True
        outpath.parent.mkdir(parents=True,exist_ok=True)
        outpath.write_text(_json.dumps(row,separators=(",",":"))+"\n")
        return eager
    source=hidden_states.detach().clone()
    torch.npu.synchronize()
    row["input_sha256"]=_hashlib.sha256(source.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    mem_before=int(torch.npu.memory_allocated())
    prod_t0=_time.perf_counter()
    prod=_extreme_run218_orig_moe(hidden_states,router_logits,shared_experts_input,input_ids,layer_name,hidden_dim_unpadded)
    torch.npu.synchronize()
    row["eager_wall_ms"]=(_time.perf_counter()-prod_t0)*1000
    row["eager_output_sha256"]=[_hashlib.sha256(v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for v in prod]
    prod_cpu=[v.detach().to(torch.float32).cpu().clone() for v in prod]
    try:
        ctrl=_extreme_run218_orig_moe(source,source,source,None,layer_name,hidden_dim_unpadded)
        torch.npu.synchronize()
        ctrl_cpu=[v.detach().to(torch.float32).cpu().clone() for v in ctrl]
        graph_state=getattr(_bi,"_extreme_run218_graph",None)
        if tag=="A":
            buf=source.clone()
            graph=torch.npu.NPUGraph()
            old_capturing=bool(getattr(ctx,"capturing",False))
            ctx.capturing=True
            try:
                t0=_time.perf_counter()
                with torch.npu.graph(graph):
                    gout=_extreme_run218_orig_moe(buf,buf,buf,None,layer_name,hidden_dim_unpadded)
                capture_s=_time.perf_counter()-t0
            finally:
                ctx.capturing=old_capturing
            graph_state={"graph":graph,"input":buf,"output":gout,"layer":str(name)}
            _bi._extreme_run218_graph=graph_state
            row["capture_s"]=capture_s
        else:
            if graph_state is None or graph_state["layer"]!=str(name):
                raise RuntimeError("missing A graph")
            refresh_t0=_time.perf_counter()
            graph_state["input"].copy_(source)
            torch.npu.synchronize()
            row["refresh_wall_ms"]=(_time.perf_counter()-refresh_t0)*1000
        torch.npu.synchronize()
        t0=_time.perf_counter()
        graph_state["graph"].replay()
        torch.npu.synchronize()
        row["replay_wall_ms"]=(_time.perf_counter()-t0)*1000
        row["graph_output_sha256"]=[_hashlib.sha256(v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for v in graph_state["output"]]
        replay_cpu=[v.detach().to(torch.float32).cpu().clone() for v in graph_state["output"]]
        row["status"]="captured" if tag=="A" else "replayed"
        row["comparisons"]=[]
        for i in range(2):
            p,c,g=prod_cpu[i],ctrl_cpu[i],replay_cpu[i]
            row["comparisons"].append({"output":i,"shape":list(p.shape),"eager_self_exact":bool(torch.equal(p,c)),"graph_exact":bool(torch.equal(p,g)),"eager_self_max_abs":float((p-c).abs().max().item()),"graph_max_abs":float((p-g).abs().max().item())})
        row["memory_delta_bytes"]=int(torch.npu.memory_allocated())-mem_before
    except Exception as exc:
        row["status"]="error"
        row["error_type"]=type(exc).__name__
        row["error"]=str(exc)[:500]
    state["captured"]=True
    outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(_json.dumps(row,separators=(",",":"))+"\n")
    return graph_state["output"] if tag=="B" and row["status"]=="replayed" else prod

"""
RUNNER=BASE_RUNNER.replace("RUN212","RUN218").replace("run212","run218")
RUNNER=RUNNER.replace('state={"out":str(out),"captured":False}','state={"out":str(out),"tag":tag,"captured":False}')
RUNNER=RUNNER.replace(" or int(num_tokens_padded)!=88","")
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("action",choices=["install","restore"]);p.add_argument("--record",required=True);a=p.parse_args()
 result={"action":a.action,"files":{}}
 if a.action=="install":
  assert not BKP.exists();BKP.mkdir()
  for key,path in FILES.items():
   before=path.read_bytes();assert b"EXTREME_LOOP055_RUN218" not in before
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
   assert b"EXTREME_LOOP055_RUN218" in before
   path.write_bytes(orig)
   result["files"][key]={"patched_sha256":sha(before),"restored_sha256":sha(orig)}
  for f in BKP.iterdir():f.unlink()
  BKP.rmdir()
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result))
if __name__=="__main__":main()
