#!/usr/bin/env python3
"""Reversible single-layer prefill MoE shadow graph experiment."""
import argparse,hashlib,json
from pathlib import Path
from loop055_run212_patch import FILES,RUNNER as BASE_RUNNER
BKP=Path("/tmp/loop055_run223_backup")
MOE=r"""
# EXTREME_LOOP055_RUN223_MOE
_extreme_run223_orig_moe = _moe_forward_shared
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
    state=getattr(_bi,"_extreme_run223_state",None)
    if state is None or state.get("captured"):
        return _extreme_run223_orig_moe(hidden_states,router_logits,shared_experts_input,input_ids,layer_name,hidden_dim_unpadded)
    ctx=get_forward_context()
    ids=ctx.input_ids
    if ids is None:
        raise RuntimeError("missing context input_ids")
    name=_resolve_layer_name(layer_name)
    rank=int(_get_tp_group().rank_in_group)
    tag=state["tag"]
    outpath=_Path(state["out"])
    row={"rank":rank,"tag":tag,"layer_name":str(name),"shape":list(hidden_states.shape),"dtype":str(hidden_states.dtype),"context_input_ids_ptr":int(ids.data_ptr()),"context_input_ids_shape":list(ids.shape),"context_input_ids_dtype":str(ids.dtype)}
    torch.npu.synchronize()
    row["context_input_ids_sha256"]=_hashlib.sha256(ids.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    if name!="model.layers.0.mlp.experts" or tuple(hidden_states.shape)!=(11,4096) or router_logits.data_ptr()!=hidden_states.data_ptr() or shared_experts_input is None or shared_experts_input.data_ptr()!=hidden_states.data_ptr() or input_ids is not None:
        row["status"]="signature_mismatch"
        outpath.parent.mkdir(parents=True,exist_ok=True)
        with (outpath.parent/f"{tag}_rank{rank}_mismatch.jsonl").open("a") as f:
            f.write(_json.dumps(row)+"\n")
        state["captured"]=True
        return _extreme_run223_orig_moe(hidden_states,router_logits,shared_experts_input,input_ids,layer_name,hidden_dim_unpadded)
    source=hidden_states.detach().clone()
    torch.npu.synchronize()
    row["input_sha256"]=_hashlib.sha256(source.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    mem_before=int(torch.npu.memory_allocated())
    prod_t0=_time.perf_counter()
    prod=_extreme_run223_orig_moe(hidden_states,router_logits,shared_experts_input,input_ids,layer_name,hidden_dim_unpadded)
    torch.npu.synchronize()
    row["eager_wall_ms"]=(_time.perf_counter()-prod_t0)*1000
    row["eager_output_sha256"]=[_hashlib.sha256(v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for v in prod]
    prod_cpu=[v.detach().to(torch.float32).cpu().clone() for v in prod]
    try:
        ctrl=_extreme_run223_orig_moe(source,source,source,None,layer_name,hidden_dim_unpadded)
        torch.npu.synchronize()
        ctrl_cpu=[v.detach().to(torch.float32).cpu().clone() for v in ctrl]
        graph_state=getattr(_bi,"_extreme_run223_graph",None)
        if tag=="A":
            buf=source.clone()
            graph=torch.npu.NPUGraph()
            old_capturing=bool(getattr(ctx,"capturing",False))
            ctx.capturing=True
            try:
                t0=_time.perf_counter()
                with torch.npu.graph(graph):
                    gout=_extreme_run223_orig_moe(buf,buf,buf,None,layer_name,hidden_dim_unpadded)
                capture_s=_time.perf_counter()-t0
            finally:
                ctx.capturing=old_capturing
            graph_state={"graph":graph,"input":buf,"output":gout,"layer":str(name),"ids_ptr":int(ids.data_ptr())}
            _bi._extreme_run223_graph=graph_state
            row["capture_s"]=capture_s
        else:
            if graph_state is None or graph_state["layer"]!=str(name):
                raise RuntimeError("missing A graph")
            if graph_state["ids_ptr"]!=int(ids.data_ptr()):
                raise RuntimeError("context input_ids address changed")
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
        if tag=="B":
            ids_backup=ids.detach().clone()
            try:
                ids.fill_(42)
                torch.npu.synchronize()
                row["mutated_input_ids_sha256"]=_hashlib.sha256(ids.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                if row["mutated_input_ids_sha256"]==row["context_input_ids_sha256"]:
                    raise RuntimeError("token-ID mutation had no effect")
                changed_eager=_extreme_run223_orig_moe(source,source,source,None,layer_name,hidden_dim_unpadded)
                torch.npu.synchronize()
                changed_eager_cpu=[v.detach().to(torch.float32).cpu().clone() for v in changed_eager]
                graph_state["input"].copy_(source)
                graph_state["graph"].replay()
                torch.npu.synchronize()
                changed_graph_cpu=[v.detach().to(torch.float32).cpu().clone() for v in graph_state["output"]]
                row["mutated_graph_output_sha256"]=[_hashlib.sha256(v.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for v in graph_state["output"]]
                row["mutated_comparisons"]=[{"output":i,"max_abs":float((changed_eager_cpu[i]-changed_graph_cpu[i]).abs().max().item()),"exact":bool(torch.equal(changed_eager_cpu[i],changed_graph_cpu[i]))} for i in range(2)]
                row["mutated_graph_output_changed"]=[a!=b for a,b in zip(row["graph_output_sha256"],row["mutated_graph_output_sha256"])]
            finally:
                ids.copy_(ids_backup)
                torch.npu.synchronize()
            row["restored_input_ids_sha256"]=_hashlib.sha256(ids.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        row["memory_delta_bytes"]=int(torch.npu.memory_allocated())-mem_before
    except Exception as exc:
        row["status"]="error"
        row["error_type"]=type(exc).__name__
        row["error"]=str(exc)[:500]
    state["captured"]=True
    outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(_json.dumps(row,separators=(",",":"))+"\n")
    return prod

"""
RUNNER=BASE_RUNNER.replace("RUN212","RUN223").replace("run212","run223")
RUNNER=RUNNER.replace('state={"out":str(out),"captured":False}','state={"out":str(out),"tag":tag,"captured":False}')
RUNNER=RUNNER.replace(" or int(num_tokens_padded)!=88","")
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("action",choices=["install","restore"]);p.add_argument("--record",required=True);a=p.parse_args()
 result={"action":a.action,"files":{}}
 if a.action=="install":
  assert not BKP.exists();BKP.mkdir()
  for key,path in FILES.items():
   before=path.read_bytes();assert b"EXTREME_LOOP055_RUN223" not in before
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
   assert b"EXTREME_LOOP055_RUN223" in before
   path.write_bytes(orig)
   result["files"][key]={"patched_sha256":sha(before),"restored_sha256":sha(orig)}
  for f in BKP.iterdir():f.unlink()
  BKP.rmdir()
 out=Path(a.record);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps(result))
if __name__=="__main__":main()
