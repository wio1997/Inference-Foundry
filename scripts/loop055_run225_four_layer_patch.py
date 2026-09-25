#!/usr/bin/env python3
"""Reversible four-layer first88 MoE graph probe."""
import argparse
import hashlib
import json
from pathlib import Path
from loop055_run212_patch import FILES

BKP = Path("/tmp/loop055_run225_backup")
MOE = r"""
# EXTREME_LOOP055_RUN225_MOE
_extreme_run225_orig_moe = _moe_forward_shared
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
    state = getattr(_bi, "_extreme_run225_state", None)
    if state is None:
        return _extreme_run225_orig_moe(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded)
    name = str(_resolve_layer_name(layer_name))
    prefix = "model.layers."
    suffix = ".mlp.experts"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return _extreme_run225_orig_moe(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded)
    middle = name[len(prefix):-len(suffix)]
    if middle not in ("0", "1", "2", "3"):
        return _extreme_run225_orig_moe(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded)
    idx = int(middle)
    if idx in state["seen"]:
        return _extreme_run225_orig_moe(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded)
    state["seen"].add(idx)
    rank = int(_get_tp_group().rank_in_group)
    tag = state["tag"]
    outpath = _Path(state["out_dir"]) / f"{tag}_rank{rank}_layer{idx}.json"
    row = {"tag": tag, "rank": rank, "layer": idx, "name": name, "shape": list(hidden_states.shape), "dtype": str(hidden_states.dtype)}
    if tuple(hidden_states.shape) != (11, 4096) or router_logits.data_ptr() != hidden_states.data_ptr() or shared_experts_input is None or shared_experts_input.data_ptr() != hidden_states.data_ptr() or input_ids is not None:
        row["status"] = "signature_mismatch"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(_json.dumps(row, separators=(",", ":")) + "\n")
        raise RuntimeError("Run225 layer signature mismatch")
    ctx = get_forward_context()
    ids = ctx.input_ids
    if idx < 3:
        if ids is None:
            raise RuntimeError("missing implicit hash input IDs")
        row["context_input_ids_ptr"] = int(ids.data_ptr())
        row["context_input_ids_shape"] = list(ids.shape)
    source = hidden_states.detach().clone()
    graph_bank = getattr(_bi, "_extreme_run225_graphs", {})
    try:
        if tag == "A":
            result = _extreme_run225_orig_moe(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded)
            buf = source.clone()
            graph = torch.npu.NPUGraph()
            old_capturing = bool(getattr(ctx, "capturing", False))
            ctx.capturing = True
            try:
                t0 = _time.perf_counter()
                with torch.npu.graph(graph):
                    gout = _extreme_run225_orig_moe(buf, buf, buf, None, layer_name, hidden_dim_unpadded)
                row["capture_s"] = _time.perf_counter() - t0
            finally:
                ctx.capturing = old_capturing
            graph_bank[idx] = {"graph": graph, "input": buf, "output": gout, "ids_ptr": int(ids.data_ptr()) if idx < 3 else None}
            _bi._extreme_run225_graphs = graph_bank
            row["status"] = "captured"
        elif tag == "B":
            result = _extreme_run225_orig_moe(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded)
            torch.npu.synchronize()
            prod_cpu = [v.detach().float().cpu().clone() for v in result]
            ctrl = _extreme_run225_orig_moe(source, source, source, None, layer_name, hidden_dim_unpadded)
            torch.npu.synchronize()
            ctrl_cpu = [v.detach().float().cpu().clone() for v in ctrl]
            gs = graph_bank[idx]
            if idx < 3 and gs["ids_ptr"] != int(ids.data_ptr()):
                raise RuntimeError("hash input-ID pointer changed")
            gs["input"].copy_(source)
            gs["graph"].replay()
            torch.npu.synchronize()
            graph_cpu = [v.detach().float().cpu().clone() for v in gs["output"]]
            row["comparisons"] = [{"output": i, "graph_exact": bool(torch.equal(prod_cpu[i], graph_cpu[i])), "graph_max_abs": float((prod_cpu[i] - graph_cpu[i]).abs().max().item()), "self_max_abs": float((prod_cpu[i] - ctrl_cpu[i]).abs().max().item())} for i in range(2)]
            row["status"] = "replayed"
        elif tag == "C":
            if not state["gate"]:
                raise RuntimeError("B parity gate absent")
            gs = graph_bank[idx]
            if idx < 3 and gs["ids_ptr"] != int(ids.data_ptr()):
                raise RuntimeError("hash input-ID pointer changed")
            gs["input"].copy_(hidden_states)
            gs["graph"].replay()
            result = gs["output"]
            row["status"] = "substituted"
        elif tag == "D":
            result = _extreme_run225_orig_moe(hidden_states, router_logits, shared_experts_input, input_ids, layer_name, hidden_dim_unpadded)
            row["status"] = "eager_control"
        else:
            raise RuntimeError("bad Run225 tag")
    except Exception as exc:
        row["status"] = "error"
        row["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        raise
    finally:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(_json.dumps(row, separators=(",", ":")) + "\n")
    return result

"""
RUNNER = r"""
# EXTREME_LOOP055_RUN225_RUNNER
if os.getenv("EXTREME_RUN225_OUT_DIR"):
    _extreme_run225_orig_forward = NPUModelRunner._model_forward
    def _extreme_run225_forward(self, num_tokens_padded, *args, **kwargs):
        import builtins as _bi, json as _json, time as _time
        from pathlib import Path as _Path
        tagfile = os.getenv("EXTREME_RUN225_TAG_FILE")
        if not tagfile or not os.path.exists(tagfile) or int(num_tokens_padded) != 88:
            return _extreme_run225_orig_forward(self, num_tokens_padded, *args, **kwargs)
        ctx = get_forward_context()
        if str(getattr(ctx, "cudagraph_runtime_mode", "")) != "NONE":
            return _extreme_run225_orig_forward(self, num_tokens_padded, *args, **kwargs)
        tag = _Path(tagfile).read_text().strip()
        rank = int(get_tp_group().rank_in_group)
        outdir = _Path(os.environ["EXTREME_RUN225_OUT_DIR"])
        stagepath = outdir / f"{tag}_rank{rank}_stage.json"
        if stagepath.exists():
            return _extreme_run225_orig_forward(self, num_tokens_padded, *args, **kwargs)
        state = {"tag": tag, "out_dir": str(outdir), "seen": set(), "gate": bool(os.getenv("EXTREME_RUN225_GATE_FILE") and os.path.exists(os.environ["EXTREME_RUN225_GATE_FILE"]))}
        _bi._extreme_run225_state = state
        stage = {"tag": tag, "rank": rank, "num_tokens_padded": int(num_tokens_padded)}
        torch.npu.synchronize()
        t0 = _time.perf_counter()
        try:
            result = _extreme_run225_orig_forward(self, num_tokens_padded, *args, **kwargs)
            torch.npu.synchronize()
            stage["status"] = "pass"
            return result
        except Exception as exc:
            stage["status"] = "error"
            stage["error"] = type(exc).__name__ + ": " + str(exc)[:300]
            raise
        finally:
            stage["completion_wall_ms"] = (_time.perf_counter() - t0) * 1000
            stage["layers_seen"] = sorted(state["seen"])
            stagepath.parent.mkdir(parents=True, exist_ok=True)
            stagepath.write_text(_json.dumps(stage, separators=(",", ":")) + "\n")
            _bi._extreme_run225_state = None
    NPUModelRunner._model_forward = _extreme_run225_forward
"""
def sha(b):
    return hashlib.sha256(b).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["install", "restore"])
    p.add_argument("--record", required=True)
    a = p.parse_args()
    result = {"action": a.action, "files": {}}
    if a.action == "install":
        assert not BKP.exists()
        BKP.mkdir()
        original = {k: path.read_bytes() for k, path in FILES.items()}
        assert all(b"EXTREME_LOOP055_RUN225" not in b for b in original.values())
        needle = "direct_register_custom_op(\n    op_name=\"moe_forward_shared\","
        assert original["moe"].decode().count(needle) == 1
        patched = {"moe": original["moe"].decode().replace(needle, MOE + needle, 1).encode(),
                   "runner": original["runner"] + RUNNER.encode()}
        for key, path in FILES.items():
            compile(patched[key], str(path), "exec")
            (BKP / key).write_bytes(original[key])
        for key, path in FILES.items():
            path.write_bytes(patched[key])
            result["files"][key] = {"original_sha256": sha(original[key]), "patched_sha256": sha(patched[key])}
    else:
        assert BKP.exists()
        for key, path in FILES.items():
            before = path.read_bytes()
            original = (BKP / key).read_bytes()
            assert b"EXTREME_LOOP055_RUN225" in before
            path.write_bytes(original)
            result["files"][key] = {"patched_sha256": sha(before), "restored_sha256": sha(original)}
        for f in BKP.iterdir():
            f.unlink()
        BKP.rmdir()
    out = Path(a.record)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
