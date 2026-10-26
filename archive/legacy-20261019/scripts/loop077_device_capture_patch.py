#!/usr/bin/env python3
"""Reversible selected-cycle route/count capture; no hot-path D2H or synchronize."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
SOURCES = {
    "moe": Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py"),
    "runtime": ROOT / "runtime/extreme_decode.py",
    "serving": ROOT / "runtime/fixed_serving.py",
}
BACKUPS = {k: Path(f"/tmp/extreme_run375_{k}.orig") for k in SOURCES}
MARKERS = {"moe": "EXTREME_RUN375_ROUTE_REFS",
           "runtime": "EXTREME_RUN375_DEVICE_STACK",
           "serving": "EXTREME_RUN375_COHORT_EXPORT"}
MOE_INSERT = """
    # EXTREME_RUN375_ROUTE_REFS
    if (__import__("os").environ.get("EXTREME_BOUND_DEVICE_CAPTURE_DIR")
            and tuple(hidden_states.shape) == (576, 4096)
            and tuple(group_list.shape) == (32,)):
        _refs = getattr(quant_apply_mlp, "_extreme_group_refs", None)
        if _refs is None:
            _refs = []
            quant_apply_mlp._extreme_group_refs = _refs
        _ptr = group_list.data_ptr()
        if not any(ptr == _ptr for ptr, _ in _refs) and len(_refs) < 256:
            _refs.append((_ptr, group_list))
"""
RUNTIME_INSERT = """
                    # EXTREME_RUN375_DEVICE_STACK
                    if (os.getenv("EXTREME_BOUND_DEVICE_CAPTURE_DIR")
                            and self.state.cycle_index in (64, 65)):
                        from vllm_ascend.ops.fused_moe import moe_mlp as _bound_mlp
                        _refs = getattr(_bound_mlp.quant_apply_mlp,
                                        "_extreme_group_refs", [])
                        if len(_refs) != 86:
                            raise RuntimeError(f"route ref count {len(_refs)} != 86")
                        _route = torch.stack([tensor for _, tensor in _refs[43:86]], dim=0)
                        _captures = getattr(self, "_bound_device_routes", None)
                        if _captures is None:
                            _captures = []
                            self._bound_device_routes = _captures
                        _captures.append((self.state.cycle_index, _route))
"""
SERVING_START = """
        # EXTREME_RUN375_COHORT_EXPORT
        if os.getenv("EXTREME_BOUND_DEVICE_CAPTURE_DIR"):
            self.runtime._bound_capture_cohort = getattr(
                self.runtime, "_bound_capture_cohort", 0) + 1
            self.runtime._bound_device_routes = []
"""
SERVING_END = """
        # EXTREME_RUN375_COHORT_EXPORT_AFTER_DRAIN
        _capture_dir = os.getenv("EXTREME_BOUND_DEVICE_CAPTURE_DIR")
        if _capture_dir:
            _captures = getattr(self.runtime, "_bound_device_routes", [])
            if [cycle for cycle, _ in _captures] != [64, 65]:
                raise RuntimeError(f"route cycles: {[cycle for cycle, _ in _captures]}")
            _rank = torch.distributed.get_rank()
            _path = Path(_capture_dir)
            _path.mkdir(parents=True, exist_ok=True)
            _cohort = 1 + len(list(_path.glob(f"rank{_rank}_cohort*.json")))
            (_path / f"rank{_rank}_cohort{_cohort}.json").write_text(json.dumps({
                "rank": _rank, "cohort": _cohort, "cycles": cycles,
                "initial_output_counts": list(self.initial_output_counts),
                "remaining": list(self.remaining),
                "accepted_counts_by_cycle_slot": counts_cpu.tolist(),
                "route_rows": [{"cycle": cycle, "counts": tensor.cpu().tolist()}
                               for cycle, tensor in _captures],
            }))
            self.runtime._bound_device_routes = []
"""

def sha(data):
    return hashlib.sha256(data).hexdigest()

def patch(key, src):
    if key == "moe":
        i = src.index("def quant_apply_mlp(\n")
        j = src.index("    input_hidden_dtype = hidden_states.dtype\n", i)
        return src[:j] + MOE_INSERT + src[j:]
    if key == "runtime":
        anchor = "                    target_output = self.target.execute(self.state)\n"
        assert src.count(anchor) == 1
        return src.replace(anchor, anchor + RUNTIME_INSERT)
    start = "        cycles = 0\n"
    end = "        counts_cpu = count_history[:cycles].cpu()\n"
    assert src.count(start) == src.count(end) == 1
    return src.replace(start, SERVING_START + start).replace(end, end + SERVING_END)

def main():
    a = argparse.ArgumentParser()
    a.add_argument("action", choices=("install", "restore"))
    a.add_argument("--record", type=Path, required=True)
    args = a.parse_args()
    rows = []
    if args.action == "install":
        for key, source in SOURCES.items():
            if BACKUPS[key].exists():
                raise SystemExit(f"backup exists: {BACKUPS[key]}")
            original = source.read_bytes()
            if MARKERS[key].encode() in original:
                raise SystemExit(f"already patched: {source}")
            edited = patch(key, original.decode()).encode()
            BACKUPS[key].write_bytes(original)
            source.write_bytes(edited)
            rows.append({"key": key, "source": str(source),
                         "original_sha256": sha(original), "patched_sha256": sha(edited)})
    else:
        for key, source in SOURCES.items():
            backup = BACKUPS[key]
            if not backup.exists() or MARKERS[key].encode() not in source.read_bytes():
                raise SystemExit(f"cannot safely restore: {source}")
            source.write_bytes(backup.read_bytes())
            backup.unlink()
            rows.append({"key": key, "source": str(source),
                         "restored_sha256": sha(source.read_bytes())})
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({"action": args.action, "files": rows}, indent=2) + "\n")

if __name__ == "__main__":
    main()
