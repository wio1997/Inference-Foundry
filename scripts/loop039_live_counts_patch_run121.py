#!/usr/bin/env python3
"""Install/restore temporary live expert-count instrumentation for Run120."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
SOURCES = {
    "moe": Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py"),
    "runtime": ROOT / "runtime/extreme_decode.py",
}
BACKUPS = {k: Path(f"/tmp/extreme_run121_{k}.orig") for k in SOURCES}
MOE_MARKER = "    # EXTREME_RUN121_CAPTURE_GROUP_LIST\n"
RUNTIME_MARKER = "                    # EXTREME_RUN121_SNAPSHOT_GROUP_LIST\n"
MOE_INSERT = '''    # EXTREME_RUN121_CAPTURE_GROUP_LIST
    if (__import__("os").environ.get("EXTREME_GMM_LIVE_COUNTS_DIR")
            and tuple(hidden_states.shape) == (576, 4096)
            and tuple(group_list.shape) == (32,)):
        _refs = getattr(quant_apply_mlp, "_extreme_group_refs", None)
        if _refs is None:
            _refs = []
            quant_apply_mlp._extreme_group_refs = _refs
        _ptr = group_list.data_ptr()
        if not any(ptr == _ptr for ptr, _ in _refs) and len(_refs) < 256:
            _refs.append((_ptr, group_list))
'''
RUNTIME_INSERT = '''                    # EXTREME_RUN121_SNAPSHOT_GROUP_LIST
                    _counts_dir = os.getenv("EXTREME_GMM_LIVE_COUNTS_DIR")
                    if _counts_dir and self.state.cycle_index in (64, 65):
                        from vllm_ascend.ops.fused_moe import moe_mlp as _probe_mlp
                        torch.npu.synchronize()
                        _refs = getattr(_probe_mlp.quant_apply_mlp,
                                        "_extreme_group_refs", [])
                        _rank = (torch.distributed.get_rank()
                                 if torch.distributed.is_initialized()
                                 else os.getpid())
                        os.makedirs(_counts_dir, exist_ok=True)
                        _rows = [{"ordinal": i, "ptr": ptr,
                                  "counts": tensor.cpu().tolist()}
                                 for i, (ptr, tensor) in enumerate(_refs)]
                        _path = os.path.join(
                            _counts_dir,
                            f"rank{_rank}_cycle{self.state.cycle_index}.json")
                        with open(_path, "w", encoding="utf-8") as _handle:
                            json.dump({"rank": _rank,
                                       "cycle": self.state.cycle_index,
                                       "ref_count": len(_refs),
                                       "rows": _rows}, _handle)
'''
def sha(data):
    return hashlib.sha256(data).hexdigest()

def patch(name, src):
    if name == "moe":
        anchor = "def quant_apply_mlp(\n"
        target = "    input_hidden_dtype = hidden_states.dtype\n"
        i = src.index(anchor)
        j = src.index(target, i)
        return src[:j] + MOE_INSERT + src[j:]
    anchor = "                    target_output = self.target.execute(self.state)\n"
    assert src.count(anchor) == 1
    return src.replace(anchor, anchor + RUNTIME_INSERT)

def main():
    a = argparse.ArgumentParser()
    a.add_argument("action", choices=("install", "restore"))
    a.add_argument("--record", type=Path, required=True)
    args = a.parse_args()
    entries = []
    if args.action == "install":
        for key, source in SOURCES.items():
            if BACKUPS[key].exists():
                raise SystemExit(f"backup exists: {BACKUPS[key]}")
            original = source.read_bytes()
            marker = MOE_MARKER if key == "moe" else RUNTIME_MARKER
            if marker.encode() in original:
                raise SystemExit(f"already patched: {source}")
            patched = patch(key, original.decode()).encode()
            BACKUPS[key].write_bytes(original)
            source.write_bytes(patched)
            entries.append({"key": key, "source": str(source),
                            "original_sha256": sha(original),
                            "patched_sha256": sha(patched)})
    else:
        for key, source in SOURCES.items():
            backup = BACKUPS[key]
            marker = MOE_MARKER if key == "moe" else RUNTIME_MARKER
            if not backup.exists() or marker.encode() not in source.read_bytes():
                raise SystemExit(f"cannot safely restore: {source}")
            original = backup.read_bytes()
            source.write_bytes(original)
            backup.unlink()
            entries.append({"key": key, "source": str(source),
                            "restored_sha256": sha(source.read_bytes())})
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({"action": args.action, "files": entries}, indent=2) + "\n")
    print(json.dumps({"action": args.action, "files": entries}))

if __name__ == "__main__":
    main()
