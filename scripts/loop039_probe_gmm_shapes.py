#!/usr/bin/env python3
"""Temporarily instrument borrowed MoE source for one shape-only diagnostic run."""
import argparse
import hashlib
import json
import os
from pathlib import Path

SOURCE = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/ops/fused_moe/moe_mlp.py")
BACKUP = Path("/tmp/extreme_loop039_moe_mlp.py.orig")
ANCHOR = "def quant_apply_mlp(\n"
TARGET = "    input_hidden_dtype = hidden_states.dtype\n"
MARKER = "    # EXTREME_LOOP039_GMM_SHAPE_PROBE\n"
INSERT = '''    # EXTREME_LOOP039_GMM_SHAPE_PROBE
    _probe_dir = __import__("os").environ.get("EXTREME_GMM_SHAPE_PROBE_DIR")
    if _probe_dir:
        import json as _probe_json
        import os as _probe_os

        def _probe_desc(value):
            if isinstance(value, (list, tuple)):
                return [_probe_desc(item) for item in value]
            if value is None:
                return None
            return {"shape": list(value.shape), "dtype": str(value.dtype)}

        _probe_row = {
            "pid": _probe_os.getpid(),
            "rank": _probe_os.environ.get("RANK"),
            "local_rank": _probe_os.environ.get("LOCAL_RANK"),
            "hidden_states": _probe_desc(hidden_states),
            "w1": _probe_desc(w1),
            "w1_scale": _probe_desc(w1_scale),
            "w2": _probe_desc(w2),
            "w2_scale": _probe_desc(w2_scale),
            "group_list": _probe_desc(group_list),
            "group_list_type": group_list_type,
            "dynamic_scale": _probe_desc(dynamic_scale),
            "w4a8_per_channel": use_w4a8_per_channel_gmm_swiglu,
            "activation": str(activation),
        }
        _probe_key = _probe_json.dumps(_probe_row, sort_keys=True)
        _probe_seen = getattr(quant_apply_mlp, "_extreme_shape_probe_seen", set())
        if _probe_key not in _probe_seen and len(_probe_seen) < 500:
            _probe_seen.add(_probe_key)
            quant_apply_mlp._extreme_shape_probe_seen = _probe_seen
            _probe_path = _probe_os.path.join(_probe_dir, f"pid{_probe_os.getpid()}.jsonl")
            with open(_probe_path, "a", encoding="utf-8") as _probe_file:
                _probe_file.write(_probe_key + "\\n")
'''
def digest(x):
    return hashlib.sha256(x).hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("install", "restore"))
    p.add_argument("--record", required=True, type=Path)
    a = p.parse_args()
    data = SOURCE.read_bytes()
    if a.action == "install":
        if MARKER.encode() in data or BACKUP.exists():
            raise SystemExit("probe already installed or backup exists")
        text = data.decode()
        section = text.index(ANCHOR)
        target = text.index(TARGET, section)
        patched = (text[:target] + INSERT + text[target:]).encode()
        BACKUP.write_bytes(data)
        SOURCE.write_bytes(patched)
        result = {"action": "install", "source": str(SOURCE), "original_sha256": digest(data), "patched_sha256": digest(patched)}
    else:
        if not BACKUP.exists() or MARKER.encode() not in data:
            raise SystemExit("probe backup missing or marker absent")
        original = BACKUP.read_bytes()
        SOURCE.write_bytes(original)
        BACKUP.unlink()
        result = {"action": "restore", "source": str(SOURCE), "original_sha256": digest(original), "restored_sha256": digest(SOURCE.read_bytes())}
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))

if __name__ == "__main__":
    main()
