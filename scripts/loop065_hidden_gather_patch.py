#!/usr/bin/env python3
"""SHA-guarded one-layer Target decode hidden-gather scheduling diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py")
BASE_SHA = "27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e"
OLD = """        hidden_states_handle = None
        overlap_hidden_states_gather = (
            has_prefill and need_gather_q_kv and _EXTRA_CTX.flash_comm_v1_enabled
        )
        if overlap_hidden_states_gather:
            hidden_states, hidden_states_handle = all_gather_async(hidden_states_local, self.tp_group)
        else:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states_local, need_gather_q_kv
            )
"""
NEW = """        hidden_states_handle = None
        from vllm.forward_context import get_forward_context
        _extreme_hidden_mode = __import__("os").getenv("EXTREME_HIDDEN_GATHER_MODE", "off")
        _extreme_hidden_selected = (
            _extreme_hidden_mode in ("immediate", "delayed")
            and not has_prefill and need_gather_q_kv
            and common_attn_metadata.num_actual_tokens == 96
            and layer_name == "model.layers.2.self_attn.attn"
            and not getattr(get_forward_context(), "is_draft_model", False)
        )
        overlap_hidden_states_gather = (
            (has_prefill and need_gather_q_kv and _EXTRA_CTX.flash_comm_v1_enabled)
            or _extreme_hidden_selected
        )
        if _extreme_hidden_selected and (
            hidden_states_local.shape != (12, 4096) or hidden_states_local.dtype != torch.bfloat16
        ):
            raise RuntimeError("hidden gather fixed shape/dtype guard failed")
        if overlap_hidden_states_gather:
            hidden_states, hidden_states_handle = all_gather_async(hidden_states_local, self.tp_group)
            if _extreme_hidden_selected:
                if not getattr(self, "_extreme_hidden_gather_logged", False):
                    print(
                        f"EXTREME_HIDDEN_GATHER rank={self.tp_rank} layer={layer_name} "
                        f"mode={_extreme_hidden_mode} input={tuple(hidden_states_local.shape)} "
                        f"output={tuple(hidden_states.shape)} pad={_EXTRA_CTX.pad_size}",
                        flush=True,
                    )
                    self._extreme_hidden_gather_logged = True
                if _extreme_hidden_mode == "immediate":
                    hidden_states_handle.wait()
                    hidden_states_handle = None
                    if _EXTRA_CTX.pad_size > 0:
                        hidden_states = hidden_states[: -_EXTRA_CTX.pad_size]
        else:
            hidden_states = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
                hidden_states_local, need_gather_q_kv
            )
"""

def sha(data):
    return hashlib.sha256(data).hexdigest()

def render(source):
    if source.count(OLD) != 1:
        raise RuntimeError("hidden-gather source anchor changed")
    changed = source.replace(OLD, NEW, 1)
    compile(changed, str(SOURCE), "exec")
    return changed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("preview", "install", "restore"))
    ap.add_argument("--record")
    a = ap.parse_args()
    original = SOURCE.read_bytes()
    if a.action in ("preview", "install"):
        if sha(original) != BASE_SHA:
            raise SystemExit("borrowed source SHA changed; refusing patch")
        changed = render(original.decode()).encode()
        print(json.dumps({"base_sha": BASE_SHA, "patched_sha": sha(changed), "patched_bytes": len(changed)}))
        if a.action == "preview":
            return
        if not a.record:
            raise SystemExit("--record required")
        record = Path(a.record)
        record.parent.mkdir(parents=True, exist_ok=True)
        backup = record.with_suffix(".original.py")
        backup.write_bytes(original)
        SOURCE.write_bytes(changed)
        record.write_text(json.dumps({"source": str(SOURCE), "backup": str(backup), "base_sha": BASE_SHA, "patched_sha": sha(changed)}, indent=2))
    else:
        if not a.record:
            raise SystemExit("--record required")
        info = json.loads(Path(a.record).read_text())
        backup = Path(info["backup"]).read_bytes()
        if sha(backup) != BASE_SHA:
            raise SystemExit("backup SHA mismatch")
        if sha(original) == BASE_SHA:
            print("already restored")
            return
        if sha(original) != info["patched_sha"]:
            raise SystemExit("patched source changed; refusing overwrite")
        SOURCE.write_bytes(backup)
        print("restored", sha(SOURCE.read_bytes()))

if __name__ == "__main__":
    main()
