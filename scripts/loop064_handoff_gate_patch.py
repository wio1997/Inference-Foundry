#!/usr/bin/env python3
"""SHA-guarded, reversible diagnostic of the fixed-cohort handoff predicate."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE = Path("/data/wio/vllm_ascend_26/framework/vllm-ascend/vllm_ascend/worker/model_runner_v1.py")
BASE_SHA = "004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba"
ANCHOR = """        if (
            _extreme_run_dir
            and use_spec_decode
            and num_reqs == 12
            and scheduler_output.total_num_scheduled_tokens == 96
            and spec_decode_common_attn_metadata is not None
            and _extreme_can_start
        ):
"""
PROBE = """
        if (_extreme_run_dir and _extreme_serve
                and os.getenv("EXTREME_HANDOFF_GATE_PROBE") == "1"
                and get_tp_group().rank_in_group == 0):
            _probe_state = getattr(self, "_extreme_gate_probe_state", None)
            if _probe_state is None:
                _probe_state = {"step": 0, "reason_counts": {}, "shape_counts": {},
                                "memberships": set(), "samples": 0, "dropped": 0}
                self._extreme_gate_probe_state = _probe_state
            _probe_state["step"] += 1
            _probe_step = _probe_state["step"]
            _probe_map = {str(k): int(v) for k, v in
                          scheduler_output.num_scheduled_tokens.items()}
            _probe_hist = {}
            for _probe_n in _probe_map.values():
                _probe_hist[_probe_n] = _probe_hist.get(_probe_n, 0) + 1
            _probe_total = int(scheduler_output.total_num_scheduled_tokens)
            _probe_conditions = {
                "spec": bool(use_spec_decode),
                "req12": num_reqs == 12,
                "total96": _probe_total == 96,
                "metadata": spec_decode_common_attn_metadata is not None,
                "new_tuple": bool(_extreme_can_start),
                "uniform8": num_reqs == 12 and len(_probe_map) == 12
                            and all(n == 8 for n in _probe_map.values()),
            }
            _probe_reason = next((name for name, ok in
                                  _probe_conditions.items()
                                  if name != "uniform8" and not ok), "ready")
            _probe_counts = _probe_state["reason_counts"]
            _probe_counts[_probe_reason] = _probe_counts.get(_probe_reason, 0) + 1
            _probe_shape = (num_reqs, _probe_total,
                            tuple(sorted(_probe_hist.items())),
                            tuple(sorted(_probe_conditions.items())))
            _probe_shape_key = str(_probe_shape)
            _probe_shapes = _probe_state["shape_counts"]
            _probe_shapes[_probe_shape_key] = _probe_shapes.get(_probe_shape_key, 0) + 1
            _probe_new_membership = _extreme_req_ids not in _probe_state["memberships"]
            if _probe_new_membership:
                _probe_state["memberships"].add(_extreme_req_ids)
            if _probe_reason == "ready" or _probe_new_membership:
                if _probe_reason == "ready" or _probe_state["samples"] < 128:
                    _probe_state["samples"] += 1
                    print("EXTREME_HANDOFF_GATE_SAMPLE " + json.dumps({
                        "step": _probe_step, "monotonic_ns": time.monotonic_ns(),
                        "reason": _probe_reason, "conditions": _probe_conditions,
                        "num_reqs": num_reqs, "scheduled_total": _probe_total,
                        "scheduled_hist": sorted(_probe_hist.items()),
                        "req_ids": list(_extreme_req_ids),
                        "scheduled_by_req": _probe_map,
                        "unscheduled_input_ids": sorted(set(_extreme_req_ids)
                                                        - set(_probe_map)),
                    }, separators=(",", ":")), flush=True)
                else:
                    _probe_state["dropped"] += 1
            if _probe_step % 256 == 0:
                _probe_top_shapes = sorted(
                    _probe_shapes.items(), key=lambda item: -item[1])[:20]
                print("EXTREME_HANDOFF_GATE_SUMMARY " + json.dumps({
                    "step": _probe_step, "monotonic_ns": time.monotonic_ns(),
                    "reason_counts": _probe_counts,
                    "top_shape_counts": _probe_top_shapes,
                    "unique_shapes": len(_probe_shapes),
                    "unique_memberships": len(_probe_state["memberships"]),
                    "samples": _probe_state["samples"],
                    "dropped": _probe_state["dropped"],
                }, separators=(",", ":")), flush=True)
"""


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def render(original: str) -> str:
    if original.count(ANCHOR) != 1:
        raise ValueError("handoff anchor not unique")
    changed = original.replace(ANCHOR, PROBE + ANCHOR)
    compile(changed, str(SOURCE), "exec")
    return changed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preview", "install", "restore"))
    parser.add_argument("--record")
    args = parser.parse_args()
    raw = SOURCE.read_bytes()
    if args.action in ("preview", "install"):
        if sha(raw) != BASE_SHA:
            raise SystemExit("borrowed runner SHA changed; refusing")
        patched = render(raw.decode()).encode()
        print(json.dumps({"base_sha": BASE_SHA, "patched_sha": sha(patched),
                          "base_bytes": len(raw), "patched_bytes": len(patched)}))
        if args.action == "preview":
            return
        if not args.record:
            raise SystemExit("--record required")
        record = Path(args.record)
        record.parent.mkdir(parents=True, exist_ok=True)
        backup = record.with_suffix(".original.py")
        backup.write_bytes(raw)
        SOURCE.write_bytes(patched)
        record.write_text(json.dumps({"source": str(SOURCE), "backup": str(backup),
                                      "base_sha": BASE_SHA, "patched_sha": sha(patched)}, indent=2))
    else:
        if not args.record:
            raise SystemExit("--record required")
        info = json.loads(Path(args.record).read_text())
        backup = Path(info["backup"]).read_bytes()
        if sha(backup) != BASE_SHA:
            raise SystemExit("backup mismatch")
        if sha(raw) == BASE_SHA:
            print("already restored")
            return
        if sha(raw) != info["patched_sha"]:
            raise SystemExit("patched source changed; refusing overwrite")
        SOURCE.write_bytes(backup)
        print("restored", sha(SOURCE.read_bytes()))

if __name__ == "__main__":
    main()
