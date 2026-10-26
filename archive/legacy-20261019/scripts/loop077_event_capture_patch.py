#!/usr/bin/env python3
"""Reversible selected-cycle NPU event capture without hot-path sync."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
SOURCES = {
    "runtime": ROOT / "runtime/extreme_decode.py",
    "proposer": ROOT / "bootstrap/vllm_dspark_handoff.py",
    "serving": ROOT / "runtime/fixed_serving.py",
}
BACKUPS = {key: Path(f"/tmp/extreme_run386_{key}.orig") for key in SOURCES}
MARKERS = {"runtime": "EXTREME_RUN386_SELECTED_EVENTS",
           "proposer": "EXTREME_RUN386_PROPOSER_EVENTS",
           "serving": "EXTREME_RUN386_EXPORT_EVENTS"}
START = """        # EXTREME_RUN386_EXPORT_EVENTS
        if os.getenv("EXTREME_BOUND_EVENT_DIR"):
            self.runtime.diagnostic_events = []
            self.runtime.proposer.dag_events = []
"""
END = """        # EXTREME_RUN386_EXPORT_EVENTS_AFTER_COHORT
        _event_dir = os.getenv("EXTREME_BOUND_EVENT_DIR")
        if _event_dir:
            _anchor = torch.npu.Event(enable_timing=True)
            _anchor_before_ns = time.time_ns()
            _anchor.record()
            _anchor.synchronize()
            _anchor_after_ns = time.time_ns()
            _runtime_events = self.runtime.diagnostic_events
            _draft_events = self.runtime.proposer.dag_events
            if len(_runtime_events) != 2 or len(_draft_events) != 2:
                raise RuntimeError(f"event counts {len(_runtime_events)}, {len(_draft_events)}")
            def _rows(source):
                return [{"cycle": 64 + i,
                         "markers": [{"label": label,
                                      "ms_before_anchor": event.elapsed_time(_anchor)}
                                     for label, event in marks]}
                        for i, marks in enumerate(source)]
            _rank = torch.distributed.get_rank()
            _path = Path(_event_dir)
            _path.mkdir(parents=True, exist_ok=True)
            _cohort = 1 + len(list(_path.glob(f"rank{_rank}_cohort*.json")))
            (_path / f"rank{_rank}_cohort{_cohort}.json").write_text(json.dumps({
                "rank": _rank, "cohort": _cohort, "cycles": cycles,
                "anchor_before_ns": _anchor_before_ns,
                "anchor_after_ns": _anchor_after_ns,
                "initial_output_counts": list(self.initial_output_counts),
                "remaining": list(self.remaining),
                "accepted_counts_by_cycle_slot": counts_cpu.tolist(),
                "runtime_events": _rows(_runtime_events),
                "dspark_events": _rows(_draft_events),
            }))
"""

def digest(data):
    return hashlib.sha256(data).hexdigest()

def patch(key, source):
    if key == "runtime":
        anchor = "if diag is not None or self._profile_dag:"
        assert source.count(anchor) == 1
        return source.replace("        markers = []\n", "        markers = []\n        _run386_selected_cycle = self.state.cycle_index in (64, 65)\n").replace(anchor, "# EXTREME_RUN386_SELECTED_EVENTS\n            if diag is not None or self._profile_dag or (os.getenv(\"EXTREME_BOUND_EVENT_DIR\") and _run386_selected_cycle):")
    if key == "proposer":
        anchor = "if self._profile_dag:"
        assert source.count(anchor) == 1
        return source.replace(anchor, "# EXTREME_RUN386_PROPOSER_EVENTS\n            if self._profile_dag or (os.getenv(\"EXTREME_BOUND_EVENT_DIR\") and state.cycle_index in (64, 65)):")
    anchor = "        counts_cpu = count_history[:cycles].cpu()\n"
    start = "        cycles = 0\n"
    assert source.count(anchor) == source.count(start) == 1
    return source.replace("import os\n", "import os\nimport time\n").replace(start, START + start).replace(anchor, anchor + END)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("install", "restore"))
    p.add_argument("--record", type=Path, required=True)
    args = p.parse_args()
    rows = []
    if args.action == "install":
        for key, file in SOURCES.items():
            if BACKUPS[key].exists() or MARKERS[key].encode() in file.read_bytes():
                raise SystemExit(f"unsafe install: {key}")
            original = file.read_bytes()
            edited = patch(key, original.decode()).encode()
            compile(edited, str(file), "exec")
            BACKUPS[key].write_bytes(original)
            file.write_bytes(edited)
            rows.append({"key": key, "source": str(file),
                         "original_sha256": digest(original), "patched_sha256": digest(edited)})
    else:
        for key, file in SOURCES.items():
            if not BACKUPS[key].exists() or MARKERS[key].encode() not in file.read_bytes():
                raise SystemExit(f"unsafe restore: {key}")
            file.write_bytes(BACKUPS[key].read_bytes())
            BACKUPS[key].unlink()
            rows.append({"key": key, "source": str(file), "restored_sha256": digest(file.read_bytes())})
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps({"action": args.action, "files": rows}, indent=2) + "\n")

if __name__ == "__main__":
    main()
