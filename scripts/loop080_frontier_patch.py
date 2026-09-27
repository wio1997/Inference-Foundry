#!/usr/bin/env python3
"""Second-stage reversible source patch for the selected Bound frontier.

The Loop079 formal ledger is installed first. `check` composes both patches
offline; `install` accepts only the pinned first-stage source bytes; `restore`
accepts only this patch's exact bytes. No service or NPU action occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import loop079_formal_ledger_patch as base

ROOT = Path("/data/wio/Inference_Foundry")
FRAME = Path("/data/wio/vllm_ascend_26/framework")
HELPER = ROOT / "scripts/loop080_frontier_server.py"
SOURCES = {
    "runner": FRAME / "vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
    "serving": ROOT / "runtime/fixed_serving.py",
    "runtime": ROOT / "runtime/extreme_decode.py",
}
RUNTIME_ORIGINAL = "eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499"
PRIMARY_PATCHED = {
    "runner": "f448c48368d8cf98b1df26bcdf5351e6761356542b686f51a441a4ba2c62a3dd",
    "serving": "5295756b0c9a3f24aac18b81d96bde3efb380ed117426e795d1f9a68db8b4238",
}
MARK = "EXTREME_LOOP080_BOUND_FRONTIER"


def once(src: str, old: str, new: str) -> str:
    return base.once(src, old, new)


def patch(key: str, src: str) -> str:
    if MARK in src:
        raise ValueError("already patched")
    if key == "runner":
        src = once(src,
            "        self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)\n",
            "        # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "        if os.getenv('EXTREME_FRONTIER_DIR'):\n"
            "            from scripts import loop080_frontier_server as _frontier\n"
            "            _frontier.prepare_ids_before(self, scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)\n"
            "        self._prepare_input_ids(scheduler_output, num_reqs, total_num_scheduled_tokens, cu_num_tokens)\n"
            "        if os.getenv('EXTREME_FRONTIER_DIR'):\n"
            "            _frontier.prepare_ids_after(self)\n")
        src = once(src,
            "        if cp_async_rebuild.positions_ready_on_device:\n",
            "        # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "        if os.getenv('EXTREME_FRONTIER_DIR'):\n"
            "            _frontier.dcp_result(self, cp_async_rebuild, with_prefill)\n"
            "        if cp_async_rebuild.positions_ready_on_device:\n")
        src = once(src,
            "            hidden_states = self._model_forward(\n"
            "                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs\n"
            "            )\n",
            "            # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "            if os.getenv('EXTREME_FRONTIER_DIR'):\n"
            "                from scripts import loop080_frontier_server as _frontier\n"
            "                _frontier.forward_before(self, scheduler_output, num_tokens_padded)\n"
            "            hidden_states = self._model_forward(\n"
            "                num_tokens_padded, input_ids, positions, intermediate_tensors, inputs_embeds, **model_kwargs\n"
            "            )\n"
            "            if os.getenv('EXTREME_FRONTIER_DIR'):\n"
            "                _frontier.forward_after(self)\n")
        src = once(src,
            "            self._copy_draft_token_ids_to_cpu(scheduler_output)\n",
            "            self._copy_draft_token_ids_to_cpu(scheduler_output)\n"
            "            # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "            if os.getenv('EXTREME_FRONTIER_DIR'):\n"
            "                from scripts import loop080_frontier_server as _frontier\n"
            "                _frontier.draft_copy_result(self, scheduler_output)\n")
        src = once(src,
            "                _formal_ledger.handoff(self, scheduler_output, _extreme_req_ids)\n",
            "                _formal_ledger.handoff(self, scheduler_output, _extreme_req_ids)\n"
            "                # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "                if os.getenv('EXTREME_FRONTIER_DIR'):\n"
            "                    from scripts import loop080_frontier_server as _frontier\n"
            "                    _frontier.handoff(self, _extreme_runtime, _extreme_req_ids)\n")
        src = once(src,
            "                ).run()\n"
            "                torch.npu.synchronize()\n"
            "                _extreme_wall_seconds = (\n"
            "                    time.perf_counter() - _extreme_wall_start\n"
            "                )\n",
            "                ).run()\n"
            "                torch.npu.synchronize()\n"
            "                _extreme_wall_seconds = (\n"
            "                    time.perf_counter() - _extreme_wall_start\n"
            "                )\n"
            "                # EXTREME_LOOP080_BOUND_FRONTIER: existing sync precedes reduction\n"
            "                if os.getenv('EXTREME_FRONTIER_DIR'):\n"
            "                    _frontier.cohort_done(self, _extreme_runtime)\n")
    elif key == "serving":
        src = once(src,
            "        cycles = 0\n"
            "        for index in range(self.max_cycles):\n",
            "        # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "        if getattr(self.runtime, '_frontier_cohort', None) == 5:\n"
            "            from scripts import loop080_frontier_server as _frontier\n"
            "            _frontier.history_begin(self, token_history, count_history)\n"
            "        cycles = 0\n"
            "        for index in range(self.max_cycles):\n")
        src = once(src,
            "            count_history[index].copy_(self.runtime.state.num_sampled)\n"
            "            cycles = index + 1\n",
            "            count_history[index].copy_(self.runtime.state.num_sampled)\n"
            "            # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "            if getattr(self.runtime, '_frontier_cohort', None) == 5:\n"
            "                _frontier.history_after_copy(self, index)\n"
            "            cycles = index + 1\n")
        src = once(src,
            "        # The last speculative preparation has no consumer in this cohort.\n",
            "        # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "        if getattr(self.runtime, '_frontier_cohort', None) == 5:\n"
            "            _frontier.history_after_loop(self, cycles)\n"
            "        # The last speculative preparation has no consumer in this cohort.\n")
    elif key == "runtime":
        src = once(src,
            "            mark(\"begin\")\n"
            "            use_scheduled = self._schedule_pending\n",
            "            mark(\"begin\")\n"
            "            # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "            if getattr(self, '_frontier_cohort', None) == 6 and self.state.cycle_index == 0:\n"
            "                from scripts import loop080_frontier_server as _frontier\n"
            "                _frontier.first_target_mark(self, 'before_prepare')\n"
            "            use_scheduled = self._schedule_pending\n")
        src = once(src,
            "            if self.kv_slot_audit is not None:\n"
            "                self.kv_slot_audit.observe(\n",
            "            # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "            if getattr(self, '_frontier_cohort', None) == 6 and self.state.cycle_index == 0:\n"
            "                _frontier.first_target_mark(self, 'before_target_consumer')\n"
            "            if self.kv_slot_audit is not None:\n"
            "                self.kv_slot_audit.observe(\n")
        src = once(src,
            "            mark(\"acceptance\")\n"
            "            if diag is not None:\n",
            "            mark(\"acceptance\")\n"
            "            # EXTREME_LOOP080_BOUND_FRONTIER\n"
            "            if getattr(self, '_frontier_cohort', None) == 6 and self.state.cycle_index == 0:\n"
            "                _frontier.first_target_mark(self, 'after_acceptance_current_stream')\n"
            "            if diag is not None:\n")
    else:
        raise ValueError(key)
    return src


def preview():
    compile(HELPER.read_bytes(), str(HELPER), "exec")
    originals = {}
    for key in ("runner", "serving"):
        current = SOURCES[key].read_bytes()
        digest = base.sha(current)
        if digest == base.SOURCES[key][1]:
            stage_one = base.patch(key, current.decode()).encode()
            if base.sha(stage_one) != PRIMARY_PATCHED[key]:
                raise ValueError(f"{key} primary patch changed")
        elif digest == PRIMARY_PATCHED[key]:
            stage_one = current
        else:
            raise ValueError(f"{key} source is neither original nor pinned primary patch: {digest}")
        originals[key] = stage_one
    originals["runtime"] = SOURCES["runtime"].read_bytes()
    if base.sha(originals["runtime"]) != RUNTIME_ORIGINAL:
        raise ValueError("runtime original SHA changed")
    files = {}
    buffers = {}
    for key, path in SOURCES.items():
        old = originals[key]
        new = patch(key, old.decode()).encode()
        compile(new, str(path), "exec")
        files[key] = dict(path=str(path), original=base.sha(old), patched=base.sha(new),
                          original_bytes=len(old), patched_bytes=len(new))
        buffers[key] = (old, new)
    return files, buffers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("check", "install", "restore"))
    ap.add_argument("--state-dir", type=Path)
    ap.add_argument("--record", required=True, type=Path)
    ap.add_argument("--offline-confirmed", action="store_true")
    args = ap.parse_args()
    if args.action in ("install", "restore") and not args.offline_confirmed:
        raise ValueError("source mutation requires guarded offline confirmation")
    if args.action in ("check", "install"):
        files, buffers = preview()
        record = dict(action=args.action, marker=MARK, helper_sha256=base.sha(HELPER.read_bytes()),
                      primary_patch_sha256=base.sha(Path(base.__file__).read_bytes()),
                      files=files, installed=False)
        if args.action == "install":
            if args.state_dir is None or args.state_dir.exists():
                raise ValueError("install requires a fresh state-dir")
            args.state_dir.mkdir(parents=True)
            for key, (old, _) in buffers.items():
                if base.sha(SOURCES[key].read_bytes()) != base.sha(old):
                    raise ValueError(f"{key} is not at pinned stage-one source")
                base.atomic_replace(args.state_dir / f"{key}.stage1.py", old)
            base.write_json(args.state_dir / "manifest.json", record)
            for key, (old, new) in buffers.items():
                path = SOURCES[key]
                if base.sha(path.read_bytes()) != base.sha(old):
                    raise ValueError(f"{key} changed during install")
                base.atomic_replace(path, new)
                if base.sha(path.read_bytes()) != base.sha(new):
                    raise ValueError(f"{key} patch verify failed")
            record["installed"] = True
    else:
        if args.state_dir is None:
            raise ValueError("restore requires state-dir")
        saved = json.loads((args.state_dir / "manifest.json").read_text())
        if saved["marker"] != MARK or set(saved["files"]) != set(SOURCES):
            raise ValueError("state manifest mismatch")
        files = saved["files"]
        for key, path in SOURCES.items():
            old = (args.state_dir / f"{key}.stage1.py").read_bytes()
            if base.sha(old) != files[key]["original"]:
                raise ValueError(f"{key} backup changed")
            if base.sha(path.read_bytes()) not in (files[key]["original"], files[key]["patched"]):
                raise ValueError(f"{key} has unknown source bytes")
        for key, path in SOURCES.items():
            old = (args.state_dir / f"{key}.stage1.py").read_bytes()
            if base.sha(path.read_bytes()) != base.sha(old):
                base.atomic_replace(path, old)
        record = dict(saved, action="restore", restored=True)
    base.write_json(args.record, record)
    print(json.dumps(dict(action=args.action, marker=MARK,
                          files=len(record["files"]), record=str(args.record),
                          sha256=base.sha(args.record.read_bytes()))))


if __name__ == "__main__":
    main()
