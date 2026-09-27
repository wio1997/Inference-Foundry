#!/usr/bin/env python3
"""SHA-pinned reversible Host timeline hooks in four borrowed Python sources."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path("/data/wio/Inference_Foundry")
FRAME = Path("/data/wio/vllm_ascend_26/framework")
SOURCES = {
    "scheduler": (FRAME / "vllm-ascend/vllm_ascend/patch/platform/patch_kv_delivery_preemption.py",
                  "5ba894b9ec003502c008e12281c260787ed64a419327c451c8e67374bf525e8d"),
    "runner": (FRAME / "vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
               "004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba"),
    "output": (FRAME / "vllm/vllm/v1/engine/output_processor.py",
               "ee10351275d90796c8b901a5f4b23d5a046ef6ee72fd2921aff2ae78ca58bd9b"),
    "api": (FRAME / "vllm/vllm/entrypoints/openai/chat_completion/serving.py",
            "860e27d548ec7bb0497a46e0356a5acc5500998944cffa5e67e336041693c067"),
}
MARK = "EXTREME_LOOP079_TIMELINE"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"expected one anchor, found {text.count(old)}: {old[:90]!r}")
    return text.replace(old, new)


def patch(key, src):
    if MARK in src:
        raise ValueError("already patched")
    if key == "scheduler":
        src = replace_once(src,
            "            # Check for stop and update request status.\n            if new_token_ids:\n",
            "            # Check for stop and update request status.\n"
            "            # EXTREME_LOOP079_TIMELINE\n"
            "            from scripts import loop079_timeline as _timeline\n"
            "            _timeline_row = _timeline.scheduler_before(\n"
            "                self, request, new_token_ids,\n"
            "                model_runner_output.extreme_bulk_output, output_is_stale)\n"
            "            if new_token_ids:\n")
        src = replace_once(src,
            "            if new_token_ids and self.structured_output_manager.should_advance(request, new_token_ids=new_token_ids):\n",
            "            # EXTREME_LOOP079_TIMELINE\n"
            "            _timeline.scheduler_after(_timeline_row, request, new_token_ids, stopped)\n"
            "            if new_token_ids and self.structured_output_manager.should_advance(request, new_token_ids=new_token_ids):\n")
    elif key == "runner":
        src = replace_once(src,
            "                _extreme_wall_start = time.perf_counter()\n"
            "                _cohort_output = FixedCohortServing(\n",
            "                # EXTREME_LOOP079_TIMELINE\n"
            "                from scripts import loop079_timeline as _timeline\n"
            "                _timeline.handoff(self, scheduler_output, _extreme_req_ids)\n"
            "                _extreme_wall_start = time.perf_counter()\n"
            "                _cohort_output = FixedCohortServing(\n")
        src = replace_once(src,
            "                ).run()\n"
            "                torch.npu.synchronize()\n",
            "                ).run()\n"
            "                torch.npu.synchronize()\n"
            "                # EXTREME_LOOP079_TIMELINE\n"
            "                _timeline.flush()\n")
    elif key == "output":
        src = replace_once(src,
            "            new_token_ids = engine_core_output.new_token_ids\n",
            "            new_token_ids = engine_core_output.new_token_ids\n"
            "            # EXTREME_LOOP079_TIMELINE\n"
            "            from scripts import loop079_timeline as _timeline\n"
            "            _timeline.output_processor(req_state, engine_core_output)\n")
        src = replace_once(src,
            "                    # AsyncLLM: put into queue for handling by generate().\n"
            "                    req_state.queue.put(request_output)\n",
            "                    # AsyncLLM: put into queue for handling by generate().\n"
            "                    req_state.queue.put(request_output)\n"
            "                    # EXTREME_LOOP079_TIMELINE\n"
            "                    _timeline.output_processor_queue(req_state, request_output)\n")
    elif key == "api":
        src = replace_once(src,
            "        created_time = int(time.time())\n"
            "        chunk_object_type: Final = \"chat.completion.chunk\"\n",
            "        created_time = int(time.time())\n"
            "        # EXTREME_LOOP079_TIMELINE\n"
            "        from scripts import loop079_timeline as _timeline\n"
            "        chunk_object_type: Final = \"chat.completion.chunk\"\n")
        src = replace_once(src,
            "                    previous_num_tokens[i] += len(output.token_ids)\n",
            "                    previous_num_tokens[i] += len(output.token_ids)\n"
            "                    # EXTREME_LOOP079_TIMELINE\n"
            "                    _timeline.api_consume(res.request_id, request_id, i,\n"
            "                                          output.token_ids, previous_num_tokens[i])\n")
        src = replace_once(src,
            "                    data = chunk.model_dump_json(exclude_unset=True)\n"
            "                    yield f\"data: {data}\\n\\n\"\n",
            "                    data = chunk.model_dump_json(exclude_unset=True)\n"
            "                    # EXTREME_LOOP079_TIMELINE\n"
            "                    _timeline.api_yield(res.request_id, request_id, i,\n"
            "                                        previous_num_tokens[i], delta_message,\n"
            "                                        output.finish_reason is not None)\n"
            "                    yield f\"data: {data}\\n\\n\"\n")
    else:
        raise ValueError(key)
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("check", "install", "restore"))
    ap.add_argument("--state-dir", type=Path)
    ap.add_argument("--record", type=Path, required=True)
    a = ap.parse_args()
    helper = ROOT / "scripts/loop079_timeline.py"
    compile(helper.read_bytes(), str(helper), "exec")
    if a.action in ("check", "install"):
        files = {}
        for key, (path, expected) in SOURCES.items():
            old = path.read_bytes()
            if sha(old) != expected:
                raise ValueError(f"{key} source SHA changed: {sha(old)}")
            new = patch(key, old.decode()).encode()
            compile(new, str(path), "exec")
            files[key] = dict(path=str(path), original=sha(old), patched=sha(new))
        record = dict(action=a.action, files=files, helper_sha256=sha(helper.read_bytes()))
        if a.action == "install":
            if a.state_dir is None or a.state_dir.exists():
                raise ValueError("install requires fresh state-dir")
            a.state_dir.mkdir(parents=True)
            for key, info in files.items():
                path = Path(info["path"])
                old = path.read_bytes()
                if sha(old) != info["original"]:
                    raise ValueError("source changed during backup")
                (a.state_dir / f"{key}.original.py").write_bytes(old)
            (a.state_dir / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
            for key, info in files.items():
                path = Path(info["path"])
                path.write_bytes(patch(key, path.read_text()).encode())
                if sha(path.read_bytes()) != info["patched"]:
                    raise ValueError(f"{key} install mismatch")
    else:
        if a.state_dir is None:
            raise ValueError("restore requires state-dir")
        record = json.loads((a.state_dir / "manifest.json").read_text())
        for key, info in record["files"].items():
            path = Path(info["path"])
            old = (a.state_dir / f"{key}.original.py").read_bytes()
            if sha(old) != info["original"] or sha(path.read_bytes()) != info["patched"]:
                raise ValueError(f"{key} backup/current SHA mismatch")
        for key, info in record["files"].items():
            path = Path(info["path"])
            path.write_bytes((a.state_dir / f"{key}.original.py").read_bytes())
            if sha(path.read_bytes()) != info["original"]:
                raise ValueError(f"{key} restore mismatch")
        record = dict(record, action="restore", restored=True)
    a.record.parent.mkdir(parents=True, exist_ok=True)
    a.record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"action": a.action, "files": len(record["files"])}))


if __name__ == "__main__":
    main()
