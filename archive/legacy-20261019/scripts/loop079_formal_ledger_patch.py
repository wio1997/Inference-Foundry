#!/usr/bin/env python3
"""SHA-pinned, reversible full48 Host-ledger source patch generator.

`check` performs source-only compilation and produces a manifest preview.
`install` and `restore` are reserved for a separately guarded live controller.
This script does not launch a service or execute an NPU operation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

ROOT = Path("/data/wio/Inference_Foundry")
FRAME = Path("/data/wio/vllm_ascend_26/framework")
HELPER = ROOT / "scripts/loop079_formal_ledger_server.py"
SOURCES = {
    "scheduler": (FRAME / "vllm-ascend/vllm_ascend/patch/platform/patch_kv_delivery_preemption.py",
                  "5ba894b9ec003502c008e12281c260787ed64a419327c451c8e67374bf525e8d"),
    "runner": (FRAME / "vllm-ascend/vllm_ascend/worker/model_runner_v1.py",
               "004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba"),
    "serving": (ROOT / "runtime/fixed_serving.py",
                "137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a"),
    "output": (FRAME / "vllm/vllm/v1/engine/output_processor.py",
               "ee10351275d90796c8b901a5f4b23d5a046ef6ee72fd2921aff2ae78ca58bd9b"),
    "api": (FRAME / "vllm/vllm/entrypoints/openai/chat_completion/serving.py",
            "860e27d548ec7bb0497a46e0356a5acc5500998944cffa5e67e336041693c067"),
}
MARK = "EXTREME_LOOP079_FORMAL_LEDGER"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_replace(path, data):
    """Durably replace one file with complete bytes or leave its old bytes.

    The temporary file is in the destination directory so rename is atomic.
    A failed write/rename never truncates the existing source. A failure after
    rename may leave the complete new file; restore accepts that known SHA.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing symlink destination: {path}")
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.formal-ledger-", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, mode)
        view = memoryview(data)
        while view:
            count = os.write(fd, view[:65536])
            if count <= 0:
                raise OSError("short atomic write")
            view = view[count:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        temp_path.unlink(missing_ok=True)


def write_json(path, value):
    atomic_replace(path, (json.dumps(value, indent=2) + "\n").encode())


def once(src, old, new):
    count = src.count(old)
    if count != 1:
        raise ValueError(f"anchor count {count}, expected 1: {old[:100]!r}")
    return src.replace(old, new)


def patch(key, src):
    if MARK in src:
        raise ValueError("already patched")
    if key == "scheduler":
        src = once(src,
            "            # Check for stop and update request status.\n            if new_token_ids:\n",
            "            # Check for stop and update request status.\n"
            "            # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "            from scripts import loop079_formal_ledger_server as _formal_ledger\n"
            "            _formal_ledger_row = _formal_ledger.scheduler_before(\n"
            "                self, request, new_token_ids,\n"
            "                model_runner_output.extreme_bulk_output, output_is_stale)\n"
            "            if new_token_ids:\n")
        src = once(src,
            "            if new_token_ids and self.structured_output_manager.should_advance(request, new_token_ids=new_token_ids):\n",
            "            # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "            _formal_ledger.scheduler_after(_formal_ledger_row, request, new_token_ids, stopped)\n"
            "            if new_token_ids and self.structured_output_manager.should_advance(request, new_token_ids=new_token_ids):\n")
    elif key == "runner":
        src = once(src,
            "                _extreme_wall_start = time.perf_counter()\n"
            "                _cohort_output = FixedCohortServing(\n",
            "                # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "                from scripts import loop079_formal_ledger_server as _formal_ledger\n"
            "                _formal_ledger.handoff(self, scheduler_output, _extreme_req_ids)\n"
            "                _extreme_wall_start = time.perf_counter()\n"
            "                _cohort_output = FixedCohortServing(\n")
        src = once(src,
            "                _row[\"pass\"] = bool(\n",
            "                # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "                _formal_ledger.runner_done(_row)\n"
            "                _row[\"pass\"] = bool(\n")
    elif key == "serving":
        src = once(src,
            "            progress = self._committed_progress()\n"
            "            self._park_completed(progress)\n",
            "            progress = self._committed_progress()\n"
            "            # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "            from scripts import loop079_formal_ledger_server as _formal_ledger\n"
            "            _formal_parked_before = tuple(self._parked)\n"
            "            self._park_completed(progress)\n"
            "            _formal_ledger.serving_cycle(index, progress, _formal_parked_before, self._parked)\n")
        src = once(src,
            "        acceptance_window_means = {}\n",
            "        # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "        _formal_ledger.serving_history(self, cycles, tokens_cpu, counts_cpu, output, staged)\n"
            "        acceptance_window_means = {}\n")
    elif key == "output":
        src = once(src,
            "        self.request_states[request_id] = req_state\n",
            "        self.request_states[request_id] = req_state\n"
            "        # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "        from scripts import loop079_formal_ledger_server as _formal_ledger\n"
            "        _formal_ledger.output_add(request, req_state)\n")
        src = once(src,
            "            new_token_ids = engine_core_output.new_token_ids\n",
            "            new_token_ids = engine_core_output.new_token_ids\n"
            "            # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "            from scripts import loop079_formal_ledger_server as _formal_ledger\n"
            "            _formal_ledger.output_receive(req_state, engine_core_output)\n")
        src = once(src,
            "                    # AsyncLLM: put into queue for handling by generate().\n"
            "                    req_state.queue.put(request_output)\n",
            "                    # AsyncLLM: put into queue for handling by generate().\n"
            "                    req_state.queue.put(request_output)\n"
            "                    # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "                    _formal_ledger.output_queue(req_state, request_output)\n")
    elif key == "api":
        src = once(src,
            "        created_time = int(time.time())\n"
            "        chunk_object_type: Final = \"chat.completion.chunk\"\n",
            "        created_time = int(time.time())\n"
            "        # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "        from scripts import loop079_formal_ledger_server as _formal_ledger\n"
            "        chunk_object_type: Final = \"chat.completion.chunk\"\n")
        src = once(src,
            "                    previous_num_tokens[i] += len(output.token_ids)\n",
            "                    previous_num_tokens[i] += len(output.token_ids)\n"
            "                    # EXTREME_LOOP079_FORMAL_LEDGER\n"
            "                    _formal_ledger.api_consume(res.request_id, request_id, i,\n"
            "                        output.token_ids, previous_num_tokens[i], delta_text,\n"
            "                        delta_message, parser, output.finish_reason)\n")
        begin = src.index("    async def chat_completion_stream_generator(")
        end = src.index("    async def chat_completion_full_generator(", begin)
        section = src[begin:end]
        section, n = re.subn(r"(?m)^([ \t]*)yield (.*)$",
                             r"\1yield _formal_ledger.api_yield(request_id, \2)", section)
        if n != 9:
            raise ValueError(f"expected 9 API yields, got {n}")
        src = src[:begin] + section + src[end:]
    else:
        raise ValueError(key)
    return src


def _preview():
    compile(HELPER.read_bytes(), str(HELPER), "exec")
    files = {}
    buffers = {}
    for key, (path, expected) in SOURCES.items():
        old = path.read_bytes()
        if sha(old) != expected:
            raise ValueError(f"{key} source SHA changed: {sha(old)} != {expected}")
        new = patch(key, old.decode()).encode()
        compile(new, str(path), "exec")
        files[key] = dict(path=str(path), original=sha(old), patched=sha(new),
                          original_bytes=len(old), patched_bytes=len(new))
        buffers[key] = (old, new)
    return files, buffers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("check", "install", "restore"))
    ap.add_argument("--state-dir", type=Path)
    ap.add_argument("--record", type=Path, required=True)
    ap.add_argument("--offline-confirmed", action="store_true",
                    help="controller has verified no service/workers before source mutation")
    args = ap.parse_args()
    if args.action in ("install", "restore") and not args.offline_confirmed:
        raise ValueError("install/restore require --offline-confirmed from guarded controller")
    if args.action == "install" and args.state_dir is not None and args.state_dir.exists():
        existing = json.loads((args.state_dir / "manifest.json").read_text())
        if existing["marker"] != MARK or existing["helper_sha256"] != sha(HELPER.read_bytes()):
            raise ValueError("existing state-dir manifest/helper differs")
        files = existing["files"]
        if set(files) != set(SOURCES):
            raise ValueError("existing state-dir source set differs")
        if not all(str(SOURCES[k][0]) == v["path"] and
                   sha(Path(v["path"]).read_bytes()) == v["patched"]
                   for k, v in files.items()):
            raise ValueError("state-dir exists but source not installed")
        record = dict(existing, action="install", installed=True, idempotent=True)
        write_json(args.record, record)
        print(json.dumps(dict(action="install", files=len(files), idempotent=True,
                              record=str(args.record), sha256=sha(args.record.read_bytes()))))
        return
    if args.action in ("check", "install"):
        files, buffers = _preview()
        record = dict(action=args.action, marker=MARK, helper=str(HELPER),
                      helper_sha256=sha(HELPER.read_bytes()), files=files,
                      installed=False)
        if args.action == "install":
            if args.state_dir is None:
                raise ValueError("install requires --state-dir")
            args.state_dir.mkdir(parents=True)
            for key, (old, _) in buffers.items():
                atomic_replace(args.state_dir / f"{key}.original.py", old)
            write_json(args.state_dir / "manifest.json", record)
            for key, (_, new) in buffers.items():
                path = Path(files[key]["path"])
                if sha(path.read_bytes()) != files[key]["original"]:
                    raise ValueError(f"{key} changed between check and install")
                atomic_replace(path, new)
                if sha(path.read_bytes()) != files[key]["patched"]:
                    raise ValueError(f"{key} install verification failed")
            record["installed"] = True
    else:
        if args.state_dir is None:
            raise ValueError("restore requires --state-dir")
        saved = json.loads((args.state_dir / "manifest.json").read_text())
        if saved["marker"] != MARK:
            raise ValueError("manifest marker mismatch")
        # Cleanup must remain possible after the helper is edited or replaced
        # while the service is stopped. Source/backup hashes and the exact
        # source set, rather than the current helper, authorize restoration.
        try:
            current_helper_sha = sha(HELPER.read_bytes())
        except FileNotFoundError:
            current_helper_sha = None
        helper_sha_drift = current_helper_sha != saved["helper_sha256"]
        files = saved["files"]
        if set(files) != set(SOURCES):
            raise ValueError("restore manifest source set differs")
        for key, info in files.items():
            old = (args.state_dir / f"{key}.original.py").read_bytes()
            if (sha(old) != info["original"] or
                    info["original"] != SOURCES[key][1] or
                    str(SOURCES[key][0]) != info["path"]):
                raise ValueError(f"{key} backup/manifest mismatch")
            current = sha(Path(info["path"]).read_bytes())
            if current not in (info["patched"], info["original"]):
                raise ValueError(f"{key} current source changed: {current}")
        already = all(sha(Path(v["path"]).read_bytes()) == v["original"] for v in files.values())
        for key, info in files.items():
            path = Path(info["path"])
            if sha(path.read_bytes()) == info["patched"]:
                atomic_replace(path, (args.state_dir / f"{key}.original.py").read_bytes())
            if sha(path.read_bytes()) != info["original"]:
                raise ValueError(f"{key} restore verification failed")
        record = dict(action="restore", files=files, restored=True,
                      idempotent=already,
                      installed_helper_sha256=saved["helper_sha256"],
                      current_helper_sha256=current_helper_sha,
                      helper_sha_drift=helper_sha_drift)
    write_json(args.record, record)
    print(json.dumps(dict(action=args.action, files=len(record["files"]),
                          record=str(args.record), sha256=sha(args.record.read_bytes()))))


if __name__ == "__main__":
    main()
