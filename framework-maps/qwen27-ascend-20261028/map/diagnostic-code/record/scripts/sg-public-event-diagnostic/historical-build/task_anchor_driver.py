#!/usr/bin/env python3
"""Explicitly gated build-only / isolated device-3 graph-anchor toy driver.

Importing this file does not import torch, compile code, initialize ACL, or touch an NPU.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

TASK = "Q27-SG-GRAPH-ANCHOR-gen20"
DEVICE = 3
EXPECTED_TORCH_NPU_VERSION = "2.10.0"
EXPECTED_TORCH_NPU_GIT = "94f8a8e6b523d7ba553e1b80d5b5248478391526"
EXPECTED_CPP_SHA256 = "0ef2e5b45078035326681c40e8bf15c742fe032b98ab38c367738fbc919c308a"
CANN_HEADER = Path("/usr/local/Ascend/cann-9.0.0/include/acl/acl_rt.h")
EXPECTED_CANN_SHA256 = "4691b8532495a556dce03d63a3ff431ddefedc982d0db6a337efeb101ec5180c"
REPLAY_COUNT = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def exclusive_outdir(outdir: Path) -> None:
    if outdir.exists():
        raise RuntimeError(f"output path must be fresh and absent: {outdir}")
    outdir.mkdir(parents=True, exist_ok=False)


def exact_preflight(torch, torch_npu, cpp_source: Path) -> dict:
    import torch_npu.version as version
    if str(getattr(torch_npu, "__version__", "")) != EXPECTED_TORCH_NPU_VERSION:
        raise RuntimeError(f"torch_npu version mismatch: {getattr(torch_npu, '__version__', None)!r}")
    if str(getattr(version, "git_version", "")) != EXPECTED_TORCH_NPU_GIT:
        raise RuntimeError(f"torch_npu git revision mismatch: {getattr(version, 'git_version', None)!r}")
    if not CANN_HEADER.is_file():
        raise RuntimeError(f"frozen CANN header missing: {CANN_HEADER}")
    header_sha = sha256(CANN_HEADER)
    if header_sha != EXPECTED_CANN_SHA256:
        raise RuntimeError(f"CANN 9.0 header SHA mismatch: {header_sha}")
    cpp_sha = sha256(cpp_source)
    if cpp_sha != EXPECTED_CPP_SHA256:
        raise RuntimeError(f"gen19 C++ source SHA mismatch: {cpp_sha}")

    pkg_root = Path(torch_npu.__file__).resolve().parent
    candidates = [pkg_root, pkg_root / "include", pkg_root.parent]
    include_roots = []
    required = "torch_npu/csrc/core/npu/NPUStream.h"
    for root in candidates:
        if (root / required).is_file():
            include_roots.append(root)
    if not include_roots:
        raise RuntimeError(f"installed package lacks required public extension header under {pkg_root}")
    cann_include = CANN_HEADER.parent.parent
    if not (cann_include / "acl/acl.h").is_file():
        raise RuntimeError("CANN include root does not contain acl/acl.h")
    libdirs = [p for p in (pkg_root / "lib", pkg_root / "lib64") if p.is_dir()]
    cann_libdirs = [p for p in (Path("/usr/local/Ascend/cann-9.0.0/lib64"),
                                Path("/usr/local/Ascend/cann-9.0.0/lib")) if p.is_dir()]
    if not libdirs or not cann_libdirs:
        raise RuntimeError(f"expected package/CANN library dirs missing: torch={libdirs}, cann={cann_libdirs}")
    torch_version = str(getattr(torch, "__version__", ""))
    return {
        "torch_version": torch_version,
        "torch_npu_version": str(torch_npu.__version__),
        "torch_npu_git": str(version.git_version),
        "torch_npu_package_root": str(pkg_root),
        "include_roots": [str(p) for p in include_roots] + [str(cann_include)],
        "torch_npu_libdirs": [str(p) for p in libdirs],
        "cann_libdirs": [str(p) for p in cann_libdirs],
        "cpp_source": {"path": str(cpp_source), "sha256": cpp_sha},
        "cann_header": {"path": str(CANN_HEADER), "sha256": header_sha},
    }


def build_only(args) -> int:
    # Explicit build-only mode imports framework metadata and invokes the CPU
    # compiler/linker; it never calls torch.npu, ACL init, or any device API.
    import torch
    import torch_npu
    cpp_source = Path(args.cpp).resolve()
    receipt = exact_preflight(torch, torch_npu, cpp_source)
    out = Path(args.outdir).resolve()
    exclusive_outdir(out)
    state = {"task": TASK, "mode": "build_only", "status": "preflight_passed",
             "pid": os.getpid(), "started_ns": time.time_ns(), "device_initialized": False,
             "scope": "CPU extension build only; no ACL/NPU API invocation", "source_identity": receipt}
    atomic_json(out / "build-state.json", state)
    build_dir = out / "build"
    build_dir.mkdir()
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    ldflags = []
    for p in receipt["torch_npu_libdirs"] + receipt["cann_libdirs"]:
        ldflags.extend([f"-L{p}", f"-Wl,-rpath,{p}"])
    ldflags.extend(["-ltorch_npu", "-lascendcl", f"-Wl,-rpath,{torch_lib}"])
    for p in receipt["include_roots"]:
        if p != str(CANN_HEADER.parent.parent):
            ldflags.append(f"-Wl,-rpath,{p}")
    os.environ["MAX_JOBS"] = "1"
    from torch.utils.cpp_extension import load
    try:
        module = load(name="q27_sg_task_anchor_gen20", sources=[str(cpp_source)],
                      build_directory=str(build_dir), extra_include_paths=receipt["include_roots"],
                      extra_cflags=["-O0", "-g0", "-std=c++17"], extra_ldflags=ldflags,
                      with_cuda=False, verbose=True, is_python_module=True)
        module_file = Path(module.__file__).resolve()
        state.update(status="build_completed", module_path=str(module_file),
                     module_sha256=sha256(module_file), finished_ns=time.time_ns())
        atomic_json(out / "build-state.json", state)
        (out / "build-receipt.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        return 0
    except BaseException as exc:
        state.update(status="build_failed", error=repr(exc), traceback=traceback.format_exc(),
                     finished_ns=time.time_ns())
        atomic_json(out / "build-state.json", state)
        raise


def sanitize_graph_dump(graph, out: Path) -> dict:
    """Retain raw private probe dump and a separate native-ID-only projection."""
    raw = out / "raw_graph_debug.json"
    try:
        graph.debug_dump(str(raw))
        raw_bytes = raw.stat().st_size
        raw_sha = sha256(raw)
        rows = json.loads(raw.read_text())
        if not isinstance(rows, list):
            raise RuntimeError("unexpected graph.debug_dump root; refusing to persist unreviewed fields")
        safe = []
        for index, row in enumerate(rows):
            args = row.get("args") if isinstance(row, dict) else None
            if not isinstance(args, dict):
                raise RuntimeError(f"graph row {index} missing args object")
            normalized = {"".join(ch.lower() for ch in str(k) if ch.isalnum()): v for k, v in args.items()}
            keys = ("modelid", "streamid", "taskid")
            if any(k not in normalized or not isinstance(normalized[k], int) for k in keys):
                raise RuntimeError(f"graph row {index} missing exact integer native key fields")
            safe.append({"Model Id": normalized["modelid"],
                         "Stream Id": normalized["streamid"],
                         "Task Id": normalized["taskid"]})
        safe_path = out / "raw_graph_native_ids.json"
        safe_path.write_text(json.dumps(safe, indent=2) + "\n")
        return {"path": safe_path.name, "rows": len(safe), "bytes": safe_path.stat().st_size,
                "sha256": sha256(safe_path), "raw_dump_discarded": False,
                "raw_dump": {"path": raw.name, "bytes": raw_bytes, "sha256": raw_sha,
                             "scope": "private isolated probe output; retained as unique raw evidence"},
                "persisted_fields": ["Model Id", "Stream Id", "Task Id"]}
    finally:
        pass


def run_toy(args) -> int:
    required = {
        "Q27_ALLOW_TASK_ANCHOR_TOY": "I_UNDERSTAND_RESERVED_DEVICE3_TOY",
        "Q27_EXPECT_WORKFLOW_LOCK_OWNER": "root",
        "Q27_EXPECT_MODEL_QUEUES_EMPTY": "1",
        "Q27_EXPECT_TASKQUEUE_ENABLED": "1",
        "Q27_EXPECT_SHARED_DEVICE_IDLE": "1",
        "TASK_QUEUE_ENABLE": "1",
    }
    missing = [k for k, v in required.items() if os.environ.get(k) != v]
    if missing:
        raise RuntimeError(f"root launch/queue assertions missing or false: {missing}")
    if not os.environ.get("Q27_EXPECT_WORKFLOW_LOCK_ID"):
        raise RuntimeError("root wrapper must provide a workflow lock identity receipt")
    out = Path(args.outdir).resolve()
    exclusive_outdir(out)
    cpp_source = Path(args.cpp).resolve()
    import torch
    import torch_npu
    receipt = exact_preflight(torch, torch_npu, cpp_source)
    state = {"task": TASK, "mode": "run_toy", "status": "preflight_passed",
             "pid": os.getpid(), "started_ns": time.time_ns(), "device": DEVICE,
             "scope": "isolated 2-op BF16 graph anchor capability toy; no serving attribution",
             "source_identity": receipt,
             "root_assertions": {k: os.environ[k] for k in required},
             "queue_mode_environment": {"TASK_QUEUE_ENABLE": os.environ["TASK_QUEUE_ENABLE"],
                                        "Q27_EXPECT_TASKQUEUE_ENABLED": os.environ["Q27_EXPECT_TASKQUEUE_ENABLED"]},
             "workflow_lock_id": os.environ["Q27_EXPECT_WORKFLOW_LOCK_ID"],
             "replays_requested": REPLAY_COUNT, "native_analysis_run": False}
    atomic_json(out / "state.json", state)

    so_dir = Path(args.extension_dir).resolve()
    if not so_dir.is_dir():
        raise RuntimeError(f"prebuilt extension directory missing: {so_dir}")
    build_receipt = so_dir / "build-receipt.json"
    if not build_receipt.is_file():
        raise RuntimeError("prebuilt extension receipt missing")
    built = json.loads(build_receipt.read_text())
    if built.get("source_identity", {}).get("cpp_source", {}).get("sha256") != EXPECTED_CPP_SHA256:
        raise RuntimeError("prebuilt extension source receipt mismatch")
    module_path = Path(built.get("module_path", ""))
    if not module_path.is_file() or sha256(module_path) != built.get("module_sha256"):
        raise RuntimeError("prebuilt extension artifact absent or SHA changed")
    name = "q27_sg_task_anchor_gen20"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load prebuilt extension module")
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)

    graph = None
    profile = None
    profile_started = False
    npu_selected = False
    try:
        if torch.npu.device_count() <= DEVICE:
            raise RuntimeError(f"device {DEVICE} unavailable: count={torch.npu.device_count()}")
        torch.npu.set_device(DEVICE)
        npu_selected = True
        free_bytes, total_bytes = torch.npu.mem_get_info(DEVICE)
        state["memory_gate"] = {"free_bytes": int(free_bytes), "total_bytes": int(total_bytes),
                                "minimum_free_bytes": 2 * 1024**3}
        if free_bytes < 2 * 1024**3:
            raise RuntimeError("device-3 free-memory gate below 2 GiB")
        original = torch.npu.current_stream(DEVICE)
        stream = torch.npu.Stream(device=DEVICE)
        if int(stream.device_index) != DEVICE or int(stream.stream_id) == int(original.stream_id):
            raise RuntimeError("public stream is not an explicit nondefault device-3 stream")
        stream.wait_stream(original)
        state["stream"] = {"device_index": int(stream.device_index), "stream_id": int(stream.stream_id),
                           "original_stream_id": int(original.stream_id),
                           "kind": "public torch.npu.Stream(device=3)"}
        state["status"] = "memory_and_stream_gates_passed"
        atomic_json(out / "state.json", state)

        with torch.npu.stream(stream):
            x = torch.ones((32, 32), device=f"npu:{DEVICE}", dtype=torch.bfloat16)
            warm = (x + 1) * 2
            stream.synchronize()
            del warm

            handler = torch_npu.profiler.tensorboard_trace_handler(
                str(out / "raw_profile"), worker_name="q27-sg-task-anchor-gen20", analyse_flag=False)
            profile = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=False, on_trace_ready=handler)
            profile.start()
            profile_started = True
            graph = torch.npu.NPUGraph()
            anchors = []  # retain both result objects until out-of-capture queue drain
            with torch.npu.graph(graph, stream=stream, capture_error_mode="global"):
                with torch.autograd.profiler.record_function("Q27_CAPTURE_ADD"):
                    y = x + 1
                    anchors.append(extension.anchor("after_add"))
                with torch.autograd.profiler.record_function("Q27_CAPTURE_MUL"):
                    z = y * 2
                    anchors.append(extension.anchor("after_mul"))
            state["capture"] = {"closed": True, "anchors_retained": len(anchors),
                                "dtype": "bfloat16", "shape": [32, 32],
                                "operation_order": ["add_scalar_1", "multiply_scalar_2"],
                                "profile_active_during_capture": True}
            atomic_json(out / "state.json", state)

            # This is after capture exit: it drains queue callbacks and device work.
            stream.synchronize()
            anchor_rows = []
            for label, result in zip(("after_add", "after_mul"), anchors):
                anchor_rows.append({"label": label, "producer_pid": os.getpid(),
                    "device_index": int(stream.device_index),
                    "producer_thread": result.producer_thread,
                    "callback_thread": result.callback_thread,
                    "callback_rc": int(result.callback_rc), "task_rc": int(result.task_rc),
                    "task_id": int(result.task_id), "stream_rc": int(result.stream_rc),
                    "acl_stream_id": int(result.acl_stream_id),
                    "capture_rc": int(result.capture_rc), "capture_status": int(result.capture_status),
                    "model_ri_nonnull": bool(result.model_ri_nonnull)})
            state["capture_anchors"] = anchor_rows
            state["queue_path_observation"] = [
                {"label": row["label"],
                 "callback_thread_differs_from_producer": row["callback_thread"] != row["producer_thread"],
                 "interpretation": ("worker_callback_observed"
                                    if row["callback_thread"] != row["producer_thread"]
                                    else "same_thread_direct_or_unexpected; queued-consumer path not confirmed")}
                for row in anchor_rows]
            state["capture_anchor_rule"] = (
                "capture classification requires zero ACL returns and nonnull model_ri; otherwise unknown; "
                "one thread-last-task value per explicit callback only, not op task inventory/completion")
            atomic_json(out / "state.json", state)
            state["native_graph_dump"] = sanitize_graph_dump(graph, out)
            atomic_json(out / "state.json", state)

            for index in range(REPLAY_COUNT):
                with torch.autograd.profiler.record_function(f"Q27_GRAPH_REPLAY_SUBMIT_{index+1}"):
                    graph.replay()
                profile.step()
            stream.synchronize()
            expected = torch.full_like(z, 4)
            state["numerical_sanity"] = {
                "passed": bool(torch.equal(z, expected)),
                "dtype": "bfloat16", "shape": [32, 32],
                "expected_scalar": 4, "after_replays": REPLAY_COUNT,
            }
            if not state["numerical_sanity"]["passed"]:
                raise RuntimeError("2-op capture/replay numerical check failed")
            profile.stop()
            profile_started = False
            if not getattr(profile, "stopped", False):
                raise RuntimeError("profiler did not confirm closed state")

        restored = torch.npu.current_stream(DEVICE)
        state["stream_restore"] = {"restored": int(restored.stream_id) == int(original.stream_id),
                                   "stream_id": int(restored.stream_id),
                                   "original_stream_id": int(original.stream_id)}
        if not state["stream_restore"]["restored"]:
            raise RuntimeError("current stream not restored after context exit")
        profile_files = []
        for p in sorted((out / "raw_profile").rglob("*")):
            if p.is_file():
                profile_files.append({"path": str(p.relative_to(out)), "bytes": p.stat().st_size,
                                      "sha256": sha256(p)})
        state["profile_files"] = profile_files
        state["status"] = "raw_profile_closed_pending_native_export"
        state["native_analysis_run"] = False
        state["finish_ns"] = time.time_ns()
        atomic_json(out / "state.json", state)
        return 0
    except BaseException as exc:
        state.update(status="failed_or_interrupted", error=repr(exc),
                     traceback=traceback.format_exc(), finish_ns=time.time_ns())
        atomic_json(out / "state.json", state)
        raise
    finally:
        if profile_started and profile is not None:
            try:
                profile.stop()
                profile_started = False
            except BaseException:
                pass
        if npu_selected:
            try:
                torch.npu.synchronize(DEVICE)
                if "original" in locals():
                    restored = torch.npu.current_stream(DEVICE)
                    state["cleanup_stream_restore"] = int(restored.stream_id) == int(original.stream_id)
            except BaseException as exc:
                state["cleanup_error"] = repr(exc)
            if graph is not None:
                try:
                    graph.reset()
                except BaseException as exc:
                    state["graph_reset_error"] = repr(exc)
            atomic_json(out / "state.json", state)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("build-only", "run-toy"):
        p = sub.add_parser(mode)
        p.add_argument("--cpp", required=True, help="exact gen19 task_anchor.cpp")
        p.add_argument("--outdir", required=True, help="fresh output directory")
        if mode == "run-toy":
            p.add_argument("--extension-dir", required=True, help="previous build-only output directory")
    args = parser.parse_args()
    if args.mode == "build-only":
        return build_only(args)
    return run_toy(args)


if __name__ == "__main__":
    sys.exit(main())
