#!/usr/bin/env python3
"""Bounded compile-only coverage for sparse-MQA metadata small-Q shapes.

The installed DeepGEMM JIT kernel computes the template arguments
``sparse_mqa_logits_metadata<block, metadata_cores, ai_cores, schedule_cores>``
from the actual Q length, then, when the runtime is in dry-run mode, calls
``jit->compile_without_load`` and returns before any launch.  This helper uses
that supported public path through
``deep_gemm.get_sparse_mqa_logits_metadata`` with CPU shape tensors.  It never
runs a model forward, never launches a kernel and never reads device data.

The Q range is bounded (default 1..512).  The compile key deduplicates the
finite metadata-core specializations, and the verification pass requires every
Q call to be a cache hit with no new cache directory and no slow call.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def sha256_file(path: str | None) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def committed_dirs(cache_dir: str) -> set[str]:
    root = os.path.join(cache_dir, "cache")
    return {
        os.path.basename(path)
        for path in glob.glob(os.path.join(root, "*"))
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, ".committed"))
    }


def cache_fingerprint(cache_dir: str) -> str:
    digest = hashlib.sha256()
    root = os.path.join(cache_dir, "cache")
    for name in sorted(committed_dirs(cache_dir)):
        digest.update(name.encode())
        meta = os.path.join(root, name, "meta.json")
        try:
            digest.update(Path(meta).read_bytes())
        except OSError:
            digest.update(b"missing-meta")
        digest.update(b"\n")
    return digest.hexdigest()


_SIG_RE = re.compile(r"sparse_mqa_logits_metadata<([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)>")


def scan_signatures(cache_dir: str) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for asc in glob.glob(os.path.join(cache_dir, "cache", "*", "kernel.asc")):
        try:
            text = Path(asc).read_text(errors="replace")
        except OSError:
            continue
        match = _SIG_RE.search(text)
        if not match:
            continue
        block, meta, ai, schedule = (int(x) for x in match.groups())
        key = "%d,%d,%d,%d" % (block, meta, ai, schedule)
        row = found.setdefault(key, {
            "block": block, "num_metadata_cores": meta,
            "num_ai_cores": ai, "num_schedule_cores": schedule,
            "cache_dirs": []})
        row["cache_dirs"].append(os.path.basename(os.path.dirname(asc)))
    return found


def signature_for(q: int, block: int, num_sms: int, vec_cores: int) -> str:
    q_blocks = (q + 3) // 4
    metadata_cores = min(num_sms * vec_cores, q_blocks)
    ai_cores = min(num_sms, q)
    schedule_cores = min(metadata_cores, (q + 511) // 512)
    return "%d,%d,%d,%d" % (block, metadata_cores, ai_cores, schedule_cores)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("DG_JIT_CACHE_DIR"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-q", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tensor-device", choices=("cpu", "meta"), default="cpu")
    parser.add_argument("--block-size", type=int, action="append", default=None)
    parser.add_argument("--adapter-source", default=None)
    args = parser.parse_args()

    if not args.cache_dir:
        raise SystemExit("--cache-dir or DG_JIT_CACHE_DIR is required")
    args.cache_dir = args.cache_dir.split(":", 1)[0]
    os.environ["DG_JIT_CACHE_DIR"] = args.cache_dir
    started = time.time()
    manifest = json.loads(Path(args.manifest).read_text())
    q_min, q_max_manifest = [int(x) for x in manifest.get("q_range", [1, 512])]
    max_q = min(int(args.max_q), q_max_manifest)
    if max_q < q_min:
        raise SystemExit("max_q below manifest q_range")
    block_sizes = args.block_size or [int(x) for x in manifest.get("block_sizes", [8])]
    capacity = int(manifest.get("capacity", 2048))
    num_kv_tokens = int(manifest.get("num_kv_tokens", 33024))
    if capacity < 1 or capacity * max(block_sizes) >= 65536:
        raise SystemExit("manifest capacity violates the 16-bit output-offset contract")

    import torch
    import torch_npu
    import deep_gemm
    from deep_gemm._C import get_dry_run, set_dry_run

    torch.npu.set_device(args.device)
    num_sms = int(deep_gemm.get_num_sms())
    # The cached production signatures show two vector cores per AI core:
    # Q=50 -> metadata 13, Q=245 -> 62, Q>=256 -> 64 with num_sms=32.
    vec_cores = 2
    before_dirs = committed_dirs(args.cache_dir)
    before_fingerprint = cache_fingerprint(args.cache_dir)
    before_sigs = scan_signatures(args.cache_dir)

    tasks = [(q, block) for block in block_sizes
             for q in range(q_min, max_q + 1)]

    def invoke(q: int, block: int):
        device = args.tensor_device
        indices = torch.empty((q, capacity), dtype=torch.int32, device=device)
        start = torch.zeros(q, dtype=torch.int32, device=device)
        end = torch.full((q,), num_kv_tokens, dtype=torch.int32, device=device)
        return deep_gemm.get_sparse_mqa_logits_metadata(
            start, end, num_kv_tokens, indices, torch.float8_e4m3fn, block)

    lock = threading.Lock()
    slow_threshold = 1.0
    compile_rows = []
    compile_errors = []
    compile_slow = 0

    def run_compile(task):
        nonlocal compile_slow
        q, block = task
        t0 = time.perf_counter()
        try:
            invoke(q, block)
            elapsed = time.perf_counter() - t0
            slow = elapsed >= slow_threshold
            with lock:
                compile_rows.append({"q": q, "block": block,
                                     "elapsed_s": round(elapsed, 6),
                                     "slow": slow})
                if slow:
                    compile_slow += 1
        except BaseException as exc:  # noqa: BLE001
            with lock:
                compile_errors.append({"q": q, "block": block, "error": repr(exc)})

    set_dry_run(True)
    compile_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.workers), thread_name_prefix="e15-sparse") as pool:
        list(pool.map(run_compile, tasks))
    compile_s = time.perf_counter() - compile_t0
    after_compile_dirs = committed_dirs(args.cache_dir)
    after_compile_fingerprint = cache_fingerprint(args.cache_dir)
    after_compile_sigs = scan_signatures(args.cache_dir)

    verify_rows = []
    verify_errors = []
    verify_slow_rows = []

    def run_verify(task):
        q, block = task
        t0 = time.perf_counter()
        try:
            invoke(q, block)
            elapsed = time.perf_counter() - t0
            row = {"q": q, "block": block, "elapsed_s": round(elapsed, 6)}
            with lock:
                verify_rows.append(row)
                if elapsed >= slow_threshold:
                    verify_slow_rows.append(row)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                verify_errors.append({"q": q, "block": block, "error": repr(exc)})

    verify_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.workers), thread_name_prefix="e15-sparse-v") as pool:
        list(pool.map(run_verify, tasks))
    verify_s = time.perf_counter() - verify_t0
    after_dirs = committed_dirs(args.cache_dir)
    after_fingerprint = cache_fingerprint(args.cache_dir)
    after_sigs = scan_signatures(args.cache_dir)
    set_dry_run(False)

    new_dirs_compile = sorted(after_compile_dirs - before_dirs)
    new_dirs_verify = sorted(after_dirs - after_compile_dirs)
    new_signatures = sorted(set(after_sigs) - set(before_sigs))
    missing_signatures = []
    for q, block in tasks:
        key = signature_for(q, block, num_sms, vec_cores)
        if key not in after_sigs:
            missing_signatures.append({"q": q, "block": block, "signature": key})
    coverage_ok = not (compile_errors or verify_errors or verify_slow_rows
                       or new_dirs_verify or missing_signatures)
    fingerprint_stable_after_verify = after_compile_fingerprint == after_fingerprint

    compiler = shutil.which("bisheng") or "/usr/local/Ascend/cann-9.2.0/bin/bisheng"
    try:
        compiler_version = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True,
            timeout=20).stdout.strip().splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        compiler_version = "unavailable: %r" % (exc,)

    helper_path = os.path.abspath(__file__)
    adapter_path = args.adapter_source or os.path.join(
        os.path.dirname(helper_path), "deepinfer_dp8_adapter.py")
    pkg = Path(deep_gemm.__file__).resolve().parent
    report = {
        "schema_version": "1.0",
        "experiment": "E15",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generated_epoch": started,
        "cache_dir": args.cache_dir,
        "manifest": os.path.abspath(args.manifest),
        "manifest_sha256": sha256_file(os.path.abspath(args.manifest)),
        "helper_sha256": sha256_file(helper_path),
        "adapter_source": os.path.abspath(adapter_path),
        "adapter_sha256": sha256_file(adapter_path),
        "max_q": max_q,
        "declared_covered_input_classes": {
            "family": "sparse_mqa_logits_metadata",
            "q_range": [q_min, max_q],
            "block_sizes": block_sizes,
            "qk_dtypes": ["float8_e4m3fn"],
            "capacity": capacity,
            "num_kv_tokens": num_kv_tokens,
            "template": "sparse_mqa_logits_metadata<block, metadata_cores, ai_cores, schedule_cores>",
            "dry_run": "jit->compile_without_load then return before launch",
        },
        "runtime": {
            "device": args.device,
            "tensor_device": args.tensor_device,
            "torch": getattr(torch, "__version__", None),
            "torch_npu": getattr(torch_npu, "__version__", None),
            "deep_gemm": getattr(deep_gemm, "__version__", None),
            "deep_gemm_C_sha256": sha256_file(
                os.path.join(os.path.dirname(deep_gemm.__file__),
                             "_C.cpython-312-aarch64-linux-gnu.so")),
            "num_sms": num_sms,
            "vec_cores_per_ai_core": vec_cores,
            "npu_arch": int(deep_gemm.get_npu_arch()),
            "compiler": compiler,
            "compiler_version": compiler_version,
            "source_hashes": {
                "sparse_mqa_logits.hpp": sha256_file(
                    str(pkg / "csrc/jit_kernels/sparse_mqa_logits.hpp")),
                "attention.hpp": sha256_file(str(pkg / "csrc/apis/attention.hpp")),
                "sparse_mqa_logits_metadata.hpp": sha256_file(
                    str(pkg / "include/deep_gemm/sparse_mqa_logits_metadata.hpp")),
                "layout_sparse_mqa_logits.hpp": sha256_file(
                    str(pkg / "include/deep_gemm/layout/sparse_mqa_logits.hpp")),
            },
            "dry_run_before": bool(get_dry_run()),
            "workers": args.workers,
        },
        "cache": {
            "dirs_before": len(before_dirs),
            "dirs_after_compile": len(after_compile_dirs),
            "dirs_after_verify": len(after_dirs),
            "new_dirs_compile": len(new_dirs_compile),
            "new_dirs_verify": len(new_dirs_verify),
            "new_dirs": new_dirs_compile + new_dirs_verify,
            "fingerprint_before": before_fingerprint,
            "fingerprint_after_compile": after_compile_fingerprint,
            "fingerprint_after_verify": after_fingerprint,
            "fingerprint_persisted": fingerprint_stable_after_verify,
            "signatures_before": sorted(before_sigs),
            "signatures_after": sorted(after_sigs),
            "new_signatures": new_signatures,
        },
        "compile": {
            "elapsed_s": round(compile_s, 3),
            "calls": len(compile_rows) + len(compile_errors),
            "slow_calls": compile_slow,
            "errors": compile_errors,
            "max_s": max((row["elapsed_s"] for row in compile_rows), default=None),
        },
        "verification": {
            "elapsed_s": round(verify_s, 3),
            "calls": len(verify_rows) + len(verify_errors),
            "slow_calls": len(verify_slow_rows),
            "slow_rows": verify_slow_rows,
            "errors": verify_errors,
            "new_dirs": new_dirs_verify,
            "max_s": max((row["elapsed_s"] for row in verify_rows), default=None),
        },
        "coverage_ok": bool(coverage_ok),
        "remaining_misses": {
            "compile_errors": compile_errors,
            "verify_errors": verify_errors,
            "verify_slow_calls": verify_slow_rows,
            "verify_new_dirs": new_dirs_verify,
            "missing_signatures": missing_signatures,
        },
        "total_elapsed_s": round(time.time() - started, 3),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    tmp.replace(out)
    print(json.dumps({
        "coverage_ok": report["coverage_ok"],
        "cache_dirs_before": report["cache"]["dirs_before"],
        "cache_dirs_after": report["cache"]["dirs_after_verify"],
        "new_dirs_compile": report["cache"]["new_dirs_compile"],
        "new_signatures": len(new_signatures),
        "verify_slow_calls": len(verify_slow_rows),
        "verify_new_dirs": len(new_dirs_verify),
        "elapsed_s": report["total_elapsed_s"],
        "out": str(out),
    }, indent=2))
    return 0 if coverage_ok else 2


if __name__ == "__main__":
    sys.exit(main())
