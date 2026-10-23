#!/usr/bin/env python3
"""Bounded compile-only DeepGEMM startup coverage for the selected text model.

This helper never runs a model forward.  It binds one NPU context, puts the
installed DeepGEMM runtime into dry-run mode, and asks the real public GEMM
entry points to instantiate/deduplicate the kernels for M=1..max_m for each
family in ``e14_gemm_families.json``.  Every call uses meta tensors with the
same shapes, strides, dtypes and quantization recipe as the production
DeepInfer ``ReplicatedLinear`` path.  A second pass re-issues every call and
rejects the readiness claim if any call still compiled (slow call or cache
delta), so a reported "covered" family is one that actually hit the persistent
cache for the declared input range.

The report is intentionally explicit about what is *not* covered: TileLang
metadata kernels, DeepEP/Engram, sparse-MQA, mega/m-grouped MoE and graph
capture are separate classes.  See the manifest's out_of_scope list.
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


_TEMPLATE_RE = re.compile(r"(fp8_gemm_impl|bf16_gemm_impl)<(.*?)>\);", re.S)


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


def scan_cached_families(cache_dir: str) -> dict[tuple[str, str, int, int], int]:
    counts: dict[tuple[str, str, int, int], int] = {}
    for asc in glob.glob(os.path.join(cache_dir, "cache", "*", "kernel.asc")):
        try:
            text = Path(asc).read_text()
        except OSError:
            continue
        match = _TEMPLATE_RE.search(text)
        if not match:
            continue
        name = "fp8" if match.group(1) == "fp8_gemm_impl" else "bf16"
        args = [part.strip() for part in match.group(2).split(",")]
        try:
            n = int(args[3])
            k = int(args[4])
            gemm_type = (args[21] if name == "fp8" else args[19]).split("::")[-1].lower()
        except (IndexError, ValueError):
            continue
        counts[(name, gemm_type, n, k)] = counts.get((name, gemm_type, n, k), 0) + 1
    return counts


def make_call(kind: str, n: int, k: int, m: int, cd_dtype: str, acc: bool,
              batch: int = 1):
    import torch

    if kind == "fp8":
        if k % 64:
            raise ValueError("fp8 K must be divisible by 64")
        if batch > 1:
            # fp8_bmm: A/B/D carry a leading batch axis, SFs are [B, MN, K/64]
            # with per-batch column-major blocks.
            a = torch.empty((batch, m, k), dtype=torch.float8_e4m3fn, device="meta")
            sfa = torch.empty_strided(
                (batch, m, k // 64), (m * (k // 64), 1, m),
                dtype=torch.int16, device="meta")
            b = torch.empty((batch, n, k), dtype=torch.float8_e4m3fn, device="meta")
            sfb = torch.empty_strided(
                (batch, n, k // 64), (n * (k // 64), 1, n),
                dtype=torch.int16, device="meta")
            d = torch.empty((batch, m, n), dtype=torch.bfloat16, device="meta")
            return (a, sfa), (b, sfb), d, None
        a = torch.empty((m, k), dtype=torch.float8_e4m3fn, device="meta")
        sfa = torch.empty_strided(
            (m, k // 64), (1, m), dtype=torch.int16, device="meta")
        b = torch.empty((n, k), dtype=torch.float8_e4m3fn, device="meta")
        sfb = torch.empty_strided(
            (n, k // 64), (1, n), dtype=torch.int16, device="meta")
        d = torch.empty((m, n), dtype=torch.bfloat16, device="meta")
        return (a, sfa), (b, sfb), d, None
    a = torch.empty((m, k), dtype=torch.bfloat16, device="meta")
    b = torch.empty((n, k), dtype=torch.bfloat16, device="meta")
    target = torch.float32 if cd_dtype == "float" else torch.bfloat16
    d = torch.empty((m, n), dtype=target, device="meta")
    return a, b, d, (d if acc else None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", default=os.environ.get("DG_JIT_CACHE_DIR"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-m", type=int, default=512)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--adapter-source", default=None)
    parser.add_argument("--only", default="", help="comma-separated family ids")
    args = parser.parse_args()

    if not args.cache_dir:
        raise SystemExit("--cache-dir or DG_JIT_CACHE_DIR is required")
    args.cache_dir = args.cache_dir.split(":", 1)[0]
    os.environ["DG_JIT_CACHE_DIR"] = args.cache_dir
    started = time.time()
    manifest = json.loads(Path(args.manifest).read_text())
    families = manifest.get("families") or []
    if args.only:
        wanted = {item for item in args.only.split(",") if item}
        families = [family for family in families if family["id"] in wanted]
    if not families:
        raise SystemExit("manifest has no families")
    max_m = min(int(args.max_m), int(manifest.get("max_m", args.max_m)))

    import torch
    import torch_npu
    import deep_gemm
    from deep_gemm._C import get_dry_run, set_dry_run

    torch.npu.set_device(args.device)
    before_dirs = committed_dirs(args.cache_dir)
    before_fingerprint = cache_fingerprint(args.cache_dir)
    before_families = scan_cached_families(args.cache_dir)

    stats: dict[str, dict] = {
        family["id"]: {"calls": 0, "slow_calls": 0, "compile_s": 0.0,
                       "errors": [], "max_s": 0.0}
        for family in families
    }
    lock = __import__("threading").Lock()
    slow_threshold = 1.0
    tasks = []
    for family in families:
        for start in range(1, max_m + 1, max(1, args.chunk)):
            tasks.append((family, start, min(max_m, start + args.chunk - 1)))

    def invoke(family, m):
        kind = family["kind"]
        if kind == "fp8":
            a, b, d, c = make_call(
                kind, family["N"], family["K"], m,
                family.get("cd_dtype", "bfloat16"),
                bool(family.get("acc")),
                int(family.get("batch") or 1))
            if family.get("gemm_type", "normal") == "batched":
                deep_gemm._C.fp8_bmm(
                    a, b, d, c,
                    recipe=tuple(family.get("recipe", [1, 1, 32])),
                    compiled_dims=family.get("compiled_dims", "nk"))
            else:
                deep_gemm.fp8_fp4_gemm_nt(
                    a, b, d, c, recipe=tuple(family.get("recipe", [1, 1, 32])),
                    compiled_dims=family.get("compiled_dims", "nk"))
        elif kind == "bf16":
            a, b, d, c = make_call(kind, family["N"], family["K"], m,
                                   family.get("cd_dtype", "bfloat16"),
                                   bool(family.get("acc")))
            deep_gemm.bf16_gemm_nt(
                a, b, d, c, compiled_dims=family.get("compiled_dims", "nk"))
        else:
            raise ValueError("unsupported kind: %r" % kind)

    def run_task(task):
        family, start, end = task
        fid = family["id"]
        local = {"calls": 0, "slow_calls": 0, "compile_s": 0.0,
                 "errors": [], "max_s": 0.0}
        for m in range(start, end + 1):
            t0 = time.perf_counter()
            try:
                invoke(family, m)
                elapsed = time.perf_counter() - t0
                local["calls"] += 1
                local["max_s"] = max(local["max_s"], elapsed)
                if elapsed >= slow_threshold:
                    local["slow_calls"] += 1
                    local["compile_s"] += elapsed
            except BaseException as exc:  # noqa: BLE001 - preserve every failure
                local["errors"].append({"m": m, "error": repr(exc)})
        with lock:
            target = stats[fid]
            target["calls"] += local["calls"]
            target["slow_calls"] += local["slow_calls"]
            target["compile_s"] = round(target["compile_s"] + local["compile_s"], 6)
            target["max_s"] = max(target["max_s"], local["max_s"])
            target["errors"].extend(local["errors"])

    set_dry_run(True)
    compile_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.workers), thread_name_prefix="e14-gemm") as pool:
        list(pool.map(run_task, tasks))
    compile_s = time.perf_counter() - compile_t0
    after_compile_dirs = committed_dirs(args.cache_dir)

    # Verification pass: every call must be a cache hit.  A slow call or any
    # new directory means coverage is not honest yet; record it, do not guess.
    verify_stats: dict[str, dict] = {
        family["id"]: {"calls": 0, "slow_calls": 0, "max_s": 0.0, "errors": []}
        for family in families
    }

    def run_verify(task):
        family, start, end = task
        local = {"calls": 0, "slow_calls": 0, "max_s": 0.0, "errors": []}
        for m in range(start, end + 1):
            t0 = time.perf_counter()
            try:
                invoke(family, m)
                elapsed = time.perf_counter() - t0
                local["calls"] += 1
                local["max_s"] = max(local["max_s"], elapsed)
                if elapsed >= slow_threshold:
                    local["slow_calls"] += 1
            except BaseException as exc:  # noqa: BLE001
                local["errors"].append({"m": m, "error": repr(exc)})
        with lock:
            target = verify_stats[family["id"]]
            target["calls"] += local["calls"]
            target["slow_calls"] += local["slow_calls"]
            target["max_s"] = max(target["max_s"], local["max_s"])
            target["errors"].extend(local["errors"])

    verify_t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.workers), thread_name_prefix="e14-verify") as pool:
        list(pool.map(run_verify, tasks))
    verify_s = time.perf_counter() - verify_t0
    after_dirs = committed_dirs(args.cache_dir)
    after_fingerprint = cache_fingerprint(args.cache_dir)
    after_families = scan_cached_families(args.cache_dir)
    set_dry_run(False)

    verify_new_dirs = sorted(after_dirs - after_compile_dirs)
    all_errors = {fid: row["errors"] for fid, row in stats.items() if row["errors"]}
    verify_errors = {fid: row["errors"] for fid, row in verify_stats.items()
                     if row["errors"]}
    verify_misses = {fid: row["slow_calls"] for fid, row in verify_stats.items()
                     if row["slow_calls"]}
    missing_families = []
    for family in families:
        key = (family["kind"], family.get("gemm_type", "normal"),
               int(family["N"]), int(family["K"]))
        if after_families.get(key, 0) <= 0:
            missing_families.append(family["id"])
    coverage_ok = not (all_errors or verify_errors or verify_misses
                       or verify_new_dirs or missing_families)

    compiler = shutil.which("bisheng") or "/usr/local/Ascend/cann-9.2.0/bin/bisheng"
    try:
        compiler_version = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True,
            timeout=20).stdout.strip().splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        compiler_version = "unavailable: %r" % (exc,)

    helper_path = os.path.abspath(__file__)
    adapter_path = args.adapter_source
    if adapter_path is None:
        adapter_path = os.path.join(os.path.dirname(helper_path),
                                    "deepinfer_dp8_adapter.py")
    report = {
        "schema_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generated_epoch": started,
        "cache_dir": args.cache_dir,
        "manifest": os.path.abspath(args.manifest),
        "manifest_sha256": sha256_file(os.path.abspath(args.manifest)),
        "helper_sha256": sha256_file(helper_path),
        "adapter_source": os.path.abspath(adapter_path) if adapter_path else None,
        "adapter_sha256": sha256_file(adapter_path),
        "max_m": max_m,
        "declared_covered_input_classes": {
            "kinds": ["fp8 dense Normal", "fp8 batched bmm", "bf16 dense Normal"],
            "families": [family["id"] for family in families],
            "m_range": [1, max_m],
            "compiled_dims": "nk (N/K compile-time, M runtime)",
            "recipe_fp8": [1, 1, 32],
            "fp8_meta_sf": "int16 packed ue8m0, column-major (MN, K/64)",
            "bf16_meta": "no scale factors",
        },
        "runtime": {
            "device": args.device,
            "torch": getattr(torch, "__version__", None),
            "torch_npu": getattr(torch_npu, "__version__", None),
            "deep_gemm": getattr(deep_gemm, "__version__", None),
            "deep_gemm_C_sha256": sha256_file(
                os.path.join(os.path.dirname(deep_gemm.__file__),
                             "_C.cpython-312-aarch64-linux-gnu.so")),
            "num_sms": int(deep_gemm.get_num_sms()),
            "npu_arch": int(deep_gemm.get_npu_arch()),
            "mk_alignment": int(deep_gemm.get_mk_alignment_for_contiguous_layout()),
            "compiler": compiler,
            "compiler_version": compiler_version,
            "source_hashes": {
                "linear.py": sha256_file("/work/src/DeepInfer/deepinfer/model_executor/layers/linear.py"),
                "attn.py": sha256_file("/work/src/DeepInfer/deepinfer/model_executor/models/v4_modules/attn.py"),
                "fp8_gemm.hpp": sha256_file(os.path.join(os.path.dirname(deep_gemm.__file__),
                                                         "csrc/jit_kernels/fp8_gemm.hpp")),
                "bf16_gemm.hpp": sha256_file(os.path.join(os.path.dirname(deep_gemm.__file__),
                                                          "csrc/jit_kernels/bf16_gemm.hpp")),
                "gemm.hpp": sha256_file(os.path.join(os.path.dirname(deep_gemm.__file__),
                                                    "csrc/apis/gemm.hpp")),
            },
            "dry_run_before": bool(get_dry_run()),
            "workers": args.workers,
        },
        "cache": {
            "dirs_before": len(before_dirs),
            "dirs_after_compile": len(after_compile_dirs),
            "dirs_after_verify": len(after_dirs),
            "new_dirs_compile": len(after_compile_dirs - before_dirs),
            "new_dirs_verify": len(verify_new_dirs),
            "new_dirs": sorted(after_compile_dirs - before_dirs) + verify_new_dirs,
            "fingerprint_before": before_fingerprint,
            "fingerprint_after_verify": after_fingerprint,
            "fingerprint_persisted": before_fingerprint == after_fingerprint,
            "families_before": {
                "%s-%s-N%d-K%d" % key: value for key, value in sorted(before_families.items())
            },
            "families_after": {
                "%s-%s-N%d-K%d" % key: value for key, value in sorted(after_families.items())
            },
        },
        "compile": {
            "elapsed_s": round(compile_s, 3),
            "calls": sum(row["calls"] for row in stats.values()),
            "slow_calls": sum(row["slow_calls"] for row in stats.values()),
            "per_family": stats,
        },
        "verification": {
            "elapsed_s": round(verify_s, 3),
            "calls": sum(row["calls"] for row in verify_stats.values()),
            "slow_calls": sum(row["slow_calls"] for row in verify_stats.values()),
            "new_dirs": verify_new_dirs,
            "per_family": verify_stats,
        },
        "coverage_ok": bool(coverage_ok),
        "remaining_misses": {
            "families": missing_families,
            "compile_errors": all_errors,
            "verify_errors": verify_errors,
            "verify_slow_calls": verify_misses,
            "verify_new_dirs": verify_new_dirs,
        },
        "out_of_scope": manifest.get("out_of_scope", []),
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
        "verify_slow_calls": report["verification"]["slow_calls"],
        "verify_new_dirs": len(verify_new_dirs),
        "elapsed_s": report["total_elapsed_s"],
        "out": str(out),
    }, indent=2))
    return 0 if coverage_ok else 2


if __name__ == "__main__":
    sys.exit(main())
