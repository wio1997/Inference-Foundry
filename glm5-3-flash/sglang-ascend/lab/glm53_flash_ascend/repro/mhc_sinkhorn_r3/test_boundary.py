"""Unexecuted R3 component proposal; run only inside the authorized container.

Example (roles, not a historical command):
  python3 test_mhc_sinkhorn_proposal.py --sinkhorn-eps 1e-6 \
    --output-dir "${RUN_ARTIFACTS}/component-01"
This is correctness/contract diagnosis, never a performance or full-model test.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

SEED = 20261022
SHAPES = [0, 1, 2, 3, 4, 7, 16, 127, 128, 129, 2048, 4096]
BOUNDARY_SHAPES = [1, 4, 128, 2048]
CASES = ["balanced", "random", "skewed", "alternating_20", "all_20", "all_minus20"]
LIMITS = {"atol": 2e-7, "rtol": 2e-6, "max_relative": 2e-5,
          "relative_floor": 1e-8, "row_column_atol": 2e-6,
          "bf16_post_max_ulp": 1}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def tensor_hash(value):
    import torch
    return sha(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def reference_normalize(comb, eps):
    value = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(19):
        value = value / (value.sum(-1, keepdim=True) + eps)
        value = value / (value.sum(-2, keepdim=True) + eps)
    return value


def bitwise_equal(torch, left, right):
    return (left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.detach().contiguous().cpu().view(torch.uint8),
                            right.detach().contiguous().cpu().view(torch.uint8)))


def numeric_metrics(torch, actual, reference):
    require(actual.dtype == reference.dtype == torch.float32, "FP32 contract")
    require(actual.shape == reference.shape, "shape contract")
    a, r = actual.detach().cpu(), reference.detach().cpu()
    require(bool(torch.isfinite(a).all()), "nonfinite candidate result")
    require(bool(torch.isfinite(r).all()), "nonfinite reference result")
    if not a.numel():
        return {"elements": 0, "max_abs": 0.0, "p99_abs": 0.0,
                "max_relative": 0.0, "p99_relative": 0.0, "max_fp32_ulp": 0,
                "row_sum_max_abs": 0.0, "column_sum_max_abs": 0.0}, {
                    "allclose": True, "relative": True, "row_column": True}
    require(bool((a > 0).all()), "positive input did not remain positive")
    difference = (a.double() - r.double()).abs()
    allowed = LIMITS["atol"] + LIMITS["rtol"] * r.double().abs()
    relative = difference / r.double().abs().clamp_min(LIMITS["relative_floor"])
    selected = relative[r.abs() >= LIMITS["relative_floor"]]
    row_error = float((a.double().sum(-1) - r.double().sum(-1)).abs().max())
    column_error = float((a.double().sum(-2) - r.double().sum(-2)).abs().max())
    metrics = {"elements": a.numel(), "max_abs": float(difference.max()),
               "p99_abs": float(torch.quantile(difference.flatten(), 0.99)),
               "max_relative": float(selected.max()) if selected.numel() else 0.0,
               "p99_relative": float(torch.quantile(relative.flatten(), 0.99)),
               "max_fp32_ulp": int((a.view(torch.int32).long() -
                                     r.view(torch.int32).long()).abs().max()),
               "row_sum_max_abs": row_error, "column_sum_max_abs": column_error}
    return metrics, {"allclose": bool((difference <= allowed).all()),
                     "relative": metrics["max_relative"] <= LIMITS["max_relative"],
                     "row_column": max(row_error, column_error) <= LIMITS["row_column_atol"]}


def bf16_ulp_metrics(torch, actual, reference):
    require(actual.dtype == reference.dtype == torch.bfloat16, "POST BF16 contract")
    require(actual.shape == reference.shape, "POST shape contract")
    a, r = actual.detach().contiguous().cpu(), reference.detach().contiguous().cpu()
    require(bool(torch.isfinite(a).all() and torch.isfinite(r).all()), "POST nonfinite")
    # All ordering work is CPU integer math. +0/-0 share an ordered value.
    def ordered(value):
        bits = value.view(torch.int16).to(torch.int32).bitwise_and(0xffff)
        magnitude = bits.bitwise_and(0x7fff)
        return torch.where(bits.bitwise_and(0x8000) != 0,
                           0x8000 - magnitude, 0x8000 + magnitude)
    distance = (ordered(a) - ordered(r)).abs()
    return {"max_bf16_ulp": int(distance.max()) if distance.numel() else 0,
            "different_bit_elements": int((a.view(torch.int16) != r.view(torch.int16)).sum()),
            "elements": a.numel()}


def git_identity(runtime_paths):
    root = Path(subprocess.check_output(
        ["git", "-C", str(runtime_paths[0].parent), "rev-parse", "--show-toplevel"],
        text=True).strip()).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    files = []
    for path in runtime_paths:
        relative = path.relative_to(root).as_posix()
        blob = subprocess.check_output(["git", "-C", str(root), "show", head + ":" + relative])
        raw = path.read_bytes()
        files.append({"path": "${CANDIDATE_ROOT}/" + relative, "sha256": sha(raw),
                      "git_blob_sha256": sha(blob), "matches_fixed_git_blob": raw == blob})
        require(raw == blob, "runtime file differs from committed Git blob: " + relative)
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain=v1",
                                      "--untracked-files=normal"], text=True)
    require(not status.strip(), "candidate worktree must be clean before this formal component test: " + status.strip())
    return root, {"head": head, "worktree_clean": True, "runtime_files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=["triton", "tilelang", "tilelang-aot"], default="triton")
    parser.add_argument("--sinkhorn-eps", type=float, required=True,
                        help="Actual caller epsilon, explicitly supplied; no guessed default")
    parser.add_argument("--rms-eps", type=float, default=1e-6,
                        help="Declared synthetic full-boundary RMS epsilon, not checkpoint certification")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["npu:0"], default="npu:0")
    parser.add_argument("--expected-commit", help="Optional exact HEAD assertion")
    args = parser.parse_args()
    for name, value in (("sinkhorn_eps", args.sinkhorn_eps), ("rms_eps", args.rms_eps)):
        if not math.isfinite(value) or value <= 0:
            parser.error(name + " must be finite and positive")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "tracebacks").mkdir()
    rows, started, fatal = [], time.time(), None

    def attempt(name, kind, operation, expect_rejection=False):
        row = {"name": name, "kind": kind, "status": "failed"}
        try:
            value = operation()
            if expect_rejection:
                raise AssertionError("invalid input was accepted")
            row.update(status="passed", **(value or {}))
        except Exception as error:
            trace_name = "tracebacks/" + name + ".txt"
            (out / trace_name).write_text(traceback.format_exc(), encoding="utf-8")
            if expect_rejection and isinstance(error, (ValueError, TypeError)):
                row.update(status="passed", expected_rejection=True,
                           exception_type=type(error).__name__, traceback=trace_name)
            else:
                row.update(error_type=type(error).__name__, error=str(error), traceback=trace_name)
        rows.append(row)
        with (out / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return row["status"] == "passed"

    try:
        # Library/identity-hook output goes to raw text, not the concise stdout JSON.
        with (out / "library-output.txt").open("x", encoding="utf-8") as library_output:
            with contextlib.redirect_stdout(library_output), contextlib.redirect_stderr(library_output):
                import torch
                import torch_npu  # noqa: F401
                os.environ["GLM53_MHC_SINKHORN_FUSED"] = "1"
                os.environ["GLM53_EXECUTION_IDENTITY_DIR"] = str(out / "runtime-identities")
                os.environ["GLM53_MHC_SINKHORN_IMPLEMENTATION"] = args.implementation
                from sglang.kernels.ops.layernorm import mhc
                import importlib
                selected_module = {"tilelang": "mhc_sinkhorn_tilelang_npu", "triton": "mhc_sinkhorn_npu", "tilelang-aot": "mhc_sinkhorn_tilelang_aot_npu"}[args.implementation]
                selected = importlib.import_module("sglang.kernels.ops.layernorm." + selected_module)
                fused = selected.fused_sinkhorn_normalize
                require(type(mhc._R3_SINKHORN_FUSED) is bool, "actual mHC opt-in bool missing")
                runtime_paths = [Path(mhc.__file__).resolve(), Path(selected.__file__).resolve()]
                _, git = git_identity(runtime_paths)
                if args.expected_commit:
                    require(git["head"] == args.expected_commit, "unexpected candidate HEAD")
                os.environ["GLM53_CANDIDATE_COMMIT"] = git["head"]
                generator = torch.Generator(device="cpu").manual_seed(SEED)
                data, data_identity = {}, []
                for size in SHAPES:
                    for case in CASES:
                        if case == "balanced":
                            logits = torch.zeros(size, 4, 4)
                        elif case == "random":
                            logits = torch.randn(size, 4, 4, generator=generator)
                        elif case == "skewed":
                            logits = torch.full((size, 4, 4), -20.0)
                            logits[:, :, 0] = 20.0
                        elif case == "alternating_20":
                            logits = (torch.arange(16).reshape(1, 4, 4) % 2 * 40.0 - 20.0).repeat(size, 1, 1)
                        else:
                            logits = torch.full((size, 4, 4), 20.0 if case == "all_20" else -20.0)
                        value = (logits.softmax(-1) + args.sinkhorn_eps).contiguous()
                        data[(size, case)] = value
                        data_identity.append({"T": size, "case": case, "dtype": str(value.dtype),
                                              "logits_sha256": tensor_hash(logits), "c0_sha256": tensor_hash(value)})
                # Generate and hash every full-boundary input before any tested NPU call.
                boundaries, boundary_identity = {}, []
                for size in BOUNDARY_SHAPES:
                    tensors = {"residual": torch.randn(size, 4, 4096, generator=generator).bfloat16(),
                               "fn": torch.randn(24, 16384, generator=generator) * 0.01,
                               "scale": torch.tensor([0.7, 1.1, 0.5]),
                               "base": torch.randn(24, generator=generator) * 0.1,
                               "post_x": torch.randn(size, 4096, generator=generator).bfloat16()}
                    boundaries[size] = tensors
                    boundary_identity.append({"T": size, "tensors": {
                        k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": tensor_hash(v)}
                        for k, v in tensors.items()}})
                torch.npu.set_device(args.device)
                require(torch.npu.current_device() == 0, "single logical-device-0 requirement")
                identity = {"schema": 1, "status": "prepared_not_passed", "created_epoch": time.time(),
                            "git": git, "script_sha256": sha(Path(__file__).read_bytes()),
                            "device": {"requested": args.device, "actual_logical_index": torch.npu.current_device(),
                                       "name": str(torch.npu.get_device_name(0))},
                            "versions": {name: importlib.metadata.version(name)
                                         for name in ("torch", "torch-npu", "triton", "triton-ascend")},
                            "argv_roles": ["python3", "${TEST_SCRIPT}", "--sinkhorn-eps", str(args.sinkhorn_eps),
                                           "--rms-eps", str(args.rms_eps), "--implementation", args.implementation, "--device", args.device,
                                           "--output-dir", "${COMPONENT_OUTPUT}"],
                            "helper_signature": str(inspect.signature(fused)), "seed": SEED, "implementation": args.implementation,
                            "boundary_reference": "Same committed mhc.hc_pre with _R3_SINKHORN_FUSED=False; flag restored in finally",
                            "initial_module_fused_flag": mhc._R3_SINKHORN_FUSED,
                            "sinkhorn_eps": args.sinkhorn_eps, "sinkhorn_repeat": 20,
                            "synthetic_rms_eps": args.rms_eps, "limits": LIMITS,
                            "normalization_data": data_identity, "synthetic_boundary_data": boundary_identity,
                            "scope": "Single NPU component only; synthetic inputs; no persistent state; not performance/model certification"}
                if args.implementation == "tilelang":
                    libraries, maps_error = selected._mapped_tilelang_libraries()
                    identity["tilelang_native_before_test"] = {"libraries": libraries, "maps_error": maps_error}
                    identity["tilelang_python_before_test"] = {name: selected._file_identity(module.__file__) for name in ("tilelang", "tilelang.jit", "tvm") if (module := sys.modules.get(name)) is not None and getattr(module, "__file__", None)}
                write_json(out / "identity.json", identity)

                def check_normalize(size, case):
                    comb = data[(size, case)].to(args.device)
                    before = comb.detach().cpu().clone()
                    reference = reference_normalize(comb, args.sinkhorn_eps)
                    actual = fused(comb, args.sinkhorn_eps, 20)
                    torch.npu.synchronize(0)
                    require(actual.device == comb.device and actual.is_contiguous(), "output device/layout")
                    require(bitwise_equal(torch, comb, before), "input mutation")
                    if size:
                        require(actual.data_ptr() != comb.data_ptr(), "output aliases input")
                        metrics, checks = numeric_metrics(torch, actual, reference)
                        require(all(checks.values()), "strict normalization mismatch: " + json.dumps(metrics))
                    else:
                        require(actual.shape == comb.shape and actual.dtype == torch.float32, "empty contract")
                        metrics = {"elements": 0}
                        checks = {"empty_shape_dtype": True}
                    return {"T": size, "case": case, "metrics": metrics,
                            "checks": checks,
                            "reference_sha256": tensor_hash(reference), "actual_sha256": tensor_hash(actual)}

                # One real compile/execute first. On failure, retain all unexecuted plans explicitly.
                capability_ok = attempt("normalize_T1_balanced", "normalization", lambda: check_normalize(1, "balanced"))
                for size in SHAPES:
                    for case in CASES:
                        if (size, case) == (1, "balanced"):
                            continue
                        name = "normalize_T" + str(size) + "_" + case
                        if capability_ok:
                            attempt(name, "normalization", lambda s=size, c=case: check_normalize(s, c))
                        else:
                            rows.append({"name": name, "kind": "normalization", "status": "not_executed",
                                         "reason": "first real compile/execute or numeric gate failed"})

                def independence():
                    comb = data[(7, "random")].to(args.device)
                    together = fused(comb, args.sinkhorn_eps)
                    apart = torch.cat([fused(comb[i:i + 1].contiguous(), args.sinkhorn_eps) for i in range(7)])
                    order = torch.tensor([6, 0, 4, 2, 1, 5, 3], device=args.device)
                    permuted = fused(comb[order].contiguous(), args.sinkhorn_eps)
                    require(bitwise_equal(torch, together, apart), "single/batch row drift")
                    require(bitwise_equal(torch, together[order], permuted), "permutation row drift")
                    require(bitwise_equal(torch, together, fused(comb, args.sinkhorn_eps)), "repeat drift")
                    return {"rows": 7, "bitwise_independent": True}
                if capability_ok:
                    attempt("row_independence", "independence", independence)
                else:
                    rows.append({"name": "row_independence", "kind": "independence", "status": "not_executed",
                                 "reason": "first normalization gate failed"})

                def full_boundary(size):
                    values = {k: v.to(args.device) for k, v in boundaries[size].items()}
                    residual, fn, scale, base = (values[k] for k in ("residual", "fn", "scale", "base"))
                    norm_weight = torch.ones(4096, dtype=torch.bfloat16, device=args.device)
                    calls = [0]
                    def profile_call(frame, event, arg):
                        if event == "call" and frame.f_code is fused.__code__:
                            calls[0] += 1
                    old_profile = sys.getprofile()
                    old_fused_flag = mhc._R3_SINKHORN_FUSED
                    def pre_call():
                        return mhc.hc_pre(
                            residual.view(size, 16384), fn, scale, base, 4,
                            args.rms_eps, args.sinkhorn_eps, 20,
                            post_mult_value=2.0, hc_norm_weight=None,
                            out_norm_weight=norm_weight, out_norm_eps=args.rms_eps)
                    try:
                        sys.setprofile(profile_call)
                        mhc._R3_SINKHORN_FUSED = False
                        layer_ref, comb_ref_flat, post_ref_flat, norm_ref = pre_call()
                        require(calls[0] == 0, "Torch reference unexpectedly invoked fused helper")
                        mhc._R3_SINKHORN_FUSED = True
                        layer, comb_flat, post_flat, norm_fused = pre_call()
                    finally:
                        mhc._R3_SINKHORN_FUSED = old_fused_flag
                        sys.setprofile(old_profile)
                    post_ref = post_ref_flat.reshape(size, 4, 1)
                    comb_ref = comb_ref_flat.reshape(size, 4, 4)
                    require(calls[0] == 1, "public hc_pre did not invoke expected fused helper once")
                    require(norm_ref is False and norm_fused is False, "norm contract changed")
                    require(bitwise_equal(torch, layer, layer_ref), "layer_input is not bitwise identical")
                    require(bitwise_equal(torch, post_flat, post_ref.reshape(size, 4)), "post is not bitwise identical")
                    comb = comb_flat.reshape(size, 4, 4)
                    metrics, checks = numeric_metrics(torch, comb, comb_ref)
                    require(all(checks.values()), "integrated comb mismatch: " + json.dumps(metrics))
                    reference_out = mhc._mhc_post_torch(values["post_x"], residual, post_ref, comb_ref)
                    actual_out = mhc._mhc_post_torch(values["post_x"], residual, post_flat.reshape(size, 4, 1), comb)
                    ulp = bf16_ulp_metrics(torch, actual_out, reference_out)
                    require(ulp["max_bf16_ulp"] <= LIMITS["bf16_post_max_ulp"], "POST exceeds BF16 ULP limit: " + json.dumps(ulp))
                    return {"T": size, "HC": 4, "H": 4096, "helper_calls": calls[0],
                            "post_bitwise": True, "layer_input_bitwise": True, "norm_fused": False,
                            "comb_metrics": metrics, "comb_checks": checks, "downstream_post": ulp}
                for size in BOUNDARY_SHAPES:
                    if capability_ok:
                        attempt("full_boundary_T" + str(size), "synthetic_full_boundary", lambda s=size: full_boundary(s))
                    else:
                        rows.append({"name": "full_boundary_T" + str(size), "kind": "synthetic_full_boundary",
                                     "status": "not_executed", "reason": "first normalization gate failed"})

                valid = torch.full((1, 4, 4), 0.25, dtype=torch.float32, device=args.device).contiguous()
                invalid_inputs = {"cpu": valid.cpu(), "bf16": valid.bfloat16(),
                                  "rank2": valid[0], "hc3": valid[:, :3, :].contiguous(),
                                  "width5": torch.ones(1, 4, 5, device=args.device),
                                  "noncontiguous": valid.transpose(1, 2)}
                for label, value in invalid_inputs.items():
                    attempt("reject_" + label, "contract", lambda v=value: fused(v, args.sinkhorn_eps), True)
                for label, repeat in (("zero", 0), ("negative", -1), ("nineteen", 19),
                                      ("twentyone", 21), ("float", 20.0), ("bool", True), ("none", None)):
                    attempt("reject_repeat_" + label, "contract", lambda r=repeat: fused(valid, args.sinkhorn_eps, r), True)
                for label, eps in (("zero", 0.0), ("negative", -1e-6), ("nan", float("nan")),
                                   ("infinite", float("inf")), ("bool", True), ("string", "1e-6")):
                    attempt("reject_eps_" + label, "contract", lambda e=eps: fused(valid, e), True)
                torch.npu.synchronize(0)
    except Exception as error:
        fatal = {"type": type(error).__name__, "error": str(error), "traceback": "fatal-traceback.txt"}
        (out / "fatal-traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("passed", "failed", "not_executed")}
    passed = fatal is None and counts["failed"] == 0 and counts["not_executed"] == 0 and counts["passed"] > 0
    result = {"schema": 1, "status": "passed" if passed else "failed", "tests": rows,
              "counts": counts, "fatal": fatal, "started_epoch": started, "ended_epoch": time.time(),
              "scope": "Component correctness/contract only; no model requests or performance measurement"}
    write_json(out / "results.json", result)
    print(json.dumps({"status": result["status"], "counts": counts,
                      "fatal": fatal is not None, "results": "${COMPONENT_OUTPUT}/results.json"}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
