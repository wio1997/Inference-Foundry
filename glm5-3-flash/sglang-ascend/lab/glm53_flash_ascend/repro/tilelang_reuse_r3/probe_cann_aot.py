"""Proposal: one fresh-process NPU0 CANN/AOT probe; unchanged 96-case boundary test."""
import argparse, contextlib, hashlib, json, os, re, runpy, subprocess, sys, traceback
from pathlib import Path

SHA = "8b4ab5f9c4eb37f64b4d9746def519a2247a959119db6e19caae0297af988945"

def inventory():
    roots = [Path(os.environ[k]).resolve() for k in ("ASCEND_HOME_PATH", "ASCEND_HOME", "ASCEND_OPP_PATH") if os.environ.get(k)]
    allowed = Path(os.environ["GLM53_MHC_AOT_LIBRARY"]).resolve()
    def cann(path): return any(Path(path).resolve().is_relative_to(root) for root in roots)
    modules = [{"name": name, "path": vars(mod).get("__file__")}
               for name, mod in list(sys.modules.items()) if mod is not None and re.search(r"(^|\.)(tilelang|tvm)(\.|$)", name)]
    paths = sorted({line.split(None, 5)[5] for line in Path("/proc/self/maps").read_text().splitlines()
                    if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith("/") and ".so" in line.split(None, 5)[5]})
    bad = [x for x in modules if x["name"].split(".")[0] == "tilelang" or not x["path"] or not cann(x["path"])]
    bad += [{"native_path": x} for x in paths if re.search(r"tilelang|(?:^|[/_])(?:lib)?tvm", x, re.I)
            and Path(x).resolve() != allowed and not cann(x)]
    return {"related_modules": modules, "mapped_so_paths": paths, "forbidden": bad,
            "cann_roots_from_selected_env": [str(x) for x in roots], "aot_library_exception": str(allowed)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-tree", "component-output-dir", "output-dir"): p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-commit", required=True); p.add_argument("--device", choices=("npu:0",), default="npu:0")
    a = p.parse_args(); tree = a.candidate_tree.resolve(); out = a.output_dir.resolve(); component = a.component_output_dir.resolve()
    if out.exists() or component.exists() or out == component or out.is_relative_to(tree) or component.is_relative_to(tree): raise ValueError("new evidence directories outside candidate are required")
    out.mkdir(parents=True); result = {"status": "failed", "device": a.device, "steps": {}, "source_files": [], "is_full_model_or_performance": False}; device_set = False
    def fail(step):
        result["steps"].setdefault(step, {}).update(status="failed", error=traceback.format_exc().splitlines()[-1])
        with (out / "tracebacks.txt").open("a") as f: f.write("\n" + step + "\n" + traceback.format_exc())
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", a.expected_commit): raise ValueError("full commit SHA required")
        git = lambda *args: subprocess.check_output(["git", "-C", str(tree), *args])
        if git("rev-parse", "HEAD").decode().strip() != a.expected_commit or git("status", "--porcelain=v1").strip(): raise ValueError("candidate HEAD must equal expected commit and be clean")
        test = tree / "lab/glm53_flash_ascend/repro/mhc_sinkhorn_r3/test_boundary.py"
        sources = [Path(__file__).resolve(), test, *sorted((tree / "python/sglang/kernels/ops/layernorm").glob("mhc*.py"))]
        for path in sources:
            data = path.read_bytes(); rel = path.relative_to(tree).as_posix()
            if data != git("cat-file", "blob", a.expected_commit + ":" + rel): raise ValueError("source Git blob mismatch: " + rel)
            result["source_files"].append({"path": rel, "sha256": hashlib.sha256(data).hexdigest()})
        result["source_commit"] = a.expected_commit
        import torch, torch_npu
        torch.npu.set_device(0); device_set = True
        cpu = torch.randint(-128, 128, (64, 64), generator=torch.Generator(device="cpu").manual_seed(20261022), dtype=torch.int8, device="cpu")
        def cast(label):
            actual = torch_npu.npu_format_cast(cpu.to(a.device), 29); torch.npu.synchronize()
            if not torch.equal(actual.cpu(), cpu): raise AssertionError("format_cast values changed")
            result["steps"][label] = {"status": "passed", "shape": [64, 64], "dtype": "int8", "format": 29}
        cast("cann_before"); result["inventory_before"] = inventory()
        if result["inventory_before"]["forbidden"]: raise RuntimeError("compiler/private TVM already loaded before boundary test")
        original = sys.argv[:]; exit_code = None
        try:
            sys.argv = [str(test), "--implementation", "tilelang-aot", "--sinkhorn-eps", "1e-6", "--output-dir", str(component), "--expected-commit", a.expected_commit]
            with (out / "component.stdout").open("w") as stdout, (out / "component.stderr").open("w") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try: runpy.run_path(str(test), run_name="__main__"); exit_code = 0
                except SystemExit as e: exit_code = 0 if e.code is None else e.code
        except Exception: fail("boundary_exception")
        finally: sys.argv = original
        summary = json.loads((component / "results.json").read_text())
        result["steps"]["boundary96"] = {"exit_code": exit_code, "counts": summary.get("counts"), "fatal": summary.get("fatal"), "status": summary.get("status")}
        try:
            mods = [m for m in list(sys.modules.values()) if m is not None and vars(m).get("_SELECTED_KERNEL_SHA") == SHA and callable(vars(m).get("fused_sinkhorn_normalize"))]
            if len(mods) != 1 or not Path(mods[0].__file__).resolve().is_relative_to(tree / "python"): raise RuntimeError("one selected candidate pure-AOT module required")
            result["runtime_identity"] = {"module": str(Path(mods[0].__file__).resolve()), "module_sha256": hashlib.sha256(Path(mods[0].__file__).read_bytes()).hexdigest(), "library": vars(mods[0]).get("_LIB_ID"), "torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", None)}
            default = torch.npu.default_stream(); aux = torch.npu.Stream()
            with torch.npu.stream(default):
                value = (torch.randn((5, 4, 4), generator=torch.Generator(device="cpu").manual_seed(20261022), device="cpu").to(a.device).softmax(-1) + 1e-6).contiguous(); saved = value.clone()
                ref = value / (value.sum(-2, keepdim=True) + 1e-6)
                for _ in range(19): ref = ref / (ref.sum(-1, keepdim=True) + 1e-6); ref = ref / (ref.sum(-2, keepdim=True) + 1e-6)
            aux.wait_stream(default)
            with torch.npu.stream(aux): actual = mods[0].fused_sinkhorn_normalize(value, 1e-6, 20)
            default.wait_stream(aux); torch.npu.synchronize(); error = float((actual - ref).abs().max().item())
            result["steps"]["stream_smoke"] = {"status": "checking", "max_abs_error": error, "input_unchanged": bool(torch.equal(value, saved)), "seed": 20261022, "scope": "one nondefault-stream routing smoke, not lifetime stress certification"}
            torch.testing.assert_close(actual, ref, atol=2e-7, rtol=2e-6)
            if not torch.equal(value, saved): raise AssertionError("stream smoke changed input")
            result["steps"]["stream_smoke"]["status"] = "passed"
        except Exception: fail("stream_smoke")
    except Exception: fail("precheck_or_probe")
    finally:
        if device_set:
            try: cast("cann_after")
            except Exception: fail("cann_after")
            try: result["inventory_after"] = inventory()
            except Exception: fail("inventory_after")
        steps = result["steps"]; boundary = steps.get("boundary96", {})
        passed = (boundary.get("exit_code") == 0 and boundary.get("status") == "passed" and boundary.get("counts") == {"passed": 96, "failed": 0, "not_executed": 0} and boundary.get("fatal") is None and
                  all(steps.get(k, {}).get("status") == "passed" for k in ("cann_before", "stream_smoke", "cann_after")) and "inventory_after" in result and not result["inventory_after"]["forbidden"])
        result["status"] = "passed_limited_probe" if passed else "failed"; (out / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"status": result["status"], "results": str(out / "probe.json"), "is_full_model_or_performance": False}))
    return 0 if result["status"] == "passed_limited_probe" else 1

if __name__ == "__main__": sys.exit(main())
