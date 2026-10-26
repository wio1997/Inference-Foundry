# SPDX-License-Identifier: MIT
"""CPU-only export of the existing LibraryGenerator cache; never import TL/TVM.

Uses the already compiled c27 public ordered tail, not a new mathematical kernel.
The public 2f22 gemm_aot example documents the same bare ctypes launcher route.
No params.pkl deserialization, compilation, dlopen or NPU execution occurs here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

UPSTREAM = "2f22c6ac17a4235aa3de5ccf25729c3553b46e02"
MATH_SOURCE = "c27da9d1b830db29237fe26f61a637a27ef5aabe"
LICENSE_SHA = "bfd7d57808112905dce724b04e7b3bebb823d3e74737d5d83e9194464015017b"
ABI = "void call(uint8_t*, uint8_t*, int64_t, aclrtStream)"
KERNEL_SHA = "8b4ab5f9c4eb37f64b4d9746def519a2247a959119db6e19caae0297af988945"
SOURCE_SHA = "c6b161af7acf52e25019a730d201811179ac73cf58101850f6c61a3a51949331"


def read_bound(path, maximum=16 * 1024 * 1024):
    before = path.stat()
    if before.st_size > maximum:
        raise ValueError("bounded input exceeds limit: " + path.name)
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("input changed during read: " + path.name)
    return data, {"name": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                  "stat_stable": True}


def inspect_elf(path, out):
    results = {}
    for label, flag in (("dynamic", "-d"), ("symbols", "--dyn-syms")):
        run = subprocess.run(["readelf", flag, "--wide", str(path)], capture_output=True,
                             text=True, timeout=30, check=False)
        (out / ("readelf-" + label + ".txt")).write_text(run.stdout + run.stderr)
        if run.returncode:
            raise ValueError("readelf " + label + " failed: " + str(run.returncode))
        results[label] = run.stdout
    needed = re.findall(r"\(NEEDED\).*?\[([^\]]+)\]", results["dynamic"])
    if any(re.search(r"tilelang|tvm", name, re.I) for name in needed):
        raise ValueError("direct TileLang/TVM dependency is outside the pure launcher contract")
    forbidden = [line for line in results["symbols"].splitlines()
                 if " UND " in line and re.search(r"tilelang|tvm", line, re.I)]
    if forbidden:
        raise ValueError("undefined TileLang/TVM symbols: " + " | ".join(forbidden))
    exported = [line for line in results["symbols"].splitlines()
                if re.search(r"\bFUNC\s+GLOBAL\s+\S+\s+(?!UND\b)\S+\s+call\s*$", line)]
    if not exported:
        raise ValueError("global defined call symbol was not found")
    return {"direct_needed": needed, "call_export": exported, "undefined_tl_tvm": forbidden}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-dir", "output-dir", "license-file"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("expected-kernel-sha256", "expected-source-sha256", "expected-tool-commit"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    if not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in
               (a.expected_kernel_sha256, a.expected_source_sha256)):
        p.error("expected file hashes must be complete lowercase SHA256")
    if a.expected_kernel_sha256 != KERNEL_SHA or a.expected_source_sha256 != SOURCE_SHA:
        p.error("this bounded exporter accepts only the already selected c27 compiled artifact")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
                                   text=True).strip()
    if head != a.expected_tool_commit or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("exporter must run from the expected committed tool tree")
    script = Path(__file__).resolve()
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=script.parent, text=True).strip())
    committed = subprocess.check_output(["git", "cat-file", "blob", head + ":" + script.relative_to(repo).as_posix()], cwd=repo)
    if committed != script.read_bytes():
        raise ValueError("exporter bytes do not match its committed Git blob")
    a.output_dir.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "status": "failed_cpu_precheck", "model_or_npu_executed": False,
              "tool_commit": head, "math_origin_commit": MATH_SOURCE, "upstream_compiler_reference": UPSTREAM,
              "cache_origin": "existing LibraryGenerator output; not JITKernel.export_library or params.pkl",
              "runtime_qualification": "PENDING: CANN-first + same 96 gates + full model",
              "call_abi": ABI, "input_output": "contiguous NPU FP32 [T,4,4]", "epsilon": 1e-6,
              "sinkhorn_repeat": 20, "target": "ascendc", "platform": "A3"}
    try:
        bound = {name: read_bound(a.cache_dir / name) for name in
                 ("kernel.cu", "wrapped_kernel.cu", "kernel_lib.so", "auto_gm_idx.pkl")}
        result["inputs"] = [identity for _, identity in bound.values()]
        for name in ("kernel.cu", "wrapped_kernel.cu"):
            if bound[name][1]["sha256"] != a.expected_source_sha256:
                raise ValueError("generated source does not match the fixed existing kernel")
        if bound["kernel_lib.so"][1]["sha256"] != a.expected_kernel_sha256:
            raise ValueError("compiled kernel SHA256 mismatch")
        if bound["kernel.cu"][0] != bound["wrapped_kernel.cu"][0]:
            raise ValueError("unexpected wrapper transformation requires separate review")
        source = bound["wrapped_kernel.cu"][0].decode("utf-8")
        signatures = re.findall(r'extern\s+"C"\s+void\s+call\s*\(([^)]*)\)', source)
        args = [re.sub(r"\s+", " ", x.strip()) for x in signatures[0].split(",")] if len(signatures) == 1 else []
        patterns = (r"uint8_t\s*\*\s*\w+", r"uint8_t\s*\*\s*\w+", r"int64_t n", r"aclrtStream stream")
        if len(args) != 4 or not all(re.fullmatch(pattern, value) for pattern, value in zip(patterns, args)):
            raise ValueError("unsupported buffer/scalar/workspace ABI; do not guess")
        if re.search(r'extern\s+"C"\s+(?:int|void)\s+init\s*\(', source):
            raise ValueError("unexpected manual init ABI")
        gm = json.loads(bound["auto_gm_idx.pkl"][0].decode("utf-8"))
        if gm not in (None, []):
            raise ValueError("auto-GM workspace is outside this minimal runtime")
        result["actual_call_declaration"] = signatures[0]
        result["auto_gm_indices"] = gm
        result["elf"] = inspect_elf(a.cache_dir / "kernel_lib.so", a.output_dir)
        # Recheck after readelf so a changed file cannot replace the inspected ELF.
        if read_bound(a.cache_dir / "kernel_lib.so")[1] != bound["kernel_lib.so"][1]:
            raise ValueError("ELF changed during CPU inspection")
        license_data, license_id = read_bound(a.license_file)
        if license_id["sha256"] != LICENSE_SHA:
            raise ValueError("MIT notice is not the fixed upstream blob")
        for name in ("kernel.cu", "wrapped_kernel.cu", "kernel_lib.so"):
            (a.output_dir / name).write_bytes(bound[name][0])
        (a.output_dir / "LICENSE-MIT.txt").write_bytes(license_data)
        result["license"] = license_id
        result["exporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result["status"] = "exported_not_executed"
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    (a.output_dir / "artifact.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "model_or_npu_executed": False,
                      "error": result.get("error"), "artifact": str(a.output_dir / "artifact.json")}))
    return 0 if result["status"] == "exported_not_executed" else 2


if __name__ == "__main__":
    sys.exit(main())
