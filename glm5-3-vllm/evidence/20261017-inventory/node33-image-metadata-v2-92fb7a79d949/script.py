#!/usr/bin/env python3
"""Inspect installed files/metadata without importing torch or initializing an NPU."""
import hashlib
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import platform


def main():
    result = {"python": platform.python_version(), "machine": platform.machine(),
              "npu_initialized": False, "model_loaded": False, "packages": {}}
    roots = {}
    for name in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers",
                 "mooncake-transfer-engine", "triton-ascend"):
        try:
            dist = md.distribution(name)
            result["packages"][name] = {"version": dist.version,
                "location": str(dist.locate_file(""))}
            roots[name] = Path(dist.locate_file(""))
        except md.PackageNotFoundError:
            result["packages"][name] = {"installed": False}
    result["version_files"] = {}
    for pattern in ("/usr/local/Ascend/ascend-toolkit/*/version.cfg",
                    "/usr/local/Ascend/ascend-toolkit/*/*/ascend_toolkit_install.info",
                    "/usr/local/Ascend/cann/*/version.cfg"):
        import glob
        for filename in glob.glob(pattern):
            p = Path(filename)
            if p.is_file() and p.stat().st_size < 32768:
                result["version_files"][str(p)] = p.read_text(errors="replace")
    result["source_evidence"] = {}
    patterns = {"vllm_ascend": ("envs.py", "worker/worker.py", "worker/worker_v1.py", "_version.py"),
                "vllm": ("_version.py",)}
    for distname, paths in patterns.items():
        spec = importlib.util.find_spec(distname)  # Top-level lookup: does not execute package code.
        if spec is None or not spec.submodule_search_locations:
            continue
        package_root = Path(next(iter(spec.submodule_search_locations)))
        result.setdefault("package_source_roots", {})[distname] = str(package_root)
        for rel in paths:
            p = package_root / rel
            if p.is_file():
                data = p.read_bytes()
                lines = data.decode(errors="replace").splitlines()
                selected = set()
                for i, line in enumerate(lines):
                    if any(t in line for t in ("V2_MODEL_RUNNER", "model_runner", "ModelRunner", "__version__", "commit_id")):
                        selected.update(range(max(0, i - 2), min(len(lines), i + 4)))
                result["source_evidence"][str(p)] = {"sha256": hashlib.sha256(data).hexdigest(),
                    "lines": [{"line": i + 1, "text": lines[i]} for i in sorted(selected)]}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
