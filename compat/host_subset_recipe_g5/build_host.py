"""Target CPU-only build of explicit four-interface host registration subset.

Creates a new task-private build directory. Never installs, edits original
sources/vendors, compiles device kernels, enables a vendor, or initializes NPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--cann-root", type=Path, required=True)
    p.add_argument("--build-dir", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=2, choices=range(1, 5))
    p.add_argument("--timeout-seconds", type=int, default=600)
    a = p.parse_args()
    if platform.machine() not in ("aarch64", "arm64"):
        raise RuntimeError("Use the parent's target aarch64 CPU container, not the client VM")
    here = Path(__file__).resolve().parent
    source, cann = a.source_root.resolve(strict=True), a.cann_root.resolve(strict=True)
    out = a.build_dir.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError("New build directory required: " + str(out))
    if out.is_relative_to(source) or out.is_relative_to(cann):
        raise ValueError("Build directory must be outside input source/toolkit")
    inputs = json.loads((here / "inputs_manifest.json").read_text())
    for item in inputs["files"]:
        path = (source / item["path"]).resolve(strict=True)
        if not path.is_relative_to(source) or sha(path) != item["sha256"]:
            raise RuntimeError("Pinned actual source mismatch: " + item["path"])
    out.mkdir(parents=True)
    result = {"status": "building", "scope": __doc__, "machine": platform.machine(),
              "source_root": str(source), "cann_root": str(cann),
              "inputs_manifest_sha256": sha(here / "inputs_manifest.json"),
              "recipe": {f: sha(here / f) for f in (
                  "CMakeLists.txt", "proto_subset.cpp", "build_host.py")},
              "commands": [], "outputs": {}, "runtime_verified": False,
              "started_unix": time.time()}

    def save():
        (out / "build_result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        # No inherited custom vendor or Torch dependency participates in compilation.
        env = dict(os.environ)
        env.pop("ASCEND_CUSTOM_OPP_PATH", None)
        commands = [
            ["cmake", "-S", str(here), "-B", str(out),
             "-DDONOR_SOURCE_ROOT=" + str(source), "-DCURRENT_CANN_ROOT=" + str(cann),
             "-DCMAKE_BUILD_TYPE=Release"],
            ["cmake", "--build", str(out), "--target", "v41_proto", "v41_tiling",
             "--parallel", str(a.jobs)],
        ]
        for index, command in enumerate(commands):
            log = out / ("configure.log" if index == 0 else "compile.log")
            record = {"argv": command, "log": str(log)}
            result["commands"].append(record)
            save()
            with log.open("w") as handle:
                process = subprocess.run(command, cwd=out, env=env, stdout=handle,
                                         stderr=subprocess.STDOUT, timeout=a.timeout_seconds)
            record["returncode"] = process.returncode
            save()
            if process.returncode:
                raise RuntimeError("CPU build failed; inspect " + str(log))
        for name in ("libcust_opsproto_rt2.0.so", "libcust_opmaster_rt2.0.so"):
            path = out / name
            result["outputs"][name] = {"sha256": sha(path), "bytes": path.stat().st_size}
            for tool, extra in (("readelf", ["-d"]), ("nm", ["-D", "-C"]), ("strings", [])):
                log = out / (name + "." + tool + ".txt")
                with log.open("w") as handle:
                    subprocess.run([tool, *extra, str(path)], cwd=out, stdout=handle,
                                   stderr=subprocess.STDOUT, check=True, timeout=30)
        result["status"] = "CPU_compiled_pending_registration_and_NPU_coexistence_acceptance"
    except Exception:
        result["status"], result["error"] = "failed", traceback.format_exc()
    result["finished_unix"] = time.time()
    save()
    print(json.dumps({"status": result["status"], "manifest": str(out / "build_result.json")}), flush=True)
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
