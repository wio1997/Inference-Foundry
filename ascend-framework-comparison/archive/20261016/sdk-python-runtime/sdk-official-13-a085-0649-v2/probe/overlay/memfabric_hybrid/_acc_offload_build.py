# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""acc_offload extend library (libmf_hybm_accoffload.so) build entry.

Compiles the AscendC kernel + torch_npu launch adapter shipped inside the NPU
wheel into a loadable extend library. Invoked via ``mfcli kernel install``;
the environment must provide CANN (ASCEND_HOME_PATH), torch and torch_npu.
"""

import fcntl
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

from .env_utils import get_ascend_home as _get_ascend_home
from .env_utils import get_package_dir as _get_package_dir
from .env_utils import get_torch_abi_flag as _get_torch_abi_flag

_WHEEL_MARKER = ".acc_offload_provision_wheel_only"
_SRC_REL = "_kernel/acc_offload"
_KERNEL_SRC_DIR = "operators"
_LAUNCH_SRC_FILE = "launch/acc_offload_operators_launch.cpp"

_ACCOFFLOAD_LIB = "libmf_hybm_accoffload.so"

_BUILD_TIMEOUT_SEC = 600
_STDERR_TRUNC = 4096

# Lock wait capped at the full build budget so a process that waits for an
# in-flight build can always outlive it instead of failing on a shorter
# fixed timeout.
_LOCK_TIMEOUT_SEC = _BUILD_TIMEOUT_SEC + 60

_SOC_CONFIG = {
    "A5": {"soc_version": "ascend950pr_9599", "abi_macro": "ACC_SOC_VERSION_A5"},
    "A3": {"soc_version": "ascend910_9382", "abi_macro": "ACC_SOC_VERSION_A3"},
}


class ExtendBuildError(Exception):
    """Carries stage and truncated stderr for extend library build failures."""

    def __init__(self, stage, exit_code, stderr_truncated):
        parts = ["extend build failed at stage '%s'" % stage]
        if exit_code is not None:
            parts.append("exit code %s" % exit_code)
        if stderr_truncated:
            parts.append("stderr: %s" % stderr_truncated)
        super().__init__(", ".join(parts))
        self.stage = stage
        self.exit_code = exit_code
        self.stderr_truncated = stderr_truncated


def _log(level, msg):
    getattr(logging.getLogger(__name__), {"INFO": "info", "WARN": "warning", "ERROR": "error"}[level])(msg)


def _echo_header():
    """Print the section header separating acc_offload output from AICPU output."""
    print("\nMemFabric Hybrid AccOffload:")


def _get_pkg_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _is_wheel_install(pkg_dir):
    return os.path.isfile(os.path.join(pkg_dir, _WHEEL_MARKER))


def _cache_root():
    return os.path.join(os.path.expanduser("~"), ".cache", "memfabric_hybrid", "extend")


def _cache_key(torch_env):
    if torch_env is None:
        return hashlib.sha256(b"none").hexdigest()[:16]
    env = "%s|%s" % (torch_env["torch_dir"], torch_env["torch_npu_dir"])
    return hashlib.sha256(env.encode("utf-8")).hexdigest()[:16]


def _lock_path(install_dir):
    digest = hashlib.sha1(install_dir.encode("utf-8")).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), "memfabric_hybrid_extend_%s.lock" % digest)


def _run_cmd(args, stage, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ExtendBuildError(stage, None, "timed out (deadline exceeded)")
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "")[:_STDERR_TRUNC]
        err = (exc.stderr or "")[:_STDERR_TRUNC]
        detail = ("stdout: %s, stderr: %s" % (out, err)) if out else err
        raise ExtendBuildError(stage, None, "timed out: %s" % detail) from exc
    except OSError as exc:
        raise ExtendBuildError(stage, None, "cannot execute '%s': %s" % (args[0], exc)) from exc
    if proc.returncode != 0:
        raise ExtendBuildError(stage, proc.returncode, proc.stderr[:_STDERR_TRUNC])


def _torch_env():
    python_bin = sys.executable or "python3"
    torch_dir = _get_package_dir("torch")
    if torch_dir is None:
        return None
    npu_dir = _get_package_dir("torch_npu")
    if npu_dir is None:
        return None
    abi_flag = _get_torch_abi_flag(python_bin)
    if abi_flag is None:
        raise ExtendBuildError(
            "check_torch_abi",
            None,
            "cannot query torch _GLIBCXX_USE_CXX11_ABI via %s, torch may be broken" % python_bin,
        )
    return {
        "torch_dir": torch_dir,
        "torch_npu_dir": npu_dir,
        "abi_flag": abi_flag,
    }


def _built(install_dir):
    """acc_offload extend lib present and non-empty (placeholder reset on wheel reinstall)."""
    path = os.path.join(install_dir, _ACCOFFLOAD_LIB)
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _fast_target(pkg_dir):
    """Package lib dir when wheel install and writable; None otherwise (no heavy probing)."""
    if not _is_wheel_install(pkg_dir):
        return None
    lib_dir = os.path.join(pkg_dir, "lib")
    if os.access(lib_dir, os.W_OK):
        return lib_dir
    return None


def _acquire_lock(lock_path, deadline):
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.close(fd)
                raise ExtendBuildError("acquire_lock", None, "timeout after %ds" % _LOCK_TIMEOUT_SEC) from exc
            time.sleep(min(remaining, 1.0))


def _release_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def _install_one(src_path, install_dir):
    dst = os.path.join(install_dir, os.path.basename(src_path))
    tmp = os.path.join(install_dir, ".tmp_%s_%s" % (os.path.basename(src_path), os.urandom(6).hex()))
    shutil.copy2(src_path, tmp)
    os.replace(tmp, dst)
    return True


def _build_kernel(tmp_dir, src_dir, soc_version, ascend_home, deadline):
    build_dir = os.path.join(tmp_dir, "kernel_build")
    _run_cmd(
        [
            "cmake",
            "-S",
            os.path.join(src_dir, _KERNEL_SRC_DIR),
            "-B",
            build_dir,
            "-DSOC_VERSION=" + soc_version,
            "-DCMAKE_BUILD_TYPE=RELEASE",
            "-DASCEND_HOME_PATH=" + ascend_home,
        ],
        "cmake_configure",
        deadline,
    )
    _run_cmd(["cmake", "--build", build_dir, "-j", str(os.cpu_count() or 8)], "cmake_build", deadline)
    kern_a = os.path.join(build_dir, "lib", "libmf_accoffload_kernel.a")
    if not os.path.isfile(kern_a):
        raise ExtendBuildError("kernel_build", None, "libmf_accoffload_kernel.a not found: %s" % kern_a)
    return kern_a


def _launch_includes(src_dir, torch_dir, npu_dir, ascend_home):
    includes = ["-isystem", os.path.join(src_dir, _KERNEL_SRC_DIR)]
    includes += ["-isystem", os.path.join(ascend_home, "include")]
    includes += ["-isystem", os.path.join(ascend_home, "include/experiment/runtime/runtime/")]
    for inc in (
        "include",
        "include/torch/csrc/api/include",
        "include/torch/csrc/utils",
        "include/c10/util",
        "include/c10/core",
        "include/ATen",
        "include/ATen/detail",
    ):
        includes += ["-isystem", os.path.join(torch_dir, inc)]
    for inc in ("include", "include/torch_npu/csrc/aten", "include/torch_npu/csrc/core/npu"):
        includes += ["-isystem", os.path.join(npu_dir, inc)]
    return includes


def _launch_link_cmd(out, obj, kern_a, torch_dir, npu_dir, ascend_home):
    return [
        "g++",
        "-shared",
        "-fPIC",
        "-o",
        out,
        obj,
        "-Wl,--whole-archive",
        kern_a,
        "-Wl,--no-whole-archive",
        "-L" + os.path.join(torch_dir, "lib"),
        "-ltorch",
        "-lc10",
        "-ltorch_python",
        "-L" + os.path.join(npu_dir, "lib"),
        "-ltorch_npu",
        "-L" + os.path.join(ascend_home, "lib64"),
        "-lopapi",
        "-Wl,-z,noexecstack",
        "-Wl,-z,relro",
        "-Wl,-z,now",
        "-Wl,-rpath,$ORIGIN",
        "-Wl,-rpath," + os.path.join(torch_dir, "lib"),
        "-Wl,-rpath," + os.path.join(npu_dir, "lib"),
        "-Wl,-rpath," + os.path.join(ascend_home, "lib64"),
    ]


def _build_launch(tmp_dir, src_dir, cfg, torch_env, ascend_home, kern_a, deadline):
    obj = os.path.join(tmp_dir, "acc_offload_operators_launch.o")
    src = os.path.join(src_dir, _LAUNCH_SRC_FILE)
    torch_dir = torch_env["torch_dir"]
    npu_dir = torch_env["torch_npu_dir"]
    _run_cmd(
        [
            "g++",
            "-c",
            "-fPIC",
            "-std=c++17",
            "-O3",
            "-fstack-protector-strong",
            "-Wno-unused-parameter",
            "-Wno-unused-function",
            "-Wunused-value",
            "-Wcast-align",
            "-Wcast-qual",
            "-Wwrite-strings",
            "-Wsign-compare",
            "-Wextra",
            "-fvisibility-inlines-hidden",
            "-ftrapv",
            torch_env["abi_flag"],
            "-D" + cfg["abi_macro"],
        ]
        + _launch_includes(src_dir, torch_dir, npu_dir, ascend_home)
        + [src, "-o", obj],
        "compile_launch",
        deadline,
    )
    out = os.path.join(tmp_dir, _ACCOFFLOAD_LIB)
    cmd = _launch_link_cmd(out, obj, kern_a, torch_dir, npu_dir, ascend_home)
    _run_cmd(cmd, "link_launch", deadline)
    return out


def _resolve_target(pkg_dir, install_dir, torch_env):
    if install_dir is not None:
        return install_dir
    if not _is_wheel_install(pkg_dir):
        return None
    lib_dir = os.path.join(pkg_dir, "lib")
    if os.access(lib_dir, os.W_OK):
        return lib_dir
    return os.path.join(_cache_root(), _cache_key(torch_env))


def _gc_cache(cache_root, keep_key):
    try:
        for name in os.listdir(cache_root):
            if name == keep_key:
                continue
            shutil.rmtree(os.path.join(cache_root, name))
    except OSError:
        pass


def _resolve_early(pkg_dir, src_dir, install_dir, force):
    """Return (target, skip) resolving install target and early-exit cases."""
    if not os.path.isdir(src_dir):
        if install_dir is not None:
            raise ExtendBuildError("check_src", None, "extend sources not found in wheel: %s" % src_dir)
        return None, True
    target = install_dir or _fast_target(pkg_dir)
    if target is not None:
        os.makedirs(target, exist_ok=True)
        if not force and _built(target):
            return target, True
    return target, False


def _build_once(target, src_dir, soc_key, torch_env, ascend_home):
    """Compile and install the extend library. Caller must hold the lock."""
    deadline = time.monotonic() + _BUILD_TIMEOUT_SEC
    cfg = _SOC_CONFIG[soc_key]
    tmp_dir = tempfile.mkdtemp(prefix="acc_offload_extend_")
    try:
        kern_a = _build_kernel(tmp_dir, src_dir, cfg["soc_version"], ascend_home, deadline)
        _install_one(_build_launch(tmp_dir, src_dir, cfg, torch_env, ascend_home, kern_a, deadline), target)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return target


def ensure_acc_offload_ext(soc_key, install_dir=None, force=False):
    """Resolve the install target, check environment and build the extend library.

    ``soc_key`` (A2/A3/A5, or None when undetectable) is resolved by the caller.
    Returns the install dir on success, or None when the build is skipped
    (sources absent or unsupported SoC/CANN/torch environment).
    Raises ExtendBuildError on build failure.
    """
    pkg_dir = _get_pkg_dir()
    src_dir = os.path.join(pkg_dir, _SRC_REL)
    target, done = _resolve_early(pkg_dir, src_dir, install_dir, force)
    if done:
        if target is not None:
            _echo_header()
            print("  Already installed. Path:\n    %s" % os.path.join(target, _ACCOFFLOAD_LIB))
        return target
    _echo_header()
    print("  Start installing, this may take minutes...")

    ascend_home = _get_ascend_home()
    if not os.path.isdir(ascend_home):
        _log("WARN", "ASCEND_HOME_PATH=%s is not a valid directory, skip extend build." % ascend_home)
        return None

    if soc_key not in ("A3", "A5"):
        _log("INFO", "acc_offload kernel only supports A3/A5 (detected %s), skipped." % soc_key)
        return None

    torch_env = _torch_env()
    if torch_env is None:
        _log("WARN", "torch/torch_npu not importable, acc_offload kernel will be skipped.")
        return None

    target = _resolve_target(pkg_dir, install_dir, torch_env)
    if target is None:
        _log("ERROR", "cannot resolve install target: not a wheel install and no --install-dir given.")
        return None
    os.makedirs(target, exist_ok=True)
    if not force and _built(target):
        return target

    lock_fd = _acquire_lock(_lock_path(target), time.monotonic() + _LOCK_TIMEOUT_SEC)
    try:
        if not force and _built(target):
            return target
        _build_once(target, src_dir, soc_key, torch_env, ascend_home)
    finally:
        _release_lock(lock_fd)
    if target.startswith(_cache_root()):
        _gc_cache(_cache_root(), os.path.basename(target))
    print("  Install done. Path:\n    %s" % os.path.join(target, _ACCOFFLOAD_LIB))
    return target


def uninstall_acc_offload_ext():
    """Reset the extend library artifacts to the fresh-wheel state.

    Restores the placeholder .so in the wheel lib dir (so a later install
    re-probes the environment) and removes user-cache builds.
    Never raises: uninstall must not fail on a partially installed state.
    """
    _echo_header()
    print("  Start uninstalling...")
    pkg_dir = _get_pkg_dir()
    lib_dir = os.path.join(pkg_dir, "lib")
    if os.path.isdir(lib_dir) and os.access(lib_dir, os.W_OK):
        _reset_placeholder(lib_dir, _ACCOFFLOAD_LIB)
    try:
        cache_root = _cache_root()
        if os.path.isdir(cache_root):
            shutil.rmtree(cache_root, ignore_errors=True)
    except OSError:
        pass
    print("  Uninstall done.")


def _reset_placeholder(lib_dir, filename):
    """Overwrite a build artifact with an empty placeholder file."""
    path = os.path.join(lib_dir, filename)
    if not os.path.exists(path):
        return
    try:
        tmp = os.path.join(lib_dir, ".tmp_uninstall_%s" % filename)
        with open(tmp, "w") as f:
            f.write("")
        os.replace(tmp, path)
    except OSError as exc:
        _log("WARN", "failed to reset %s: %s" % (path, exc))


def _lib_status(lib_dir):
    """Return (state, detail) for the extend library in the given directory."""
    lib_path = os.path.join(lib_dir, _ACCOFFLOAD_LIB)
    if _built(lib_dir):
        return "installed", lib_path
    return "not installed", "run 'mfcli kernel install' to build it"


def acc_offload_info():
    """Print the acc_offload extend library install status. Never raises."""
    _echo_header()
    lib_dir = os.path.join(_get_pkg_dir(), "lib")
    if not _is_wheel_install(_get_pkg_dir()):
        print("  [not available]")
        return
    state, detail = _lib_status(lib_dir)
    if os.access(lib_dir, os.W_OK):
        if state == "installed":
            print("  Installed. Path: %s" % detail)
        else:
            print("  Not installed. %s" % detail)
        return
    cache_dir = os.path.join(_cache_root(), _cache_key(_torch_env()))
    state, detail = _lib_status(cache_dir)
    if state == "installed":
        print("  Installed (in user cache). Path: %s" % detail)
    else:
        print("  Not installed. %s" % detail)
