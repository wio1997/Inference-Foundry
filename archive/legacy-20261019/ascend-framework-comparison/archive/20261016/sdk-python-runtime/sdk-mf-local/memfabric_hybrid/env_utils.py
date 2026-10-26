# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Environment probes shared across memfabric_hybrid tooling.

Covers Ascend CANN path, SoC type detection (via ACL or npu-smi) and
torch/torch_npu package discovery. Kept import-light so callers such as
``mfcli`` do not drag in torch.
"""

import importlib.util
import os
import re
import shutil
import subprocess


def get_ascend_home():
    """Resolve the CANN home directory, defaulting to the standard toolkit path."""
    path = os.environ.get("ASCEND_HOME_PATH")
    if not path:
        path = os.environ.get("ASCEND_TOOLKIT_HOME")
    if not path:
        path = "/usr/local/Ascend/ascend-toolkit/latest"
    return path


def get_package_dir(name):
    """Return the top-level directory of an installed package, or None."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return None
    if spec.submodule_search_locations:
        return str(spec.submodule_search_locations[0])
    return os.path.dirname(spec.origin)


def _detect_soc_from_acl():
    try:
        import acl

        soc = acl.get_soc_name()
    except Exception:
        return None
    return soc if isinstance(soc, str) and soc else None


def _detect_soc_from_npu_smi():
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        return None
    try:
        proc = subprocess.run([npu_smi, "info"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = proc.stdout or ""
    m = re.search(r"Ascend(?:9|3)[0-9]+[A-Za-z0-9]*", text)
    if m:
        return m.group(0)
    m = re.search(r"(?:9|3)[0-9]{2}[A-Za-z0-9]*", text)
    if m and m.group(0).startswith(("910", "950", "310")):
        return m.group(0)
    return None


def get_soc_name():
    """Detect the SoC name via ACL, falling back to npu-smi output. None when unavailable."""
    soc = _detect_soc_from_acl()
    if soc is None:
        soc = _detect_soc_from_npu_smi()
    return soc


def _map_soc_key(soc):
    if not soc:
        return None
    if "950" in soc or "910_95" in soc:
        return "A5"
    if "910B" in soc:
        return "A2"
    if "910" in soc:
        return "A3"
    return None


def get_soc_key():
    """Map the current SoC to its Ascend generation key (A2/A3/A5), or None when undetectable."""
    return _map_soc_key(get_soc_name())


def get_torch_abi_flag(python_bin):
    """Query torch's C++ ABI flag by running a snippet in a child interpreter.

    mfcli never imports torch itself, so the ABI cannot be read from
    sys.modules; run the same query as the legacy install script instead.
    """
    try:
        proc = subprocess.run(
            [python_bin, "-c", "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    abi = proc.stdout.strip()
    if abi not in ("0", "1"):
        return None
    return "-D_GLIBCXX_USE_CXX11_ABI=%s" % abi
