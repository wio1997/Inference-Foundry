#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import sys
import ctypes

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
sys.path.append(current_dir)
libs_path = os.path.join(current_dir, 'lib')
# Keep handles to prevent GC-triggered dlclose before process exit (avoids double-free).
_lib_handles = []
for lib in ["libmf_hybm_core.so", "libmf_smem.so", "libmf_acc_offload.so"]:
    _lib_handles.append(ctypes.CDLL(os.path.join(libs_path, lib), mode=ctypes.RTLD_LOCAL))

# Preload optional dlopen-ed libraries (may not be packaged depending on build options)
# Resolve by soname first (same search order as libmf_hybm_core's internal dlopen) to
# share a single instance, then fall back to the packaged lib; keep handles until exit.
optional_libs = ["libboundscheck.so", "libhcom.so", "libetcd_client_v3.so"]
for lib in optional_libs:
    handle = None
    try:
        handle = ctypes.CDLL(lib, mode=ctypes.RTLD_LOCAL)
    except OSError:
        try:
            handle = ctypes.CDLL(os.path.join(libs_path, lib), mode=ctypes.RTLD_LOCAL)
        except OSError:
            pass
    if handle is not None:
        _lib_handles.append(handle)


def get_include_path():
    package_dir = os.path.dirname(__file__)
    include_path = os.path.join(package_dir, 'include')
    return os.path.abspath(include_path)


def get_lib_path():
    package_dir = os.path.dirname(__file__)
    lib_path = os.path.join(package_dir, 'lib')
    return os.path.abspath(lib_path)


from _pymf_transfer import TransferEngine, TransferOpcode, TransDataOpType, create_config_store
from _pymf_hybrid import (
    bm,
    shm,
    initialize,
    uninitialize,
    set_log_level,
    set_extern_logger,
    get_last_err_msg,
    set_conf_store_tls,
    set_conf_store_tls_key,
    get_and_clear_last_err_msg,
)
from _pymf_acc_offload import offload
from mf_acc_offload import (
    empty,
    sparse_copy,
    group_pack_copy,
    kv_exchange_copy,
    entry_gather,
)

offload.empty = empty
offload.sparse_copy = sparse_copy
offload.group_pack_copy = group_pack_copy
offload.kv_exchange_copy = kv_exchange_copy
offload.entry_gather = entry_gather


__all__ = [
    'TransferEngine',
    'TransferOpcode',
    'TransDataOpType',
    'create_config_store',
    'bm',
    'shm',
    'offload',
    'initialize',
    'uninitialize',
    'set_log_level',
    'set_extern_logger',
    'get_last_err_msg',
    'set_conf_store_tls',
    'set_conf_store_tls_key',
    'get_and_clear_last_err_msg',
]
__all__ += ['get_include_path', 'get_lib_path']
