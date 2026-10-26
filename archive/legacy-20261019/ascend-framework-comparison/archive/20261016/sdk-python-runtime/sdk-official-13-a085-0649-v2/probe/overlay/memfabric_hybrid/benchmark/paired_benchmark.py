# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Measure producer writes before waiting for a consumer to measure remote reads."""

import logging
import time

import torch

import memfabric_hybrid as mf
from memfabric_hybrid import bm

from .bm_band_width_benchmark import _GVA_SIZE, _check_wrong_data_count, _generate_data
from .benchmark_sync import BenchmarkSync

LOGGER = logging.getLogger(__name__)
_ALIGNMENT = 4096


def _require_ok(ret, operation):
    if ret != 0:
        raise RuntimeError(f"{operation} failed, ret={ret}, detail={mf.get_last_err_msg()}")


def _initialize_bm(params, rank_id, device_id):
    cfg = bm.BmConfig()
    cfg.auto_ranking = False
    cfg.rank_id = rank_id
    cfg.start_store = rank_id == 0
    cfg.set_nic(params["nic"])
    _require_ok(bm.initialize(params["store_url"], params["world_size"], device_id, cfg), "bm.initialize")


def _create_pool(params):
    handle = bm.create2(
        id=0,
        local_dram_size=_GVA_SIZE,
        max_dram_size=_GVA_SIZE,
        local_hbm_size=params["local_hbm_size"],
        max_hbm_size=params["local_hbm_size"],
        data_op_type=params["bm_data_op_type"],
    )
    return handle


def _create_buffer(handle, params, rank_id):
    total_len = params["copy_size"] * params["batch_size"]
    source_rank = rank_id if params["direction"] == "write" else rank_id - params["local_rank_size"]
    expected = torch.zeros(total_len // 4, dtype=torch.int32)
    _generate_data(expected, source_rank)
    raw = torch.zeros((total_len + _ALIGNMENT) // 4, dtype=torch.int32, device="npu")
    offset = (-raw.data_ptr() % _ALIGNMENT) // 4
    tensor = raw[offset : offset + total_len // 4]
    if params["direction"] == "write":
        tensor.copy_(expected)
    torch.npu.synchronize()
    if params["protocol"] != "sdma":
        _require_ok(handle.register(tensor.data_ptr(), total_len), "handle.register")
    return raw, tensor, expected


def _copy_arguments(params, source, destination, copy_type=bm.BmCopyType.G2L):
    offsets = [index * params["copy_size"] for index in range(params["batch_size"])]
    return {
        "src_addrs": [source + offset for offset in offsets],
        "dst_addrs": [destination + offset for offset in offsets],
        "sizes": [params["copy_size"]] * params["batch_size"],
        "count": params["batch_size"],
        "type": copy_type,
        "flags": 0,
    }


def _measure_copies(handle, copy_args, params, rank_id, device_id):
    elapsed_ns = 0
    start_ns = time.monotonic_ns()
    for _ in range(params["copy_count"]):
        start = time.monotonic_ns()
        ret = handle.copy_data_batch(**copy_args)
        elapsed_ns += time.monotonic_ns() - start
        _require_ok(ret, "copy_data_batch")
    end_ns = time.monotonic_ns()
    elapsed_ns = max(1, elapsed_ns)
    total_bytes = params["copy_size"] * params["batch_size"] * params["copy_count"]
    return {
        "rank_id": rank_id,
        "device_id": device_id,
        "peer_rank_id": rank_id if params["direction"] == "write" else rank_id - params["local_rank_size"],
        "type_name": params["type_name"],
        "total_time_us": elapsed_ns / 1000,
        "avg_time_us": elapsed_ns / params["copy_count"] / 1000,
        "bandwidth_gbps": total_bytes / elapsed_ns,
        "total_kb": total_bytes // 1024,
        "wrong_bytes": 0,
        "start_ns": start_ns,
        "end_ns": end_ns,
    }


def _measure_writer(handle, params, rank_id, device_id, coordinator, buffers):
    _, tensor, expected = buffers
    gva = handle.peer_rank_ptr(rank_id, params["peer_mem_type"])
    if not gva:
        raise RuntimeError(f"source pool address is unavailable, rank={rank_id}")
    copy_args = _copy_arguments(params, tensor.data_ptr(), gva, bm.BmCopyType.L2G)
    for _ in range(params["warmup_count"]):
        _require_ok(handle.copy_data_batch(**copy_args), "warmup write")
    # Clear the destination after warmup so a no-op measured write cannot pass.
    total_len = params["copy_size"] * params["batch_size"]
    tensor.zero_()
    torch.npu.synchronize()
    _require_ok(handle.copy_data(tensor.data_ptr(), gva, total_len, bm.BmCopyType.L2G, 0), "clear source pool")
    tensor.copy_(expected)
    torch.npu.synchronize()
    coordinator.barrier("write warmup complete", range(params["local_rank_size"]))
    result = _measure_copies(handle, copy_args, params, rank_id, device_id)
    tensor.zero_()
    torch.npu.synchronize()
    _require_ok(handle.copy_data(gva, tensor.data_ptr(), total_len, bm.BmCopyType.G2L, 0), "verify write")
    _verify_data(expected, tensor, rank_id)
    return result


def _verify_data(expected, tensor, rank_id):
    wrong_bytes = _check_wrong_data_count(expected, tensor.cpu()) * 4
    if wrong_bytes:
        raise RuntimeError(f"data verification failed, rank={rank_id}, wrong_bytes={wrong_bytes}")


def _warmup_reader(handle, params, rank_id, tensor):
    source_rank = rank_id - params["local_rank_size"]
    gva = handle.peer_rank_ptr(source_rank, params["peer_mem_type"])
    if not gva:
        raise RuntimeError(f"peer pool address is unavailable, peer_rank={source_rank}")
    copy_args = _copy_arguments(params, gva, tensor.data_ptr())
    for _ in range(params["warmup_count"]):
        _require_ok(handle.copy_data_batch(**copy_args), "warmup read")
    # A no-op measured read must not pass verification using a warmup payload.
    tensor.zero_()
    torch.npu.synchronize()
    return copy_args


def _execute_reader(handle, params, rank_id, device_id, coordinator, buffers):
    _, tensor, expected = buffers
    copy_args = _warmup_reader(handle, params, rank_id, tensor)
    coordinator.barrier("warmup complete")
    result = _measure_copies(handle, copy_args, params, rank_id, device_id)
    coordinator.barrier("copies complete")
    _verify_data(expected, tensor, rank_id)
    return result


def _execute_session(handle, params, rank_id, device_id, coordinator, buffers, result_queue):
    writer = params["direction"] == "write"
    if writer:
        result = _measure_writer(handle, params, rank_id, device_id, coordinator, buffers)
        result_queue.put({"event": "measurement", "result": result})
        # Publish configuration only after writing. Readers validate all writer
        # configurations before joining BM, keeping link setup out of write timing.
        coordinator.check_config(params)
    coordinator.barrier("joined")
    if writer:
        coordinator.barrier("warmup complete")
        coordinator.barrier("copies complete")
    else:
        result = _execute_reader(handle, params, rank_id, device_id, coordinator, buffers)
    return result


def _cleanup(handle, bm_inited, mf_inited, rank_id):
    operations = []
    if handle is not None:
        operations.append(handle.destroy)
    if bm_inited:
        operations.append(lambda: bm.uninitialize(0))
    if mf_inited:
        operations.append(mf.uninitialize)
    for operation in operations:
        try:
            operation()
        except Exception as error:  # noqa: BLE001 - preserve the original rank error
            LOGGER.error("cleanup failed, rank=%s, error=%s", rank_id, error)


def run_paired_rank(params, rank_id, device_id, result_queue, sync):
    coordinator = None
    handle = None
    bm_inited = mf_inited = False
    buffers = None
    try:
        torch.npu.set_device(device_id)
        coordinator = BenchmarkSync(params, rank_id)
        if params["direction"] == "read":
            coordinator.check_config(params)
        _require_ok(mf.initialize(), "mf.initialize")
        mf_inited = True
        mf.set_log_level(2)
        _initialize_bm(params, rank_id, device_id)
        bm_inited = True
        if bm.bm_rank_id() != rank_id:
            raise RuntimeError(f"actual BM rank does not match requested rank={rank_id}")
        handle = _create_pool(params)
        _require_ok(handle.join(), "handle.join")
        buffers = _create_buffer(handle, params, rank_id)
        result = _execute_session(handle, params, rank_id, device_id, coordinator, buffers, result_queue)
        coordinator.barrier("verified")
        _require_ok(handle.leave(), "handle.leave")
        handle.destroy()
        handle = None
        bm.uninitialize(0)
        bm_inited = False
        mf.uninitialize()
        mf_inited = False
        coordinator.barrier("cleaned up")
        result_queue.put(result)
    except Exception as error:  # noqa: BLE001 - propagate runtime and device failures to the parent
        LOGGER.error("paired benchmark failed, rank=%s, device=%s, error=%s", rank_id, device_id, error)
        result_queue.put({"rank_id": rank_id, "error": str(error)})
        sync.abort()
        if coordinator is not None:
            coordinator.fail(str(error))
    finally:
        _cleanup(handle, bm_inited, mf_inited, rank_id)
        del buffers
