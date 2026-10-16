# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""BM (Big Memory) mode benchmark runner for write/read bandwidth tests."""

import logging
import multiprocessing as mp
import sys
import threading
import time
from queue import Empty

import torch

import memfabric_hybrid as mf
from memfabric_hybrid import bm

LOGGER = logging.getLogger(__name__)

_GVA_SIZE = 4 * 1024 * 1024 * 1024
_RANDOM_MULTIPLIER = 23
_RANDOM_INCREMENT = 17
_NEGATIVE_RATIO_DIVISOR = 3
_BARRIER_TIMEOUT_SEC = 120


def _sync_ranks(sync, rank_id, result_queue, stage):
    """Wait for all local ranks."""
    try:
        sync.wait(timeout=_BARRIER_TIMEOUT_SEC)
        return True
    except threading.BrokenBarrierError:
        result_queue.put({"rank_id": rank_id, "error": f"rank barrier broken at {stage}"})
        return False
    except Exception as e:  # noqa: BLE001 - surface any barrier failure as a rank error
        result_queue.put({"rank_id": rank_id, "error": f"rank barrier failed at {stage}: {e}"})
        return False


def _generate_data(tensor, rank_id):
    arr = tensor.flatten()
    base = rank_id
    mod = 2**15 - 1
    for i in range(len(arr)):
        base = (base * _RANDOM_MULTIPLIER + _RANDOM_INCREMENT) % mod
        if (i + rank_id) % _NEGATIVE_RATIO_DIVISOR == 0:
            arr[i] = -base
        else:
            arr[i] = base


def _check_wrong_data_count(expected, actual):
    if torch.equal(expected, actual):
        return 0
    return int((expected.flatten() != actual.flatten()).sum().item())


def _run_bm_rank(params, rank_id, device_id, result_queue, sync):
    def _fail(message):
        """Report a rank error and release the peers blocked on the barrier."""
        result_queue.put({"rank_id": rank_id, "error": message})
        try:
            sync.abort()
        except Exception:  # noqa: BLE001 - abort is best-effort cleanup
            pass

    torch.npu.set_device(device_id)
    stream = torch.npu.Stream()

    ret = mf.initialize()
    if ret != 0:
        _fail(f"mf.initialize failed, ret={ret}")
        return

    mf.set_log_level(2)

    bm_inited = False
    handle = None
    try:
        cfg = bm.BmConfig()
        cfg.auto_ranking = False
        cfg.rank_id = rank_id
        cfg.start_store = rank_id == 0
        cfg.set_nic(params["nic"])
        ret = bm.initialize(params["store_url"], params["world_size"], device_id, cfg)
        if ret != 0:
            _fail(f"bm.initialize failed, ret={ret}")
            return
        bm_inited = True

        # Contribute HBM to the pool
        local_hbm_size = params["local_hbm_size"]
        handle = bm.create2(
            id=0,
            local_dram_size=_GVA_SIZE,
            max_dram_size=_GVA_SIZE,
            local_hbm_size=local_hbm_size,
            max_hbm_size=local_hbm_size,
            data_op_type=params["bm_data_op_type"],
        )
        ret = handle.join()
        if ret != 0:
            _fail(f"handle.join failed, ret={ret}")
            return

        # All ranks must have joined before anyone starts copying
        if not _sync_ranks(sync, rank_id, result_queue, "after join"):
            return

        peer_rank_id = (rank_id + params["local_rank_size"]) % params["world_size"]
        if params["local_rank_size"] == params["world_size"]:
            peer_rank_id = (rank_id + 1) % params["world_size"]

        peer_mem_type = params["peer_mem_type"]
        copy_size = params["copy_size"]
        batch_size = params["batch_size"]
        _ALIGNMENT = 4096
        total_len = copy_size * batch_size
        _padded_elems = total_len // 4 + _ALIGNMENT // 4

        cpu_tensor = torch.zeros(total_len // 4, dtype=torch.int32)
        _generate_data(cpu_tensor, rank_id)

        local_tensor_raw = torch.zeros(_padded_elems, dtype=torch.int32, device="npu")
        _raw_ptr = local_tensor_raw.data_ptr()
        _align = (_ALIGNMENT - (_raw_ptr % _ALIGNMENT)) % _ALIGNMENT
        local_tensor = local_tensor_raw[_align // 4 : _align // 4 + total_len // 4]
        local_tensor.copy_(cpu_tensor)
        local_ptr = local_tensor.data_ptr()

        if params["protocol"] != "sdma":
            ret = handle.register(local_ptr, total_len)
            if ret != 0:
                _fail(f"handle.register failed, ret={ret}")
                return

        gva_peer = handle.peer_rank_ptr(peer_rank_id, peer_mem_type)
        if gva_peer == 0:
            _fail("peer_rank_ptr returned 0")
            return

        src_addrs = []
        dst_addrs = []
        sizes = []
        for i in range(batch_size):
            offset = i * copy_size
            if params["direction"] == "write":
                src_addrs.append(local_ptr + offset)
                dst_addrs.append(gva_peer + offset)
            else:
                src_addrs.append(gva_peer + offset)
                dst_addrs.append(local_ptr + offset)
            sizes.append(copy_size)

        copy_count = params["copy_count"]
        total_time_ns = 0
        # Wall-clock window of this rank's copy loop
        window_start_ns = time.monotonic_ns()

        for _ in range(copy_count):
            start = time.monotonic_ns()
            ret = handle.copy_data_batch(
                src_addrs=src_addrs,
                dst_addrs=dst_addrs,
                sizes=sizes,
                count=batch_size,
                type=params["bm_copy_type"],
                flags=0,
            )
            total_time_ns += time.monotonic_ns() - start
            if ret != 0:
                _fail(f"copy_data_batch failed, ret={ret}")
                return

        window_end_ns = time.monotonic_ns()

        total_time_us = total_time_ns // 1000
        total_bytes = copy_count * total_len
        bandwidth = total_bytes / total_time_us / 1000.0
        avg_time_us = total_time_us // copy_count

        wrong_bytes = 0
        try:
            local_cpu = local_tensor.cpu()

            check_npu_raw = torch.zeros(_padded_elems, dtype=torch.int32, device="npu")
            _check_raw_ptr = check_npu_raw.data_ptr()
            _check_align = (_ALIGNMENT - (_check_raw_ptr % _ALIGNMENT)) % _ALIGNMENT
            check_npu = check_npu_raw[_check_align // 4 : _check_align // 4 + total_len // 4]
            check_ptr = check_npu.data_ptr()

            if params["protocol"] != "sdma":
                ret = handle.register(check_ptr, total_len)
                if ret != 0:
                    LOGGER.warning("register check buffer failed, ret=%s, rank=%s, skip verify", ret, rank_id)
                    raise RuntimeError("register check buffer failed")

            check_src = []
            check_dst = []
            for i in range(batch_size):
                offset = i * copy_size
                check_src.append(gva_peer + offset)
                check_dst.append(check_ptr + offset)

            ret = handle.copy_data_batch(
                src_addrs=check_src,
                dst_addrs=check_dst,
                sizes=sizes,
                count=batch_size,
                type=bm.BmCopyType.G2L,
                flags=0,
            )
            if ret != 0:
                LOGGER.warning("verify copy failed, ret=%s, rank=%s", ret, rank_id)
            else:
                check_cpu = check_npu.cpu()
                wrong_count = _check_wrong_data_count(local_cpu, check_cpu)
                wrong_bytes = wrong_count * 4
        except Exception as e:
            LOGGER.warning("data verification error: %s, rank=%s", e, rank_id)

        # Keep every peer's slices alive until all ranks finished copying and
        # verifying; leave() drops this rank's slices for the whole group.
        if not _sync_ranks(sync, rank_id, result_queue, "before leave"):
            return

        result_queue.put(
            {
                "type_name": params["type_name"],
                "rank_id": rank_id,
                "device_id": device_id,
                "copy_size": copy_size,
                "copy_count": copy_count,
                "batch_size": batch_size,
                "total_time_us": total_time_us,
                "avg_time_us": avg_time_us,
                "bandwidth_gbps": bandwidth,
                "total_kb": (total_bytes // 1024),
                "wrong_bytes": wrong_bytes,
                "start_ns": window_start_ns,
                "end_ns": window_end_ns,
            }
        )

        ret = handle.leave()
        if ret != 0:
            _fail(f"handle.leave failed, ret={ret}")
            return

        handle.destroy()

    except Exception as e:
        _fail(str(e))
    finally:
        if bm_inited:
            bm.uninitialize(0)
        try:
            stream.synchronize()
        except Exception:
            pass
        torch.npu.synchronize()
        mf.uninitialize()


_PROTOCOL_TO_DATA_OP = {
    "sdma": bm.BmDataOpType.SDMA,
    "device_rdma": bm.BmDataOpType.DEVICE_RDMA,
    "device_urma": bm.BmDataOpType.DEVICE_URMA,
    "device_uboe": bm.BmDataOpType.DEVICE_UBOE,
    "host_rdma": bm.BmDataOpType.HOST_RDMA,
    "host_urma": bm.BmDataOpType.HOST_URMA,
    "host_tcp": bm.BmDataOpType.HOST_TCP,
}


def _get_bm_params(args):
    store_url = args.ip
    protocol = args.protocol
    bm_data_op_type = _PROTOCOL_TO_DATA_OP[protocol]

    local_mem_type = bm.BmMemType.DEVICE
    # The peer side defaults to its DRAM slice, which on a single node is just
    # another range of the same physical host memory.
    peer_mem_type = bm.BmMemType.DEVICE if args.peer_mem_type == "device" else bm.BmMemType.HOST

    if args.command == "write":
        bm_copy_type = bm.BmCopyType.L2G
        direction = "write"
        type_name = "D2RD" if args.peer_mem_type == "device" else "D2RH"
    else:
        bm_copy_type = bm.BmCopyType.G2L
        direction = "read"
        type_name = "RD2D" if args.peer_mem_type == "device" else "RH2D"

    return {
        "store_url": store_url,
        "nic": args.nic,
        "protocol": protocol,
        "bm_data_op_type": bm_data_op_type,
        "local_mem_type": local_mem_type,
        "peer_mem_type": peer_mem_type,
        "bm_copy_type": bm_copy_type,
        "direction": direction,
        "type_name": type_name,
        "world_size": args.world_size,
        "local_rank_size": args.local_rank_size,
        "copy_size": args.copysize,
        "copy_count": args.copy_count,
        "batch_size": args.batchsize,
        "local_hbm_size": args.local_hbm_size,
        "peer_mem_name": args.peer_mem_type,
        "paired": args.paired,
        "sync_host": getattr(args, "sync_host", None),
        "sync_port": args.sync_port,
        "sync_timeout": args.sync_timeout,
        "warmup_count": args.warmup_count,
    }


def run_bm(args):
    params = _get_bm_params(args)
    local_rank_size = params["local_rank_size"]

    # A peer-HBM copy needs a non-empty peer HBM slot.
    if args.peer_mem_type == "device" and args.local_hbm_size <= 0:
        LOGGER.error(
            "benchmark: peer_mem_type=device requires local_hbm_size > 0, "
            "got local_hbm_size=%s, peer_mem_type=%s, rank_start=%s",
            args.local_hbm_size,
            args.peer_mem_type,
            args.rank_start,
        )
        return 1

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    sync = ctx.Barrier(local_rank_size)
    processes = _spawn_ranks(ctx, params, args, result_queue, sync)
    reporter = _writer_reporter(args) if args.paired and args.command == "write" else None
    results = _collect_results(processes, result_queue, sync, args.rank_start, reporter)
    errors = [r for r in results if "error" in r]
    if errors:
        for err in errors:
            LOGGER.error("rank %s: %s", err.get("rank_id", "?"), err["error"])
        return 1
    expected = set(range(args.rank_start, args.rank_start + local_rank_size))
    if len(results) != local_rank_size or {r["rank_id"] for r in results} != expected:
        LOGGER.error("missing or duplicate rank results, expected=%s, received=%s", expected, results)
        return 1
    if args.paired and args.command == "write":
        print("Remote read verification OK; source pool released.", flush=True)
        return 0
    if args.paired:
        print("Paired read: source writes, warmup, synchronization and verification are not timed.")
    _print_bm_results(results, args)
    return 0


def _spawn_ranks(ctx, params, args, result_queue, sync):
    target = _run_bm_rank
    if args.paired:
        from .paired_benchmark import run_paired_rank

        target = run_paired_rank
    processes = []
    for local_idx in range(args.local_rank_size):
        process = ctx.Process(
            target=target,
            args=(params, args.rank_start + local_idx, args.device + local_idx, result_queue, sync),
        )
        process.start()
        processes.append(process)
    return processes


def _writer_reporter(args):
    measurements = {}
    expected = set(range(args.rank_start, args.rank_start + args.local_rank_size))
    printed = False

    def report(result):
        nonlocal printed
        measurements[result["rank_id"]] = result
        if not printed and set(measurements) == expected:
            _print_bm_results(list(measurements.values()), args)
            print("Write benchmark complete; keeping source pool alive while waiting for remote read.", flush=True)
            printed = True

    return report


def _record_result(result, results, on_measurement):
    if result.get("event") == "measurement":
        if on_measurement is not None:
            on_measurement(result["result"])
    else:
        results.append(result)


def _collect_results(processes, result_queue, sync, rank_start, on_measurement=None):
    results = []
    while any(process.exitcode is None for process in processes):
        try:
            _record_result(result_queue.get(timeout=0.2), results, on_measurement)
        except Empty:
            pass
        if any(process.exitcode not in (None, 0) for process in processes):
            sync.abort()
    for index, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            results.append({"rank_id": rank_start + index, "error": f"process exited with code {process.exitcode}"})
    while True:
        try:
            _record_result(result_queue.get_nowait(), results, on_measurement)
        except Empty:
            break
    return results


def _print_bm_results(results, args):
    copy_size = args.copysize
    copy_count = args.copy_count
    batch_size = args.batchsize
    sep = "-" * 57

    print()
    print(f"  protocol:{args.protocol}  copy_size:{copy_size}  copy_count:{copy_count}  batch_size:{batch_size}")
    print(sep + " Band Width Test " + sep)
    print(
        f"{'Type':<8}{'NPU':<8}{'Rank':<8}{'TotalSize(KB)':<20}"
        f"{'TotalTime(us)':<20}{'AvgTime(us)':<20}{'BW(GB/s)':<20}{'WrongBytes(B)':<16}"
    )

    for r in sorted(results, key=lambda x: x.get("device_id", 0)):
        print(
            f"{r.get('type_name', '?'):<8}{r['device_id']:<8}{r['rank_id']:<8}"
            f"{r['total_kb']:<20}{r['total_time_us']:<20}{r['avg_time_us']:<20}"
            f"{r['bandwidth_gbps']:<20.3f}{r.get('wrong_bytes', 0):<16}"
        )

    if results:
        total_kb_sum = sum(r["total_kb"] for r in results)
        starts = [r["start_ns"] for r in results if r.get("start_ns")]
        ends = [r["end_ns"] for r in results if r.get("end_ns")]
        summed_us = sum(r["total_time_us"] for r in results)
        if starts and ends:
            # Aggregate over the real wall-clock span, not over the sum of the
            # per-rank bandwidths: the sum is only valid when the ranks actually
            # overlap. Interleaved ranks would otherwise report the link twice.
            wall_us = max(1, (max(ends) - min(starts)) // 1000)
            total_bw = total_kb_sum * 1024 / wall_us / 1000.0
            overlapped = wall_us < summed_us * 3 // 4
        else:
            wall_us = 0
            total_bw = sum(r["bandwidth_gbps"] for r in results)
            overlapped = True
        print(sep)
        print(f"{'Total':<8}{'':<8}{'':<8}{total_kb_sum:<20}{wall_us:<20}{'':<20}{total_bw:<20.3f}")
        if wall_us and not overlapped:
            print(
                f"  note: ranks did not overlap (wall {wall_us}us vs summed {summed_us}us); "
                f"each rank saw the whole link, so per-rank BW is a single-rank rate"
            )

    print(sep + "-" * 57)
    sys.stdout.flush()
