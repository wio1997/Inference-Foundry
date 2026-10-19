#!/usr/bin/env python3
"""Single-node, model-free 2-rank HCCL bounded probe (initial-condition communication diagnosis).

Run on the Ascend host with two visible devices, e.g. ASCEND_RT_VISIBLE_DEVICES=0,1 python3 hccl_probe.py
Bounded on purpose: int8 dist.send/dist.recv in both directions, then bf16 all_reduce(SUM), over the fixed
SIZES below; no other operator, model, or config sweep. Only per-rank raw ns arrays are produced; arrays
are never stitched across ranks, and one rank's send/recv interval is not a pure one-way link latency nor a
PD latency. Real KV-cache / EP message shapes, isolation, and contention are not represented here.
"""
import json
import multiprocessing as mp
import os
import queue as queue_mod
import socket
import statistics
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.multiprocessing as torch_mp

SIZES = [1024, 65536, 1048576, 16777216, 67108864]  # payload bytes
WARMUP, ITERS = 3, 12
WORLD_SIZE = 2
INIT_TIMEOUT_S = 60
SENTINEL = -100  # int8, never equal to a sender fill value 1+it
COLLECT_DEADLINE_S, JOIN_TIMEOUT_S = 1800, 120


def _free_port():
    """Ask the OS for an unused 127.0.0.1 TCP port to use as the HCCL init_method."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _median(values): return statistics.median(values)


def _device_name():
    try:
        return str(torch.npu.get_device_name())
    except Exception:
        return None


def _timed_send_recv(rank, sender_rank, cases):
    """One direction per call; every rank calls with identical arguments."""
    import torch.distributed as dist

    receiver_rank = (sender_rank + 1) % WORLD_SIZE
    is_sender = rank == sender_rank
    for size in SIZES:
        buf = torch.empty(size, dtype=torch.int8, device=f'npu:{rank}')
        ns, checks = [], []
        for it in range(WARMUP + ITERS):
            expected = 1 + it
            buf.fill_(expected if is_sender else SENTINEL)
            torch.npu.synchronize()  # fill done before barrier and timing
            dist.barrier()
            torch.npu.synchronize()  # barrier completion is outside the timed interval
            start = time.perf_counter_ns()
            if is_sender:
                dist.send(buf, dst=receiver_rank)
            else:
                dist.recv(buf, src=sender_rank)
            torch.npu.synchronize()
            stop = time.perf_counter_ns()
            ok = None if is_sender else bool(torch.all(buf == expected).item())
            if ok is False:
                raise RuntimeError('send/recv payload verification failed')
            dist.barrier()  # keep verification and the next fill out of the timing window
            if it >= WARMUP:
                ns.append(stop - start)
                checks.append(ok)
        cases.append({
            'op': 'send_recv', 'direction': f'{sender_rank}->{receiver_rank}',
            'role': 'send' if is_sender else 'recv', 'dtype': 'int8', 'payload_bytes': size,
            'timing': 'perf_counter_ns(dist.send|recv + torch.npu.synchronize)',
            'ns': ns, 'min_ns': min(ns), 'median_ns': _median(ns),
            'correctness': None if is_sender else {
                'checks': len(checks), 'passed': sum(1 for c in checks if c), 'all_ok': all(checks)},
        })


def _timed_all_reduce(rank, cases):
    import torch.distributed as dist

    expected = float(sum(r + 1 for r in range(WORLD_SIZE)))
    for size in SIZES:
        buf = torch.empty(size // 2, dtype=torch.bfloat16, device=f'npu:{rank}')
        ns, checks = [], []
        for it in range(WARMUP + ITERS):
            buf.fill_(float(rank + 1))
            torch.npu.synchronize()
            dist.barrier()
            torch.npu.synchronize()  # barrier completion is outside the timed interval
            start = time.perf_counter_ns()
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            torch.npu.synchronize()
            stop = time.perf_counter_ns()
            ok = bool(torch.all(buf == expected).item())
            if not ok:
                raise RuntimeError('all_reduce verification failed')
            dist.barrier()
            if it >= WARMUP:
                ns.append(stop - start)
                checks.append(ok)
        cases.append({
            'op': 'all_reduce', 'direction': 'all', 'role': 'rank', 'dtype': 'bf16', 'payload_bytes': size,
            'timing': 'perf_counter_ns(dist.all_reduce + torch.npu.synchronize)',
            'ns': ns, 'min_ns': min(ns), 'median_ns': _median(ns),
            'correctness': {'checks': len(checks), 'passed': sum(1 for c in checks if c), 'all_ok': all(checks)},
        })


def _worker(rank, world_size, init_method, result_queue):
    result = {'rank': rank, 'logical_device': f'npu:{rank}', 'device_name': None, 'cases': [], 'error': None,
              'torch_version': None, 'torch_npu_version': None, 'init_ns': None,
              'ascend_rt_visible_devices': os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}
    dist = None
    try:
        import torch_npu  # noqa: F401  (registers the npu backend)
        import torch.distributed as dist

        result['torch_version'] = torch.__version__
        result['torch_npu_version'] = getattr(torch_npu, '__version__', None)
        torch.set_num_threads(1)
        torch.npu.set_device(rank)
        result['device_name'] = _device_name()
        start = time.perf_counter_ns()
        dist.init_process_group(backend='hccl', init_method=init_method, rank=rank,
                                world_size=world_size, timeout=timedelta(seconds=INIT_TIMEOUT_S))
        result['init_ns'] = time.perf_counter_ns() - start
        for sender_rank in range(WORLD_SIZE):
            _timed_send_recv(rank, sender_rank, result['cases'])
        _timed_all_reduce(rank, result['cases'])
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        try:
            if dist is not None and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            result['destroy_error'] = traceback.format_exc()
    result_queue.put(result)


def _collect(result_queue, procs, deadline_s):
    """Drain results while children run, so no child blocks on a full queue during join."""
    results, deadline = [], time.monotonic() + deadline_s
    while len(results) < len(procs) and time.monotonic() < deadline:
        try:
            results.append(result_queue.get(timeout=5.0))
        except queue_mod.Empty:
            if not any(proc.is_alive() for proc in procs):
                break
    return results


def main():
    init_method = f'tcp://127.0.0.1:{_free_port()}'
    result_queue = mp.get_context('spawn').Queue()
    procs, spawn_error = [], None
    controls = {'HCCL_INTER_HCCS_DISABLE': 'FALSE', 'HCCL_INTRA_ROCE_ENABLE': '0',
                'HCCL_OP_EXPANSION_MODE': 'AI_CPU'}
    for key, value in controls.items():
        if os.environ.get(key) != value:
            raise RuntimeError('SDMA policy requires %s=%s' % (key, value))
    try:
        context = torch_mp.spawn(_worker, args=(WORLD_SIZE, init_method, result_queue),
                                 nprocs=WORLD_SIZE, join=False)
        procs = context.processes
    except Exception:
        spawn_error = traceback.format_exc()
    results = _collect(result_queue, procs, COLLECT_DEADLINE_S)
    for proc in procs:
        proc.join(timeout=JOIN_TIMEOUT_S)
        if proc.is_alive():
            proc.terminate()
    exitcodes = [proc.exitcode for proc in procs]
    ok = (spawn_error is None and len(results) == WORLD_SIZE
          and all(result['error'] is None for result in results)
          and all(code == 0 for code in exitcodes))
    report = {
        'probe': 'hccl_2rank_bounded_probe', 'kind': 'initial_condition_communication_diagnosis',
        'ok': ok, 'spawn_error': spawn_error, 'init_method': init_method, 'world_size': WORLD_SIZE,
        'sizes_bytes': SIZES, 'warmup_iterations': WARMUP, 'timed_iterations': ITERS,
        'transport_controls': controls, 'sdma_runtime_path_verified': False,
        'physical_visible_string': os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        'visibility_note': 'set by the caller; rank r runs on npu:r inside that visible set',
        'rank_mapping': {str(r): f'npu:{r}' for r in range(WORLD_SIZE)},
        'exitcodes': exitcodes, 'rank_results': sorted(results, key=lambda result: result['rank']),
        'timing_note': 'each ns value is that rank\'s own perf_counter_ns interval around dist.send/recv '
                       'or dist.all_reduce plus torch.npu.synchronize; no cross-process time is combined',
        'semantics': [
            'verification runs after timing: receiver torch.all for send/recv, every rank for all_reduce',
            'a single rank\'s send/recv interval is neither a pure one-way link latency nor a PD latency',
            'initial-condition diagnosis only; real KV-cache/EP message shapes, isolation, and contention '
            'are not represented and must be recorded outside this run',
        ],
    }
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
