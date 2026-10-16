#!/usr/bin/env python3
"""Independent DDR provider; keep it alive while model clients use the pool."""
import argparse
import ctypes
import json
import os
import signal
import threading
from pathlib import Path

from memcache_hybrid import DistributedObjectStore, LocalConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--meta', required=True)
    parser.add_argument('--config-store', required=True)
    parser.add_argument('--hcom-url', required=True)
    parser.add_argument('--world-size', type=int, default=128)
    parser.add_argument('--dram', default='16GB')
    parser.add_argument('--max-dram', default='64GB')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--numa-node', type=int)
    parser.add_argument('--ready-file', required=True)
    args = parser.parse_args()
    if args.numa_node is not None:
        # Process-local policy only: distribute provider allocations across host
        # NUMA nodes instead of relying on the default automatic allocator.
        cpus = set()
        raw = Path(f'/sys/devices/system/node/node{args.numa_node}/cpulist').read_text().strip()
        for part in raw.split(','):
            ends = list(map(int, part.split('-')))
            cpus.update(range(ends[0], ends[-1] + 1))
        cpus &= os.sched_getaffinity(0)
        if not cpus:
            raise RuntimeError(('No CPUs available for selected NUMA node', args.numa_node))
        os.sched_setaffinity(0, cpus)
        libnuma = ctypes.CDLL('libnuma.so.1', use_errno=True)
        libnuma.numa_available.restype = ctypes.c_int
        if libnuma.numa_available() < 0:
            raise RuntimeError('NUMA policy unavailable')
        libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
        libnuma.numa_set_preferred.restype = None
        libnuma.numa_set_preferred(args.numa_node)
        mode = ctypes.c_int()
        mask = ctypes.c_ulong()
        libnuma.get_mempolicy.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_ulong),
                                        ctypes.c_ulong, ctypes.c_void_p, ctypes.c_ulong]
        if libnuma.get_mempolicy(ctypes.byref(mode), ctypes.byref(mask), 64, None, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot verify process NUMA memory policy')
        if mode.value != 1 or not mask.value & (1 << args.numa_node):
            raise RuntimeError(('NUMA preferred policy not applied', mode.value, mask.value))
        print('PROVIDER_NUMA_POLICY', args.numa_node, sorted(cpus), flush=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    cfg = LocalConfig()
    cfg.meta_service_url = args.meta
    cfg.config_store_url = args.config_store
    cfg.hcom_url = args.hcom_url
    cfg.protocol = 'device_sdma'
    cfg.dram_size = args.dram
    cfg.max_dram_size = args.max_dram
    cfg.hbm_size = '0'
    cfg.max_hbm_size = '0'
    cfg.world_size = args.world_size
    cfg.log_level = 'info'
    ready = Path(args.ready_file)
    if ready.exists():
        raise RuntimeError('Remove the old readiness file after checking its provider PID')
    pidfile = ready.with_suffix('.pid')
    pidfile.write_text(str(os.getpid()))
    store = DistributedObjectStore()
    try:
        ret = store.setup(cfg)
        if ret != 0:
            raise RuntimeError(('setup', ret))
        ret = store.init(args.device, True)
        if ret != 0:
            raise RuntimeError(('init', ret))
        # MemFabric init installs native signal handlers; restore our graceful
        # shutdown handlers after it returns, before advertising readiness.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        ready.write_text(json.dumps({'ready': True, 'pid': os.getpid(), 'dram': args.dram,
                                    'world_size': args.world_size, 'device': args.device, 'numa_node': args.numa_node, 'max_dram': args.max_dram, 'max_dram_bytes': int(args.max_dram), 'dram_bytes': int(args.dram)}))
        print('POOL_READY', flush=True)
        stop.wait()
    finally:
        ret = store.close()
        ready.unlink(missing_ok=True)
        pidfile.unlink(missing_ok=True)
        print('POOL_CLOSE_RESULT', ret, flush=True)
        if ret not in (None, 0):
            raise RuntimeError(('close', ret))


if __name__ == '__main__':
    main()
