#!/usr/bin/env python3
"""Independent DDR provider; keep it alive while model clients use the pool."""
import argparse
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
    parser.add_argument('--world-size', type=int, default=128)
    parser.add_argument('--dram', default='16GB')
    parser.add_argument('--max-dram', default='64GB')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--ready-file', required=True)
    args = parser.parse_args()
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    cfg = LocalConfig()
    cfg.meta_service_url = args.meta
    cfg.config_store_url = args.config_store
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
    store = DistributedObjectStore()
    try:
        ret = store.setup(cfg)
        if ret != 0:
            raise RuntimeError(('setup', ret))
        ret = store.init(args.device, True)
        if ret != 0:
            raise RuntimeError(('init', ret))
        ready.write_text(json.dumps({'ready': True, 'pid': os.getpid(), 'dram': args.dram,
                                    'world_size': args.world_size, 'device': args.device}))
        print('POOL_READY', flush=True)
        stop.wait()
    finally:
        ret = store.close()
        ready.unlink(missing_ok=True)
        print('POOL_CLOSE_RESULT', ret, flush=True)
        if ret not in (None, 0):
            raise RuntimeError(('close', ret))


if __name__ == '__main__':
    main()
