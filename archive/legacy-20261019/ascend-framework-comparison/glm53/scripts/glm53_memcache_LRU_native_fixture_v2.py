"""Isolated native SDK LRU counterfactual inside owned full-device container; no model or production pool calls."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


BASE = Path('/run/glm53/diagnostics/20261017T2020-memcache-LRU-native-v2')


def case(mode):
    assert mode in ('control', 'exist_touch')
    assert all((Path('/dev') / ('davinci' + str(index))).exists() for index in range(16))
    assert importlib.metadata.version('memcache_hybrid') == '1.3.0'
    root = BASE / mode
    root.mkdir(exist_ok=False)
    meta_config = root / 'meta.conf'
    meta_config.write_text('''ock.mmc.meta_service_url = tcp://127.0.0.1:29880
ock.mmc.meta_service.config_store_url = tcp://127.0.0.1:29881
ock.mmc.meta_service.metrics_url = http://127.0.0.1:29882
ock.mmc.log_level = info
ock.mmc.meta_service.lru_shard_count = 1
ock.mmc.evict_threshold_high = 85
ock.mmc.evict_threshold_low = 50
ock.mmc.evict_check_interval_seconds = 0
''')
    environment = os.environ.copy()
    environment['MMC_META_CONFIG_PATH'] = str(meta_config)
    log = (root / 'meta-console.txt').open('wb')
    meta = subprocess.Popen([sys.executable, '-c', 'import memcache_hybrid as mc,sys;sys.exit(mc.MetaService.main())'], env=environment, stdout=log, stderr=subprocess.STDOUT)
    store = None
    result = {'mode': mode, 'model_requests': 0, 'visible_NPU_devices': 16, 'native_context_required': True, 'protocol': 'host_shm', 'production_pool_contacted': False, 'put_rcs': [], 'touch_results': []}
    try:
        deadline = time.monotonic() + 25
        while True:
            assert meta.poll() is None, 'Isolated Meta exited before readiness'
            try:
                with socket.create_connection(('127.0.0.1', 29880), timeout=.3):
                    break
            except OSError:
                assert time.monotonic() < deadline, 'Isolated Meta readiness deadline'
                time.sleep(.2)
        import memcache_hybrid as mc
        assert mc.__file__.startswith('/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/')
        cfg = mc.LocalConfig()
        for field, value in {'meta_service_url': 'tcp://127.0.0.1:29880', 'config_store_url': 'tcp://127.0.0.1:29881', 'log_level': 'info', 'world_size': 2, 'protocol': 'host_shm', 'dram_size': '64MB', 'max_dram_size': '64MB', 'hbm_size': '0GB', 'max_hbm_size': '0GB'}.items():
            assert hasattr(cfg, field), field
            setattr(cfg, field, value)
        store = mc.DistributedObjectStore()
        result['setup_rc'] = store.setup(cfg)
        assert result['setup_rc'] == 0
        result['init_rc'] = store.init(0, init_bm=True)
        assert result['init_rc'] == 0
        pattern = b'L05-native-CPU-only-'
        payload = pattern * (8 * 1024**2 // len(pattern))
        assert 7 * 1024**2 < len(payload) <= 8 * 1024**2
        result['payload_bytes'] = len(payload)
        result['payload_sha256'] = hashlib.sha256(payload).hexdigest()
        for index in range(10):
            if mode == 'exist_touch' and index:
                touched = store.batch_is_exist(['glm53-L05-isolated-0'])
                result['touch_results'].append(touched)
                assert touched == [1], touched
            rc = store.put('glm53-L05-isolated-' + str(index), payload)
            result['put_rcs'].append(rc)
            assert rc == 0, (index, rc)
        result['oldest_key_exists_after_pressure'] = store.batch_is_exist(['glm53-L05-isolated-0'])
        if mode == 'exist_touch':
            restored = store.get('glm53-L05-isolated-0')
            result['retained_payload_bytes'] = len(restored)
            result['retained_payload_sha256'] = hashlib.sha256(restored).hexdigest()
            assert restored == payload
        expected = [0] if mode == 'control' else [1]
        assert result['oldest_key_exists_after_pressure'] == expected, result
        result['case_passed'] = True
    except BaseException as error:
        result['case_passed'] = False
        result['error'] = repr(error)
        raise
    finally:
        if store is not None:
            try:
                result['close_rc'] = store.close()
            except Exception as error:
                result['close_error'] = repr(error)
        if meta.poll() is None:
            meta.terminate()
            try:
                meta.wait(timeout=15)
            except subprocess.TimeoutExpired:
                meta.kill()
                meta.wait(timeout=5)
                result['isolated_meta_required_kill'] = True
        result['meta_exit_code'] = meta.returncode
        log.close()
        raw = (root / 'meta-console.txt').read_bytes()
        result['meta_log_bytes'] = len(raw)
        result['meta_log_sha256'] = hashlib.sha256(raw).hexdigest()
        (root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print('GLM53_LRU_CPU_CASE=' + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=['control', 'exist_touch'])
    args = parser.parse_args()
    if args.case:
        case(args.case)
        return
    rows = []
    for mode in ('control', 'exist_touch'):
        process = subprocess.run([sys.executable, __file__, '--case', mode], capture_output=True, text=True, timeout=100)
        rows.append({'mode': mode, 'rc': process.returncode, 'stdout_tail': process.stdout[-16000:], 'stderr_tail': process.stderr[-8000:]})
        if process.returncode != 0:
            break
    passed = len(rows) == 2 and all(row['rc'] == 0 for row in rows)
    (BASE / 'result.json').write_text(json.dumps({'cases': rows, 'passed': passed, 'model_requests': 0, 'visible_NPU_devices': 16, 'native_context_required': True}, indent=2) + '\n')
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
