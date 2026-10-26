"""Qualify documented SDK byte APIs on the existing SDMA pool; no model calls."""
import hashlib, importlib.metadata, json, time

def main():
    import memcache_hybrid as sdk
    assert importlib.metadata.version('memcache_hybrid') == '1.3.0'
    assert sdk.__file__.startswith('/usr/local/python3.12.13/lib/python3.12/site-packages/memcache_hybrid/')
    config = sdk.LocalConfig()
    config.meta_service_url = 'tcp://172.16.12.34:29780'
    config.config_store_url = 'tcp://172.16.12.34:29781'
    config.protocol = 'device_sdma'
    config.world_size = 128
    config.dram_size = '0GB'
    config.max_dram_size = '160GB'
    config.hbm_size = config.max_hbm_size = '0GB'
    config.log_level = 'info'
    store = sdk.DistributedObjectStore()
    result = {'project_id': 'glm53', 'model_requests': 0, 'init_bm': True,
              'dram_size': '0GB', 'protocol': 'device_sdma', 'cases': [], 'pass': False}
    result['setup_rc'] = store.setup(config)
    assert result['setup_rc'] == 0
    keys = []
    try:
        result['init_rc'] = store.init(0, init_bm=True)
        assert result['init_rc'] == 0
        # Match the actual native SG K/V/indexer and vLLM page object sizes.
        for index, size in enumerate([10354688, 1294336, 2555904, 12369920]):
            key = 'glm53-sdk-readback-20261017T1722-' + str(index)
            assert store.is_exist(key) == 0, 'Unique fixture namespace already exists'
            keys.append(key)
            value = (bytes(range(256)) * ((size + 255) // 256))[:size]
            expected = hashlib.sha256(value).hexdigest()
            row = {'bytes': size, 'expected_sha256': expected, 'reads': []}
            result['cases'].append(row)
            row['put_rc'] = store.put(key, value)
            assert row['put_rc'] == 0
            del value
            assert store.batch_is_exist([key]) == [1]
            info = store.batch_get_key_info([key])[0]
            assert info is not None and info.size() == size and any(info.gva_list())
            for _ in range(2):
                begin = time.monotonic_ns()
                value = store.get(key)
                row['reads'].append({'bytes': len(value), 'type': type(value).__name__,
                                     'sha256': hashlib.sha256(value).hexdigest(),
                                     'elapsed_ns': time.monotonic_ns() - begin})
                assert isinstance(value, bytes) and len(value) == size
                assert row['reads'][-1]['sha256'] == expected
                del value
            row['remove_rc'] = store.remove(key)
            assert row['remove_rc'] == 0 and store.is_exist(key) == 0
            keys.remove(key)
        result['pass'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        for key in keys:
            try:
                result.setdefault('failure_cleanup', []).append({'key': key, 'remove_rc': store.remove(key)})
            except Exception as error:
                result.setdefault('failure_cleanup', []).append({'key': key, 'error': repr(error)})
        result['close_rc'] = store.close()
        result['pass'] = bool(result['pass'] and result['close_rc'] == 0)
        print('GLM53_SDK_HOST_READBACK=' + json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
