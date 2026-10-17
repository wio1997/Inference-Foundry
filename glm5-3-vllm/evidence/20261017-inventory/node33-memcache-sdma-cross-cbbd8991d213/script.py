#!/usr/bin/env python3
"""Two-node MemCache device_sdma buffer correctness probe; no Mooncake/model/SSD."""
import ctypes
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import socket
import time

os.environ['ASCEND_ENABLE_USE_FABRIC_MEM'] = '1'
os.environ['ACL_OP_INIT_MODE'] = '1'
os.environ['ASCEND_LOCAL_COMM_RES'] = '{"version":"1.2"}'

import torch
import torch_npu  # noqa: F401,E402


def main():
    rank = int(os.environ['GLM53_EXTERNAL_RANK'])
    assert rank in (0, 1) and torch.npu.device_count() == 1
    torch.npu.set_device(0)
    acl = ctypes.CDLL('libascendcl.so')
    logic, physical = ctypes.c_int(), ctypes.c_int()
    assert acl.aclrtGetLogicDevIdByUserDevId(0, ctypes.byref(logic)) == 0
    assert acl.aclrtGetPhyDevIdByLogicDevId(logic.value, ctypes.byref(physical)) == 0
    assert str(physical.value) == os.environ['GLM53_PHYSICAL_DEVICE_IDS']
    config = ('ock.mmc.meta_service_url = tcp://172.16.12.32:27400\n'
              'ock.mmc.local_service.config_store_url = tcp://172.16.12.32:27401\n'
              'ock.mmc.local_service.world_size = 2\n'
              'ock.mmc.local_service.protocol = device_sdma\n'
              'ock.mmc.local_service.dram.size = 1GB\n'
              'ock.mmc.local_service.hbm.size = 0GB\n'
              'ock.mmc.local_service.max.dram.size = 1GB\n'
              'ock.mmc.local_service.storage.enabled = false\n'
              'ock.mmc.log_level = info\n'
              'ock.mmc.log_output_target = screen\n')
    path = Path('/out/mmc-local.conf')
    path.write_text(config)
    os.environ['MMC_LOCAL_CONFIG_PATH'] = str(path)
    versions = {p: md.version(p) for p in ('memcache_hybrid', 'memfabric_hybrid')}
    assert versions == {'memcache_hybrid': '1.2.0', 'memfabric_hybrid': '1.2.0'}
    from memcache_hybrid import DistributedObjectStore
    store = DistributedObjectStore()
    init_rc = store.init(0, init_bm=True)
    assert init_rc == 0, ('store.init', init_rc)
    sizes = [65536, 1048576]
    pattern = (torch.arange(max(sizes), dtype=torch.int32, device='npu:0') % 97 - 50).to(torch.int8)
    source, target = torch.empty_like(pattern), torch.empty_like(pattern)
    register_rc = [store.register_buffer(t.data_ptr(), t.numel()) for t in (source, target)]
    assert register_rc == [0, 0], register_rc
    addr = (os.environ['GLM53_MASTER_ADDR'], int(os.environ['GLM53_MASTER_PORT']))
    listener = None
    if rank == 0:
        listener = socket.socket()
        listener.bind(addr)
        listener.listen(1)
        listener.settimeout(120)
        connection, _ = listener.accept()
    else:
        deadline = time.monotonic()+120
        while True:
            try:
                connection = socket.create_connection(addr, timeout=3)
                break
            except (ConnectionRefusedError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
    connection.settimeout(120)
    wire = connection.makefile('rwb')

    def exchange(value):
        wire.write(json.dumps(value).encode()+b'\n')
        wire.flush()
        line = wire.readline(65536)
        assert line
        return json.loads(line)

    checks = []
    for source_rank in (0, 1):
        for case, size in enumerate(sizes):
            key = 'glm53-exec-memcache-20261017-s%d-b%d' % (source_rank, size)
            source.copy_(pattern+source_rank*11+case)
            target.fill_(-128)
            torch.npu.synchronize()
            expected_sha = hashlib.sha256(source[:size].cpu().numpy().tobytes()).hexdigest()
            row = {'source_rank': source_rank, 'bytes': size, 'key': key, 'expected_sha256': expected_sha}
            if rank == source_rank:
                row['put_rc'] = store.put_from(key, source.data_ptr(), size, 0)  # L2G
                assert row['put_rc'] == 0, row
            peer = exchange(row)
            for field in ('source_rank', 'bytes', 'key', 'expected_sha256'):
                assert row[field] == peer[field]
            exists = store.batch_is_exist([key])
            assert exists == [1], ('exist', exists, key)
            info = store.batch_get_key_info([key])[0]
            row['object'] = {'size': info.size(), 'gva': info.gva_list(),
                             'location': info.loc_list(), 'media_type': info.type_list()}
            if rank != source_rank:
                row['get_rc'] = store.batch_get_into([key], [target.data_ptr()], [size], 1)  # G2L
                assert row['get_rc'] == [0], row
                torch.npu.synchronize()
                actual_sha = hashlib.sha256(target[:size].cpu().numpy().tobytes()).hexdigest()
                row.update(actual_sha256=actual_sha, payload_valid=actual_sha == expected_sha)
                assert row['payload_valid'], row
            assert exchange({'complete': key}) == {'complete': key}
            checks.append(row)
    unregister_rc = [store.unregister_buffer(t.data_ptr(), t.numel()) for t in (source, target)]
    assert unregister_rc == [0, 0], unregister_rc
    report = {'rank': rank, 'physical': physical.value, 'versions': versions,
              'config': config, 'init_rc': init_rc, 'register_rc': register_rc,
              'unregister_rc': unregister_rc, 'checks': checks,
              'scope': 'synthetic HBM to MemCache pool to peer HBM correctness; not model PD or performance',
              'runtime_sdma_path_verified': False}
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    wire.close()
    connection.close()
    if listener:
        listener.close()
    del store


if __name__ == '__main__':
    main()
