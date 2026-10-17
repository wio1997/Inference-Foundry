#!/usr/bin/env python3
"""Check large layered SDK copies with a per-key reference, without model weights."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid

os.environ['ASCEND_ENABLE_USE_FABRIC_MEM'] = '1'
os.environ['ACL_OP_INIT_MODE'] = '1'
os.environ['ASCEND_LOCAL_COMM_RES'] = '{"version":"1.2"}'

import numpy as np
import torch
import torch_npu  # noqa: F401

OBJECT_BYTES = 6979584
ENTRIES = 512
ENTRY_BYTES = OBJECT_BYTES//ENTRIES
MAX_KEYS = 31


def main():
    rank = int(os.environ['GLM53_EXTERNAL_RANK'])
    assert rank in (0, 1) and torch.npu.device_count() == 1
    assert OBJECT_BYTES == ENTRIES*ENTRY_BYTES and ENTRY_BYTES % 64 == 0
    torch.npu.set_device(0)
    runtime = ctypes.CDLL('libascendcl.so')
    logical, physical = ctypes.c_int(), ctypes.c_int()
    assert runtime.aclrtGetLogicDevIdByUserDevId(0, ctypes.byref(logical)) == 0
    assert runtime.aclrtGetPhyDevIdByLogicDevId(logical.value, ctypes.byref(physical)) == 0
    assert str(physical.value) == os.environ['GLM53_PHYSICAL_DEVICE_IDS']
    out = Path('/out')
    config = ('ock.mmc.meta_service_url = tcp://172.16.12.32:27410\n'
              'ock.mmc.local_service.config_store_url = tcp://172.16.12.32:27411\n'
              'ock.mmc.local_service.world_size = 2\n'
              'ock.mmc.local_service.protocol = device_sdma\n'
              'ock.mmc.local_service.dram.size = 1GB\n'
              'ock.mmc.local_service.hbm.size = 0GB\n'
              'ock.mmc.local_service.max.dram.size = 1GB\n'
              'ock.mmc.local_service.storage.enabled = false\n'
              'ock.mmc.log_level = info\n')
    (out/'mmc-local.conf').write_text(config)
    os.environ['MMC_LOCAL_CONFIG_PATH'] = str(out/'mmc-local.conf')
    from memcache_hybrid import DistributedObjectStore
    store = DistributedObjectStore()
    assert store.init(0, init_bm=True) == 0
    source_stride, target_stride = ENTRY_BYTES+64, ENTRY_BYTES+128
    source = ((torch.arange(MAX_KEYS*ENTRIES*source_stride, device='npu:0', dtype=torch.int32)
               +rank*37) % 251).to(torch.int8)
    target = torch.empty(MAX_KEYS*ENTRIES*target_stride, device='npu:0', dtype=torch.int8)
    torch.npu.synchronize()
    register = [store.register_buffer(t.data_ptr(), t.numel()) for t in (source, target)]
    assert register == [0, 0], register
    source_cpu = source.cpu().numpy().reshape(MAX_KEYS, ENTRIES, source_stride)
    expected = [hashlib.sha256(np.ascontiguousarray(source_cpu[i, :, :ENTRY_BYTES]).tobytes()).hexdigest()
                for i in range(MAX_KEYS)]
    del source_cpu
    listener = None
    address = (os.environ['GLM53_MASTER_ADDR'], int(os.environ['GLM53_MASTER_PORT']))
    if rank == 0:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address)
        listener.listen(1)
        listener.settimeout(120)
        connection, _ = listener.accept()
    else:
        deadline = time.monotonic()+120
        while True:
            try:
                connection = socket.create_connection(address, timeout=3)
                break
            except (ConnectionRefusedError, TimeoutError):
                assert time.monotonic() < deadline
                time.sleep(1)
    connection.settimeout(240)
    wire = connection.makefile('rwb')

    def exchange(value):
        wire.write(json.dumps(value).encode()+b'\n')
        wire.flush()
        line = wire.readline(1048576)
        assert line
        return json.loads(line)

    nonce = uuid.uuid4().hex if rank == 0 else None
    hello = exchange({'rank': rank, 'nonce': nonce})
    nonce = nonce if rank == 0 else hello['nonce']

    def buffers(tensor, stride, count):
        return [[tensor.data_ptr()+(i*ENTRIES+j)*stride for j in range(ENTRIES)] for i in range(count)]

    checks = []
    for sender in (0, 1):
        for count in (7, 31):
            keys = [f'glm53-layer-{nonce}-{sender}-{count}-{i}' for i in range(count)]
            sizes = [[ENTRY_BYTES]*ENTRIES for _ in keys]
            row = {'sender': sender, 'keys': count, 'buffers_per_key': ENTRIES,
                   'flattened_buffers': count*ENTRIES, 'object_bytes': OBJECT_BYTES,
                   'logical_payload_bytes': count*OBJECT_BYTES}
            if rank == sender:
                result = store.batch_put_from_layers(keys, buffers(source, source_stride, count), sizes, 0)
                row['put_results'] = result
                assert result == [0]*count, result
                control = {'keys': keys, 'expected_sha256': expected[:count]}
            else:
                control = {'ready': True}
            peer = exchange(control)
            if rank != sender:
                assert peer['keys'] == keys
                infos = store.batch_get_key_info(keys)
                row['metadata_sizes'] = [i.size() for i in infos]
                assert row['metadata_sizes'] == [OBJECT_BYTES]*count
                row['methods'] = []
                for mode in ('one-batch', 'one-key-at-a-time'):
                    target.fill_(-128)
                    # Establish destination readiness explicitly. This tests
                    # SDK gather/scatter fidelity, not framework stream races.
                    torch.npu.synchronize()
                    destinations = buffers(target, target_stride, count)
                    started = time.monotonic()
                    if mode == 'one-batch':
                        ret = store.batch_get_into_layers(keys, destinations, sizes, 1)
                    else:
                        ret = []
                        for k, a, s in zip(keys, destinations, sizes):
                            ret.extend(store.batch_get_into_layers([k], [a], [s], 1))
                    torch.npu.synchronize()
                    sdk_elapsed = time.monotonic()-started
                    copied = target.cpu().numpy().reshape(MAX_KEYS, ENTRIES, target_stride)
                    actual = [hashlib.sha256(np.ascontiguousarray(copied[i, :, :ENTRY_BYTES]).tobytes()).hexdigest()
                              for i in range(count)]
                    guard_valid = bool(np.all(copied[:count, :, ENTRY_BYTES:] == -128)
                                       and np.all(copied[count:] == -128))
                    item = {'method': mode, 'sdk_results': ret, 'actual_sha256': actual,
                            'expected_sha256': peer['expected_sha256'],
                            'every_key_payload_equal': actual == peer['expected_sha256'],
                            'destination_padding_untouched': guard_valid,
                            'diagnostic_sdk_elapsed_seconds': sdk_elapsed}
                    row['methods'].append(item)
                    del copied
                (out/f'check-s{sender}-n{count}.json').write_text(json.dumps(row, indent=2)+'\n')
            assert exchange({'complete': keys[-1]}) == {'complete': keys[-1]}
            checks.append(row)
    unregister = [store.unregister_buffer(t.data_ptr(), t.numel()) for t in (source, target)]
    assert unregister == [0, 0]
    torch.npu.synchronize()
    del source, target
    torch.npu.empty_cache()
    close = store.close()
    wire.close()
    connection.close()
    if listener:
        listener.close()
    report = {'rank': rank, 'physical_die': physical.value, 'checks': checks, 'close_return': close,
              'register': register, 'unregister': unregister,
              'passed': all(m['sdk_results'] == [0]*c['keys'] and m['every_key_payload_equal']
                            and m['destination_padding_untouched'] for c in checks for m in c.get('methods', [])),
              'scope': 'No model weights. Same measured object bytes as current GLM128-token KV object;512 noncontiguous buffers is a representative stress layout, not a claim of the actual model entry count. Fenced destination; SDMA SDK transport, CPU checksum readback and TCP metadata/barriers separate. Timing is diagnostic, not service/peak bandwidth or a universal lower bound.'}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    assert close == 0 and report['passed']


if __name__ == '__main__':
    main()
