#!/usr/bin/env python3
"""Bounded cross-node NPU-buffer READ using the installed connector's exact TE protocol."""
import ctypes
import json
import os
import socket
import time

import torch
import torch_npu  # noqa: F401
from mooncake.engine import TransferEngine


def main():
    rank = int(os.environ['GLM53_EXTERNAL_RANK'])
    assert rank in (0, 1) and torch.npu.device_count() == 1
    for key, value in {'HCCL_INTER_HCCS_DISABLE': 'FALSE', 'HCCL_INTRA_ROCE_ENABLE': '0',
                       'HCCL_INTRA_PCIE_ENABLE': '1', 'HCCL_OP_EXPANSION_MODE': 'AI_CPU'}.items():
        assert os.environ.get(key) == value
    torch.npu.set_device(0)
    acl = ctypes.CDLL('libascendcl.so')
    logic, physical = ctypes.c_int(), ctypes.c_int()
    assert acl.aclrtGetLogicDevIdByUserDevId(0, ctypes.byref(logic)) == 0
    assert acl.aclrtGetPhyDevIdByLogicDevId(logic.value, ctypes.byref(physical)) == 0
    assert str(physical.value) == os.environ['GLM53_PHYSICAL_DEVICE_IDS']
    ip = os.environ['HCCL_IF_IP']
    te = TransferEngine()
    initialize_rc = te.initialize(ip, 'P2PHANDSHAKE', 'ascend', '')
    assert initialize_rc == 0, ('initialize', initialize_rc)
    sizes = [65536, 1048576, 16777216]
    source = torch.empty(max(sizes), dtype=torch.int8, device='npu:0')
    target = torch.empty_like(source)
    registrations = [te.register_memory(t.data_ptr(), t.numel()) for t in (source, target)]
    assert registrations == [0, 0], registrations
    local = {'rank': rank, 'physical': physical.value, 'session': ip+':'+str(te.get_rpc_port()),
             'source': source.data_ptr(), 'bytes': source.numel()}
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

    remote = exchange(local)
    assert remote['rank'] == 1-rank and remote['bytes'] == max(sizes)
    checks = []
    for source_rank in (0, 1):
        for case, size in enumerate(sizes):
            expected = 31+source_rank*13+case
            source.fill_(expected)
            target.fill_(-7)
            torch.npu.synchronize()
            phase = {'source_rank': source_rank, 'bytes': size, 'expected': expected}
            assert exchange(phase) == phase
            result = dict(phase, receiver=rank != source_rank)
            if rank != source_rank:
                # Mooncake calls these src/dst even for READ: the first list is
                # LOCAL destination, the second is REMOTE source. This matches
                # pull_worker._append_block_transfer_addresses at the audited SHA.
                rc = te.batch_transfer_sync_read(remote['session'], [target.data_ptr()], [remote['source']], [size])
                assert rc == 0, ('transfer', rc)
                torch.npu.synchronize()
                valid = bool(torch.all(target[:size] == expected).item())
                assert valid, ('payload mismatch', phase)
                result.update(returncode=rc, payload_valid=valid)
            peer = exchange({'complete': phase})
            assert peer == {'complete': phase}
            checks.append(result)
    result = {'rank': rank, 'local': local, 'remote': remote, 'protocol': 'ascend',
              'metadata_server': 'P2PHANDSHAKE', 'initialize_rc': initialize_rc,
              'register_rc': registrations, 'checks': checks,
              'runtime_sdma_path_verified': False,
              'scope': 'standalone installed Mooncake TE buffer test; not model PD; analyze retained logs for path'}
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(result, indent=2)+'\nGLM53_RESULT_JSON_END', flush=True)
    wire.close()
    connection.close()
    if listener:
        listener.close()
    # Explicitly release the TE while registered tensors and NPU context exist.
    del te


if __name__ == '__main__':
    main()
