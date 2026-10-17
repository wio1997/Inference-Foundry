#!/usr/bin/env python3
"""Isolate SDK init/close from tensor allocation and payload transfers."""
import faulthandler
import json
import os
from pathlib import Path

faulthandler.enable()
os.environ['ASCEND_ENABLE_USE_FABRIC_MEM'] = '1'
os.environ['ACL_OP_INIT_MODE'] = '1'
os.environ['ASCEND_LOCAL_COMM_RES'] = '{"version":"1.2"}'

import torch
import torch_npu  # noqa: F401,E402


def main():
    assert torch.npu.device_count() == 1
    torch.npu.set_device(0)
    config = ('ock.mmc.meta_service_url = tcp://172.16.12.32:27400\n'
              'ock.mmc.local_service.config_store_url = tcp://172.16.12.32:27401\n'
              'ock.mmc.local_service.world_size = 2\n'
              'ock.mmc.local_service.protocol = device_sdma\n'
              'ock.mmc.local_service.dram.size = 1GB\n'
              'ock.mmc.local_service.hbm.size = 0GB\n'
              'ock.mmc.local_service.max.dram.size = 1GB\n'
              'ock.mmc.local_service.storage.enabled = false\n'
              'ock.mmc.log_level = info\n')
    path = Path('/out/mmc-local.conf')
    path.write_text(config)
    os.environ['MMC_LOCAL_CONFIG_PATH'] = str(path)
    from memcache_hybrid import DistributedObjectStore
    store = DistributedObjectStore()
    init_rc = store.init(0, init_bm=True)
    assert init_rc == 0
    close_rc = store.close()
    assert close_rc == 0
    del store
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({
        'init_rc': init_rc, 'close_rc': close_rc, 'scope': 'no NPU tensors/buffer registration/payload transfer',
        'npu_torch_allocated': torch.npu.memory_allocated(), 'npu_torch_reserved': torch.npu.memory_reserved(),
        'config': config})+'\nGLM53_RESULT_JSON_END', flush=True)


if __name__ == '__main__':
    main()
