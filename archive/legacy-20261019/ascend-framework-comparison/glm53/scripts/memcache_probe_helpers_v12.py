"""Known-key native MemCache setup for a bounded SDK coexistence probe."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

META_PORT = 38400
CACHE_STORE_PORT = 38401
METRICS_PORT = 38402
KEY = 'glm53/sdk-coexist/20261017-v12/known-4mib'


def cache_setup(role, source, state, control, event):
    directory = Path('/tmp/glm53-memcache-v12')
    directory.mkdir(exist_ok=False)
    meta_url = f'tcp://{source}:{META_PORT}'
    store_url = f'tcp://{source}:{CACHE_STORE_PORT}'
    meta_config = directory / 'meta.conf'
    local_config = directory / 'local.conf'
    meta_config.write_text(
        f'ock.mmc.meta_service_url = {meta_url}\n'
        f'ock.mmc.meta_service.config_store_url = {store_url}\n'
        f'ock.mmc.meta_service.metrics_url = http://{source}:{METRICS_PORT}\n'
        'ock.mmc.log_level = info\n')
    local_config.write_text(
        f'ock.mmc.meta_service_url = {meta_url}\n'
        f'ock.mmc.local_service.config_store_url = {store_url}\n'
        'ock.mmc.log_level = debug\n'
        'ock.mmc.local_service.world_size = 2\n'
        'ock.mmc.local_service.protocol = device_sdma\n'
        'ock.mmc.local_service.dram.size = 1GB\n')
    os.environ['MMC_META_CONFIG_PATH'] = str(meta_config)
    os.environ['MMC_LOCAL_CONFIG_PATH'] = str(local_config)
    meta = None
    meta_log = None
    if role == 'Prefill':
        meta_root='/opt/glm53/meta-sdk13'
        meta_env=os.environ.copy()
        meta_env['PYTHONPATH']=meta_root+':'+meta_env.get('PYTHONPATH','')
        inherited=[p for p in meta_env.get('LD_LIBRARY_PATH','').split(':') if p and '/tmp/glm53-sdk-alias' not in p and '/site-packages/memfabric_hybrid' not in p and '/site-packages/memcache_hybrid' not in p]
        meta_env['LD_LIBRARY_PATH']=meta_root+'/memfabric_hybrid/lib:'+meta_root+'/memcache_hybrid/lib:'+':'.join(inherited)
        meta_env['PYTHONDONTWRITEBYTECODE']='1'
        meta_code="import memcache_hybrid as mc, pathlib, sys; assert mc.__file__.startswith('/opt/glm53/meta-sdk13/'); print('GLM53_META_SDK='+mc.__file__+' VERSION='+pathlib.Path(mc.__file__).with_name('VERSION').read_text(),flush=True); sys.exit(mc.MetaService.main())"
        meta = subprocess.Popen([sys.executable,'-c',meta_code],env=meta_env,
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        def forward_meta():
            for line in meta.stdout:
                print('GLM53_META_LOG='+line.rstrip('\n'), flush=True)
        meta_log = threading.Thread(target=forward_meta, daemon=True)
        meta_log.start()
        deadline = time.monotonic() + 30
        while True:
            if meta.poll() is not None:
                raise RuntimeError(f'MetaService exited: {meta.returncode}; raw log forwarded')
            try:
                with socket.create_connection((source, META_PORT), timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('MemCache MetaService readiness timeout')
                time.sleep(.1)
        state['meta_ready'] = True
        event('memcache_meta_ready', pid=meta.pid, meta_url=meta_url, config_store=store_url)
    mc=load_memcache()
    store = mc.DistributedObjectStore()
    rc = store.init(0, init_bm=True)
    event('memcache_init_completed', rc=rc, protocol='device_sdma', dram_bytes=1073741824,
          config_sha256=__import__('hashlib').sha256(local_config.read_bytes()).hexdigest())
    if rc != 0:
        raise RuntimeError(f'MemCache init failed: {rc}')
    return mc, store, meta, meta_log


def cache_close(role, store, meta, meta_log, ptr, size, event):
    rc = store.unregister_buffer(ptr, size)
    event('memcache_unregister_completed', rc=rc)
    if rc != 0:
        raise RuntimeError(f'MemCache unregister failed: {rc}')
    rc = store.close()
    event('memcache_close_completed', rc=rc)
    if rc != 0:
        raise RuntimeError(f'MemCache close failed: {rc}')
    if meta is not None:
        maps=Path('/proc/%s/maps'%meta.pid).read_text()
        event('memcache_meta_library_maps',paths=sorted(set(line.split()[-1] for line in maps.splitlines() if 'libmf_' in line or '_pymmc' in line)))
        meta.terminate()
        try:
            rc = meta.wait(timeout=10)
        except subprocess.TimeoutExpired:
            meta.kill()
            meta.wait(timeout=5)
            raise RuntimeError('Owned MetaService failed graceful stop')
        meta_log.join(timeout=2)
        event('memcache_meta_stopped', rc=rc)
        if rc not in (0, -15):
            raise RuntimeError(f'MetaService stop failed: {rc}')
