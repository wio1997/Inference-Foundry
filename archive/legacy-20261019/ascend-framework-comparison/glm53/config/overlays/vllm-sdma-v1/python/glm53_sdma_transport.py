"""Project-scoped vLLM pull-connector adapter. SDMA only; no fallback.

Prepared locally; deployment requires separate frozen launcher and real PD gates.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import threading
import time

log = logging.getLogger(__name__)


def audit(event: str, **fields) -> None:
    log.info('GLM53_PD_SDMA=%s', json.dumps({
        'event': event, 'pid': os.getpid(), 'thread_id': threading.get_ident(),
        'monotonic_ns': time.monotonic_ns(), **fields,
    }, sort_keys=True))


class GlobalTE:
    def __init__(self):
        self._lock = threading.RLock()
        self._configuration = None
        self._engine = None
        self._hostname = None
        self._registered = None
        self._closed = False
        self._kv_refs = None

    def configure(self, *, role: str, device_id: int, session_port: int, store_url: str,
                  sdk_log_level: int = 2):
        if role not in ('Prefill', 'Decode') or device_id < 0:
            raise ValueError('Invalid GLM53 SDMA role/device')
        if not 1024 <= session_port <= 65535 or not store_url.startswith('tcp://'):
            raise ValueError('Invalid GLM53 SDMA session/store endpoint')
        if sdk_log_level not in (0, 1, 2, 3):
            raise ValueError('Invalid SDK log level')
        config = (role, device_id, session_port, store_url, sdk_log_level)
        with self._lock:
            if self._closed:
                raise RuntimeError('GLM53 SDMA engine has been closed')
            if self._configuration is not None and self._configuration != config:
                raise RuntimeError('GLM53 SDMA engine cannot change role/device/endpoints')
            self._configuration = config

    def get_transfer_engine(self, hostname: str, device_name=None):
        with self._lock:
            if self._closed:
                raise RuntimeError('GLM53 SDMA engine has been closed')
            if self._configuration is None:
                raise RuntimeError('GLM53 SDMA requires explicit role/device/endpoints')
            if self._engine is not None:
                if hostname != self._hostname:
                    raise RuntimeError('GLM53 SDMA cannot change hostname')
                return self
            import memfabric_hybrid as mf
            from importlib.metadata import version
            if version('memfabric_hybrid') != '1.3.0' or not mf.__file__.startswith('/opt/glm53/sdk13/'):
                raise RuntimeError('GLM53 serving adapter requires the frozen private SDK1.3 overlay')

            role, device_id, port, store_url, sdk_log_level = self._configuration
            # Verified in the fixed images: nested enum exists in 1.2 and 1.3.
            op = mf.TransferEngine.TransDataOpType.SDMA
            if op.value != 1:
                raise RuntimeError('Unexpected SDK SDMA value; refuse transfer')
            engine = mf.TransferEngine()
            mf.set_log_level(sdk_log_level)
            mf.set_conf_store_tls(False, '')
            session = f'{hostname}:{port}'
            rc = engine.initialize(store_url, session, role, device_id, op,
                                   store_server_role='Prefill')
            audit('initialize_complete', role=role, device_id=device_id,
                  protocol='SDMA', protocol_value=op.value, session=session,
                  store_url=store_url, rc=rc)
            if rc != 0:
                raise RuntimeError(f'GLM53 SDMA initialize failed: {rc}; {mf.get_last_err_msg()}')
            actual_port = engine.get_rpc_port()
            if actual_port != port:
                raise RuntimeError(f'SDK advertised port {actual_port} differs from unique session {port}')
            self._engine, self._hostname = engine, hostname
            self._session = session
            audit('loaded_libraries', paths=sorted({
                line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                if any(s in line for s in ('libmf_', 'libhcom', '_pymf_transfer', 'libascend_hal'))
            }))
            return self

    def pin_kv_caches(self, kv_caches):
        """Keep backing allocations alive until SDK and cache shutdown complete."""
        self._kv_refs = kv_caches

    def close(self):
        with self._lock:
            if self._engine is None:
                return
            audit('shutdown_begin', session=self._session)
            self._engine.destroy()
            self._engine.unInitialize()
            self._engine = None
            self._closed = True
            audit('shutdown_complete', session=self._session)

    def release_kv_pins(self):
        if self._engine is not None:
            raise RuntimeError('Cannot release KV backing allocations before SDMA shutdown')
        self._kv_refs = None

    def get_rpc_port(self):
        if self._engine is None:
            raise RuntimeError('SDMA engine not initialized')
        return self._engine.get_rpc_port()

    def register_buffer(self, ptrs: list[int], sizes: list[int]):
        if len(ptrs) != len(sizes) or any(x <= 0 for x in ptrs + sizes):
            raise ValueError('Invalid SDMA memory registration descriptors')
        descriptors = tuple(zip(ptrs, sizes))
        with self._lock:
            if self._engine is None:
                raise RuntimeError('SDMA engine not initialized')
            if self._registered is not None:
                if descriptors != self._registered:
                    raise RuntimeError('SDMA buffer registration changed after initialization')
                return
            # SDK performs alignment/overlap merging within one batch.
            rc = self._engine.batch_register_memory(ptrs, sizes)
            audit('register_complete', session=self._session, rc=rc,
                  allocations=len(sizes), allocation_bytes=sum(sizes))
            if rc != 0:
                raise RuntimeError(f'GLM53 SDMA batch registration failed: {rc}')
            self._registered = descriptors

    def batch_transfer_sync_read(self, session_id, local_buffers, peer_buffers, lengths):
        if not (len(local_buffers) == len(peer_buffers) == len(lengths)):
            raise ValueError('SDMA READ descriptor lengths differ')
        if any(x <= 0 for x in list(local_buffers) + list(peer_buffers) + list(lengths)):
            raise ValueError('SDMA READ descriptor is nonpositive')
        if self._engine is None or self._registered is None:
            raise RuntimeError('SDMA READ before engine/buffer registration')
        begin = time.monotonic_ns()
        rc = self._engine.batch_transfer_sync_read(session_id, local_buffers, peer_buffers, lengths)
        audit('read_complete', local_session=self._session, remote_session=session_id,
              payload_bytes_completed=sum(lengths) if rc == 0 else None,
              payload_bytes_requested=sum(lengths), physical_link_bytes=None,
              ranges=len(lengths), rc=rc,
              elapsed_ns=time.monotonic_ns() - begin)
        if rc != 0:
            raise RuntimeError(f'GLM53 SDMA READ failed: {rc}; no fallback permitted')
        return 0


global_te = GlobalTE()
