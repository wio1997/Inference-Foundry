# SPDX-License-Identifier: Apache-2.0
"""Candidate Flash-only B103 MF1.3 PUSH adapter; runtime qualification required.

Bound to the native ASCEND registry's AscendTransferEngine symbol. TCP carries
SDK control metadata. All transferred tensor bytes use typed SDMA. No fallback.
"""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
import threading
import time


def _event(kind, **fields):
    from sglang.srt.disaggregation.ascend.flash_b103_pd_events import append
    record=append(kind, **fields)
    print('GLM53_FLASH_B103_SDMA '+json.dumps(record,sort_keys=True),flush=True)


class AscendTransferEngine:
    def __init__(self, hostname, npu_id, disaggregation_mode):
        import torch
        import memfabric_hybrid as mf
        from sglang.srt.disaggregation.utils import DisaggregationMode
        self.hostname = hostname
        self.npu_id = int(npu_id)
        self.role = {DisaggregationMode.PREFILL: 'Prefill',
                     DisaggregationMode.DECODE: 'Decode'}[disaggregation_mode]
        if os.environ.get('ASCEND_MF_TRANSFER_PROTOCOL', 'sdma').strip().lower() != 'sdma':
            raise RuntimeError('Flash PD requires explicit typed SDMA')
        self.store_url = os.environ['ASCEND_MF_STORE_URL']
        if not self.store_url.startswith('tcp://'):
            raise ValueError('Flash requires its explicit TCP control-store URL')
        assert hashlib.sha256(pathlib.Path(mf.__file__).read_bytes()).hexdigest() == (
            '0e1faeefec1f7295285e2e60b6a43b6476ee31b343615cac62ad6a3d8e3896e9')
        self._torch = torch
        self._lock = threading.RLock()
        self._closing = False
        self._closed = False
        self._peer_regions = {}
        self._registered_ptrs = set()
        if os.environ.get('FLASH_SDMA_VERIFY_BYTES')!='1':
            raise ValueError('This bounded PD diagnostic requires every payload SHA')
        from sglang.srt.disaggregation.ascend import sdma_payload_hash
        self._regions=[]
        self._hasher=sdma_payload_hash.PayloadHasher(self.npu_id,self._regions)
        self._hash_port=19100+self.npu_id
        self._hash_server=(sdma_payload_hash.start_hash_server(
            self._hasher,hostname,self._hash_port,{'172.16.12.46'})
            if self.role=='Decode' else None)
        self._hash_source_sha256=hashlib.sha256(pathlib.Path(sdma_payload_hash.__file__).read_bytes()).hexdigest()
        self._probe = None
        mf.set_conf_store_tls(False, '')
        mf.set_log_level(0)
        self.engine = mf.TransferEngine()
        self.protocol = mf.TransDataOpType.SDMA
        initial_session = f'{hostname}:{self.engine.get_rpc_port()}'
        ret = self.engine.initialize(self.store_url, initial_session, self.role,
                                     self.npu_id, data_op_type=self.protocol,
                                     store_server_role='Decode')
        if ret != 0:
            raise RuntimeError(f'Flash SDMA initialize failed: {ret}')
        port = int(self.engine.get_rpc_port())
        if not 0 < port < 65536:
            raise RuntimeError('Flash SDMA has no valid listening port')
        self.session_id = f'{hostname}:{port}'
        libs = {}
        for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
            p = pathlib.Path(line.split()[-1])
            if p.is_file() and any(s in p.name for s in ('libmf_', '_pymf_transfer')):
                libs[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        _event('initialized', device=self.npu_id, role=self.role,
               session_id=self.session_id, protocol=str(self.protocol),
               store_server_role='Decode', store_url=self.store_url,
               payload_hash_required=True,payload_hash_source_sha256=self._hash_source_sha256,
               hash_control_port=self._hash_port if self.role=='Decode' else None,
               loaded_libraries=libs,
               source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())

    @staticmethod
    def _vectors(ptrs, lengths):
        ptrs, lengths = list(map(int, ptrs)), list(map(int, lengths))
        if len(ptrs) != len(lengths) or not ptrs:
            raise ValueError('Empty or mismatched SDMA vectors')
        if any(p <= 0 or n <= 0 for p, n in zip(ptrs, lengths)):
            raise ValueError('Nonpositive SDMA pointer/length')
        return ptrs, lengths

    def batch_register(self, ptrs, lengths):
        ptrs, lengths = self._vectors(ptrs, lengths)
        with self._lock:
            if self._closing: raise RuntimeError('Flash SDK engine is closing or closed')
            # Keep a private aligned buffer alive for genuine recovery reads.
            # Include it in the native batch that merges aligned registrations.
            if self._probe is None:
                t = self._torch
                align = 2097152
                self._probe_slab = t.empty(2 * align, dtype=t.uint8,
                                           device=f'npu:{self.npu_id}')
                offset = (-self._probe_slab.data_ptr()) % align
                self._probe = self._probe_slab[offset:offset + align]
                self._probe.zero_()
                t.npu.synchronize()
                ptrs.append(self._probe.data_ptr())
                lengths.append(align)
            ret = self.engine.batch_register_memory(ptrs, lengths)
            _event('registered_batch', device=self.npu_id, ptrs=ptrs,
                   lengths=lengths, ret=ret, probe_ptr=self._probe.data_ptr())
            if ret != 0:
                raise RuntimeError(f'Flash SDMA batch register failed: {ret}')
            self._registered_ptrs.update(p for p in ptrs if p != self._probe.data_ptr())
            self._regions.extend((p,n) for p,n in zip(ptrs,lengths)
                                 if (p,n) not in self._regions)
            return ret

    def register(self, ptr, length):
        return self.batch_register([ptr], [length])

    def batch_deregister(self, ptrs):
        with self._lock:
            for ptr in dict.fromkeys(map(int, ptrs)):
                if ptr not in self._registered_ptrs:
                    raise ValueError('Unowned or already deregistered Flash region')
                ret = self.engine.unregister_memory(ptr)
                _event('unregistered', device=self.npu_id, ptr=ptr, ret=ret)
                if ret != 0:
                    raise RuntimeError(f'Flash SDMA unregister failed: {ret}')
                self._registered_ptrs.remove(ptr)
            if not self._registered_ptrs and self._probe is not None:
                ret = self.engine.unregister_memory(self._probe.data_ptr())
                _event('probe_unregistered', device=self.npu_id,
                       ptr=self._probe.data_ptr(), ret=ret)
                if ret != 0:
                    raise RuntimeError(f'Flash SDMA probe unregister failed: {ret}')
                self._probe = None
                self._probe_slab = None
                self._peer_regions.clear()
            return 0

    def deregister(self, ptr):
        return self.batch_deregister([ptr])

    def batch_transfer_sync(self, session_id, buffers, peer_buffer_addresses, lengths):
        buffers, lengths = self._vectors(buffers, lengths)
        remote, lengths2 = self._vectors(peer_buffer_addresses, lengths)
        assert lengths == lengths2
        with self._lock:
            if self._closing: raise RuntimeError('Flash SDK engine is closing or closed')
            import urllib.request
            from sglang.srt.disaggregation.ascend.flash_b103_pd_events import tag
            active=tag()
            if active is None:raise ValueError('No actual serialized request tag for PD transfer')
            peers=json.loads(pathlib.Path('/run-evidence/sdma-peer-map.json').read_text())
            peer=peers[str(session_id)]
            rank=int(peer['device_id'])
            if not 0<=rank<16 or rank!=self.npu_id:
                raise ValueError('Initial Flash PD requires matching TP16 ranks')
            expected_url='http://172.16.12.47:'+str(19100+rank)+'/hash'
            if peer['url']!=expected_url:raise ValueError('Unexpected Flash hash peer')
            before=self._hasher.hash_ranges(buffers,lengths)
            def remote_hash():
                request=urllib.request.Request(expected_url,
                    data=json.dumps({'addresses':remote,'sizes':lengths}).encode(),
                    headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=45) as response:return json.load(response)
            # A known nonempty peer range may be used even if this write fails.
            # It comes from the native registered destination vectors, never a
            # guessed address or a zero-length success shortcut.
            for ptr, n in zip(remote, lengths):
                if n >= 64:
                    self._peer_regions[str(session_id)] = ptr
                    break
            from sglang.srt.managers.flash_pd_lifetime_audit import before_actual_write
            before_actual_write(self, buffers, lengths, active)
            _event('push_submitted', device=self.npu_id, remote_session=str(session_id),
                   local=buffers, remote=remote, lengths=lengths, bytes=sum(lengths))
            ret = self.engine.batch_transfer_sync_write(str(session_id), buffers, remote, lengths)
            _event('push_complete', device=self.npu_id, remote_session=str(session_id),
                   bytes=sum(lengths), segments=len(lengths), ret=ret)
            if ret!=0:
                raise RuntimeError(f'Flash SDMA PUSH failed: {ret}')
            destination=remote_hash()
            after=self._hasher.hash_ranges(buffers,lengths)
            mismatch=[i for i,(a,b,c) in enumerate(zip(before['range_sha256'],
                         destination['range_sha256'],after['range_sha256'])) if not a==b==c]
            passed=(len(before['range_sha256'])==len(destination['range_sha256'])==len(after['range_sha256'])==len(lengths)
                    and before['sha256']==destination['sha256']==after['sha256'] and not mismatch)
            _event('payload_checksum',device=self.npu_id,remote_device=rank,
                   remote_session=str(session_id),tag=active,passed=passed,
                   bytes=sum(lengths),segments=len(lengths),
                   p_before_sha256=before['sha256'],d_after_sha256=destination['sha256'],
                   p_after_sha256=after['sha256'],mismatch_indices=mismatch,
                   range_sha256=before['range_sha256'],explicit_device_sync=True)
            if not passed:raise RuntimeError('Flash actual PD payload byte mismatch')
            # Native manager handles nonzero status as a failed request/session.
            return 0 if ret == 0 else -1

    def transfer_sync(self, session_id, buffer, peer_buffer_address, length):
        return self.batch_transfer_sync(session_id, [buffer], [peer_buffer_address], [length])

    def send_probe(self, peer_session_id):
        with self._lock:
            if self._closing: return -1
            ptr = self._peer_regions.get(str(peer_session_id))
            if ptr is None or self._probe is None:
                _event('probe_unavailable', device=self.npu_id,
                       remote_session=str(peer_session_id), ret=-1)
                return -1
            ret = self.engine.batch_transfer_sync_read(str(peer_session_id),
                    [self._probe.data_ptr()], [ptr], [64])
            _event('probe64_complete', device=self.npu_id,
                   remote_session=str(peer_session_id), local=self._probe.data_ptr(),
                   remote=ptr, bytes=64, ret=ret)
            return 0 if ret == 0 else -1

    def close(self):
        from sglang.srt.managers.flash_pd_sdk_shutdown import close_engine
        return close_engine(self)

    def get_session_id(self):
        return self.session_id

    def get_rpc_port(self):
        return self.engine.get_rpc_port()
