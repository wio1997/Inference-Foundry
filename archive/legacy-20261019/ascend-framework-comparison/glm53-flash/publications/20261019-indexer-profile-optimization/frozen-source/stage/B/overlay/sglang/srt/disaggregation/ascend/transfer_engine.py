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


def audit_chunks(local,remote,lengths,*,max_bytes=32*1024*1024,max_ranges=1024,max_piece=8*1024*1024):
    """Partition audit views only; never split/retry/change the SDK transfer.

    Exactly cover each paired native range with matching byte offsets. Bound
    every hash RPC below the unchanged R049 helper's64MiB/4096range limits.
    """
    if not len(local)==len(remote)==len(lengths) or min(max_bytes,max_ranges,max_piece)<=0:
        raise ValueError('Invalid audit partition')
    left=[];right=[];sizes=[];total=0
    for l,r,n in zip(local,remote,lengths):
        if any(type(x) is not int or x<=0 for x in (l,r,n)):raise ValueError('Audit vectors must be positive integers')
        offset=0
        while offset<n:
            take=min(n-offset,max_piece,max_bytes-total)
            left.append(l+offset);right.append(r+offset);sizes.append(take);offset+=take;total+=take
            if total==max_bytes or len(sizes)==max_ranges:
                yield left,right,sizes
                left=[];right=[];sizes=[];total=0
    if sizes:yield left,right,sizes

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
        self._next_audit_call = 0
        self._registered_ptrs = set()
        audit=os.environ.get('FLASH_SDMA_VERIFY_BYTES')
        if audit not in ('0','1'):raise ValueError('Explicit Flash SDMA byte-audit mode required')
        self.byte_audit=audit=='1'
        self._regions=[]
        self._hasher=self._hash_server=self._hash_port=None
        self._hash_source_sha256=None
        self._performance_stats={'calls':0,'successful_calls':0,'failed_calls':0,'bytes':0,'successful_bytes':0,'ranges':0,'producer_fences':0}
        if self.byte_audit:
            from sglang.srt.disaggregation.ascend import sdma_payload_hash
            self._hasher=sdma_payload_hash.PayloadHasher(self.npu_id,self._regions)
            self._hash_port=19100+self.npu_id
            self._hash_server=(sdma_payload_hash.start_hash_server(
                self._hasher,hostname,self._hash_port,{'172.16.12.46'})
                if self.role=='Decode' else None)
            self._hash_source_sha256=hashlib.sha256(pathlib.Path(sdma_payload_hash.__file__).read_bytes()).hexdigest()
        self._probe = None
        mf.set_conf_store_tls(False, '')
        mf.set_log_level(2)
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
               payload_hash_required=self.byte_audit,payload_hash_source_sha256=self._hash_source_sha256,
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
            if not self.byte_audit:
                # The hasher used to supply an implicit producer fence. Keep an
                # explicit device completion fence before the real SDK PUSH.
                # Source owners remain held by the existing native manager and
                # its writer-first shutdown protocol until the sync call ends.
                self._torch.npu.set_device(self.npu_id)
                self._torch.npu.synchronize(self.npu_id)
                self._performance_stats['producer_fences']+=1
                for ptr,n in zip(remote,lengths):
                    if n>=64:self._peer_regions[str(session_id)]=ptr;break
                self._performance_stats['calls']+=1
                self._performance_stats['bytes']+=sum(lengths)
                self._performance_stats['ranges']+=len(lengths)
                try:
                    ret = self.engine.batch_transfer_sync_write(str(session_id), buffers, remote, lengths)
                except BaseException:
                    self._performance_stats['failed_calls']+=1
                    raise
                if ret!=0:
                    self._performance_stats['failed_calls']+=1
                    raise RuntimeError(f'Flash SDMA PUSH failed: {ret}')
                self._performance_stats['successful_calls']+=1
                self._performance_stats['successful_bytes']+=sum(lengths)
                return 0
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
            chunks=list(audit_chunks(buffers,remote,lengths))
            call_id=self._next_audit_call;self._next_audit_call+=1
            before=[self._hasher.hash_ranges(ls,ns) for ls,rs,ns in chunks]
            def remote_hash(addresses,sizes):
                request=urllib.request.Request(expected_url,
                    data=json.dumps({'addresses':addresses,'sizes':sizes}).encode(),
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
            for index,((ls,rs,ns),source) in enumerate(zip(chunks,before)):
                destination=remote_hash(rs,ns)
                after=self._hasher.hash_ranges(ls,ns)
                mismatch=[i for i,(a,b,c) in enumerate(zip(source['range_sha256'],
                             destination['range_sha256'],after['range_sha256'])) if not a==b==c]
                passed=(len(source['range_sha256'])==len(destination['range_sha256'])==len(after['range_sha256'])==len(ns)
                        and source['sha256']==destination['sha256']==after['sha256'] and not mismatch)
                _event('payload_checksum',device=self.npu_id,remote_device=rank,
                       remote_session=str(session_id),tag=active,passed=passed,
                       call_id=call_id,audit_chunk=index,audit_chunks=len(chunks),
                       native_bytes=sum(lengths),native_segments=len(lengths),
                       bytes=sum(ns),segments=len(ns),lengths=ns,local=ls,remote=rs,
                       p_before_sha256=source['sha256'],d_after_sha256=destination['sha256'],
                       p_after_sha256=after['sha256'],mismatch_indices=mismatch,
                       range_sha256=source['range_sha256'],explicit_device_sync=True)
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
        result=close_engine(self)
        if not self.byte_audit and not getattr(self,'_stats_emitted',False):
            self._stats_emitted=True
            _event('performance_SDMA_summary',device=self.npu_id,role=self.role,
                   protocol=str(self.protocol),byte_audit=False,
                   stats=dict(self._performance_stats))
        return result

    def get_session_id(self):
        return self.session_id

    def get_rpc_port(self):
        return self.engine.get_rpc_port()
