# SPDX-License-Identifier: Apache-2.0
"""Strict MF1.2 SDMA backend for native Flash MooncakeConnectorV2 only.

Source-only candidate; DP/graph/MTP/native hybrid cache remain unqualified.
The store backend keeps its separate native Mooncake engine. No automatic
RDMA fallback and no CPU tensor hashing/copies in byte-audit-off mode.
"""
from __future__ import annotations
import atexit,hashlib,json,os,pathlib,threading,time


_EVENT_LOCK=threading.Lock()

def event(kind,**fields):
    root=pathlib.Path('/run-evidence/transport-events');root.mkdir(exist_ok=True)
    row={'project_id':'glm53-flash','kind':kind,'pid':os.getpid(),'monotonic_ns':time.monotonic_ns(),**fields}
    with _EVENT_LOCK:
        with (root/(str(os.getpid())+'.jsonl')).open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')


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


class FlashPerformanceSdmaEngine:
    def __init__(self,hostname,role,device_id,rank_info):
        from vllm_ascend.distributed.kv_transfer.utils.flash_native_sdk_bindings import load_bindings
        hybrid,transfer,_=load_bindings();TransferEngine=transfer.TransferEngine
        set_conf_store_tls=hybrid.set_conf_store_tls;set_log_level=hybrid.set_log_level
        TransDataOpType=transfer.TransDataOpType
        if (role,hostname) not in [('Prefill','172.16.12.14'),('Decode','172.16.12.15')]:raise ValueError('Outside owned Flash P14/D15 transport')
        store=os.environ['ASCEND_MF_STORE_URL']
        if store!='tcp://172.16.12.14:19048':raise ValueError('Wrong frozen Flash control store')
        audit=os.environ.get('FLASH_SDMA_VERIFY_BYTES')
        if audit not in ('0','1'):raise ValueError('Explicit byte audit mode is required')
        self.byte_audit=audit=='1';self.device_id=int(device_id);self.role=role;self.rank_info=dict(rank_info);self._regions=[];self._lock=threading.Lock();self._producer_seen=False;self._next_call_sequence=0
        self._stats={'calls':0,'successful_calls':0,'failed_calls':0,'ranges':0,'bytes':0,'producer_fences':0,'successful_bytes':0};self._engine=TransferEngine();self.protocol=TransDataOpType.SDMA
        set_log_level(2);set_conf_store_tls(False,'')
        self.session_id=f'{hostname}:{self._engine.get_rpc_port()}'
        ret=self._engine.initialize(store,self.session_id,role,self.device_id,data_op_type=self.protocol,store_server_role='Prefill')
        if ret!=0:raise RuntimeError(f'Strict Flash SDMA initialize failed: {ret}')
        self.session_id=f'{hostname}:{self._engine.get_rpc_port()}';self.source_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
        import ctypes
        self._acl=ctypes.CDLL('libascendcl.so');self._acl.aclrtSetDevice.argtypes=[ctypes.c_int32];self._acl.aclrtSetDevice.restype=ctypes.c_int32
        self.hasher=self.hash_server=self.hash_control_port=None
        if self.byte_audit:
            from vllm_ascend.distributed.kv_transfer.utils import sdma_payload_hash
            self.hasher=sdma_payload_hash.PayloadHasher(self.device_id,self._regions)
            if role=='Prefill':
                # DP-visible logical device IDs repeat; ports use native DP/TP
                # identity instead. This never changes tensor/state indices.
                self.hash_control_port=18500+self.rank_info['global_device_ordinal']
                self.hash_server=sdma_payload_hash.start_hash_server(self.hasher,hostname,self.hash_control_port,{'172.16.12.15'})
        event('strict_SDMA_initialized',role=role,logical_device=self.device_id,rank_info=self.rank_info,protocol=str(self.protocol),session_id=self.session_id,byte_audit=self.byte_audit,hash_control_port=self.hash_control_port,source_sha256=self.source_sha256)
        atexit.register(self.emit_summary)

    def get_rpc_port(self):return self._engine.get_rpc_port()

    def register_memory(self,ptr,size):
        ptr,size=int(ptr),int(size)
        if ptr<=0 or size<=0:raise ValueError('Invalid native registration')
        return self.register_memories([ptr],[size])

    def register_memories(self,ptrs,sizes):
        if not ptrs or len(ptrs)!=len(sizes) or any(type(x) is not int or x<=0 for x in [*ptrs,*sizes]):raise ValueError('Invalid native registration vectors')
        # Native MF1.2 batch API owns page alignment/coalescing. Logical cache
        # tensors may share a2MiB allocator page. Preserve all original logical
        # byte intervals for qualification hashing and all transfer addresses.
        ret=self._engine.batch_register_memory(ptrs,sizes)
        if ret!=0:raise RuntimeError(f'Strict SDMA batch register failed: {ret}')
        self._regions.extend(zip(ptrs,sizes))
        event('native_batch_registration',role=self.role,logical_device=self.device_id,logical_ranges=len(ptrs),logical_bytes=sum(sizes),SDK_calls=1,ret=ret)
        return ret

    def synchronize_producer_stream(self):
        if self.role!='Prefill':raise ValueError('Only producer may publish its completion fence')
        import torch
        if torch.npu.current_device()!=self.device_id:raise ValueError('Producer completion device differs from owned SDK')
        torch.npu.current_stream().synchronize()
        with self._lock:self._stats['producer_fences']+=1
        if not self._producer_seen:
            self._producer_seen=True;event('native_producer_completion_fence_seen',role=self.role,logical_device=self.device_id,rank_info=self.rank_info)

    def batch_transfer_sync_read(self,session_id,local,remote,lengths):
        if self.role!='Decode' or len(local)!=len(remote) or len(local)!=len(lengths) or not lengths or any(int(n)<=0 for n in lengths):raise ValueError('Invalid native consumer transfer vectors')
        host,sep,port=str(session_id).rpartition(':')
        if sep!=':' or host!='172.16.12.14' or not port.isdecimal() or not 0<int(port)<65536:raise ValueError('Outside owned Prefill SDK session')
        if self._acl.aclrtSetDevice(self.device_id)!=0:raise RuntimeError('Consumer ACL device setup failed')
        with self._lock:
            call_id=self._next_call_sequence;self._next_call_sequence+=1
        before=chunks=None
        if self.byte_audit:
            import urllib.request
            peer=json.loads(pathlib.Path('/run-evidence/sdma-peer-map.json').read_text())[str(session_id)]
            if type(peer['global_device_ordinal']) is not int or not 0<=peer['global_device_ordinal']<16:raise ValueError('Bad native producer peer ordinal')
            expected='http://172.16.12.14:'+str(18500+int(peer['global_device_ordinal']))+'/hash'
            if peer['url']!=expected:raise ValueError('Wrong qualification hash peer')
            def remote_hash(addresses,sizes):
                body=json.dumps({'addresses':addresses,'sizes':sizes}).encode()
                if len(body)>128*1024:raise ValueError('Qualification hash control body exceeds bound')
                q=urllib.request.Request(expected,data=body,headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(q,timeout=120) as response:return json.load(response)
            chunks=list(audit_chunks(local,remote,lengths))
            before=[remote_hash(rs,ns) for ls,rs,ns in chunks]
        with self._lock:self._stats['calls']+=1;self._stats['ranges']+=len(lengths);self._stats['bytes']+=sum(lengths)
        try:
            # ONE original native SDK read, unchanged vectors. Audit partition
            # never fragments payload, retries a transfer, or changes indices.
            ret=self._engine.batch_transfer_sync_read(session_id,local,remote,lengths)
            if ret!=0:raise RuntimeError(f'Strict Flash SDMA read failed: {ret}')
            if self.byte_audit:
                for index,((ls,rs,ns),pre) in enumerate(zip(chunks,before)):
                    own=self.hasher.hash_ranges(ls,ns);post=remote_hash(rs,ns)
                    passed=len(pre['range_sha256'])==len(own['range_sha256'])==len(post['range_sha256'])==len(ns) and pre['range_sha256']==own['range_sha256']==post['range_sha256'] and pre['sha256']==own['sha256']==post['sha256']
                    event('qualification_payload_checksum',role=self.role,logical_device=self.device_id,rank_info=self.rank_info,remote_session=str(session_id),call_id=call_id,audit_chunk=index,audit_chunks=len(chunks),native_ranges=len(lengths),native_bytes=sum(lengths),bytes=sum(ns),ranges=len(ns),passed=passed,local=ls,remote=rs,lengths=ns,range_sha256=own['range_sha256'])
                    if not passed:raise RuntimeError('Actual hybrid SDMA byte mismatch')
        except BaseException:
            with self._lock:self._stats['failed_calls']+=1
            raise
        with self._lock:self._stats['successful_calls']+=1;self._stats['successful_bytes']+=sum(lengths)
        return ret

    def emit_summary(self):
        with self._lock:stats=dict(self._stats)
        event('strict_SDMA_summary',role=self.role,logical_device=self.device_id,rank_info=self.rank_info,protocol=str(self.protocol),session_id=self.session_id,byte_audit=self.byte_audit,source_sha256=self.source_sha256,**stats)
        # Native worker/process owns the engine lifetime. Do not guess or call
        # an unqualified SDK1.2 destruction API or free buffers during transfers.


class GlobalFlashPerformanceTE:
    def __init__(self):
        self.transfer_engine=None;self.is_register_buffer=False;self.transfer_engine_lock=threading.Lock();self.register_buffer_lock=threading.Lock();self._binding=None

    def configure(self,role,device_id,rank_info):
        if role not in ('Prefill','Decode') or type(device_id) is not int or not 0<=device_id<16:raise ValueError('Invalid native role/device')
        r=dict(rank_info);tp,dp,ti,di=(r[x] for x in ['tp_size','dp_size','tp_rank','dp_rank'])
        if any(type(x) is not int for x in (tp,dp,ti,di,r['global_device_ordinal'])):raise ValueError('Native topology must be integral')
        if (role,tp,dp) not in [('Prefill',8,2),('Decode',1,16)]:raise ValueError('Only official P DP2/TP8 D DP16/TP1 candidate topology')
        if tp*dp!=16 or not 0<=ti<tp or not 0<=di<dp or r['global_device_ordinal']!=di*tp+ti:raise ValueError('Native topology outside one owned16-die node')
        binding=(role,device_id,r)
        with self.transfer_engine_lock:
            if self._binding is not None and binding!=self._binding:raise ValueError('SDK owner cannot change')
            self._binding=binding

    def get_transfer_engine(self,hostname,device_name=None):
        with self.transfer_engine_lock:
            if self._binding is None:raise ValueError('Native role/device binding missing')
            if self.transfer_engine is None:self.transfer_engine=FlashPerformanceSdmaEngine(hostname,*self._binding)
            return self.transfer_engine

    def register_buffer(self,ptrs,sizes):
        if len(ptrs)!=len(sizes):raise ValueError('Native registration vectors differ')
        with self.register_buffer_lock:
            if self.is_register_buffer:return
            if self.transfer_engine is None:raise ValueError('SDK initialization missing')
            self.transfer_engine.register_memories(list(ptrs),list(sizes))
            self.is_register_buffer=True

global_te=GlobalFlashPerformanceTE()
