# SPDX-License-Identifier: Apache-2.0
# The pointer-table gather layout follows vLLM Ascend's Apache-2.0
# vllm_ascend/ops/triton/engram_lookup.py. This implementation keeps signed
# INT8 codes and raw E8M0 scales independent, matching this ModelSlim checkpoint.
"""Read-only checkpoint pages registered for direct NPU Engram lookup.

MAP_PRIVATE mappings are never written. Clean file pages are shared by DP
processes by the OS; no named shared-memory objects or expanded tables are
created. Only head shards are registered, and all device uses must finish
before close unregisters the mapping.
"""
import ctypes
from functools import cache
import mmap
import os

import torch
import torch.distributed as dist
import triton
import triton.language as tl

CHUNK_ROWS = 1 << 22
ACL_HOST_REG_MAPPED = 0x2
ACL_HOST_REG_PINNED = 0x10000000


@cache
def _library():
    lib = ctypes.CDLL('libascendcl.so')
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


class FileUvaRows:
    """Own a registered view of a contiguous byte-preserving table shard."""
    def __init__(self, path, tensor_offset, first, last, width, device):
        if not 0 <= first < last or width < 1:
            raise ValueError('expected a nonempty contiguous row shard')
        self.rows, self.width = last-first, width
        self.registered = False
        self.mapping = None
        self.ptrs = None
        self.device = device
        self.lib = _library()
        start = tensor_offset+first*width
        self.bytes = self.rows*width
        aligned = start//mmap.PAGESIZE*mmap.PAGESIZE
        delta = start-aligned
        self.tensor_delta = delta
        # PROT_WRITE is needed to expose the buffer address to ctypes. No
        # consumer writes; MAP_PRIVATE additionally protects checkpoint files.
        with open(path, 'rb') as handle:
            self.mapping = mmap.mmap(handle.fileno(), delta+self.bytes,
                flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=aligned)
        self._register(delta)

    def _register(self, delta):
        self.pointer = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(self.mapping)))
        try:
            rc = self.lib.aclrtHostRegisterV2(self.pointer, len(self.mapping),
                ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED)
            if rc:
                raise RuntimeError(f'aclrtHostRegisterV2 mapping failed: rc={rc} bytes={len(self.mapping)}')
            self.registered = True
            address = ctypes.c_void_p()
            rc = self.lib.aclrtHostGetDevicePointer(self.pointer, ctypes.byref(address), 0)
            if rc:
                raise RuntimeError(f'aclrtHostGetDevicePointer failed: rc={rc}')
            self.ptrs = torch.tensor([address.value+delta+row*self.width
                for row in range(0, self.rows, CHUNK_ROWS)],dtype=torch.int64,device=self.device)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.registered:
            if self.ptrs is not None:
                torch.npu.synchronize(self.device)
            rc = self.lib.aclrtHostUnregister(self.pointer)
            if rc:
                # Retain the mapping if CANN still owns the registration.
                raise RuntimeError(f'aclrtHostUnregister failed: rc={rc}')
            self.registered = False
        self.ptrs = None
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if getattr(self,'fd',None) is not None:
            os.close(self.fd)
            self.fd = None


class SharedMemfdRows(FileUvaRows):
    """One anonymous shared host shard for a local DP pair, no /dev/shm quota.

The leader streams original checkpoint bytes into a memfd. The peer opens
that exact live descriptor through the shared container's /proc namespace.
Both map MAP_SHARED and register their own address; CANN write-pinning then
retains the same physical pages instead of making MAP_PRIVATE COW copies.
"""
    def __init__(self,path,tensor_offset,first,last,width,device,group):
        if not 0<=first<last or width<1:
            raise ValueError('expected a nonempty contiguous row shard')
        self.rows,self.width=last-first,width
        self.bytes=self.rows*width
        self.tensor_delta=0
        self.registered=False
        self.mapping=self.ptrs=self.fd=None
        self.device,self.lib=device,_library()
        leader=dist.get_global_rank(group,0)
        payload=[None]
        if dist.get_rank(group)==0:
            try:
                self.fd=os.memfd_create('wio-engram-shard',os.MFD_CLOEXEC)
                os.ftruncate(self.fd,self.bytes)
                self.mapping=mmap.mmap(self.fd,self.bytes,flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ|mmap.PROT_WRITE)
                with open(path,'rb',buffering=0) as source:
                    source.seek(tensor_offset+first*width)
                    done=0
                    while done<self.bytes:
                        view=memoryview(self.mapping)[done:min(done+8*1024**2,self.bytes)]
                        try:read=source.readinto(view)
                        finally:view.release()
                        if not read:raise EOFError('incomplete checkpoint shard')
                        done+=read
                payload[0]={'fd_path':f'/proc/{os.getpid()}/fd/{self.fd}'}
            except BaseException as exc:
                payload[0]={'error':f'{type(exc).__name__}: {exc}'}
                self.close()
        dist.broadcast_object_list(payload,src=leader,group=group)
        if 'error' in payload[0]:
            raise RuntimeError('Engram shared host fill failed: '+payload[0]['error'])
        error=None
        try:
            if self.fd is None:
                self.fd=os.open(payload[0]['fd_path'],os.O_RDWR)
                self.mapping=mmap.mmap(self.fd,self.bytes,flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ|mmap.PROT_WRITE)
            self._register(0)
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}'
        errors=[None]*dist.get_world_size(group)
        dist.all_gather_object(errors,error,group=group)
        if any(x is not None for x in errors):
            self.close()
            raise RuntimeError('Engram shared host registration failed: '+repr(errors))


@triton.jit
def _e8m0(scale):
    exponent = scale.to(tl.int32) & 0xFF
    bits = exponent << 23
    bits = tl.where(exponent == 0, 0x00400000, bits)
    bits = tl.where(exponent == 255, 0x7FC00000, bits)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _lookup(codes_ptrs, scales_ptrs, ids, output,
            rows, vocab_start, vocab_end, ids_stride,
            HEAD_START: tl.constexpr, HEADS: tl.constexpr,
            CHUNK: tl.constexpr, WIDTH: tl.constexpr, GROUP: tl.constexpr):
    row = tl.program_id(0)
    if row < rows:
        token = row//HEADS
        head = row%HEADS
        index = tl.load(ids+token*ids_stride+HEAD_START+head).to(tl.int64)
        owned = (index >= vocab_start) & (index < vocab_end)
        local_row = tl.where(owned, index-vocab_start, 0)
        chunk = local_row//CHUNK
        offset = local_row%CHUNK
        codes = tl.load(codes_ptrs+chunk).to(tl.pointer_type(tl.int8))
        scales = tl.load(scales_ptrs+chunk).to(tl.pointer_type(tl.uint8))
        col = tl.arange(0, WIDTH)
        group = tl.arange(0, WIDTH//GROUP)
        value = tl.load(codes+offset*WIDTH+col).to(tl.float32)
        scale = _e8m0(tl.load(scales+offset*(WIDTH//GROUP)+group))
        value = tl.reshape(value,(WIDTH//GROUP,GROUP))*scale[:,None]
        value = tl.reshape(value,(WIDTH,)).to(tl.bfloat16)
        tl.store(output+row*WIDTH+col,tl.where(owned,value,tl.zeros_like(value)))


def lookup(codes, scales, ids, *, vocab_start, vocab_end,
           head_start=0, heads=24, output=None):
    """Return [tokens, owned_heads, 256] exact BF16 rows on the NPU."""
    if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('expected device integer [tokens, hash_heads] indices')
    if codes.width != 256 or scales.width != 8 or codes.rows != scales.rows:
        raise ValueError('expected signed INT8 width256 with group32 E8M0 scales')
    if vocab_end-vocab_start != codes.rows or head_start < 0 or heads < 1 or head_start+heads > ids.shape[1]:
        raise ValueError('invalid owned vocab or head range')
    shape=(ids.shape[0],heads,256)
    if output is None:
        output=torch.empty(shape,dtype=torch.bfloat16,device=ids.device)
    if output.shape != shape or output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise ValueError('expected contiguous BF16 output of the exact lookup shape')
    if ids.shape[0]:
        _lookup[(ids.shape[0]*heads,)](codes.ptrs,scales.ptrs,ids,output,
            ids.shape[0]*heads,vocab_start,vocab_end,ids.stride(0),
            HEAD_START=head_start,HEADS=heads,CHUNK=CHUNK_ROWS,WIDTH=256,GROUP=32,num_warps=4)
    return output
