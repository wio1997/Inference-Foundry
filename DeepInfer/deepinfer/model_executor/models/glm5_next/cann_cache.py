# SPDX-License-Identifier: Apache-2.0
"""Explicit GLM CANN cache views and non-speculative tail metadata preparation.

No CANN/Triton import, allocation of a serving cache, operator call, resource
claim or scheduler integration. Main/index backing currently requires dense
logical pages. Tail pages may be padded independently; their physical stride
must be carried into every history/write address. A layout with logical block16
is valid for the unpaged reference but cannot bind this kernel128 interface.
"""
from dataclasses import dataclass

import torch

from .cache import DSACacheLayout


def _integer(value,name,minimum=0):
    if type(value) is not int or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


@dataclass(frozen=True)
class CANNCacheGeometry:
    layout: DSACacheLayout
    tail_page_bytes: int
    kernel_block_size: int = 128

    def __post_init__(self):
        if type(self.kernel_block_size) is not int or self.kernel_block_size!=128 or self.layout.logical_block_size%128 or 128%self.layout.pool_size:
            raise ValueError('GLM CANN requires kernel128 and logical blocks divisible by128')
        _integer(self.tail_page_bytes,'tail_page_bytes',1)
        row_bytes=2*self.layout.index_dim*4
        if self.tail_page_bytes%row_bytes or self.tail_page_bytes<self.layout.tail_capacity*row_bytes:
            raise ValueError('tail page must fit capacity and align to complete FP32 K/gate rows')

    @property
    def main_page_bytes(self):return self.layout.logical_block_size*self.layout.latent_dim*2
    @property
    def index_page_bytes(self):return self.layout.compressed_block_size*self.layout.index_dim*2
    @property
    def tail_page_rows(self):return self.tail_page_bytes//(2*self.layout.index_dim*4)
    @property
    def kernel_blocks_per_logical(self):return self.layout.logical_block_size//128
    @property
    def index_kernel_rows(self):return 128//self.layout.pool_size


def _raw(raw,pages,page_bytes,name):
    _integer(pages,'pages')
    if raw.dtype!=torch.uint8 or raw.ndim!=1 or not raw.is_contiguous() or raw.storage_offset()!=0:
        raise ValueError(f'{name} backing must be a contiguous zero-offset UINT8 vector')
    if raw.numel()!=pages*page_bytes:
        raise ValueError(f'{name} backing bytes must exactly match explicit dense page geometry')


def view_main_index(main_raw,index_raw,pages,geometry):
    """Aliased kernel views. Padding of main/index pages is deliberately rejected.

    Never infer extra logical token rows from page padding. A production padded
    allocator needs a separately validated kernel/stride contract.
    """
    _raw(main_raw,pages,geometry.main_page_bytes,'main')
    _raw(index_raw,pages,geometry.index_page_bytes,'index')
    if main_raw.device!=index_raw.device:raise ValueError('main/index backing devices must agree')
    blocks=pages*geometry.kernel_blocks_per_logical
    main=main_raw.view(torch.bfloat16).view(blocks,128,1,geometry.layout.latent_dim)
    index=index_raw.view(torch.bfloat16).view(blocks,geometry.index_kernel_rows,1,geometry.layout.index_dim)
    rope=torch.as_strided(main,size=(*main.shape[:-1],0),stride=main.stride())
    return main,index,rope


def view_tail(raw,pages,geometry):
    _raw(raw,pages,geometry.tail_page_bytes,'tail')
    width=2*geometry.layout.index_dim
    return torch.as_strided(raw.view(torch.float32),
        size=(pages,geometry.layout.tail_capacity,width),
        stride=(geometry.tail_page_rows*width,width,1))


def pack_tail_planes(planes):
    if planes.ndim!=3 or planes.shape[0]!=2 or planes.dtype!=torch.float32:
        raise ValueError('unpaged tail requires FP32 [2,capacity,index_dim]')
    return torch.cat((planes[0],planes[1]),dim=-1)


def unpack_tail_rows(rows):
    if rows.ndim!=2 or rows.shape[-1]%2 or rows.dtype!=torch.float32:
        raise ValueError('CANN tail payload requires FP32 [capacity,2*index_dim]')
    width=rows.shape[-1]//2
    return torch.stack((rows[:,:width],rows[:,width:]))


def expand_kernel_block_table(logical_ids,geometry,*,num_logical_pages):
    _integer(num_logical_pages,'num_logical_pages')
    if logical_ids.ndim!=2 or logical_ids.dtype not in (torch.int32,torch.int64):
        raise ValueError('logical block table must be an integer matrix')
    if bool(((logical_ids < -1)|(logical_ids>=num_logical_pages)).any()):
        raise ValueError('logical block table contains out-of-range IDs')
    factor=geometry.kernel_blocks_per_logical
    expanded=logical_ids.long()[...,None]*factor+torch.arange(factor,device=logical_ids.device)
    expanded=torch.where(logical_ids[...,None]>=0,expanded,-1)
    if expanded.numel() and expanded.max().item()>torch.iinfo(torch.int32).max:raise ValueError('expanded kernel ID exceeds INT32')
    return expanded.reshape(logical_ids.shape[0],logical_ids.shape[1]*factor).int()


@dataclass(frozen=True)
class TailPlan:
    history_indices: torch.Tensor  # [requests,pool], INT64 physical row IDs
    write_indices: torch.Tensor  # [requests,pool], -1 marks no write
    state_indices: torch.Tensor  # [requests,pool], flattened CANN state rows
    start_offsets: torch.Tensor  # [requests], INT32 within logical pool
    state_table: torch.Tensor  # initial/history then per-request scratch IDs
    page_rows: int


def plan_tail(starts,lengths,page_ids,geometry,*,num_tail_pages):
    """CPU-light non-speculative planner, mirroring installed metadata equations.

    One request owns one independent tail page; no MTP retained window, PCP/DCP,
    graph buffers or prefix reuse is implemented. Complete-pool shifts align
    history to the beginning of the current pool, not to the last pool tokens.
    Caller must supply already validated request history (including position
    tags when importing an unpaged state) and coordinate actual cache writes.
    """
    _integer(num_tail_pages,'num_tail_pages')
    if starts.ndim!=1 or lengths.shape!=starts.shape or page_ids.shape!=starts.shape:
        raise ValueError('tail starts/lengths/page IDs must be matching vectors')
    if any(t.dtype not in (torch.int32,torch.int64) or t.device!=starts.device for t in (starts,lengths,page_ids)):
        raise ValueError('tail metadata must be same-device integer vectors')
    if bool(((starts<0)|(lengths<0)).any()) or bool(((page_ids < -1)|(page_ids>=num_tail_pages)).any()):
        raise ValueError('invalid tail token counts or page IDs')
    if bool(((lengths>0)&(page_ids<0)).any()):raise ValueError('active requests require an owned tail page')
    assigned=page_ids[page_ids>=0]
    if assigned.unique().numel()!=assigned.numel():raise ValueError('requests cannot share tail pages')
    n=starts.numel();pool=geometry.layout.pool_size;capacity=geometry.layout.tail_capacity
    starts=starts.long();lengths=lengths.long();pages=page_ids.long();cols=torch.arange(pool,device=starts.device)
    history=pages.clamp_min(0)[:,None]*geometry.tail_page_rows+((starts-starts%pool)[:,None]+cols)%capacity
    end=starts%pool+lengths
    request_rows=torch.arange(1,n+1,device=starts.device)
    state=(request_rows+torch.where(end>=pool,n,0))[:,None]*pool+cols
    pos=(starts+lengths-end%pool)[:,None]+cols
    writes=pages[:,None]*geometry.tail_page_rows+pos%capacity
    writes=torch.where((lengths[:,None]>0)&(cols<(end%pool)[:,None])&(pages[:,None]>=0),writes,-1)
    max_length=int(lengths.max()) if n else 0
    table=(request_rows+n)[:,None].expand(n,1+(max_length+pool-1)//pool).clone().int()
    if n:table[:,0]=request_rows.int()
    return TailPlan(history,writes,state,(starts%pool).int(),table,geometry.tail_page_rows)


def prepare_tail_state(tail,plan,pool_size):
    """Private history snapshot; reserved/scratch rows start at deterministic0.

    Installed preparation leaves scratch uninitialized; only copied history
    rows are comparable before the native operator. This function does not run
    that operator or populate its output scratch.
    """
    _integer(pool_size,'pool_size',2)
    if tail.ndim!=3:raise ValueError('tail must be a three-dimensional view')
    width=tail.shape[-1]
    if tail.dtype!=torch.float32 or tail.stride(2)!=1 or tail.stride(1)!=width or tail.stride(0)!=plan.page_rows*width:
        raise ValueError('tail view/metadata must preserve the physical page row stride')
    if plan.history_indices.ndim!=2 or plan.history_indices.shape[1]!=pool_size or plan.history_indices.dtype!=torch.int64 or plan.history_indices.device!=tail.device:
        raise ValueError('invalid history index matrix')
    n=plan.history_indices.shape[0]
    result=tail.new_zeros((1+2*n,pool_size,width))
    if not n:return result
    if not tail.shape[0]:raise ValueError('nonempty history needs backing tail pages')
    rows=(tail.shape[0]-1)*plan.page_rows+tail.shape[1]
    indices=plan.history_indices.flatten()
    if bool(((indices<0)|(indices>=rows)).any()):raise ValueError('history row outside tail backing')
    # An index into page padding is never a valid raw tail history entry.
    if bool(((indices%plan.page_rows)>=tail.shape[1]).any()):raise ValueError('history row points into page padding')
    flat=torch.as_strided(tail,size=(rows,width),stride=(width,1))
    result[1:n+1]=torch.index_select(flat,0,indices).reshape(n,pool_size,width)
    return result
