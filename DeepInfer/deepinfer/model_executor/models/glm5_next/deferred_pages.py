# SPDX-License-Identifier: Apache-2.0
"""Explicit CPU shadow-page bindings for validated unpaged DSA transitions.

No native paged attention/allocator is implemented. Whole supplied CPU backing
buffers must be private to one active request/layer; sharing pages or native
stream work requires a different validated ownership/commit interface.
"""
from dataclasses import dataclass
from types import MappingProxyType

import torch

from .cache import DSAState,validate_dsa_state
from .cann_cache import view_main_index,view_tail,pack_tail_planes
from .paged_commit import stage_dsa_writes


@dataclass(frozen=True)
class CPUPageBinding:
    main_raw: torch.Tensor
    index_raw: torch.Tensor
    tail_raw: torch.Tensor
    geometry: object
    num_main_pages: int
    num_tail_pages: int
    logical_table: torch.Tensor
    tail_page_id: int

    def __post_init__(self):
        tensors=(self.main_raw,self.index_raw,self.tail_raw,self.logical_table)
        if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' for t in tensors):raise ValueError('deferred page binding is CPU-only')
        if self.geometry.layout.lookahead:raise ValueError('deferred CPU pages do not support speculation')
        view_main_index(self.main_raw,self.index_raw,self.num_main_pages,self.geometry)
        view_tail(self.tail_raw,self.num_tail_pages,self.geometry)
        if self.logical_table.ndim!=2 or self.logical_table.shape[0]!=1 or self.logical_table.dtype not in (torch.int32,torch.int64) or self.logical_table.numel()>4096:
            raise ValueError('one-request bounded integer logical table required')
        if bool(((self.logical_table < -1)|(self.logical_table>=self.num_main_pages)).any()):raise ValueError('logical page ID out of range')
        if type(self.tail_page_id) is not int or not -1<=self.tail_page_id<self.num_tail_pages:raise ValueError('tail page ID out of range')
        self._require_table()
        object.__setattr__(self,'logical_table',self.logical_table.clone())

    def _require_table(self):
        ids=self.logical_table
        if ids.ndim!=2 or ids.shape[0]!=1 or ids.numel()>4096 or ids.dtype not in (torch.int32,torch.int64):
            raise ValueError('one-request bounded integer logical table required')
        active=ids[ids>=0]
        if bool(((ids < -1)|(ids>=self.num_main_pages)).any()) or active.unique().numel()!=active.numel():
            raise ValueError('CPU binding logical pages must be in range and unique')

    def ranges(self):
        return tuple((t.data_ptr(),t.data_ptr()+t.numel()) for t in (self.main_raw,self.index_raw,self.tail_raw) if t.numel())

    @torch.inference_mode()
    def require_snapshot(self,state,*,max_history_tokens=4096):
        if not isinstance(state,DSAState) or any(t.device.type!='cpu' for t in (state.latent,state.compressed,state.tail,state.tail_positions)):
            raise ValueError('CPU page history requires a CPU DSA snapshot')
        self._require_table()
        layout=self.geometry.layout;validate_dsa_state(state,layout)
        if type(max_history_tokens) is not int or max_history_tokens<1 or state.tokens_seen>max_history_tokens:
            raise ValueError('CPU page history validation bound exceeded')
        main,index,_=view_main_index(self.main_raw,self.index_raw,self.num_main_pages,self.geometry)
        main=main.view(-1,layout.latent_dim);index=index.view(-1,layout.index_dim)
        slots=[];pool_slots=[];B=layout.logical_block_size;p=layout.pool_size
        for position in range(state.tokens_seen):
            if position//B>=self.logical_table.shape[1]:raise ValueError('CPU page history has no logical page')
            page=int(self.logical_table[0,position//B])
            if page<0:raise ValueError('CPU page history has an unassigned page')
            slots.append(page*B+position%B)
            if (position+1)%p==0:pool_slots.append(page*(B//p)+(position%B)//p)
        if not torch.equal(main[slots],state.latent) or not torch.equal(index[pool_slots],state.compressed):
            raise ValueError('CPU page history disagrees with request snapshot')
        if state.tokens_seen%p:
            if self.tail_page_id<0:raise ValueError('partial pool requires a private tail page')
            tail=view_tail(self.tail_raw,self.num_tail_pages,self.geometry)[self.tail_page_id]
            for position in range(state.tokens_seen//p*p,state.tokens_seen):
                slot=position%p;expected=torch.cat((state.tail[0,slot],state.tail[1,slot]))
                if not torch.equal(tail[slot],expected):raise ValueError('CPU page tail disagrees with request snapshot')

    @torch.inference_mode()
    def stage(self,previous,following):
        layout=self.geometry.layout
        for state in (previous,following):
            if not isinstance(state,DSAState) or any(t.device.type!='cpu' for t in (state.latent,state.compressed,state.tail,state.tail_positions)):
                raise ValueError('deferred pages require explicit CPU DSA snapshots')
            validate_dsa_state(state,layout)
        self.require_snapshot(previous)
        start,stop=previous.tokens_seen,following.tokens_seen
        if stop<start or stop-start>128:raise ValueError('DSA page transition cannot rewind or exceed128 new tokens')
        if not torch.equal(previous.latent,following.latent[:start]) or not torch.equal(previous.compressed,following.compressed[:start//layout.pool_size]):
            raise ValueError('DSA page transition changed committed history')
        p=layout.pool_size;count=stop-start;width=2*layout.index_dim
        scratch=following.tail.new_zeros((3,p,width))
        scratch[2 if start%p+count>=p else 1]=pack_tail_planes(following.tail)
        return stage_dsa_writes(self.main_raw,self.index_raw,self.tail_raw,self.geometry,
            num_main_pages=self.num_main_pages,num_tail_pages=self.num_tail_pages,
            logical_table=self.logical_table,starts=torch.tensor([start]),lengths=torch.tensor([count]),
            tail_page_ids=torch.tensor([self.tail_page_id]),latent=following.latent[start:],
            compressed=following.compressed[start//p:],tail_state=scratch)


def capture_page_bindings(config,layout,bindings,existing=(),*,state=None):
    """Reserve whole CPU backings for this local request; no allocator claim."""
    expected={s.layer_id for s in config.layers if s.attention=='deepseek_sparse_attention'}
    if not isinstance(bindings,dict) or set(bindings)!=expected or any(type(b) is not CPUPageBinding or b.geometry.layout!=layout for b in bindings.values()):
        raise ValueError('explicit CPU page bindings must cover every DSA layer exactly')
    intervals=[interval for b in bindings.values() for interval in b.ranges()]
    # Existing owners were already validated against one another.
    occupied=[interval for owner in existing for b in owner.values() for interval in b.ranges()]
    all_intervals=sorted(intervals+occupied)
    if any(b>c for (a,b),(c,d) in zip(all_intervals,all_intervals[1:])):
        raise ValueError('active CPU page bindings must have exclusive nonoverlapping backing')
    if state is not None:
        for layer,binding in bindings.items():binding.require_snapshot(state.layers[layer])
    return MappingProxyType(dict(bindings))
