# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU-only staged GLM CANN page-write correctness interface.

No native operator, attention, allocator or resource claim. Request integration
is restricted to explicit synchronous CPU write batches.
Callers must exclusively own all affected pages and complete staging producers
before use. CPU rollback is synchronous; it establishes no asynchronous device
transaction guarantee. This is not a production cache-copy implementation.
"""
import torch

from .cann_cache import view_main_index,view_tail,plan_tail


def _signature(t):
    return (t.data_ptr(),t.shape,t.stride(),t.storage_offset(),t.untyped_storage().nbytes())


def _write_rows(target,indices,values):
    target.index_copy_(0,indices,values)


class _RowWrite:
    def __init__(self,target,indices,values):
        if indices.ndim!=1 or indices.dtype!=torch.int64 or indices.device.type!='cpu':raise ValueError('CPU INT64 destination rows required')
        if values.shape!=(len(indices),target.shape[1]) or values.dtype!=target.dtype or values.device!=target.device:raise ValueError('cache write payload geometry/dtype/device mismatch')
        if bool(((indices<0)|(indices>=len(target))).any()) or indices.unique().numel()!=indices.numel():raise ValueError('destination rows must be unique and in range')
        if not bool(torch.isfinite(values).all()):raise ValueError('cache write payload must be finite')
        self.target,self.indices,self.after=target,indices.clone(),values.clone()
        self.before=target.index_select(0,indices).clone();self.signature=_signature(target)

    def require_unchanged(self):
        if _signature(self.target)!=self.signature:raise ValueError('cache backing changed after staging')
        # Old unused cache rows can contain any bit pattern, including NaNs.
        current=self.target.index_select(0,self.indices).contiguous().view(torch.uint8)
        if not torch.equal(current,self.before.contiguous().view(torch.uint8)):
            raise ValueError('cache destination changed after staging')


class DSAWriteBatch:
    """Own private before/after rows. commit/abort are terminal and single-use.

    Validates all destinations before the first write, then restores attempted
    rows if a synchronous CPU write fails. No other writer may run concurrently.
    Unrelated pages are not cloned or restored; metadata token commits and native
    scratch side effects remain the caller's separate responsibility.
    """
    def __init__(self,writes):
        self._writes=tuple(writes);self._state='staged'

    def summary(self):
        return {'state':self._state,'scope':'CPU_staged_page_writes_only',
                'rows':{name:len(write.indices) for name,write in zip(('main','index','tail'),self._writes)},
                'payload_bytes':sum(w.after.numel()*w.after.element_size() for w in self._writes),
                'native_execution_validated':False}

    def commit(self):
        return commit_dsa_batches((self,))[0]

    def abort(self):
        if self._state!='staged':raise ValueError('page write batch is already terminal')
        self._state='aborted'


@torch.inference_mode()
def commit_dsa_batches(batches,*,max_batches=11,max_payload_bytes=16*1024*1024):
    """Precheck all CPU layer rows, then write/rollback the entire group.

    Callers hold exclusive process/page ownership through metadata publication.
    Empty groups are no-ops. This adds no native async/crash atomicity guarantee.
    """
    if any(type(v) is not int or v<1 for v in (max_batches,max_payload_bytes)):
        raise ValueError('positive explicit layer/payload commit bounds required')
    if not isinstance(batches,tuple) or len(batches)>max_batches or any(type(b) is not DSAWriteBatch for b in batches):
        raise ValueError('bounded explicit tuple of CPU DSA write batches required')
    if len({id(b) for b in batches})!=len(batches) or any(b._state!='staged' for b in batches):
        raise ValueError('page batches must be unique and nonterminal')
    writes=tuple(w for b in batches for w in b._writes)
    if sum(w.after.numel()*w.after.element_size() for w in writes)>max_payload_bytes:
        raise ValueError('cross-layer payload byte bound exceeded')
    summaries=tuple(dict(b.summary(),state='committed') for b in batches)
    attempted=[]
    try:
        intervals=[]
        for w in writes:
            if any(t.device.type!='cpu' for t in (w.target,w.indices,w.before,w.after)):
                raise ValueError('cross-layer page commit is CPU-only')
            w.require_unchanged()
            row_bytes=w.target.shape[1]*w.target.element_size()
            for row in w.indices.tolist():
                start=w.target.data_ptr()+row*w.target.stride(0)*w.target.element_size()
                intervals.append((start,start+row_bytes))
        intervals.sort()
        if any(b>c for (a,b),(c,d) in zip(intervals,intervals[1:])):
            raise ValueError('cross-layer write destinations overlap')
        for w in writes:
            if len(w.indices):attempted.append(w);_write_rows(w.target,w.indices,w.after)
    except Exception:
        for w in reversed(attempted):w.target.index_copy_(0,w.indices,w.before)
        for b in batches:b._state='aborted'
        raise
    for b in batches:b._state='committed'
    return summaries


@torch.inference_mode()
def stage_dsa_writes(main_raw,index_raw,tail_raw,geometry,*,num_main_pages,num_tail_pages,
                     logical_table,starts,lengths,tail_page_ids,latent,compressed,tail_state,
                     max_tokens=128,max_requests=8,max_table_entries=4096,max_payload_bytes=2*1024*1024):
    """Stage supplied outputs, preserving CANN physical page/tail strides.

    latent is packed in request/token order; compressed contains only rows whose
    positions end a complete pool, in the same order. tail_state is the completed
    CANN preparation/operator scratch [1+2*requests,pool,2*index_dim]. Only the
    final incomplete-pool rows are copied. No tail compression is computed here.
    Main/index logical page IDs share a namespace; tail IDs are independent.
    Full allocator ownership/prefix sharing checks are outside this interface.
    """
    tensors=(main_raw,index_raw,tail_raw,logical_table,starts,lengths,tail_page_ids,latent,compressed,tail_state)
    if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' for t in tensors):raise ValueError('staged cache commit is CPU-only; native async path is unverified')
    if any(type(v) is not int or v<1 for v in (max_tokens,max_requests,max_table_entries,max_payload_bytes)):raise ValueError('positive explicit staging bounds required')
    if geometry.layout.lookahead!=0:raise ValueError('staged commit supports non-speculative tail only')
    if starts.ndim!=1 or lengths.shape!=starts.shape or tail_page_ids.shape!=starts.shape:raise ValueError('matching request metadata vectors required')
    if any(t.dtype not in (torch.int32,torch.int64) for t in (starts,lengths,tail_page_ids,logical_table)):raise ValueError('integer page/request metadata required')
    n=len(starts)
    if n>max_requests or logical_table.ndim!=2 or logical_table.shape[0]!=n or logical_table.numel()>max_table_entries:raise ValueError('request/table staging bound or geometry exceeded')
    counts=lengths.tolist();offsets=starts.tolist()
    if any(s<0 or count<0 or s+count>2**31-1 for s,count in zip(offsets,counts)):raise ValueError('invalid or overflowing INT32 token positions')
    if sum(counts)>max_tokens:raise ValueError('token staging bound exceeded')
    main,index,_=view_main_index(main_raw,index_raw,num_main_pages,geometry);tail=view_tail(tail_raw,num_tail_pages,geometry)
    if bool(((logical_table < -1)|(logical_table>=num_main_pages)).any()):raise ValueError('logical table page ID out of range')
    plan=plan_tail(starts,lengths,tail_page_ids,geometry,num_tail_pages=num_tail_pages)
    block,pool=geometry.layout.logical_block_size,geometry.layout.pool_size
    main_rows=[];index_rows=[];owners={}
    for request,(start,count) in enumerate(zip(offsets,counts)):
        for position in range(start,start+count):
            logical=position//block
            if logical>=logical_table.shape[1]:raise ValueError('logical table does not cover all new positions')
            page=int(logical_table[request,logical])
            if page<0:raise ValueError('new position requires an owned main/index page')
            owner=(request,logical)
            if page in owners and owners[page]!=owner:raise ValueError('active requests/logical blocks cannot alias a destination page')
            owners[page]=owner;slot=page*block+position%block;main_rows.append(slot)
            if (position+1)%pool==0:index_rows.append(page*(block//pool)+(position%block)//pool)
    main_ids=torch.tensor(main_rows,dtype=torch.int64);index_ids=torch.tensor(index_rows,dtype=torch.int64)
    width=2*geometry.layout.index_dim
    if tail_state.shape!=(1+2*n,pool,width) or tail_state.dtype!=torch.float32:raise ValueError('completed tail scratch geometry/dtype mismatch')
    valid=plan.write_indices>=0;tail_ids=plan.write_indices[valid]
    # Padding is physical storage, never an eligible tail destination.
    if bool(((tail_ids%geometry.tail_page_rows)>=geometry.layout.tail_capacity).any()):raise ValueError('tail destination points into page padding')
    tail_rows=max(0,(num_tail_pages-1)*geometry.tail_page_rows+geometry.layout.tail_capacity) if num_tail_pages else 0
    tail_flat=torch.as_strided(tail,size=(tail_rows,width),stride=(width,1))
    payload_bytes=latent.numel()*latent.element_size()+compressed.numel()*compressed.element_size()+len(tail_ids)*width*4
    if payload_bytes>max_payload_bytes:raise ValueError('cache payload staging byte bound exceeded')
    values=tail_state.reshape(-1,width).index_select(0,plan.state_indices[valid])
    targets=(main.view(-1,geometry.layout.latent_dim),index.view(-1,geometry.layout.index_dim),tail_flat)
    # Reject overlapping raw buffers, even if their particular row ranges differ.
    intervals=[(t.data_ptr(),t.data_ptr()+t.numel()) for t in (main_raw,index_raw,tail_raw) if t.numel()]
    if any(max(a,c)<min(b,d) for i,(a,b) in enumerate(intervals) for c,d in intervals[i+1:]):raise ValueError('main/index/tail raw backing must not overlap')
    writes=(_RowWrite(targets[0],main_ids,latent),_RowWrite(targets[1],index_ids,compressed),_RowWrite(targets[2],tail_ids,values))
    return DSAWriteBatch(writes)
