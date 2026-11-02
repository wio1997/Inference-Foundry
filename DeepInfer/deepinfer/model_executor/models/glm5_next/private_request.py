# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU single-layer DSA prepare/validate/publish correctness interface.

No live allocator, native cache installation, resource lease or device event.
Each candidate owns full bounded CPU backing clones, so preparing/abandoning it
does not touch committed bytes. Publication replaces one in-process frame under
a nonblocking request lock. This is not whole-decoder/text atomicity, durable
recovery, async rollback or a production allocation/memory strategy. Arbitrary
callback side effects remain outside this protocol.
"""
from contextlib import contextmanager
from dataclasses import dataclass,field
from threading import Lock
import torch
from .cache import DSAState,validate_dsa_state
from .cann_cache import pack_tail_planes
from .deferred_pages import CPUPageBinding
from .private_pages import plan_private_dsa_pages,prepare_private_dsa_cpu,gather_private_latent_cpu,gather_private_index_cpu


@dataclass(frozen=True)
class DSARequestStamp:
    request_id: str
    revision: int
    tokens_seen: int
    logical_table: tuple[int,...]
    tail_page_id: int
    _owner: object = field(repr=False)


@dataclass(frozen=True)
class PrivateDSASnapshot:
    stamp: DSARequestStamp
    state: DSAState
    main_raw: torch.Tensor
    index_raw: torch.Tensor
    tail_raw: torch.Tensor


class CandidateDSAReadView:
    """CPU logical-token access only; gather returns private result buffers."""
    def __init__(self,base,stage,geometry):
        self._base,self._stage,self._geometry=base,stage,geometry

    @property
    def positions(self):return tuple(range(self._stage.plan.start,self._stage.plan.end))
    @property
    def logical_table(self):return self._stage.plan.read_table
    @property
    def tokens_seen(self):return self._stage.plan.end

    @torch.inference_mode()
    def gather(self,token_ids):
        return gather_private_latent_cpu(self._base.main_raw,self._stage,self._geometry,token_ids)


    def require_sfa_plan_cpu(self,plan,*,expected_identity):
        """Bind a composite attempt plan to this exact staged CPU candidate.

        The outer owner checks attempt/revision. This view additionally checks
        the actual request and complete staged page plan, not just tensor shapes.
        No raw backing or mutable stage object is exposed by this API.
        """
        from .sfa_inputs import NoPESFAInputPlan,SFAInputIdentity
        if type(plan) is not NoPESFAInputPlan or type(expected_identity) is not SFAInputIdentity or expected_identity!=plan.identity or plan.request_id!=self._base.stamp.request_id or plan.candidate!=self._stage.plan:
            raise ValueError('stale/foreign projected candidate plan/identity')

    @torch.inference_mode()
    def execute_rank_dsa_cpu(self,query_qk,indices,*,plan,expected_identity,**callbacks_and_bounds):
        """Use the explicit CPU chain on private packet copies; no raw export."""
        self.require_sfa_plan_cpu(plan,expected_identity=expected_identity)
        from .rank_dsa import execute_rank_dsa_cpu
        return execute_rank_dsa_cpu(query_qk,self._base.main_raw,self._stage,indices,plan=plan,expected_identity=expected_identity,**callbacks_and_bounds)

    @property
    def layout(self):return self._geometry.layout

    @torch.inference_mode()
    def gather_compressed(self,pool_ids,**bounds):
        """Copy logical completed-pool rows; this does not perform selection."""
        return gather_private_index_cpu(self._base.index_raw,self._stage,self._geometry,pool_ids,**bounds)

    @torch.inference_mode()
    def select_pools(self,query,head_weights,*,max_history_pools=4096,max_score_elements=262144):
        """Explicit CPU selection with per-query visibility and causal raw tail.

        All completed candidate pools may be present, including later pools in
        this append. Query positions are fixed to this staged step; the existing
        reference masks future pools and deterministically breaks CPU ties.
        """
        from .sparse_reference import select_pools_reference
        n=self.tokens_seen//self.layout.pool_size
        if type(max_history_pools) is not int or max_history_pools<1 or n>max_history_pools:raise ValueError('completed pool history bound exceeded')
        compressed=self.gather_compressed(torch.arange(n,dtype=torch.int64),max_selected=max_history_pools)
        return select_pools_reference(query,head_weights,compressed,torch.tensor(self.positions,dtype=torch.int64),self.layout,max_score_elements=max_score_elements)


class PreparedDSABranch:
    """Owner-local handle. Public output is copied; one publication attempt."""
    def __init__(self,base_stamp,candidate,output):
        self._base_stamp,self._candidate,self._output=base_stamp,candidate,output
        self._status='prepared'

    @property
    def base_stamp(self):return self._base_stamp
    @property
    def candidate_stamp(self):return self._candidate.stamp
    @property
    def status(self):return self._status

    @property
    @torch.inference_mode()
    def output(self):return self._output.clone()


class PrivateDSARequest:
    """One CPU DSA layer, explicit completed producers and attention callback.

    Tables/occupied sets describe CPU fixtures, not actual native reservations.
    Source backings are cloned on open, candidate preparation and publication;
    max_backing_bytes bounds each frame, not process peak. Old snapshots and
    caller-owned buffers remain stable. Native OOT/global context is untouched.
    """
    @torch.inference_mode()
    def __init__(self,request_id,state,main_raw,index_raw,tail_raw,logical_table,tail_page_id,*,geometry,
                 num_main_pages,num_tail_pages,output_width,max_context_tokens=4096,max_backing_bytes=16*1024**2):
        if not isinstance(request_id,str) or not request_id:raise ValueError('nonempty request ID required')
        if not isinstance(state,DSAState):raise ValueError('explicit DSA state required')
        for n in (output_width,max_context_tokens,max_backing_bytes):
            if type(n) is not int or n<1:raise ValueError('positive explicit request/output/backing budgets required')
        if not isinstance(logical_table,tuple):raise ValueError('immutable logical table required')
        if geometry.layout.lookahead:raise ValueError('prepared request excludes speculative/lookahead state')
        self.geometry,self.num_main_pages,self.num_tail_pages=geometry,num_main_pages,num_tail_pages
        self.output_width,self.max_context_tokens,self.max_backing_bytes=output_width,max_context_tokens,max_backing_bytes
        self._owner,self._lock,self._closed=object(),Lock(),False
        stamp=DSARequestStamp(request_id,0,state.tokens_seen,logical_table,tail_page_id,self._owner)
        self._frame=self._capture(stamp,state,main_raw,index_raw,tail_raw)

    def _require_open(self):
        if self._closed:raise RuntimeError('DSA request is closed')

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):raise RuntimeError('DSA request is busy; concurrent/reentrant use unsupported')
        try:self._require_open();yield
        finally:self._lock.release()

    def _require_expected(self,expected):
        if type(expected) is not DSARequestStamp or expected!=self._frame.stamp:
            raise ValueError('stale/foreign expected position, revision or page table')

    def _capture(self,stamp,state,main,index,tail):
        tensors=(main,index,tail,state.latent,state.compressed,state.tail,state.tail_positions)
        if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' for t in tensors):raise ValueError('prepared DSA request is CPU-only')
        if sum(t.numel()*t.element_size() for t in (main,index,tail))>self.max_backing_bytes:raise ValueError('frame backing byte budget exceeded')
        if state.tokens_seen!=stamp.tokens_seen or state.tokens_seen>self.max_context_tokens:raise ValueError('state position/context differs')
        b=self.geometry.layout.logical_block_size
        if len(stamp.logical_table)!=(state.tokens_seen+b-1)//b:raise ValueError('active table must match exact visible prefix')
        binding=CPUPageBinding(main,index,tail,self.geometry,self.num_main_pages,self.num_tail_pages,
            torch.tensor([stamp.logical_table],dtype=torch.int32),stamp.tail_page_id)
        binding.require_snapshot(state,max_history_tokens=self.max_context_tokens)
        return PrivateDSASnapshot(stamp,state.clone(),main.clone(),index.clone(),tail.clone())

    @property
    def stamp(self):
        with self._exclusive():return self._frame.stamp

    @torch.inference_mode()
    def snapshot(self):
        with self._exclusive():
            f=self._frame
            return PrivateDSASnapshot(f.stamp,f.state.clone(),f.main_raw.clone(),f.index_raw.clone(),f.tail_raw.clone())

    @torch.inference_mode()
    def prepare(self,following,private_ids,*,tail_private,occupied_main,occupied_tail,expected_stamp,attention):
        """Complete producers -> private page view -> explicit CPU attention.

        following must preserve all committed latent/compressed history. The
        callback sees logical read access only. No callback for zero tokens.
        Failed prepare returns no branch and leaves the current frame unchanged.
        """
        with self._exclusive():
            self._require_expected(expected_stamp)
            if not callable(attention):raise ValueError('explicit attention callback required')
            f=self._frame;layout=self.geometry.layout
            if not isinstance(following,DSAState) or any(t.device.type!='cpu' for t in (following.latent,following.compressed,following.tail,following.tail_positions)):
                raise ValueError('explicit completed CPU DSA state required')
            validate_dsa_state(following,layout);start=f.stamp.tokens_seen;stop=following.tokens_seen
            if not start<=stop<=min(start+128,self.max_context_tokens):raise ValueError('bounded append-only following state required')
            if not torch.equal(f.state.latent,following.latent[:start]) or not torch.equal(f.state.compressed,following.compressed[:start//layout.pool_size]):
                raise ValueError('following state rewrites committed history')
            if stop==start and (not torch.equal(f.state.tail,following.tail) or not torch.equal(f.state.tail_positions,following.tail_positions)):
                raise ValueError('empty transition changes committed tail state')
            owned=following.clone() if stop>start else f.state
            plan=plan_private_dsa_pages(start,stop-start,f.stamp.logical_table,private_ids,geometry=self.geometry,
                num_main_pages=self.num_main_pages,num_tail_pages=self.num_tail_pages,occupied_main=occupied_main,occupied_tail=occupied_tail,
                tail_source=f.stamp.tail_page_id if f.stamp.tail_page_id>=0 else None,tail_private=tail_private)
            if not plan.tokens:
                return PreparedDSABranch(f.stamp,f,torch.empty(0,self.output_width,dtype=torch.bfloat16))
            stage=prepare_private_dsa_cpu(f.main_raw,f.index_raw,f.tail_raw,plan,self.geometry,
                owned.latent[start:],owned.compressed[start//layout.pool_size:],pack_tail_planes(owned.tail) if plan.tokens else None)
            view=CandidateDSAReadView(f,stage,self.geometry)
            if plan.tokens:
                output=attention(view);self._validate_output(output,plan.tokens);output=output.clone()
            else:output=torch.empty(0,self.output_width,dtype=torch.bfloat16)
            # TEST/REFERENCE CPU installation into full candidate CLONES ONLY.
            # No bytes at serving/global addresses are written.
            main,index,tail=f.main_raw.clone(),f.index_raw.clone(),f.tail_raw.clone()
            for row,page in enumerate(plan.private_ids):
                main.reshape(self.num_main_pages,-1)[page].copy_(stage.main[row]);index.reshape(self.num_main_pages,-1)[page].copy_(stage.index[row])
            if plan.tokens:tail.reshape(self.num_tail_pages,-1)[plan.tail_private].copy_(stage.tail[0])
            next_stamp=DSARequestStamp(f.stamp.request_id,f.stamp.revision+int(bool(plan.tokens)),stop,plan.read_table,
                plan.tail_private if plan.tokens else f.stamp.tail_page_id,self._owner)
            candidate=self._capture(next_stamp,owned,main,index,tail)
            return PreparedDSABranch(f.stamp,candidate,output)

    @torch.inference_mode()
    def prepared_snapshot(self,branch,*,expected_stamp):
        """Copy a prepared frame for a composite owner; publish nothing here.

        The composite caller must validate all layers/output before replacing
        its single frame. This does not consume or publish the layer branch.
        """
        with self._exclusive():
            self._require_expected(expected_stamp)
            if type(branch) is not PreparedDSABranch or branch.base_stamp!=self._frame.stamp or branch.status!='prepared':
                raise ValueError('stale/foreign/consumed DSA branch')
            c=branch._candidate
            if c.stamp._owner is not self._owner or c.stamp.request_id!=self._frame.stamp.request_id:
                raise ValueError('candidate owner differs')
            return self._capture(c.stamp,c.state,c.main_raw,c.index_raw,c.tail_raw)

    def _validate_output(self,output,tokens):
        if not isinstance(output,torch.Tensor) or output.device.type!='cpu' or output.dtype!=torch.bfloat16 or output.shape!=(tokens,self.output_width) or not bool(torch.isfinite(output).all()):
            raise ValueError('attention result must be finite CPU BF16 [new_tokens,output_width]')

    @torch.inference_mode()
    def publish(self,branch,*,expected_stamp,validate):
        """Explicit final validation before one in-process layer-frame swap.

        validate receives an output COPY and may raise. It must establish any
        required downstream checks before this layer publishes; this API cannot
        make separate layer/request/native side effects atomic. Rejected handles
        cannot be retried. Other branches from the old stamp become stale.
        """
        with self._exclusive():
            self._require_expected(expected_stamp)
            if type(branch) is not PreparedDSABranch or branch.base_stamp!=self._frame.stamp or branch.status!='prepared':raise ValueError('stale/foreign/consumed DSA branch')
            if not callable(validate):raise ValueError('explicit final validation callback required')
            try:
                c=branch._candidate;count=c.stamp.tokens_seen-self._frame.stamp.tokens_seen
                if c.stamp._owner is not self._owner or c.stamp.request_id!=self._frame.stamp.request_id or not 0<=count<=128 or c.stamp.revision!=self._frame.stamp.revision+int(bool(count)):
                    raise ValueError('candidate owner/revision/position differs')
                self._validate_output(branch._output,count)
                result=branch._output.clone()
                if not count:branch._status='published';return result
                owned=self._capture(c.stamp,c.state,c.main_raw,c.index_raw,c.tail_raw)
                validate(result.clone())
                self._frame=owned;branch._status='published'
                return result
            except BaseException:
                branch._status='rejected';raise

    def close(self):
        if self._closed:return
        with self._exclusive():self._closed=True;self._frame=None
