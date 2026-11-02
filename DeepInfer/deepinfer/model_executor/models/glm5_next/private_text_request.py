# SPDX-License-Identifier: Apache-2.0
"""One synchronous CPU frame for KDA state, all DSA pages and text position.

Explicit per-step DSA producers and page readers run in the existing prepared
stack. All candidate layer frames, FFNs, final norm and head must succeed before
one composite reference is replaced. No per-layer publish() is called. Whole
CPU pools are cloned for correctness; no native allocator, events, registration,
serving memory strategy, async/crash rollback or performance claim is implied.
Callbacks must not mutate external state; their side effects are not undone.
"""
from dataclasses import dataclass,field
from types import MappingProxyType
import torch
from .cache import DSAState
from .deferred_pages import capture_page_bindings
from .hybrid_decoder import PreparedDSAAttention
from .private_request import PrivateDSARequest,PrivateDSASnapshot
from .request_state import MixedRequestState
from .text_request import TextStepResult
from .projected_reader import CPUProjectedDSAReader
from .rank_dsa import CPUProjectedDSAOperation,CPUProjectedDSAResult
from .private_pages import plan_private_dsa_pages


@dataclass(frozen=True)
class PrivateTextStamp:
    request_id: str
    revision: int
    tokens_seen: int
    pages: tuple  # (layer_id, immutable logical table, tail page ID)
    _owner: object = field(repr=False)


@dataclass(frozen=True)
class _Frame:
    stamp: PrivateTextStamp
    state: MixedRequestState
    pages: object


def _same_dsa(a,b):
    return a.tokens_seen==b.tokens_seen and all(torch.equal(getattr(a,n),getattr(b,n)) for n in ('latent','compressed','tail','tail_positions'))


def _copy_page(f):
    return PrivateDSASnapshot(f.stamp,f.state.clone(),f.main_raw.clone(),f.index_raw.clone(),f.tail_raw.clone())


class PrivateTextRequest:
    """Opened by PreparedTextRunner; shares its nonblocking serialization lock.

    Producer signature: (DSAStep, private DSAState) -> completed DSAState.
    Attention signature: (DSAStep, CandidateDSAReadView) -> BF16 [T,H,V].
    CPUProjectedDSAReader explicitly returns already projected BF16 [T,4096].
    Its plan binds one fresh composite attempt and bypasses mla.finish once.
    page_plans maps every DSA layer to explicit private_ids,tail_private,
    occupied_main,occupied_tail. These are CPU fixture IDs, not native leases.
    A stamp identifies this in-process owner/revision and all visible tables.
    """
    @torch.inference_mode()
    def __init__(self,runner,state,bindings,producers,attentions,*,max_backing_bytes):
        runner._validate_components()
        if type(max_backing_bytes) is not int or max_backing_bytes<1:raise ValueError('positive total backing bound required')
        if runner.boundary.norm_weight.device.type!='cpu' or runner.head.weight.device.type!='cpu':raise ValueError('private text request is CPU-only')
        self._runner,self._request_id,self._owner=runner,state.request_id,object()
        self._closed,self._page_bindings=False,None  # separate from CPU shadow writes
        self._max_backing_bytes=max_backing_bytes
        owned=MixedRequestState.capture(state.request_id,runner.stack.config,runner.stack.layout,state.layers)
        if owned.tokens_seen!=state.tokens_seen or owned.tokens_seen>runner.max_context_tokens:raise ValueError('request token/context mismatch')
        for s in owned.layers.values():
            tensors=(s.latent,s.compressed,s.tail,s.tail_positions) if isinstance(s,DSAState) else (*s.conv,s.recurrent)
            if any(t.device.type!='cpu' for t in tensors):raise ValueError('private text state is CPU-only')
        bindings=capture_page_bindings(runner.stack.config,runner.stack.layout,bindings,state=owned)
        ids=set(bindings)
        if set(producers)!=ids or set(attentions)!=ids or any(not callable(f) for d in (producers,attentions) for f in d.values()):
            raise ValueError('explicit producers and attention readers must cover every DSA layer')
        self._producers,self._attentions=dict(producers),dict(attentions)
        self._projected_specs={}
        for layer,reader in self._attentions.items():
            if type(reader) is CPUProjectedDSAReader:
                reader.validate_config(runner.stack.config,layer)
                self._projected_specs[layer]=reader.specification
        if sum(t.numel()*t.element_size() for b in bindings.values() for t in (b.main_raw,b.index_raw,b.tail_raw))>max_backing_bytes:
            raise ValueError('total frame backing budget exceeded')
        self._bindings=dict(bindings);pages={}
        for layer,b in bindings.items():
            if not isinstance(runner.stack.layers[layer].attention,PreparedDSAAttention):raise ValueError('private route requires prepared DSA attention')
            q=self._layer_owner(layer,owned.layers[layer],b.main_raw,b.index_raw,b.tail_raw,tuple(b.logical_table[0].tolist()),b.tail_page_id)
            try:pages[layer]=q.snapshot()
            finally:q.close()
        # Do not retain caller raw backing tensors after capturing private pages.
        self._geometry={i:(b.geometry,b.num_main_pages,b.num_tail_pages) for i,b in bindings.items()};self._bindings=None
        stamp=PrivateTextStamp(state.request_id,0,owned.tokens_seen,self._page_ids(pages),self._owner)
        self._frame=_Frame(stamp,owned,MappingProxyType(pages))

    def _layer_owner(self,layer,state,main,index,tail,table,tail_id):
        if self._bindings is not None:
            b=self._bindings[layer];g,n,p=b.geometry,b.num_main_pages,b.num_tail_pages
        else:g,n,p=self._geometry[layer]
        mla=self._runner.stack.layers[layer].attention.mla
        return PrivateDSARequest(self.request_id,state,main,index,tail,table,tail_id,geometry=g,num_main_pages=n,num_tail_pages=p,
            output_width=4096 if layer in self._projected_specs else mla.num_heads*mla.value_dim,max_context_tokens=self._runner.max_context_tokens,max_backing_bytes=self._max_backing_bytes)

    @staticmethod
    def _page_ids(pages):return tuple((i,f.stamp.logical_table,f.stamp.tail_page_id) for i,f in sorted(pages.items()))

    @property
    def request_id(self):return self._request_id

    def _require_open(self):
        if self._closed or self._runner._requests.get(self.request_id) is not self:raise RuntimeError('request has no active ownership')

    @property
    def stamp(self):
        with self._runner._exclusive():self._require_open();return self._frame.stamp

    @torch.inference_mode()
    def snapshot(self):
        with self._runner._exclusive():
            self._require_open();r=self._runner
            return MixedRequestState.capture(self.request_id,r.stack.config,r.stack.layout,self._frame.state.layers)

    @torch.inference_mode()
    def page_snapshot(self):
        with self._runner._exclusive():
            self._require_open();return MappingProxyType({i:_copy_page(f) for i,f in self._frame.pages.items()})

    @torch.inference_mode()
    def step(self,token_ids,*,expected_stamp,page_plans):
        r=self._runner
        with r._exclusive():
            self._require_open();r._validate_components();base=self._frame
            if type(expected_stamp) is not PrivateTextStamp or expected_stamp!=base.stamp:raise ValueError('stale/foreign text position, revision or page tables')
            if not isinstance(token_ids,torch.Tensor) or token_ids.device.type!='cpu' or token_ids.ndim!=1:raise ValueError('one-dimensional CPU token tensor required')
            count=len(token_ids);start=base.stamp.tokens_seen
            if count>min(128,r.boundary.max_tokens,r.head.max_tokens) or start+count>r.max_context_tokens:raise ValueError('text step exceeds token/context bound')
            expected=set(base.pages) if count else set()
            if not isinstance(page_plans,dict) or set(page_plans)!=expected:raise ValueError('page plans must cover exactly the active DSA layers')
            # Validate all reservation fields before embedding/projections.
            plans={}
            for i,p in page_plans.items():
                if not isinstance(p,dict) or set(p)!={'private_ids','tail_private','occupied_main','occupied_tail'}:raise ValueError('explicit private page plan fields required')
                plans[i]=dict(p)
            # A fresh owner nonce prevents reusing a result from an abandoned
            # attempt, even if a retry has the same revision/tables/token count.
            rank_plans={};attempt_owner=object()
            for layer in base.pages:
                reader=self._attentions[layer];projected=type(reader) is CPUProjectedDSAReader
                if projected!=(layer in self._projected_specs) or projected and reader.specification!=self._projected_specs[layer]:raise ValueError('projected reader mode/specification changed')
                if projected and count:
                    reader.validate_config(r.stack.config,layer);p=plans[layer];f=base.pages[layer];g,n,tail_pages=self._geometry[layer]
                    candidate=plan_private_dsa_pages(start,count,f.stamp.logical_table,p['private_ids'],geometry=g,num_main_pages=n,num_tail_pages=tail_pages,occupied_main=p['occupied_main'],occupied_tail=p['occupied_tail'],tail_source=f.stamp.tail_page_id if f.stamp.tail_page_id>=0 else None,tail_private=p['tail_private'])
                    rank_plans[layer]=reader.bind_candidate(candidate,request_id=self.request_id,revision=base.stamp.revision,attempt_owner=attempt_owner)
            hidden=r.boundary.embed(token_ids)
            if not count:
                return TextStepResult(start,start,hidden.clone(),hidden.new_empty((0,len(r.head.weight)),dtype=r.head.head_dtype),r.head.vocab_start,r.head.vocab_end)
            candidates={};owners={}
            def operation(layer):
                def run(step,state,**projected_binding):
                    if layer in candidates:raise ValueError('DSA layer executed more than once')
                    f=base.pages[layer]
                    if not _same_dsa(state,f.state):raise ValueError('DSA step input disagrees with composite frame')
                    q=self._layer_owner(layer,f.state,f.main_raw,f.index_raw,f.tail_raw,f.stamp.logical_table,f.stamp.tail_page_id);owners[layer]=q
                    following=self._producers[layer](step,state)
                    if not isinstance(following,DSAState) or following.tokens_seen!=start+count:raise ValueError('DSA producer position disagrees with text step')
                    mla=r.stack.layers[layer].attention.mla
                    rank_plan=rank_plans.get(layer)
                    if rank_plan is not None and (projected_binding.get('plan') is not rank_plan or projected_binding.get('expected_identity')!=rank_plan.identity):raise ValueError('projected operation binding differs')
                    def attend(view):
                        if rank_plan is not None:
                            return self._attentions[layer](step,view,plan=rank_plan,expected_identity=rank_plan.identity).hidden
                        value=self._attentions[layer](step,view)
                        if not isinstance(value,torch.Tensor) or value.shape!=(count,mla.num_heads,mla.value_dim):raise ValueError('private attention head/value geometry mismatch')
                        return value.flatten(1)
                    p=plans[layer];branch=q.prepare(following,p['private_ids'],tail_private=p['tail_private'],occupied_main=p['occupied_main'],occupied_tail=p['occupied_tail'],expected_stamp=q.stamp,attention=attend)
                    candidate=q.prepared_snapshot(branch,expected_stamp=q.stamp);candidates[layer]=candidate
                    output=CPUProjectedDSAResult(rank_plan.identity,branch.output) if rank_plan is not None else branch.output.reshape(count,mla.num_heads,mla.value_dim)
                    return output,candidate.state.clone()
                rank_plan=rank_plans.get(layer)
                return CPUProjectedDSAOperation(rank_plan,rank_plan.identity,run,self._attentions[layer].query_payload,self._attentions[layer].query_backend) if rank_plan is not None else run
            try:
                streams,next_state=r.stack(hidden,base.state,dsa_operations={i:operation(i) for i in base.pages})
                if set(candidates)!=set(base.pages):raise ValueError('missing DSA candidate')
                owned=MixedRequestState.capture(self.request_id,r.stack.config,r.stack.layout,next_state.layers)
                if next_state.request_id!=self.request_id or next_state.tokens_seen!=start+count or owned.tokens_seen!=start+count:raise ValueError('mixed state position/request mismatch')
                if any(not _same_dsa(owned.layers[i],f.state) for i,f in candidates.items()):raise ValueError('mixed DSA state disagrees with prepared pages')
                hidden=r.boundary.final_hidden(streams);logits=r.head(hidden)
                if hidden.device.type!='cpu' or logits.device.type!='cpu' or not bool(torch.isfinite(hidden).all()) or not bool(torch.isfinite(logits).all()):raise ValueError('finite CPU text output required')
                if logits.shape!=(count,len(r.head.weight)) or logits.dtype!=r.head.head_dtype:raise ValueError('head output geometry/dtype mismatch')
                result=TextStepResult(start,start+count,hidden.clone(),logits.clone(),r.head.vocab_start,r.head.vocab_end)
                stamp=PrivateTextStamp(self.request_id,base.stamp.revision+1,start+count,self._page_ids(candidates),self._owner)
                frame=_Frame(stamp,owned,MappingProxyType(candidates))
                # All fallible work, including temporary owner cleanup, precedes swap.
            finally:
                for q in owners.values():q.close()
            self._frame=frame
            return result

    def close(self):
        with self._runner._exclusive():
            if not self._closed:
                self._require_open();del self._runner._requests[self.request_id]
                self._frame=None;self._closed=True
