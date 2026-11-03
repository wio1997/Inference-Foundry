# SPDX-License-Identifier: Apache-2.0
"""Owned bounded text steps over explicit prepared components.

No model factory/registry, tokenizer, sampling, HTTP, native paged cache binding or
native readiness/authorization. A complete output step publishes private state
only after embedding, decoder, final norm and selected head all succeed. This
covers explicit in-memory state and optional declared synchronous CPU page
writes; arbitrary callback/native side effects remain outside the protocol.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

import torch
from torch import nn

from .hybrid_decoder import PreparedDecoderStack
from .request_state import MixedRequestState
from .text_boundary import GlmTextBoundary,GlmLogitsShard
from .deferred_pages import capture_page_bindings
from .paged_commit import commit_dsa_batches


@dataclass(frozen=True)
class TextStepResult:
    token_start: int
    token_stop: int
    hidden: torch.Tensor
    logits: torch.Tensor
    vocab_start: int
    vocab_end: int


class PreparedTextRunner(nn.Module):
    """Serialize explicit components; callers must close each request owner.

    max_context_tokens bounds logical unpaged history, not measured memory.
    No default components or CPU substitutes are constructed here. Successful
    binding (including test fixtures) is not a claim of native/model readiness.
    """
    def __init__(self,stack,boundary,head,*,max_context_tokens=128,max_requests=1,native_tokenwise=False):
        super().__init__()
        if not isinstance(stack,PreparedDecoderStack) or not isinstance(boundary,GlmTextBoundary) or not isinstance(head,GlmLogitsShard):
            raise ValueError('explicit prepared stack, text boundary and head shard required')
        if any(type(n) is not int or n<1 for n in (max_context_tokens,max_requests)):
            raise ValueError('positive explicit context/request ownership bounds required')
        if type(native_tokenwise) is not bool:raise ValueError('explicit boolean native_tokenwise required')
        self.native_tokenwise=native_tokenwise
        self.stack,self.boundary,self.head=stack,boundary,head
        self.max_context_tokens,self.max_requests=max_context_tokens,max_requests
        self._lock,self._requests=Lock(),{}
        self._rank_commit_enabled,self._rank_commit_group=False,None
        self._rank_batch_limit=0
        self._rank_batch_layer_major=False
        self._validate_components()

    def enable_rank_commit_agreement(self, *, group=None):
        """Opt in to aligned single-token head/commit agreement.

        Every member must call steps in the same order with matching shapes and
        reach the end of the stack. Errors before or inside stack collectives,
        peer loss, failed collectives and callback side effects are not recovered.
        Synchronous page bindings and multi-token steps are unsupported.
        """
        import torch.distributed as dist
        with self._exclusive():
            if self._requests:raise RuntimeError('enable agreement before opening requests')
            if not dist.is_available() or not dist.is_initialized() or dist.get_world_size(group)<2:
                raise ValueError('initialized multi-rank process group required')
            self._rank_commit_group=group
            self._rank_commit_enabled=True

    def enable_rank_batch_commit(self, *, max_tokens, layer_major_dense_kda=False):
        """Explicit token-major rank transaction, 1..8 rows; no page writes.

        All ranks must opt in and call in the same order. Live-rank failures
        before stack entry, in the head, and while capturing state are agreed.
        Process/device loss and errors inside stack collectives are not recovered.
        Optional layer-major execution requires explicit KDA chunk layers;
        DSA and sparse MoE calls retain their tokenwise fallbacks.
        """
        with self._exclusive():
            if self._requests:raise RuntimeError('enable batch agreement before opening requests')
            if not self._rank_commit_enabled or not self.native_tokenwise:
                raise ValueError('rank batch requires tokenwise rank commit agreement')
            if type(max_tokens) is not int or not 1<=max_tokens<=8:
                raise ValueError('rank batch bound must be 1..8')
            if type(layer_major_dense_kda) is not bool:raise ValueError('layer-major flag must be boolean')
            if layer_major_dense_kda:
                selected=0
                for layer in self.stack.layers:
                    if not layer.native_tokenwise:raise ValueError('all native layers must preserve tokenwise fallback')
                    if getattr(layer,'native_chunk_prefill',False):
                        if layer.attention_kind!='linear_attention' or getattr(layer.attention,'backend',None)!='ascendc_chunk_prefill':
                            raise ValueError('layer-major supports only explicit KDA chunks')
                        selected+=1
                if not selected:raise ValueError('layer-major requires an explicit KDA chunk layer')
            self._rank_batch_layer_major=layer_major_dense_kda
            self._rank_batch_limit=max_tokens

    def _validate_components(self):
        m=self.boundary.manifest;c=self.stack.config
        if (c.hidden_size,c.vocab_size,c.hf_config['text_config']['hc_mult'])!=(m.hidden_size,m.vocab_size,m.streams):
            raise ValueError('stack/text boundary geometry mismatch')
        if c.hf_config['text_config']['rms_norm_eps']!=m.norm_eps:raise ValueError('final RMS epsilon mismatch')
        if self.head.weight.shape[1]!=m.hidden_size or not 0<=self.head.vocab_start<=self.head.vocab_end<=m.vocab_size:
            raise ValueError('head shard does not belong to text vocabulary')
        if self.head.weight.device!=self.boundary.norm_weight.device or self.head.scale!=m.logit_scale:
            raise ValueError('head/text device or scale mismatch')
        for layer in self.stack.layers:
            for name in ('attn_hc','ffn_hc'):
                hc=getattr(layer,name)
                if (hc.hidden_size,hc.streams)!=(m.hidden_size,m.streams):raise ValueError('decoder stream geometry mismatch')

    @torch.inference_mode()
    def open_private_request(self,state,*,page_bindings,producers,attentions,max_backing_bytes=64*1024**2):
        """Separate CPU private-page route; never combines shadow bindings.

        Explicit producers complete DSA state; explicit attention reads staged
        private pages before the remaining decoder/head. No native fallback.
        """
        from .private_text_request import PrivateTextRequest
        with self._exclusive():
            if not isinstance(state,MixedRequestState):raise ValueError('explicit complete request snapshot required')
            if state.request_id in self._requests or len(self._requests)>=self.max_requests:
                raise ValueError('request ownership ID/count conflict')
            request=PrivateTextRequest(self,state,page_bindings,producers,attentions,max_backing_bytes=max_backing_bytes)
            self._requests[state.request_id]=request
            return request

    @contextmanager
    def _exclusive(self):
        if not self._lock.acquire(blocking=False):raise RuntimeError('prepared text runner is busy; concurrent use is unsupported')
        try:yield
        finally:self._lock.release()

    @torch.inference_mode()
    def open_request(self,state,*,page_bindings=None):
        with self._exclusive():
            if not isinstance(state,MixedRequestState):raise ValueError('explicit complete request snapshot required')
            if state.request_id in self._requests:raise ValueError('request ID already has a current owner')
            if len(self._requests)>=self.max_requests:raise ValueError('request ownership count exceeds explicit bound')
            owned=MixedRequestState.capture(state.request_id,self.stack.config,self.stack.layout,state.layers)
            if owned.tokens_seen!=state.tokens_seen or owned.tokens_seen>self.max_context_tokens:
                raise ValueError('request token count is inconsistent or exceeds context bound')
            bindings=None if page_bindings is None else capture_page_bindings(self.stack.config,self.stack.layout,page_bindings,
                tuple(q._page_bindings for q in self._requests.values() if q._page_bindings is not None),state=owned)
            request=TextRequest(self,owned,bindings);self._requests[owned.request_id]=request
            return request


class TextRequest:
    """Private state owner. snapshot() copies; stale step offsets reject early."""
    def __init__(self,runner,state,page_bindings=None):
        self._runner,self._state,self._request_id=runner,state,state.request_id
        self._closed=False;self._page_bindings=page_bindings

    @property
    def request_id(self):return self._request_id

    def _require_open(self):
        if self._closed or self._runner._requests.get(self.request_id) is not self:
            raise RuntimeError('request has no active ownership')

    @torch.inference_mode()
    def snapshot(self):
        with self._runner._exclusive():
            self._require_open();r=self._runner
            return MixedRequestState.capture(self.request_id,r.stack.config,r.stack.layout,self._state.layers)

    def _rank_batch_step(self, token_ids, expected_tokens_seen):
        """Execute under runner lock; publish only after all rank votes."""
        import hashlib
        import torch.distributed as dist
        r=self._runner;group=r._rank_commit_group;device=r.boundary.norm_weight.device
        size=dist.get_world_size(group);failure=None
        try:
            self._require_open();r._validate_components()
            if self._page_bindings is not None:raise ValueError('rank batch excludes page bindings')
            if type(expected_tokens_seen) is not int or expected_tokens_seen!=self._state.tokens_seen:
                raise ValueError('stale or invalid expected request token position')
            if not isinstance(token_ids,torch.Tensor) or token_ids.ndim!=1 or token_ids.device.type!='cpu' or token_ids.dtype not in (torch.int32,torch.int64):
                raise ValueError('rank batch requires CPU integer token vector')
            count=len(token_ids);start=self._state.tokens_seen
            if not 1<=count<=r._rank_batch_limit or count>min(r.boundary.max_tokens,r.head.max_tokens) or start+count>r.max_context_tokens:
                raise ValueError('rank batch exceeds explicit token/context bound')
            if bool(((token_ids<0)|(token_ids>=r.boundary.manifest.vocab_size)).any()):raise ValueError('rank batch token out of vocabulary')
            ids=token_ids.to(dtype=torch.int64).tolist()
            request_hash=int.from_bytes(hashlib.sha256(self.request_id.encode()).digest()[:7],'big')
            layer_mask=0;single_mask=0;projection_mask=0
            if r._rank_batch_layer_major:
                for layer in r.stack.layers:
                    if not layer.native_tokenwise:raise ValueError('layer-major fallback configuration changed')
                    if getattr(layer,'native_chunk_prefill',False):
                        if (type(layer.layer_id) is not int or not 0<=layer.layer_id<63 or layer.attention_kind!='linear_attention'
                                or getattr(layer.attention,'backend',None)!='ascendc_chunk_prefill'):
                            raise ValueError('invalid layer-major chunk configuration')
                        layer_mask |= 1<<layer.layer_id
                        single=getattr(layer.attention,'chunk_prefill_single',False)
                        if type(single) is not bool:raise ValueError('singleton prefill mode must be boolean')
                        if single:single_mask |= 1<<layer.layer_id
                        projection=getattr(layer.attention,'reference_sharded_projection',False)
                        if type(projection) is not bool:raise ValueError('reference projection mode must be boolean')
                        if projection:projection_mask |= 1<<layer.layer_id
                    elif getattr(layer.attention,'chunk_prefill_single',False) or getattr(layer.attention,'reference_sharded_projection',False):
                        raise ValueError('experimental KDA modes require a selected KDA layer')
                if not layer_mask:raise ValueError('layer-major chunk configuration is empty')
                if single_mask and count!=1:raise ValueError('singleton prefill mode requires one token')
            elif any(getattr(layer.attention,'reference_sharded_projection',False) is not False for layer in r.stack.layers):
                raise ValueError('reference projection requires layer-major rank preflight')
            metadata=[0,count,start,r._rank_batch_limit,request_hash,r.head.vocab_start,r.head.vocab_end,int(r._rank_batch_layer_major),layer_mask,single_mask,projection_mask]
        except Exception as exc:
            failure=exc;metadata=[1,0,0,0,0,0,0,0,0,0,0];ids=[]
        header=torch.tensor(metadata,dtype=torch.int64,device=device);headers=[torch.empty_like(header) for _ in range(size)]
        dist.all_gather(headers,header,group=group)
        if any(int(h[0].item()) for h in headers) or not all(torch.equal(headers[0],h) for h in headers[1:]):
            raise ValueError('rank batch preflight failed or metadata differs') from failure
        values=torch.tensor(ids+[0]*(8-len(ids)),dtype=torch.int64,device=device);copies=[torch.empty_like(values) for _ in range(size)]
        dist.all_gather(copies,values,group=group)
        if not all(torch.equal(copies[0],v) for v in copies[1:]):raise ValueError('rank batch token inputs differ')
        def agree(exc, message):
            status=torch.tensor([int(exc is not None)],dtype=torch.int32,device=device)
            dist.all_reduce(status,op=dist.ReduceOp.MAX,group=group)
            if int(status.item()):raise RuntimeError(message) from exc
        failure=None
        try:embedded=r.boundary.embed(token_ids)
        except Exception as exc:failure=exc
        agree(failure,'rank batch embedding failed')
        working=self._state;hidden_rows=[];logit_rows=[]
        if r._rank_batch_layer_major:
            streams,working=r.stack(embedded,working)
            stream_rows=streams.split(1,dim=0)
        else:
            stream_rows=None
        for index,row in enumerate(embedded.split(1,dim=0)):
            if stream_rows is None:
                streams,working=r.stack(row,working)
            else:
                streams=stream_rows[index]
            failure=None
            try:
                hidden=r.boundary.final_hidden(streams);logits=r.head(hidden)
            except Exception as exc:failure=exc
            agree(failure,'rank batch head failed')
            hidden_rows.append(hidden);logit_rows.append(logits)
        failure=None
        try:
            committed=MixedRequestState.capture(self.request_id,r.stack.config,r.stack.layout,working.layers)
            if working.request_id!=self.request_id or working.tokens_seen!=start+count or committed.tokens_seen!=start+count:
                raise ValueError('decoder returned the wrong request commit')
            result=TextStepResult(start,start+count,torch.cat(hidden_rows),torch.cat(logit_rows),r.head.vocab_start,r.head.vocab_end)
        except Exception as exc:failure=exc
        agree(failure,'rank batch commit preparation failed')
        self._state=committed
        return result

    @torch.inference_mode()
    def step(self,token_ids,*,expected_tokens_seen):
        r=self._runner
        with r._exclusive():
            if r._rank_batch_limit:return self._rank_batch_step(token_ids,expected_tokens_seen)
            failure=None
            try:
                self._require_open();r._validate_components()
                if type(expected_tokens_seen) is not int or expected_tokens_seen!=self._state.tokens_seen:
                    raise ValueError('stale or invalid expected request token position')
                if not isinstance(token_ids,torch.Tensor) or token_ids.ndim!=1:
                    raise ValueError('text step requires a one-dimensional token tensor')
                count=len(token_ids);start=self._state.tokens_seen
                if r._rank_commit_enabled and (count!=1 or self._page_bindings is not None):
                    raise ValueError('rank commit agreement requires one token and no page bindings')
                if r._rank_commit_enabled:
                    if token_ids.device.type!='cpu' or token_ids.dtype not in (torch.int32,torch.int64):
                        raise ValueError('rank commit requires CPU integer token vector')
                    if bool(((token_ids<0)|(token_ids>=r.boundary.manifest.vocab_size)).any()):
                        raise ValueError('rank commit token out of vocabulary')
                    import hashlib
                    identity=int.from_bytes(hashlib.sha256(self.request_id.encode()).digest()[:7],'big')
                if count>min(r.boundary.max_tokens,r.head.max_tokens) or start+count>r.max_context_tokens:
                    raise ValueError('text step exceeds explicit token/context bound')
                if r._rank_commit_enabled and any(getattr(layer.attention,'reference_sharded_projection',False) is not False for layer in r.stack.layers):
                    raise ValueError('reference projection requires layer-major rank preflight')
            except Exception as exc:
                failure=exc
            if r._rank_commit_enabled:
                # Validate both local state and shared request identity/position
                # before ranks enter any stack collective or publish state.
                import torch.distributed as dist
                if failure is None:
                    metadata=[0,start,count,identity,int(token_ids[0].item())]
                else:
                    metadata=[1,0,0,0,0]
                header=torch.tensor(metadata,dtype=torch.int64,device=r.boundary.norm_weight.device)
                headers=[torch.empty_like(header) for _ in range(dist.get_world_size(r._rank_commit_group))]
                dist.all_gather(headers,header,group=r._rank_commit_group)
                if any(int(item[0].item()) for item in headers) or not all(torch.equal(headers[0],item) for item in headers[1:]):
                    raise ValueError('rank commit preflight failed') from failure
            elif failure is not None:
                raise failure
            if r._rank_commit_enabled:
                failure=None
                try:hidden=r.boundary.embed(token_ids)
                except Exception as exc:failure=exc
                status=torch.tensor([int(failure is not None)],dtype=torch.int32,device=r.boundary.norm_weight.device)
                dist.all_reduce(status,op=dist.ReduceOp.MAX,group=r._rank_commit_group)
                if int(status.item()):raise RuntimeError('rank embedding preparation failed') from failure
            else:
                hidden=r.boundary.embed(token_ids)
            if count:
                if r.native_tokenwise and count>1:
                    # Preserve token-major execution order across the complete
                    # stack; publish only after every token/head succeeds.
                    working=self._state;hidden_rows=[];logit_rows=[]
                    for row in hidden.split(1,dim=0):
                        streams,working=r.stack(row,working)
                        value=r.boundary.final_hidden(streams)
                        hidden_rows.append(value);logit_rows.append(r.head(value))
                    hidden=torch.cat(hidden_rows);logits=torch.cat(logit_rows);next_state=working
                else:
                    streams,next_state=r.stack(hidden,self._state)
                    if not r._rank_commit_enabled:
                        hidden=r.boundary.final_hidden(streams);logits=r.head(hidden)
                # On the explicit rank path, all peers must have finished the
                # stack's collectives before entering this prepare/agreement.
                # Do not publish state if any peer's head or capture fails.
                if r._rank_commit_enabled:
                    import torch.distributed as dist
                    failure=None
                    try:
                        hidden=r.boundary.final_hidden(streams);logits=r.head(hidden)
                        committed=MixedRequestState.capture(self.request_id,r.stack.config,r.stack.layout,next_state.layers)
                        if next_state.request_id!=self.request_id or next_state.tokens_seen!=start+count or committed.tokens_seen!=start+count:
                            raise ValueError('decoder returned the wrong request commit')
                    except Exception as exc:
                        failure=exc
                    status=torch.tensor([int(failure is not None)],dtype=torch.int32,device=streams.device)
                    dist.all_reduce(status,op=dist.ReduceOp.MAX,group=r._rank_commit_group)
                    if int(status.item()):
                        raise RuntimeError('rank head/commit preparation failed') from failure
                else:
                    # Own another copy before publication; callbacks can retain
                    # returned buffers. Arbitrary callback side effects persist.
                    committed=MixedRequestState.capture(self.request_id,r.stack.config,r.stack.layout,next_state.layers)
                    if next_state.request_id!=self.request_id or next_state.tokens_seen!=start+count or committed.tokens_seen!=start+count:
                        raise ValueError('decoder returned the wrong request commit')
            else:
                logits=hidden.new_empty((0,len(r.head.weight)),dtype=r.head.head_dtype)
                committed=self._state
            result=TextStepResult(start,start+count,hidden,logits,r.head.vocab_start,r.head.vocab_end)
            if count and self._page_bindings is not None:
                batches=tuple(binding.stage(self._state.layers[layer],committed.layers[layer])
                              for layer,binding in sorted(self._page_bindings.items()))
                commit_dsa_batches(batches)
            self._state=committed
            return result

    def close(self):
        with self._runner._exclusive():
            if not self._closed:
                self._require_open();del self._runner._requests[self.request_id]
                self._state=None;self._page_bindings=None;self._closed=True
