# SPDX-License-Identifier: Apache-2.0
"""Persistent native full-model graph for the target four-request DP decode.

CPU Engram lookup/history and API sampling stay outside the graph. IDs,
lengths and selected Engram rows are copied into persistent inputs. All model
layers, native MC2, cache writes, indexers and attention run in one NPU graph.
No external inference runtime is used.
"""
import time
import os

import torch
import torch.nn.functional as F

from deepinfer.npu_ops.dsa910c import DSABatch
from .ascend_w8a8_packed import PackedForward
from .w8a8_math910c import project, rms_norm_mixed
from .w8a8_paged_cache910c import CacheWritePlan
from .zos_device_step910c import guard_legacy,guard_execute
from .zos_step910c import (prepare_target_inputs,replay_prepared_target,
                          publish_target_host,discard_target_inputs)


class NativeDecodeGraph:
    def __init__(self,model,slots,query_tokens=1):
        self.model=model
        self.slots=list(slots)
        self.count=len(slots)
        self.query_tokens=query_tokens
        self.tokens=self.count*query_tokens
        self.inputs=torch.zeros(self.tokens+self.count,dtype=torch.int32,device=model.device)
        self.cpu_inputs=torch.empty_like(self.inputs,device='cpu',pin_memory=True)
        self.device_input_override=None
        self.device_input_override_applied=0
        self.device_engram_override=None
        self.device_engram_override_applied=0
        self.device_engram_fast_override=None
        self.device_engram_fast_calls=0
        self.device_engram_fast_checked=0
        self.requests=torch.tensor(slots,dtype=torch.int32,device=model.device)
        self.query_requests=self.requests.repeat_interleave(query_tokens)
        self.query_offsets=torch.arange(query_tokens,dtype=torch.int32,device=model.device)
        self.scratch_rows=torch.arange(self.tokens,dtype=torch.int32,device=model.device)
        self.request_rows=torch.arange(self.count,dtype=torch.int64,device=model.device)
        self.cu_query=torch.arange(self.count+1,dtype=torch.int32,device=model.device)*query_tokens
        self.valid=torch.ones(self.tokens,dtype=torch.bool,device=model.device)
        self.tables={name:{key:plane.table(slots) for key,plane in getattr(model.cache,name).items()}
                     for name in ('swa','compressed','index_key','index_scale')}
        self.rows={i:torch.empty((self.tokens,getattr(model.engrams[str(i)].projection,'input_size',
                    model.engrams[str(i)].projection.weight.shape[1])),
                    dtype=torch.bfloat16,device=model.device) for i in model.cfg['engram_layer_ids']}
        self.even={}
        for layer in model.layers:
            if layer.layer_id in model.cfg['kv_source_layer_ids'] and layer.ratio==2:
                self.even[layer.layer_id]=tuple(torch.zeros((self.count,weight.shape[0]),dtype=torch.float32,device=model.device)
                    for weight in (layer.compressor_wkv_weight,layer.compressor_wgate_weight))
        self.even_before={key:tuple(torch.empty_like(t) for t in pair) for key,pair in self.even.items()}
        self.even_rows={}
        self.graph=None
        self.replays=0
        self.captures=0
        self.prepared=False
        self._zos_ticket=None
        self._zos_generation=0
        self._zos_commit_required=False
        self.staging_seconds=0.
        self.engram_host_rows={}
        self.engram_copy_done={}
        self.engram_pinned_calls=0
        self.engram_pinned_bytes=0
        self.engram_device_ids={}
        self.engram_cpu_ids={}
        self.engram_hash_host=None
        self.engram_hash_context=None
        self.engram_hash_output=None
        self.engram_hash_copy_done=None
        self.engram_hash_calls=0
        self.engram_hash_tokens=0

    def reset_sequence(self):
        guard_legacy(self,'reset')
        if (self._zos_ticket is not None or self._zos_commit_required or
                getattr(self.model,'_zos_prepared_frame',None) is not None or
                getattr(self.model,'_zos_commit_frame',None) is not None):
            raise RuntimeError('cannot reset graph with outstanding target publication/commit')
        self.prepared=False
        self.replays=0
        self.staging_seconds=0.
        self.engram_pinned_calls=0
        self.engram_pinned_bytes=0
        self.engram_hash_calls=0
        self.engram_hash_tokens=0

    def _stage_engram(self,histories,chunks):
        if not self.rows:return
        if os.environ.get('DEEPINFER_910C_ENGRAM_DEVICE_HASH')!='1':
            hashes=[history.batch(chunk) for history,chunk in zip(histories,chunks)]
            for layer,buffer in self.rows.items():
                self._stage_engram_rows(layer,torch.cat([entry[layer] for entry in hashes]),buffer)
            return
        runtime=self.model.engram_uva_runtime
        device_hash=self.model.engram_device_hash
        if os.environ.get('DEEPINFER_910C_ENGRAM_UVA')!='1' or runtime is None or device_hash is None:
            raise RuntimeError('device hash requires the enabled registered UVA lookup')
        from deepinfer.npu_ops.engram_hash910c import raw_contexts
        original_lengths=[len(history.history) for history in histories]
        try:
            fast=self.device_engram_fast_override
            self.device_engram_fast_override=None
            contexts=None
            if fast is None:
                contexts=raw_contexts(histories,chunks)
            else:
                context,check=fast
                if (self.count!=4 or self.query_tokens!=6 or self.tokens!=24 or
                        tuple(context.shape)!=(24,4) or context.dtype!=torch.int32 or
                        context.device!=self.model.device or not context.is_contiguous()):
                    raise ValueError('invalid fixed24 device Engram windows')
                if self.device_engram_override is not None:
                    raise ValueError('conflicting Engram window overrides')
                if check:
                    contexts=raw_contexts(histories,chunks)
                    if context.cpu().tolist()!=contexts:
                        raise ValueError('device Engram windows differ from CPU')
                    self.device_engram_fast_checked+=1
                else:
                    # Match raw_contexts' all-or-nothing validation and keep
                    # the full CPU history for later rollback and fallback.
                    if len(histories)!=len(chunks) or not histories:
                        raise ValueError('history/chunk mismatch')
                    for history,chunk in zip(histories,chunks):
                        if not chunk or any(type(x) is not int or not 0<=x<len(history.token_map) for x in chunk):
                            raise ValueError('invalid history chunk')
                    for history,chunk in zip(histories,chunks):history.history.extend(chunk)
                self.device_engram_fast_calls+=1
            if contexts is not None and len(contexts)!=self.tokens:
                raise ValueError('raw history windows must match the physical graph token count')
            if self.engram_hash_host is None:
                self.engram_hash_host=torch.empty((self.tokens,device_hash.width),dtype=torch.int32,device='cpu',pin_memory=True)
                self.engram_hash_context=torch.empty_like(self.engram_hash_host,device=self.model.device)
                self.engram_hash_output=torch.empty((len(device_hash.layers),self.tokens,device_hash.heads),dtype=torch.int32,device=self.model.device)
            # A diagnostic may provide device-constructed windows. CPU
            # histories still advance normally and serve as the exact oracle.
            override=self.device_engram_override
            self.device_engram_override=None
            if fast is not None:
                context=fast[0]
            elif override is not None and override.cpu().tolist()==contexts:
                context=override
                self.device_engram_override_applied+=1
            else:
                # A frame may be staged repeatedly. Do not overwrite pinned
                # host windows until the previous asynchronous H2D completed.
                if self.engram_hash_copy_done is not None:self.engram_hash_copy_done.synchronize()
                self.engram_hash_host.copy_(torch.tensor(contexts,dtype=torch.int32))
                self.engram_hash_context.copy_(self.engram_hash_host,non_blocking=True)
                event=torch.npu.Event();event.record();self.engram_hash_copy_done=event
                context=self.engram_hash_context
            hashes=device_hash.compute(context,self.engram_hash_output)
            for layer,buffer in self.rows.items():runtime.rows(layer,hashes[layer],buffer)
        except BaseException:
            for history,length in zip(histories,original_lengths):del history.history[length:]
            raise
        self.engram_hash_calls+=1
        self.engram_hash_tokens+=self.tokens

    def _stage_engram_rows(self,layer,indices,buffer):
        if os.environ.get('DEEPINFER_910C_ENGRAM_UVA')=='1':
            runtime=self.model.engram_uva_runtime
            if runtime is None:raise RuntimeError('UVA mode requires registered host tables')
            if layer not in self.engram_device_ids:
                self.engram_device_ids[layer]=torch.empty(indices.shape,dtype=torch.int64,device=buffer.device)
                self.engram_cpu_ids[layer]=torch.empty(indices.shape,dtype=torch.int64,device='cpu',pin_memory=True)
            self.engram_cpu_ids[layer].copy_(indices)
            self.engram_device_ids[layer].copy_(self.engram_cpu_ids[layer],non_blocking=True)
            runtime.rows(layer,self.engram_device_ids[layer],buffer)
            return
        if os.environ.get('DEEPINFER_910C_ENGRAM_PINNED_STAGING')!='1':
            buffer.copy_(self.model.reader.engram_rows(layer,indices,self.model.device).reshape(buffer.shape))
            return
        # Reuse page-locked inputs; preserve their lifetime until H2D ends.
        # This removes a synchronous temporary device allocation and a D2D
        # copy. Full-graph replay still starts after all its inputs are queued.
        if layer not in self.engram_host_rows:
            self.engram_host_rows[layer]=torch.empty_like(buffer,device='cpu',pin_memory=True)
        if layer in self.engram_copy_done:self.engram_copy_done[layer].synchronize()
        rows=self.model.reader.engram_rows(layer,indices,'cpu').reshape(buffer.shape)
        self.engram_host_rows[layer].copy_(rows)
        buffer.copy_(self.engram_host_rows[layer],non_blocking=True)
        event=torch.npu.Event();event.record();self.engram_copy_done[layer]=event
        self.engram_pinned_calls+=1
        self.engram_pinned_bytes+=buffer.numel()*buffer.element_size()

    def _prepare_sequence(self):
        guard_legacy(self,'seed')
        for source,pair in self.even.items():
            for tensor in pair:
                tensor.zero_()
            pending=self.model.pending.get(source,{})
            for row,slot in enumerate(self.slots):
                if slot in pending:
                    for target,value in zip(pair,pending[slot]):
                        target[row:row+1].copy_(value)
        self.prepared=True

    def _stage(self,chunks):
        guard_legacy(self,'stage')
        model=self.model
        if len(chunks)!=self.count or any(len(chunk)!=self.query_tokens for chunk in chunks):
            raise ValueError('verification chunks must match the fixed graph shape')
        tokens=[token for chunk in chunks for token in chunk]
        positions=[model.next_positions[slot] for slot in self.slots]
        if max(positions)+self.query_tokens>model.max_tokens:
            raise ValueError('verification lookahead exceeds native context capacity')
        override=self.device_input_override
        if override is not None:
            if (self.count!=4 or self.query_tokens!=6 or self.tokens!=24 or
                    tuple(override.shape)!=(4,6) or override.dtype!=torch.int32 or
                    override.device!=model.device or not override.is_contiguous()):
                raise ValueError('invalid explicitly staged fixed24 target input')
            # Published device state and draft tokens form the target input;
            # only the still-CPU-managed positions need a host transfer.
            self.cpu_inputs[self.tokens:].copy_(torch.tensor(
                [p+self.query_tokens for p in positions],dtype=torch.int32))
            self.inputs[self.tokens:].copy_(self.cpu_inputs[self.tokens:],non_blocking=True)
            self.inputs[:self.tokens].copy_(override.reshape(-1))
            self.device_input_override=None
            self.device_input_override_applied+=1
        else:
            self.cpu_inputs.copy_(torch.tensor(tokens+[p+self.query_tokens for p in positions],dtype=torch.int32))
            self.inputs.copy_(self.cpu_inputs,non_blocking=True)
        if model.histories:self._stage_engram([model.histories[slot] for slot in self.slots],chunks)

    @staticmethod
    def _plan(plane,positions,requests):
        indices=plane.physical_indices(positions[:,None],requests).squeeze(-1).long()
        return CacheWritePlan(indices,positions.shape[0])

    def compressed_plans(self,source,source_positions):
        ratio=self.model.cfg['compress_ratios'][source]
        logical=torch.div(source_positions,ratio,rounding_mode='floor')
        requests=self.query_requests
        if ratio==2:
            # Only odd tokens complete a real C2 pair. Temporary even-token
            # rows use distinct coordinates in the reserved scratch request.
            # This avoids aliased index_copy_ destinations for multi-token
            # verification while keeping every active request's cache causal.
            completed=self.positions.remainder(2)==1
            logical=torch.where(completed,logical,self.scratch_rows)
            requests=torch.where(completed,requests,self.model.requests-1)
        return tuple(self._plan(getattr(self.model.cache,name)[source],logical,requests)
                     for name in ('compressed','index_key','index_scale'))

    def compress(self,layer,x,positions):
        c=layer.config
        if layer.ratio==1:
            latent=rms_norm_mixed(project(x,layer.compressor_wkv_weight),layer.compressor_norm_weight,c['rms_norm_eps'])[0]
            return latent,positions
        current=project(x.float(),layer.compressor_wkv_weight.float()).reshape(self.count,self.query_tokens,-1)
        gate=project(x.float(),layer.compressor_wgate_weight.float()).reshape(self.count,self.query_tokens,-1)
        self.even_rows[layer.layer_id]=(current,gate)
        previous,previous_gate=self.even[layer.layer_id]
        for saved,value in zip(self.even_before[layer.layer_id],(previous,previous_gate)):
            saved.copy_(value)
        left=torch.cat((previous[:,None],current[:,:-1]),1)
        left_gate=torch.cat((previous_gate[:,None],gate[:,:-1]),1)
        pairs=torch.stack((left,current),2)
        scores=torch.stack((left_gate,gate),2)
        pooled=(pairs*scores.softmax(2)).sum(2).reshape(self.tokens,-1).bfloat16()
        latent=rms_norm_mixed(pooled,layer.compressor_norm_weight,c['rms_norm_eps'])[0]
        last=self.query_tokens-1-positions.reshape(self.count,self.query_tokens)[:,-1].remainder(2)
        valid=(last>=0)[:,None]
        last=last.clamp_min(0).long()
        previous.copy_(torch.where(valid,current[self.request_rows,last],previous))
        previous_gate.copy_(torch.where(valid,gate[self.request_rows,last],previous_gate))
        return latent,torch.div(positions,2,rounding_mode='floor')*2

    def _forward(self):
        from deepinfer.npu_ops.rope910c import begin_forward
        begin_forward()
        model=self.model
        used=self.inputs[self.tokens:]
        positions=(used[:,None]-self.query_tokens+self.query_offsets[None]).flatten()
        self.positions=positions
        batch=DSABatch((self.query_tokens,)*self.count,(model.max_tokens,)*self.count,
            self.cu_query,used,position_values=positions)
        plan=self._plan(model.cache.swa[0],positions,self.query_requests)
        state=PackedForward(batch,self.slots,positions,plan,self.tables['swa'][0],
            mc2_global_bs=model.mc2_global_bs,decode_frame=self)
        trunk=model.embedding.index_select(0,self.inputs[:self.tokens].long())[:,None].expand(-1,model.cfg['hc_mult'],-1).contiguous()
        previous=torch.zeros((self.tokens,model.cfg['hc_mult'],1),dtype=torch.float32,device=model.device)
        previous[:,0]=1.
        draft_features=[]
        for i,layer in enumerate(model.layers):
            if str(i) in model.engrams:
                trunk=model.engrams[str(i)].forward_rows(trunk,self.rows[i],model.engram_rotation,self.valid)
            if model.dspark is not None and i in model.dspark.target_layer_ids:
                draft_features.append(trunk.float().mean(-2).bfloat16())
            trunk,previous=layer(trunk,previous,cache=model.cache,pending=model.pending,state=state)
        if model.dspark is not None:
            model.dspark.update_main(draft_features,positions,self.query_requests,observer=self)
        hidden=(trunk.float()*previous).sum(-2).bfloat16()
        hidden=rms_norm_mixed(hidden,model.norm,model.cfg['rms_norm_eps'])[0]
        head=model.target_head if model.target_vocab_parallel and self.query_tokens>1 else model.head
        return project(hidden,head).float()

    @torch.inference_mode()
    def step(self,chunks):
        guard_legacy(self,'step')
        if (self._zos_ticket is not None or self._zos_commit_required or
                getattr(self.model,'_zos_prepared_frame',None) is not None or
                getattr(self.model,'_zos_commit_frame',None) is not None):
            raise RuntimeError('original step cannot bypass a held target ticket/commit')
        if not self.prepared:
            self._prepare_sequence()
        staged_at=time.perf_counter()
        self._stage(chunks)
        self.staging_seconds+=time.perf_counter()-staged_at
        return self._execute_graph()

    def prepare_target_inputs(self,token_input,window,*,held=(),collectives_authorized=False):
        return prepare_target_inputs(self,token_input,window,held=held,collectives_authorized=collectives_authorized)

    def replay_prepared_target(self,ticket):
        return replay_prepared_target(self,ticket)

    def publish_target_host(self,ticket,chunks):
        return publish_target_host(self,ticket,chunks)

    def discard_target_inputs(self,ticket):
        return discard_target_inputs(self,ticket)

    def _execute_graph(self,*,advance_host_positions=True):
        guard_execute(self,advance_host_positions)
        if self.graph is None:
            # Repeated writes at the same position are idempotent. C2 even
            # rows are invisible and odd steps retain the preceding even KV.
            # This warms real decode kernels without advancing CPU histories.
            initial={key:tuple(t.clone() for t in pair) for key,pair in self.even.items()}
            def restore_even():
                for key,pair in initial.items():
                    for target,value in zip(self.even[key],pair):
                        target.copy_(value)
            for _ in range(2):
                restore_even()
                self._forward()
            torch.npu.synchronize()
            restore_even()
            self.graph=torch.npu.NPUGraph()
            with torch.npu.graph(self.graph):
                self.output=self._forward()
            self.captures+=1
            restore_even()
        else:
            for layer in self.model.layers:
                layer.cann_attention_calls+=1
        # Capture records the model; it does not produce a usable inference
        # output. Launch once before consuming the first captured output too.
        if getattr(self,'forward_event_timing',False):
            start_event=torch.npu.Event(enable_timing=True)
            end_event=torch.npu.Event(enable_timing=True)
            start_event.record()
        self.graph.replay()
        if getattr(self,'forward_event_timing',False):
            end_event.record()
            self.last_forward_events=(start_event,end_event)
        self.replays+=1
        for layer in self.model.layers:
            if layer.fused_mc2_decode:
                layer.mc2.fused_calls+=1
            if layer.fused_o_proj and layer.fused_o_proj_decode:
                layer.fused_o_proj_decode_calls+=1
        if advance_host_positions:
            for slot in self.slots:
                self.model.next_positions[slot]+=self.query_tokens
        return self.output

    def commit_prefixes(self,bases,commits,device_pending=None):
        """Restore C2/Engram lengths after exact target rejection sampling.

        Keep target-written causal cache rows. Future rows are masked by the
        committed lengths; they cannot overwrite live SWA history in this
        enlarged ring. Restore the actual last even C2 activation, avoiding
        a second complete model forward solely to repair compressor state.
        """
        guard_legacy(self,'commit')
        if (self._zos_ticket is not None or getattr(self.model,'_zos_prepared_frame',None) is not None or
                getattr(self.model,'_zos_commit_frame',None) not in (None,self)):
            raise RuntimeError('publish held target history before committing prefixes')
        model=self.model
        if device_pending is not None:
            if self.count!=4 or self.query_tokens not in (1,6) or set(device_pending)!=set(self.even_rows):
                raise ValueError('invalid device C2 pending map')
            for source,(value,gate,valid,status) in device_pending.items():
                before=self.even_before[source]
                if (value.shape!=before[0].shape or gate.shape!=before[1].shape or
                        value.device!=model.device or gate.device!=model.device):
                    raise ValueError('invalid device C2 payload')
        for row,(slot,base,count) in enumerate(zip(self.slots,bases,commits)):
            end=base+count
            model.next_positions[slot]=end
            if model.histories:
                del model.histories[slot].history[end:]
            for source,values in self.even_rows.items():
                pending=model.pending.setdefault(source,{})
                if end%2:
                    if device_pending is None:
                        pending[slot]=tuple(value[row:row+1,count-1].clone() if count
                            else before[row:row+1].clone()
                            for value,before in zip(values,self.even_before[source]))
                    else:
                        pending[slot]=tuple(value[row:row+1] for value in device_pending[source][:2])
                else:
                    pending.pop(slot,None)
        self.prepared=False
        self._zos_commit_required=False
        self.model._zos_commit_frame=None
