# SPDX-License-Identifier: Apache-2.0
"""Padded target graphs with genuinely variable per-request DSpark queries.

Real request boundaries/positions are staged into persistent inputs. Padding
uses the fifth, private scratch request; its logits never reach the sampler.
The 40-layer target receives each request's actual query length through its
device cu_seqlens, so low-confidence positions need not be computed.
"""
import time

import torch
import torch.nn.functional as F

from deepinfer.npu_ops.dsa910c import DSABatch
from .ascend_w8a8_packed import PackedForward
from .w8a8_graph910c import NativeDecodeGraph
from .w8a8_math910c import rms_norm_mixed


class DSparkVarlenGraph(NativeDecodeGraph):
    def __init__(self,model,capacity):
        if capacity not in (8,) or model.requests!=5:
            raise ValueError('variable DSpark uses four real slots plus one scratch slot')
        super().__init__(model,list(range(5)))
        self.capacity=capacity
        self.tokens=capacity
        self.inputs=torch.zeros(3*capacity+11,dtype=torch.int32,device=model.device)
        self.cpu_inputs=torch.empty_like(self.inputs,device='cpu',pin_memory=True)
        self.ids=self.inputs[:capacity]
        self.used=self.inputs[capacity:capacity+5]
        self.cu_query=self.inputs[capacity+5:capacity+11]
        self.query_requests=self.inputs[capacity+11:2*capacity+11]
        self.positions=self.inputs[2*capacity+11:]
        self.scratch_rows=torch.arange(capacity,dtype=torch.int32,device=model.device)
        self.previous_rows=(torch.arange(capacity,dtype=torch.int64,device=model.device)-1).clamp_min(0)
        self.valid=torch.ones(capacity,dtype=torch.bool,device=model.device)
        self.rows={i:torch.empty((capacity,buffer.shape[-1]),dtype=buffer.dtype,
                                device=model.device) for i,buffer in self.rows.items()}
        self.host_lengths=None
        self.host_cu=None
        self.host_starts=None

    def _stage(self,chunks):
        if len(chunks)!=4 or any(not 1<=len(row)<=6 for row in chunks):
            raise ValueError('four DSpark query prefixes of length1..6 required')
        total=sum(map(len,chunks))
        if total>=self.capacity:
            raise ValueError('variable target graph needs a positive scratch-query span')
        lengths=list(map(len,chunks))+[self.capacity-total]
        starts=[self.model.next_positions[slot] for slot in range(4)]+[1024]
        if any(start+length>self.model.max_tokens for start,length in zip(starts,lengths)):
            raise ValueError('variable verification exceeds context capacity')
        cu=[0]
        for length in lengths:cu.append(cu[-1]+length)
        rows=[list(row) for row in chunks]+[[self.model.cfg['dspark_noise_token_id']] * lengths[4]]
        positions=[pos for start,length in zip(starts,lengths) for pos in range(start,start+length)]
        requests=[slot for slot,length in enumerate(lengths) for _ in range(length)]
        ids=[token for row in rows for token in row]
        values=ids+[start+length for start,length in zip(starts,lengths)]+cu+requests+positions
        self.cpu_inputs.copy_(torch.tensor(values,dtype=torch.int32))
        self.inputs.copy_(self.cpu_inputs,non_blocking=True)
        self.model.histories[4].reset()
        hashes=[self.model.histories[slot].batch(row) for slot,row in enumerate(rows) if row]
        for layer,buffer in self.rows.items():
            indices=torch.cat([entry[layer] for entry in hashes])
            buffer.copy_(self.model.reader.engram_rows(layer,indices,self.model.device).reshape(self.capacity,-1))
        self.host_lengths,self.host_starts,self.host_cu=lengths,starts,cu

    def compress(self,layer,x,positions):
        if layer.ratio==1:
            latent=rms_norm_mixed(F.linear(x,layer.compressor_wkv_weight),
                layer.compressor_norm_weight,layer.config['rms_norm_eps'])[0]
            return latent,positions
        current=F.linear(x.float(),layer.compressor_wkv_weight.float())
        gate=F.linear(x.float(),layer.compressor_wgate_weight.float())
        self.even_rows[layer.layer_id]=(current,gate)
        previous,previous_gate=self.even[layer.layer_id]
        for saved,value in zip(self.even_before[layer.layer_id],(previous,previous_gate)):
            saved.copy_(value)
        requests=self.query_requests.long()
        offsets=torch.arange(self.capacity,dtype=torch.int32,device=x.device)-self.cu_query.index_select(0,requests)
        left=torch.where((offsets==0)[:,None],previous.index_select(0,requests),
                         current.index_select(0,self.previous_rows))
        left_gate=torch.where((offsets==0)[:,None],previous_gate.index_select(0,requests),
                              gate.index_select(0,self.previous_rows))
        pairs=torch.stack((left,current),1)
        scores=torch.stack((left_gate,gate),1)
        pooled=(pairs*scores.softmax(1)).sum(1).bfloat16()
        latent=rms_norm_mixed(pooled,layer.compressor_norm_weight,layer.config['rms_norm_eps'])[0]
        return latent,torch.div(positions,2,rounding_mode='floor')*2

    def _forward(self):
        from deepinfer.npu_ops.rope910c import begin_forward
        begin_forward()
        model=self.model
        # Host tuples supply static tiling bounds and token count only. The
        # five actual lengths are carried by persistent cu_query/used inputs.
        query_bounds=(0,0,0,0,self.capacity)
        batch=DSABatch(query_bounds,(model.max_tokens,)*5,self.cu_query,
                       self.used,position_values=self.positions)
        plan=self._plan(model.cache.swa[0],self.positions,self.query_requests)
        state=PackedForward(batch,self.slots,self.positions,plan,self.tables['swa'][0],
                            mc2_global_bs=model.mc2_global_bs,decode_frame=self)
        trunk=model.embedding.index_select(0,self.ids.long())[:,None].expand(
            -1,model.cfg['hc_mult'],-1).contiguous()
        previous=torch.zeros((self.capacity,model.cfg['hc_mult'],1),dtype=torch.float32,device=model.device)
        previous[:,0]=1.
        features=[]
        for i,layer in enumerate(model.layers):
            if str(i) in model.engrams:
                trunk=model.engrams[str(i)].forward_rows(trunk,self.rows[i],model.engram_rotation,self.valid)
            if i in model.dspark.target_layer_ids:
                features.append(trunk.float().mean(-2).bfloat16())
            trunk,previous=layer(trunk,previous,cache=model.cache,pending=model.pending,state=state)
        model.dspark.update_main(features,self.positions,self.query_requests)
        hidden=(trunk.float()*previous).sum(-2).bfloat16()
        hidden=rms_norm_mixed(hidden,model.norm,model.cfg['rms_norm_eps'])[0]
        return F.linear(hidden,model.head).float()

    @torch.inference_mode()
    def step(self,chunks):
        if not self.prepared:self._prepare_sequence()
        staged=time.perf_counter()
        self._stage(chunks)
        self.staging_seconds+=time.perf_counter()-staged
        if self.graph is None:
            for _ in range(2):self._forward()
            torch.npu.synchronize()
            self.graph=torch.npu.NPUGraph()
            with torch.npu.graph(self.graph):self.output=self._forward()
            self.captures+=1
        else:
            for layer in self.model.layers:layer.cann_attention_calls+=1
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
            if layer.fused_mc2_decode:layer.mc2.fused_calls+=1
            if layer.fused_o_proj and layer.fused_o_proj_decode:
                layer.fused_o_proj_decode_calls+=1
        for slot,(start,length) in enumerate(zip(self.host_starts,self.host_lengths)):
            self.model.next_positions[slot]=start+length
        return self.output

    def request_logits(self,output):
        return [output[self.host_cu[slot]:self.host_cu[slot+1]] for slot in range(4)]

    def commit_prefixes(self,bases,commits):
        model=self.model
        for row,(base,count) in enumerate(zip(bases,commits)):
            end=base+count
            model.next_positions[row]=end
            del model.histories[row].history[end:]
            for source,values in self.even_rows.items():
                pending=model.pending.setdefault(source,{})
                if end%2:
                    offset=self.host_cu[row]+count-1
                    pending[row]=tuple(value[offset:offset+1].clone() if count
                        else before[row:row+1].clone()
                        for value,before in zip(values,self.even_before[source]))
                else:pending.pop(row,None)
        self.prepared=False
