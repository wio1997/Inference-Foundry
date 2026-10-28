# SPDX-License-Identifier: Apache-2.0
"""Native W8A8 decoder layer with BF16 SWA/source-shared compression and mHC.

The eager short-context path is exact with respect to sparse key selection
while every available compressed key fits index_topk. INT8 operators and
HCCL run only when explicitly constructing/executing this layer on an NPU.
"""
import torch
import torch.distributed as dist

from deepinfer.npu_ops import mhc910c as mhc
from deepinfer.npu_ops.attention910c import prefill
from deepinfer.npu_ops.moe910c import dispatch_combine,route
from deepinfer.npu_ops.w8a8 import AscendW8A8Expert
from .w8a8_math910c import rms_norm_mixed,rope_frequencies,device_rope_frequencies,rotate_partial,compress_step


class AscendW8A8Layer(torch.nn.Module):
    def __init__(self,reader,device,*,layer_id=0,group=None):
        super().__init__();self.reader=reader;self.device=torch.device(device);self.layer_id=layer_id;self.group=group
        self.config=reader.config['text_config'];c=self.config
        self.ratio=c['compress_ratios'][layer_id]
        self.kv_source=max((s for s in c['kv_source_layer_ids'] if s<=layer_id),default=None) if self.ratio else None
        if self.ratio not in (0,1,2) or (self.ratio and self.kv_source is None):raise ValueError('invalid source compression topology')
        self.prefix=f'layers.{layer_id}';self.expert_cache=torch.nn.ModuleDict()
        self.projections=torch.nn.ModuleDict({name:reader.load_linear(f'{self.prefix}.attn.{name}',device) for name in ('wq_a','wq_b','wkv','wo_b')})
        for name in ['hc_attn_fn','hc_attn_scale','hc_attn_base','hc_ffn_fn','hc_ffn_scale','hc_ffn_base','attn_norm.weight','ffn_norm.weight','attn.q_norm.weight','attn.kv_norm.weight','attn.attn_sink','ffn.gate.weight','ffn.gate.bias']:
            value=reader.tensor(f'{self.prefix}.{name}')[0].to(device)
            if name in ('attn.attn_sink','ffn.gate.weight','ffn.gate.bias'):value=value.float()
            self.register_buffer(name.replace('.','_'),value)
        weight=reader.linear_payload(f'{self.prefix}.attn.wo_a')[0]
        self.register_buffer('wv',weight.to(device).reshape(c['o_groups'],c['o_lora_rank'],-1))
        self.shared_expert=self._load_expert('shared_experts')
        self.register_buffer('rope_freq',device_rope_frequencies(c,device,compressed=bool(self.ratio)))
        if layer_id in c['kv_source_layer_ids']:
            for name in ('wkv.weight','norm.weight')+ (('wgate.weight',) if self.ratio==2 else ()):
                value=reader.tensor(f'{self.prefix}.attn.compressor.{name}')[0].to(device)
                if self.ratio==2 and name!='norm.weight':value=value.float()
                self.register_buffer('compressor_'+name.replace('.','_'),value)

    def load_local_experts(self):
        """Make all owned expert INT8 weights resident, once per engine load."""
        world=dist.get_world_size(self.group) if dist.is_initialized() else 1
        rank=dist.get_rank(self.group) if dist.is_initialized() else 0
        count=self.config['n_routed_experts']
        if count%world:raise ValueError('expert count must divide EP world size')
        for expert_id in range(rank*(count//world),(rank+1)*(count//world)):
            key=str(expert_id)
            if key not in self.expert_cache:self.expert_cache[key]=self._load_expert(f'experts.{expert_id}')

    def _load_expert(self,expert):
        prefix=f'{self.prefix}.ffn.{expert}'
        payload=[self.reader.linear_payload(prefix+f'.w{i}') for i in (1,3,2)]
        if any(scale is None for _,scale in payload):raise ValueError('expected actual W8A8 expert payload')
        return AscendW8A8Expert(*[value.to(self.device) for pair in payload for value in pair],limit=self.config['swiglu_limit'])

    def _pre(self,trunk,previous,kind):
        c=self.config;m=c['hc_mult'];x=trunk.unsqueeze(0)
        value=mhc.mhc_pre_apply_mix(x,previous.unsqueeze(0)).squeeze(0)
        mixes=mhc.mhc_pre_norm_fn(x,getattr(self,f'hc_{kind}_fn'),None,c['rms_norm_eps'])
        pre,post,comb=mhc.mhc_pre_split_mixes(mixes,getattr(self,f'hc_{kind}_scale'),getattr(self,f'hc_{kind}_base'),m,2.,c['hc_eps'])
        comb=mhc.sinkhorn_normalize(comb,c['hc_sinkhorn_iters'],c['hc_eps'])
        return value,pre.squeeze(0),post.squeeze(0),comb.squeeze(0)

    def attention(self,x,position,cache):
        if x.shape[0]!=1:raise ValueError('eager integration accepts one sequence position per step')
        c=self.config
        q_a=self.projections['wq_a'](x)
        q_a,_=rms_norm_mixed(q_a,self.attn_q_norm_weight,c['rms_norm_eps'])
        q=self.projections['wq_b'](q_a).reshape(1,c['num_attention_heads'],c['head_dim'])
        kv=self.projections['wkv'](x)
        kv,_=rms_norm_mixed(kv,self.attn_kv_norm_weight,c['rms_norm_eps'])
        q=rotate_partial(q,position,self.rope_freq)
        kv=rotate_partial(kv,position,self.rope_freq)
        window=cache.swa.get(self.layer_id)
        window=kv if window is None else torch.cat((window,kv))
        cache.swa[self.layer_id]=window=window[-c['sliding_window']:]
        if self.layer_id in c['kv_source_layer_ids']:
            pending=cache.pending.setdefault(self.layer_id,[])
            latent,source_pos=compress_step(x,position,self.ratio,self.compressor_wkv_weight,getattr(self,'compressor_wgate_weight',None),self.compressor_norm_weight,c['rms_norm_eps'],pending)
            if latent is not None:
                latent=rotate_partial(latent,source_pos,self.rope_freq)
                old=cache.compressed.get(self.layer_id)
                cache.compressed[self.layer_id]=latent if old is None else torch.cat((old,latent))
        if self.ratio:
            long=cache.compressed.get(self.kv_source)
            if long is not None:
                if long.shape[0]>c['index_topk']:raise ValueError('compressed key count exceeds exact select-all context limit')
                window=torch.cat((window,long))
        indices=torch.arange(window.shape[0],device=x.device,dtype=torch.int32)[None]
        context,_=prefill(q,window[:,None,:],indices,c['head_dim']**-.5,self.attn_attn_sink)
        context=rotate_partial(context,position,self.rope_freq,inverse=True)
        grouped=context.reshape(1,c['o_groups'],-1)
        projected=torch.einsum('bgr,gdr->bgd',grouped.float(),self.wv.float()).bfloat16()
        return self.projections['wo_b'](projected.flatten(1))

    def routed(self,x,router_input):
        c=self.config
        logits=torch.nn.functional.linear(router_input,self.ffn_gate_weight)
        ids,weights=route(logits,self.ffn_gate_bias,c['num_experts_per_tok'],1.)
        def local_expert(expert_id,tokens,_weights):
            key=str(expert_id)
            if key not in self.expert_cache:self.expert_cache[key]=self._load_expert(f'experts.{expert_id}')
            return self.expert_cache[key](tokens)
        output=dispatch_combine(x,ids,weights,c['n_routed_experts'],local_expert,self.group,weight_on_output=True)
        return output,ids,weights

    def forward(self,trunk,previous,*,position,cache,return_stages=False):
        if not 0<=position<cache.max_tokens or trunk.shape!=(1,self.config['hc_mult'],self.config['hidden_size']):
            raise ValueError('invalid sequence position or mHC trunk shape')
        c=self.config
        x,pre,post,comb=self._pre(trunk,previous,'attn')
        x,_=rms_norm_mixed(x,self.attn_norm_weight,c['rms_norm_eps'])
        attn=self.attention(x,position,cache)
        after_attn=mhc.mhc_post(attn,trunk,post,comb)
        x,next_pre,post,comb=self._pre(after_attn,pre,'ffn')
        x,router_x=rms_norm_mixed(x,self.ffn_norm_weight,c['rms_norm_eps'])
        shared=self.shared_expert(x);routed,ids,weights=self.routed(x,router_x)
        ffn=(shared.float()+routed.float()*c['routed_scaling_factor']).bfloat16()
        output=mhc.mhc_post(ffn,after_attn,post,comb)
        if return_stages:return output,next_pre,{'attention':attn,'after_attention':after_attn,'ffn_input':x,'shared_expert':shared,'routed_experts_unscaled':routed,'ffn':ffn,'output_trunk':output,'next_pre_mix':next_pre,'topk_ids':ids,'topk_weights':weights}
        return output,next_pre


class AscendW8A8SwaLayer(AscendW8A8Layer):
    """Position-zero probe wrapper around the same complete production layer."""
    def __init__(self,reader,device,*,layer_id=0,group=None):
        if reader.config['text_config']['compress_ratios'][layer_id]!=0:raise ValueError('SWA probe requires an uncompressed layer')
        super().__init__(reader,device,layer_id=layer_id,group=group)

    def forward(self,trunk,previous,*,position=0,return_stages=False):
        from .w8a8_math910c import W8A8SequenceCache
        if position!=0:raise ValueError('SWA probe only accepts position zero')
        return super().forward(trunk,previous,position=0,cache=W8A8SequenceCache(1),return_stages=return_stages)
