# SPDX-License-Identifier: Apache-2.0
"""910C INT8 compute, matching Ascend W8A8_DYNAMIC weight contracts.

Operator sequence/layout derived from vLLM-Ascend w8a8_dynamic.py. There is no
BF16 weight fallback: INT8 weights feed quant_matmul/grouped_matmul directly.
"""
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def configure_native_runtime():
    """Match the explicit Ascend platform setup at the NPU boundary only."""
    import torch_npu
    torch_npu.npu.config.allow_internal_format=True
    torch_npu.npu.set_compile_mode(jit_compile=False)


class AscendW8A8Linear(torch.nn.Module):
    def __init__(self,weight_nk,scale,*,nz=True):
        super().__init__()
        import torch_npu
        configure_native_runtime()
        if weight_nk.ndim!=2 or weight_nk.dtype!=torch.int8 or scale.shape!=(weight_nk.shape[0],):
            raise ValueError('expected INT8 weight [N,K] and channel scales [N]')
        if scale.dtype not in (torch.bfloat16,torch.float32) or scale.device!=weight_nk.device:
            raise ValueError('weight and BF16/F32 scales must share a device')
        self.input_size=weight_nk.shape[1];self.output_size=weight_nk.shape[0]
        weight=weight_nk.T.contiguous()
        if nz:
            torch_npu.npu.config.allow_internal_format=True
            weight=torch_npu.npu_format_cast(weight,29)
        self.weight=torch.nn.Parameter(weight,requires_grad=False)
        self.register_buffer('weight_scale',scale.contiguous())

    def forward(self,x):
        import torch_npu
        if x.dtype!=torch.bfloat16 or x.shape[-1]!=self.input_size:raise ValueError('expected BF16 input with matching K')
        shape=x.shape[:-1];flat=x.reshape(-1,self.input_size)
        if flat.shape[0]==0:return x.new_empty((*shape,self.output_size))
        quantized,token_scale=torch_npu.npu_dynamic_quant(flat,dst_type=torch.int8)
        output=torch_npu.npu_quant_matmul(quantized,self.weight,self.weight_scale,pertoken_scale=token_scale,output_dtype=x.dtype)
        return output.reshape(*shape,self.output_size)


class AscendW8A8Expert(torch.nn.Module):
    """Single local expert with actual grouped INT8 matmuls and BF16 SwiGLU.

Expert dispatch and output route weighting belong to the caller. GMM1 output
is BF16 before clamp/SwiGLU and dynamic INT8 quantization for GMM2, matching the
unfused vLLM-Ascend W8A8 compute decomposition.
"""
    def __init__(self,w1,s1,w3,s3,w2,s2,*,limit=10.,nz=True):
        super().__init__();self.limit=limit;self._group_counts={}
        import torch_npu
        configure_native_runtime()
        if any(w.dtype!=torch.int8 for w in (w1,w3,w2)) or w1.shape!=w3.shape or w2.shape!=w1.T.shape:
            raise ValueError('invalid INT8 expert shapes')
        if any(w.ndim!=2 for w in (w1,w3,w2)) or any(s.shape!=(w.shape[0],) for w,s in ((w1,s1),(w3,s3),(w2,s2))):
            raise ValueError('expert scales must have one entry per output channel')
        if len({t.device for t in (w1,s1,w3,s3,w2,s2)})!=1 or any(s.dtype!=torch.bfloat16 for s in (s1,s3,s2)):
            raise ValueError('expert weights and BF16 scales must share a device')
        self.hidden=w1.shape[1];self.intermediate=w1.shape[0]
        first=torch.cat((w1,w3),0).T.contiguous().unsqueeze(0)
        second=w2.T.contiguous().unsqueeze(0)
        if nz:
            torch_npu.npu.config.allow_internal_format=True
            first=torch_npu.npu_format_cast(first,29);second=torch_npu.npu_format_cast(second,29)
        self.first=torch.nn.Parameter(first,requires_grad=False);self.second=torch.nn.Parameter(second,requires_grad=False)
        # This unfused A8W8 GMM returns BF16 and CANN requires BF16 scale.
        # The vendor's FP32 first scale belongs to its fused/int32 branch.
        self.register_buffer('first_scale',torch.cat((s1,s3)).reshape(1,-1).contiguous())
        self.register_buffer('second_scale',s2.reshape(1,-1).contiguous())

    def _gmm(self,x,weight,scale,*,return_quantized=False):
        import torch_npu
        codes,token_scale=torch_npu.npu_dynamic_quant(x,dst_type=torch.int8)
        # Create device constants asynchronously. Request arrival order can
        # change the packed prefill shape after warmup; a host-to-device
        # torch.tensor copy here would synchronize once in every model layer.
        groups=self._group_counts.get(x.shape[0])
        if groups is None:
            groups=torch.full((1,),x.shape[0],dtype=torch.int64,device=x.device)
            self._group_counts[x.shape[0]]=groups
        output=torch_npu.npu_grouped_matmul(x=[codes],weight=[weight],scale=[scale],per_token_scale=[token_scale],group_list=groups,group_list_type=0,split_item=2,group_type=0,output_dtype=x.dtype)[0]
        return (output,codes,token_scale) if return_quantized else output

    def forward(self,x,*,return_stages=False):
        if x.ndim!=2 or x.shape[1]!=self.hidden or x.dtype!=torch.bfloat16:raise ValueError('expected BF16 [tokens,H]')
        if x.shape[0]==0:return torch.empty_like(x)
        if return_stages:
            combined,q1,s1=self._gmm(x,self.first,self.first_scale,return_quantized=True)
        else:combined=self._gmm(x,self.first,self.first_scale)
        gate,up=combined.chunk(2,-1)
        gate=gate.float();up=up.float()
        if self.limit is not None:gate=gate.clamp_max(self.limit);up=up.clamp(-self.limit,self.limit)
        activated=(torch.nn.functional.silu(gate)*up).bfloat16()
        if return_stages:
            output,q2,s2=self._gmm(activated,self.second,self.second_scale,return_quantized=True)
            return output,{'combined':combined,'input_codes':q1,'input_scale':s1,'activated':activated,'second_codes':q2,'second_scale':s2}
        return self._gmm(activated,self.second,self.second_scale)


class AscendW8A8ExpertBank(torch.nn.Module):
    """Resident local expert bank; two grouped matmuls for all received routes.

    Follows the pinned Ascend W8A8 dynamic grouped-matmul path. Group counts
    include zero-token experts. Route weights remain FP32 after GMM2.
    """
    def __init__(self, experts):
        super().__init__()
        import torch_npu
        experts=list(experts)
        self.count=len(experts)
        self.limit=experts[0].limit
        for name in ('first','second'):
            nd=torch.cat([torch_npu.npu_format_cast(getattr(e,name),2) for e in experts],0)
            self.register_parameter(name,torch.nn.Parameter(torch_npu.npu_format_cast(nd,29),requires_grad=False))
        for name in ('first_scale','second_scale'):
            self.register_buffer(name,torch.cat([getattr(e,name) for e in experts],0).contiguous())

    def prepare_fused_mc2(self):
        """Prepare the pinned dispatch_ffn_combine scale ABI once at load.

        The Ascend W8A8 implementation sign-extends the FP32 bit pattern to
        INT64; this is a bit representation, not a numeric scale conversion.
        The existing NZ INT8 banks already have its [E,K,N] weight layout.
        """
        if hasattr(self, 'fused_first_scale'):
            return
        for source, target in (('first_scale', 'fused_first_scale'),
                               ('second_scale', 'fused_second_scale')):
            scale = getattr(self, source)
            packed = scale.detach().cpu().float().contiguous().view(torch.int32).long()
            self.register_buffer(target, packed.flatten().to(scale.device))
        self.register_buffer('fused_expert_tokens',
                             torch.zeros(self.count, dtype=torch.int32, device=self.first.device))

    def forward(self,x,local_ids):
        if not x.shape[0]:
            return torch.empty_like(x)
        order=torch.argsort(local_ids.float())
        values=x.index_select(0,order).contiguous()
        groups=torch.bincount(local_ids,minlength=self.count).cumsum(0)
        result=self.forward_grouped(values,groups)
        output=torch.empty_like(result)
        output.index_copy_(0,order,result)
        return output

    def forward_grouped(self,x,groups,*,group_list_type=0):
        """Consume routes already grouped by the native MC2 dispatcher."""
        import torch_npu
        codes,scale=torch_npu.npu_dynamic_quant(x,dst_type=torch.int8)
        combined=torch_npu.npu_grouped_matmul(x=[codes],weight=[self.first],scale=[self.first_scale],
            per_token_scale=[scale],group_list=groups,group_list_type=group_list_type,
            split_item=2,group_type=0,output_dtype=x.dtype)[0]
        gate,up=combined.chunk(2,-1)
        gate,up=gate.float(),up.float()
        if self.limit is not None:
            gate=gate.clamp_max(self.limit);up=up.clamp(-self.limit,self.limit)
        activated=(torch.nn.functional.silu(gate)*up).bfloat16()
        codes,scale=torch_npu.npu_dynamic_quant(activated,dst_type=torch.int8)
        result=torch_npu.npu_grouped_matmul(x=[codes],weight=[self.second],scale=[self.second_scale],
            per_token_scale=[scale],group_list=groups,group_list_type=group_list_type,
            split_item=2,group_type=0,output_dtype=x.dtype)[0]
        return result
