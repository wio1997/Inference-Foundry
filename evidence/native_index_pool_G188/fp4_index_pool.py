"""Native 68-byte/page-slot FP4 index storage for low-ratio NPU pools.

Preserves generic [page*64 payload | page*4 UE8M0 scale] wire layout.
Only the unsupported CUDA quantize/store route is implemented in torch.
"""
import torch
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool
from sglang.kernels.ops.quantization.fp4_fake_quant import block_scale,round_fp4


def quantize_index_k(x):
    assert x.ndim==2 and x.shape[1]==128 and x.dtype==torch.bfloat16
    scale=block_scale(x,32,6.0,6.0*2.0**-126)
    normalized=x.float().reshape(-1,4,32)/scale.unsqueeze(-1)
    rounded=round_fp4(normalized.clamp(-6.0,6.0)).reshape(-1,128)
    magnitude=rounded.abs()
    code=torch.zeros_like(magnitude,dtype=torch.uint8)
    for boundary in [.25,.75,1.25,1.75,2.5,3.5,5.0]:code=code+(magnitude>boundary).to(torch.uint8)
    code=code|(((rounded<0)&(code!=0)).to(torch.uint8)<<3)
    payload=code[:,0::2]|(code[:,1::2]<<4)
    exponents=((scale.contiguous().view(torch.int32)>>23)&255).to(torch.uint8)
    return payload,exponents


class NPUFP4IndexerPool(DeepSeekV4IndexerPool):
    def __init__(self,*args,**kwargs):
        kwargs['use_fp4_indexer']=True
        super().__init__(*args,**kwargs)
        assert self.index_head_dim==128 and not self.uses_aiter_fp4_layout
        self.index_k_rne=True
        from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE
        self.dequant_table=DSV4_DEQUANT_FP4_TABLE.to(self.device)

    def get_index_k_dequant(self,layer_id,slots=None):
        if slots is None:slots=torch.arange(self.size,device=self.device)
        payload,packed_scale=self.get_index_k_fp4(layer_id,slots)
        u=payload.view(torch.uint8)
        codes=torch.stack([u&15,(u>>4)&15],dim=-1)
        values=self.dequant_table[codes.long()].flatten(1)
        exponents=(packed_scale.unsqueeze(-1)>>(torch.arange(4,device=self.device)*8))&255
        scales=(exponents.to(torch.int32)<<23).view(torch.float32).repeat_interleave(32,dim=-1)
        return (values*scales).to(torch.bfloat16)

    def set_index_fp4(self,layer_id,loc,cache_k):
        payload,scale=quantize_index_k(cache_k.contiguous())
        buf=self.index_k_with_scale_buffer[layer_id-self.start_layer]
        slots=loc.to(torch.int64);pages=(slots//self.page_size).unsqueeze(-1);offset=slots%self.page_size
        columns=offset.unsqueeze(-1)*64+torch.arange(64,device=loc.device)
        scale_columns=self.page_size*64+offset.unsqueeze(-1)*4+torch.arange(4,device=loc.device)
        buf[pages,columns]=payload
        buf[pages,scale_columns]=scale


def make_low_ratio_index_pool(size,page_size,dtype,index_head_dim,layer_num,device,enable_memory_saver,global_page_size=None):
    return NPUFP4IndexerPool(size,page_size,dtype,index_head_dim,layer_num,device,enable_memory_saver,global_page_size=global_page_size)
