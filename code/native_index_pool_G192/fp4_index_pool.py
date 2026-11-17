"""Native 68-byte/page-slot FP4 index storage for low-ratio NPU pools.

Preserves generic [page*64 payload | page*4 UE8M0 scale] wire layout.
The fixed-image native Triton quantize/store route is exercised unchanged.
"""
import torch
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4IndexerPool


class NPUFP4IndexerPool(DeepSeekV4IndexerPool):
    def __init__(self,*args,**kwargs):
        kwargs['use_fp4_indexer']=True
        super().__init__(*args,**kwargs)
        assert self.index_head_dim==128 and not self.uses_aiter_fp4_layout
        self.index_k_rne=True
        from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE
        self.dequant_table=DSV4_DEQUANT_FP4_TABLE.to(self.device)

    def get_index_k_dequant(
            self, layer_id: int, slots: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """Dequantized bf16 [n, index_head_dim] index K; `slots` None reads the pool."""
            from sglang.srt.layers.quantization.fp8 import DSV4_DEQUANT_FP4_TABLE

            assert self.use_fp4_indexer, "dequant readback only applies to the fp4 layout"
            buf = self.index_k_with_scale_buffer[layer_id - self.start_layer]
            if slots is None:
                slots = torch.arange(self.size, device=buf.device)
            slots = slots.to(torch.int64)
            # Page layout: see get_index_k_fp4.
            p = self.page_size
            page, off = (slots // p).unsqueeze(-1), slots % p
            payload_cols = (off * 64).unsqueeze(-1) + torch.arange(64, device=buf.device)
            scale_cols = (p * 64 + off * 4).unsqueeze(-1) + torch.arange(
                4, device=buf.device
            )
            u = buf[page, payload_cols].view(torch.uint8)  # [n, 64]
            codes = torch.stack([u & 0x0F, (u >> 4) & 0x0F], dim=-1)  # [n, 64, 2]
            vals = self.dequant_table[codes.long()].flatten(
                1
            )  # [n, 128]
            exps = buf[page, scale_cols].to(torch.int32) & 0xFF  # [n, 4]
            scales = torch.exp2(exps.float() - 127).repeat_interleave(32, dim=-1)
            return (vals * scales).to(torch.bfloat16)

    def set_index_fp4(self,layer_id,loc,cache_k):
        return super().set_index_fp4(layer_id,loc,cache_k)


def make_low_ratio_index_pool(size,page_size,dtype,index_head_dim,layer_num,device,enable_memory_saver,global_page_size=None):
    return NPUFP4IndexerPool(size,page_size,dtype,index_head_dim,layer_num,device,enable_memory_saver,global_page_size=global_page_size)
