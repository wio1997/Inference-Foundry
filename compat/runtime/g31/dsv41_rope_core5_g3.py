"""CANN9.2 core5 RoPE, exact standard library; no custom-first lookup."""
from functools import cache
from pathlib import Path
import os
CORE5_LIBRARY='/usr/local/Ascend/cann-9.2.0/lib64/libopapi_transformer.so'

@cache
def _operator():
 import torch
 if not hasattr(torch,'npu') or not torch.npu.is_initialized():raise RuntimeError('Initialize target NPU first')
 ext=Path(os.environ['SGLANG_V41_ROPE_CORE5_EXTENSION'])
 if not ext.is_absolute() or not ext.is_file():raise ValueError('Expected explicit currentTorch2.13 core5 extension')
 if not Path(CORE5_LIBRARY).is_file():raise ValueError('Exact standard CANN9.2 library missing')
 torch.ops.load_library(str(ext))
 return torch.ops.sglang_v41_rope_core5_g3.inplace_partial_rotary_mul

def inplace_partial_rotary_mul(x,cos,sin,*,rotary_mode='interleave',partial_slice=None):
 if rotary_mode!='interleave':raise ValueError('V41 uses interleave only')
 if partial_slice is None:partial_slice=[0,x.shape[-1]]
 _operator()(x,cos,sin,rotary_mode,partial_slice)
 return x
