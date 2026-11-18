"""Bit-preserving NPU gather of complex64 frequencies via an FP32 view."""
import torch
def gather(freqs,index):
 if freqs.device.type=='npu' and freqs.dtype==torch.complex64:
  return torch.view_as_complex(torch.view_as_real(freqs)[index].contiguous())
 return freqs[index]
