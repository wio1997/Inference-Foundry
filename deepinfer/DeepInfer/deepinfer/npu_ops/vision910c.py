"""NPU leaf interfaces for the unchanged original BF16 vision model."""
import torch
import torch_npu


def _npu(*values):
    if any(not isinstance(value, torch.Tensor) or value.device.type != 'npu' for value in values):
        raise TypeError('vision computation requires NPU tensors')
    if len({value.device for value in values}) > 1:
        raise ValueError('vision tensors must share one NPU')


def norm(x, weight, residual=None, eps=1e-6):
    _npu(x, weight, *(() if residual is None else (residual,)))
    if x.shape[-1:] != weight.shape or (residual is not None and residual.shape != x.shape):
        raise ValueError('invalid original vision RMSNorm dimensions')
    # Preserve the original BF16 materialization boundary before statistics.
    added = x if residual is None else x + residual
    output = torch_npu.npu_rms_norm(added, weight, epsilon=eps)[0]
    return output if residual is None else (output, added)


def dtype_to_str(dtype):
    return {torch.bfloat16:'bfloat16', torch.float32:'float32'}[dtype]


def swiglu_forward(x, out_dtype):
    _npu(x)
    if out_dtype != dtype_to_str(x.dtype) or x.shape[-1] % 2:
        raise ValueError('invalid original vision SwiGLU input')
    return torch_npu.npu_swiglu(x, dim=-1)


def apply_rotary(q, rope, k, *, interleaved=False):
    _npu(q, rope, k)
    if q.shape != k.shape or q.ndim != 3 or rope.shape != (q.shape[0], q.shape[-1]) or q.shape[-1] % 2:
        raise ValueError('invalid original vision RoPE dimensions')
    cos, sin = (part[:,None] for part in rope.float().chunk(2, dim=-1))
    for value in (q, k):
        if interleaved:
            first, second = value[...,0::2].float(), value[...,1::2].float()
            output = torch.stack((first*cos-second*sin, second*cos+first*sin), -1).flatten(-2)
        else:
            first, second = value.float().chunk(2, dim=-1)
            output = torch.cat((first*cos-second*sin, second*cos+first*sin), dim=-1)
        value.copy_(output.to(value.dtype))
