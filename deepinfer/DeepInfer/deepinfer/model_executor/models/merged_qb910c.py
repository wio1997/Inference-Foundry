"""Official shared-input Q_b/index_Q_b projection using unchanged W8A8 operands."""
import torch


def concatenate_payloads(weights, scales):
    if len(weights) != 2 or len(scales) != 2:
        raise ValueError("two original Q_b projections are required")
    first, second = weights
    a, b = scales
    if (first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]
            or first.dtype != torch.int8 or second.dtype != torch.int8 or first.device != second.device):
        raise ValueError("expected same-device INT8 [N,K] matrices with common K")
    if (a.shape != (first.shape[0],) or b.shape != (second.shape[0],)
            or a.dtype != b.dtype or a.dtype not in (torch.bfloat16, torch.float32)
            or a.device != first.device or b.device != second.device):
        raise ValueError("channel scales must preserve the original dtype, device and output order")
    return torch.cat((first, second), dim=0).contiguous(), torch.cat((a, b)).contiguous()


class MergedQB(torch.nn.Module):
    def __init__(self, primary, index):
        super().__init__()
        import torch_npu
        from deepinfer.npu_ops.w8a8 import AscendW8A8Linear

        if type(primary) is not AscendW8A8Linear or type(index) is not AscendW8A8Linear:
            raise ValueError("only the validated bias-free dynamic W8A8 modules can be merged")
        if primary.input_size != index.input_size:
            raise ValueError("main and index query must share the normalized q_a input")
        self.primary_size, self.index_size = primary.output_size, index.output_size
        originals = tuple(torch_npu.npu_format_cast(layer.weight, 2).T.contiguous() for layer in (primary, index))
        weight, scale = concatenate_payloads(originals, (primary.weight_scale, index.weight_scale))
        self.linear = AscendW8A8Linear(weight, scale, nz=True)
        self.input_size = self.linear.input_size
        self.output_size = self.linear.output_size

    def forward(self, normalized_q_a):
        joined = self.linear(normalized_q_a)
        return joined.split((self.primary_size, self.index_size), dim=-1)
