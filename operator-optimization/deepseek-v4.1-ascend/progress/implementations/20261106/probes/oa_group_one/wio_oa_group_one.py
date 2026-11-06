"""One observed BF16 OA geometry; every other native API case falls back."""
import torch
import torch_npu

SHAPE_X = (24, 1, 4096)
SHAPE_W = (1, 4096, 1024)


def eligible(m):
    return (m['permitted'] and m['shape_x'] == SHAPE_X and m['shape_w'] == SHAPE_W
            and m['dtype_x'] == m['dtype_w'] == 'torch.bfloat16'
            and m['device_type'] == 'npu' and m['same_device']
            and m['contiguous_x'] and m['contiguous_w']
            and m['format_x'] == m['format_w'] == 2
            and m['bias_is_none'] and m['scale_is_none']
            and m['perm_x1'] == (1, 0, 2) and m['perm_x2'] == (0, 1, 2)
            and m['perm_y'] == (1, 0, 2) and m['batch_split_factor'] == 1)


def metadata(x, weight, *, permitted, bias, scale, perm_x1, perm_x2, perm_y, batch_split_factor):
    # Skip device-format queries for unsupported geometry, dtype or device.
    basic=(permitted and tuple(x.shape)==SHAPE_X and tuple(weight.shape)==SHAPE_W
           and x.dtype==weight.dtype==torch.bfloat16 and x.device.type=='npu' and x.device==weight.device)
    return dict(permitted=permitted, shape_x=tuple(x.shape), shape_w=tuple(weight.shape),
                dtype_x=str(x.dtype), dtype_w=str(weight.dtype),device_type=x.device.type,
                same_device=x.device==weight.device,contiguous_x=x.is_contiguous(),contiguous_w=weight.is_contiguous(),
                format_x=torch_npu.get_npu_format(x) if basic else None,
                format_w=torch_npu.get_npu_format(weight) if basic else None,
                bias_is_none=bias is None,scale_is_none=scale is None,
                perm_x1=tuple(perm_x1),perm_x2=tuple(perm_x2),perm_y=tuple(perm_y),batch_split_factor=batch_split_factor)


def project(x, weight, *, bias=None, scale=None, perm_x1=(1,0,2), perm_x2=(0,1,2),
            perm_y=(1,0,2), batch_split_factor=1, _wio_permitted=True):
    flags=dict(bias=bias,scale=scale,perm_x1=perm_x1,perm_x2=perm_x2,perm_y=perm_y,batch_split_factor=batch_split_factor)
    if eligible(metadata(x,weight,permitted=_wio_permitted,**flags)):
        # This selects a different native reduction implementation. Numerical
        # identity is a probe result, never a consequence of the shape proof.
        return torch.mm(x.view(24,4096),weight[0]).view(24,1,1024)
    return torch_npu.npu_transpose_batchmatmul(x,weight,**flags)
