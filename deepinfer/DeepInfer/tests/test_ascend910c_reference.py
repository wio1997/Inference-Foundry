"""Run explicitly: DEEPINFER_ASCEND_910C_REFERENCE=1 python -m pytest -q ...

Uses CPU PyTorch byte decoding as an independent golden and real NPU compute.
"""
import os

import pytest
import torch
import torch_npu

from deepinfer import PLATFORM
from deepinfer.npu_ops.compat910c import decode_block_weight, decode_e2m1_bytes, decode_e4m3_bytes

pytestmark = pytest.mark.skipif(not PLATFORM.is_910c_reference, reason="requires explicit 910C profile")


def test_all_fp8_encodings():
    bits = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    expected = bits.view(torch.float8_e4m3fn).float()
    actual = decode_e4m3_bytes(bits.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(actual.signbit()[~expected.isnan()], expected.signbit()[~expected.isnan()])


def test_all_fp4_byte_pairs():
    bits = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    table = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    expected = torch.stack((table[(bits.int() & 15).long()], table[(bits.int() >> 4).long()]), -1).flatten()
    actual = decode_e2m1_bytes(bits.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.signbit(), expected.signbit())


@pytest.mark.parametrize('fmt', ['e4m3', 'e2m1'])
@pytest.mark.parametrize('scale_fmt', ['ue8m0', 'fp32'])
def test_block_decode(fmt, scale_fmt):
    gen = torch.Generator().manual_seed(41)
    n,k=33,96
    data=torch.randint(0, 126 if fmt=='e4m3' else 256, (n,k if fmt=='e4m3' else k//2), generator=gen,dtype=torch.uint8)
    sf=torch.tensor([[125,127,129],[128,126,124]],dtype=torch.uint8)
    sf_float=torch.pow(2.,sf.float()-127.)
    scales=sf if scale_fmt=='ue8m0' else sf_float
    if fmt=='e4m3': values=data.view(torch.float8_e4m3fn).float()
    else:
        table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
        values=torch.stack((table[(data.int()&15).long()],table[(data.int()>>4).long()]),-1).flatten(-2)
    golden=(values*sf_float.repeat_interleave(32,0).repeat_interleave(32,1)[:n,:k]).bfloat16()
    result=decode_block_weight(data.npu(),scales.npu(),fmt=fmt,scale_fmt=scale_fmt)
    torch.testing.assert_close(result.cpu(),golden,rtol=0,atol=0)


@pytest.mark.parametrize('shape',[(1,128),(7,5120),(2,3,128)])
@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
@pytest.mark.parametrize('has_residual',[False,True])
def test_norm_dispatch(shape,dtype,has_residual):
    from deepinfer.model_executor.layers.tilelang_ops.norm import norm
    gen=torch.Generator().manual_seed(17)
    x=torch.randn(shape,generator=gen).to(dtype)
    w=torch.randn((shape[-1],),generator=gen).to(dtype)
    res=torch.randn(shape,generator=gen).to(dtype) if has_residual else None
    values=x if res is None else x+res
    golden=((values.float()*torch.rsqrt(values.float().square().mean(-1,keepdim=True)+1e-6)*w.float())*1.5).to(dtype)
    xn=x.npu();rn=None if res is None else res.npu();out=torch.empty_like(xn)
    result=norm(xn,w.npu(),rn,out=out,return_bf16=True,out_scale=1.5)
    if has_residual:
        result,resout=result
        torch.testing.assert_close(resout.cpu(),values,rtol=0,atol=0)
    value,bf16=result
    assert value.data_ptr()==out.data_ptr()
    torch.testing.assert_close(value.cpu(),golden,rtol=3e-5 if dtype==torch.float32 else .012,atol=2e-5 if dtype==torch.float32 else .015)
    torch.testing.assert_close(bf16.cpu(),golden.bfloat16(),rtol=.012,atol=.015)


@pytest.mark.parametrize('shape',[(1,128,17),(7,32,16),(3,64,33)])
@pytest.mark.parametrize('use_fp32',[False,True])
@pytest.mark.parametrize('accumulate',[False,True])
def test_replicated_linear(shape,use_fp32,accumulate):
    from deepinfer.model_executor.layers.linear import ReplicatedLinear
    m,k,n=shape;gen=torch.Generator().manual_seed(24)
    x=torch.randn((m,k),generator=gen).bfloat16()
    w=torch.randn((n,k),generator=gen).bfloat16()
    bias=torch.randn(n,generator=gen).to(torch.float32 if use_fp32 else torch.bfloat16)
    layer=ReplicatedLinear(k,n,use_fp32_output=use_fp32,weight=torch.nn.Parameter(w.npu(),requires_grad=False),bias=torch.nn.Parameter(bias.npu(),requires_grad=False))
    dtype=torch.float32 if use_fp32 else torch.bfloat16
    acc=torch.randn((m,n),generator=gen).to(dtype) if accumulate else None
    out_acc=None if acc is None else layer._allocate_output((m,n),dtype,x.npu().device)
    if out_acc is not None: out_acc.copy_(acc.npu())
    expected=x.float()@w.float().T
    if acc is not None:expected+=acc.float()
    expected=expected.to(dtype)+bias
    out=layer(x.npu(),out_acc)
    torch.testing.assert_close(out.cpu(),expected,rtol=4e-4 if use_fp32 else .012,atol=1e-3 if use_fp32 else .125)
    if out_acc is not None: assert out.data_ptr()==out_acc.data_ptr()


def test_reference_rejects_unsupported_formats():
    from deepinfer.npu_ops.compat910c import norm
    x=torch.ones((1,32),device='npu',dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError):norm(x,x[0],fmt='e4m3',num_per_channels=32)
    with pytest.raises(TypeError):decode_block_weight(torch.ones((32,32),device='npu',dtype=torch.uint8),torch.ones((1,1),device='npu',dtype=torch.int16),fmt='e4m3')
