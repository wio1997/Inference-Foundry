"""Compare the public backend against upstream FlashMLA's CPU reference.

Extract only reference functions to avoid importing vendor device extensions.
The reference source remains unchanged and runs on CPU.
"""
import ast
import __future__
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_npu

from deepinfer import PLATFORM
from deepinfer.model_executor.layers.attention_backend import prefill_attn_varlen_sparse, decode_mla_mqa_sparse

pytestmark=pytest.mark.skipif(not PLATFORM.is_910c_reference,reason='explicit 910C reference profile required')
REF_PATH=Path(__file__).parents[1]/'vendor/FlashMLA/tests/ref.py'
source=ast.parse(REF_PATH.read_text())
functions=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef)],type_ignores=[])
ns={'torch':torch};exec(compile(functions,str(REF_PATH),'exec',flags=__future__.annotations.compiler_flag),ns)


def check(actual,expected):
    out,lse=actual;refout,reflse=expected
    torch.testing.assert_close(out.cpu(),refout,rtol=.008,atol=.008)
    torch.testing.assert_close(lse.cpu(),reflse,rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize('heads,dim',[(1,128),(7,512),(64,512)])
@pytest.mark.parametrize('with_sink',[False,True])
@pytest.mark.parametrize('with_length',[False,True])
def test_sparse_prefill_upstream_reference(heads,dim,with_sink,with_length):
    g=torch.Generator().manual_seed(11);m=17;k=31;topk=13
    q=torch.randn((m,heads,dim),generator=g).bfloat16();kv=torch.randn((k,1,dim),generator=g).bfloat16()
    indices=torch.randint(0,k,(m,topk),generator=g,dtype=torch.int32)
    indices[0].fill_(-1);indices[1,0]=-77;indices[2,1]=k+900;indices[3,2]=indices[3,1]
    lengths=torch.randint(0,topk+1,(m,),generator=g,dtype=torch.int32) if with_length else None
    sink=torch.randn(heads,generator=g) if with_sink else None
    if sink is not None:sink[0]=float('inf')
    p=SimpleNamespace(topk=topk,s_q=m,s_kv=k,d_qk=dim,h_q=heads,d_v=dim)
    t=SimpleNamespace(q=q,kv=kv,indices=indices[:,None,:],topk_length=lengths,attn_sink=sink,sm_scale=dim**-.5)
    ref=ns['ref_sparse_attn_fwd'](p,t)
    actual=prefill_attn_varlen_sparse(q.npu(),kv.npu(),indices.npu(),dim**-.5,None if sink is None else sink.npu(),None if lengths is None else lengths.npu())
    check(actual,(ref[0],ref[3]))
    assert torch.isinf(actual[1][0]).all() and (actual[0][0]==0).all()


def cache_pair(kind,seed):
    g=torch.Generator().manual_seed(seed);blocks,blocksize,dim=3,8,512
    if kind=='bf16':
        values=torch.randn((blocks,blocksize,1,dim),generator=g).bfloat16();return values,values.float()
    if kind=='fp8':
        data=torch.randint(0,126,(blocks,blocksize,1,512),generator=g,dtype=torch.uint8)
        scales=torch.randint(122,128,(blocks,blocksize,1,16),generator=g,dtype=torch.uint8)
        golden=data.view(torch.float8_e4m3fn).float()*torch.pow(2.,scales.float()-127).repeat_interleave(32,-1)
    else:
        data=torch.randint(0,256,(blocks,blocksize,1,256),generator=g,dtype=torch.uint8)
        scales=torch.randint(16,56,(blocks,blocksize,1,32),generator=g,dtype=torch.uint8)
        table=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.])
        golden=torch.stack((table[(data.int()&15).long()],table[(data.int()>>4).long()]),-1).flatten(-2)*scales.view(torch.float8_e4m3fn).float().repeat_interleave(16,-1)
    return torch.cat((data,scales),-1),golden


@pytest.mark.parametrize('kind',['bf16','fp8','fp4'])
@pytest.mark.parametrize('extra_kind',[None,'fp8','fp4'])
@pytest.mark.parametrize('with_sink',[False,True])
def test_sparse_decode_upstream_reference(kind,extra_kind,with_sink):
    m,h,d,topk=17,7,512,9;g=torch.Generator().manual_seed(88)
    q=torch.randn((m,h,d),generator=g).bfloat16();cache,golden=cache_pair(kind,21)
    indices=torch.randint(0,24,(m,topk),generator=g,dtype=torch.int32);indices[0].fill_(-1);indices[3,1]=indices[3,2]
    lengths=torch.randint(0,topk+1,(m,),generator=g,dtype=torch.int32);lengths[0]=0
    sink=torch.randn(h,generator=g) if with_sink else None
    extra=extra_indices=extra_lengths=None;extra_scope=None
    if extra_kind is not None:
        extra,extra_golden=cache_pair(extra_kind,22);extra_indices=torch.randint(0,24,(m,5),generator=g,dtype=torch.int32);extra_indices[0].fill_(-1)
        extra_lengths=torch.randint(0,6,(m,),generator=g,dtype=torch.int32);extra_lengths[0]=0
        extra_scope=SimpleNamespace(indices_in_kvcache=extra_indices[:,None,:],blocked_k=extra_golden,topk_length=extra_lengths)
    p=SimpleNamespace(h_kv=1,decode=SimpleNamespace(b=m),s_q=1,h_q=h,d_qk=d,d_v=d)
    scope=SimpleNamespace(indices_in_kvcache=indices[:,None,:],blocked_k=golden,topk_length=lengths)
    t=SimpleNamespace(kv_scope=scope,extra_kv_scope=extra_scope,q=q[:,None],sm_scale=d**-.5,attn_sink=sink)
    refout,reflse=ns['ref_sparse_attn_decode'](p,t)
    actual=decode_mla_mqa_sparse(q.npu(),cache.npu(),d,indices.npu(),None,d**-.5,None if sink is None else sink.npu(),index_length=lengths.npu(),key_cache_extra=None if extra is None else extra.npu(),indices_in_kvcache_extra=None if extra_indices is None else extra_indices.npu(),index_length_extra=None if extra_lengths is None else extra_lengths.npu())
    check(actual,(refout[:,0],reflse[:,:,0]))


def test_invalid_decode_is_explicit_error():
    q=torch.ones((1,1,512),device='npu',dtype=torch.bfloat16);cache=torch.ones((1,8,1,512),device='npu',dtype=torch.bfloat16)
    for idx in (-2,8):
        with pytest.raises(ValueError):decode_mla_mqa_sparse(q,cache,512,torch.tensor([[idx]],device='npu',dtype=torch.int32),None,.1)


def test_empty_prefill():
    q=torch.empty((0,7,512),device='npu',dtype=torch.bfloat16)
    out,lse=prefill_attn_varlen_sparse(q,torch.empty((0,1,512),device='npu',dtype=torch.bfloat16),torch.empty((0,0),device='npu',dtype=torch.int32),.1)
    assert out.shape==(0,7,512) and lse.shape==(0,7)


def test_decode_ignores_garbage_after_length():
    q=torch.ones((1,1,512),device='npu',dtype=torch.bfloat16)
    cache=torch.ones((1,8,1,512),device='npu',dtype=torch.bfloat16)
    indices=torch.tensor([[0,999999,-123]],device='npu',dtype=torch.int32)
    out,lse=decode_mla_mqa_sparse(q,cache,512,indices,None,.1,index_length=torch.tensor([1],device='npu',dtype=torch.int32))
    torch.testing.assert_close(out.cpu(),torch.ones((1,1,512),dtype=torch.bfloat16),rtol=0,atol=0)
    torch.testing.assert_close(lse.cpu(),torch.tensor([[51.2]]),rtol=1e-6,atol=1e-6)
