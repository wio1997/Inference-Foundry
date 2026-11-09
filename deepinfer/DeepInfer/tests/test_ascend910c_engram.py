"""Engram primitives compared with unchanged TileKernels CPU golden functions."""
import ast
import os
from pathlib import Path

import pytest
import torch
import torch_npu

from deepinfer import PLATFORM
from deepinfer.npu_ops.engram910c import gate,hashes

pytestmark=pytest.mark.skipif(not PLATFORM.is_910c_reference,reason='explicit 910C profile required')
source=Path(os.environ['TILE_KERNELS_REFERENCE_SOURCE'])/'tile_kernels/torch/engram.py'
tree=ast.parse(source.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
ref={'torch':torch};exec(compile(tree,str(source),'exec'),ref)


@pytest.mark.parametrize('image',[False,True])
@pytest.mark.parametrize('count',[0,1,17])
def test_hash_signed_overflow_and_mask(count,image):
    g=torch.Generator().manual_seed(42);tokens=torch.randint(0,100000,(count,4),dtype=torch.int32,generator=g)
    multipliers=torch.randint(1,2**61,(2,4),dtype=torch.int64,generator=g)*2+1
    sizes=torch.tensor([[[101,103],[107,109],[113,127]],[[131,137],[139,149],[151,157]]],dtype=torch.int32)
    offsets=ref['make_offsets'](sizes);mask=torch.arange(count)%3==0 if image else None
    expected=ref['engram_hash_ref'](tokens,multipliers,sizes,offsets,mask)
    actual=hashes(tokens.npu(),multipliers.npu(),sizes.npu(),offsets.npu(),None if mask is None else mask.npu())
    torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)


@pytest.mark.parametrize('size',[128,5120])
@pytest.mark.parametrize('zero',[False,True])
@pytest.mark.parametrize('image',[False,True])
def test_gate(size,zero,image):
    g=torch.Generator().manual_seed(19);x=torch.randn((5,4,size),generator=g).bfloat16();kv=torch.randn((5,5,size),generator=g).bfloat16()
    if zero:x.zero_();kv.zero_()
    wh=torch.randn((4,size),generator=g).bfloat16();we=torch.randn((4,size),generator=g).bfloat16();mask=torch.tensor([False,True,False,True,False]) if image else None
    expected=ref['engram_gate_ref'](x,kv,wh,we,1e-6,1e-20,image_token_mask=mask)
    actual=gate(x.npu(),kv.npu(),wh.npu(),we.npu(),image_mask=None if mask is None else mask.npu())
    torch.testing.assert_close(actual.cpu(),expected,rtol=.008,atol=.008)
    if mask is not None:torch.testing.assert_close(actual.cpu()[mask],x[mask],rtol=0,atol=0)
