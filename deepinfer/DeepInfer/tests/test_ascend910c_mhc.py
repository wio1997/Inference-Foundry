"""Compare public mHC interfaces against pinned upstream CPU golden functions."""
import ast
import os
from pathlib import Path

import pytest
import torch
import torch_npu
from deepinfer import PLATFORM
from deepinfer.model_executor.models.v4_modules import mhc_ops as ops
from deepinfer.model_executor.models.v4_modules.mhc_modeling import mhc_pre_big_fuse

pytestmark=pytest.mark.skipif(not PLATFORM.is_910c_reference,reason='explicit 910C profile required')
refpath=Path(os.environ['TILE_KERNELS_REFERENCE_SOURCE'])/'tile_kernels/torch/mhc.py'
ref={};exec(compile(ast.parse(refpath.read_text()),str(refpath),'exec'),ref)

def compare(actual,expected):
    torch.testing.assert_close(actual.cpu(),expected,rtol=.008 if expected.dtype==torch.bfloat16 else 3e-5,atol=.008 if expected.dtype==torch.bfloat16 else 2e-5)

@pytest.mark.parametrize('size',[128,5120])
@pytest.mark.parametrize('splits',[1,4])
def test_prenorm_and_big_fuse(size,splits):
    g=torch.Generator().manual_seed(13);m=4
    x=torch.randn((1,3,m,size),generator=g).bfloat16();fn=torch.randn((m*(m+2),m*size),generator=g)/100
    scale=torch.tensor([.01,.02,.03]);base=torch.randn((m*(m+2),),generator=g)
    expected=ref['mhc_pre_norm_fn_ref'](x,fn,None,1e-6)
    actual=ops.mhc_pre_norm_fn(x.npu(),fn.npu(),None,1e-6,n_splits=splits);compare(actual,expected)
    pre,post,comb=ref['mhc_pre_split_mixes_ref'](expected,scale,base,m,2.,1e-6)
    comb=ref['sinkhorn_normalize_ref'](comb,repeat=20,eps=1e-6)
    mixed=ref['mhc_pre_apply_mix_ref'](x,pre)
    a,b,c=mhc_pre_big_fuse(x.npu(),fn.npu(),scale.npu(),base.npu(),1e-6,1e-6,1e-6,2.,20,n_splits=splits)
    compare(a,post);compare(b,comb);compare(c,mixed)

@pytest.mark.parametrize('repeat',[1,20])
def test_post_expand_head_sinkhorn(repeat):
    g=torch.Generator().manual_seed(21);hidden=torch.randn((1,7,128),generator=g).bfloat16();res=ref['expand_to_mhc_ref'](hidden,4)
    compare(ops.expand_to_mhc(hidden.npu(),4),res)
    mix=torch.randn((1,7,4),generator=g);scale=torch.ones(1)*.2;base=torch.randn((4,),generator=g)
    compare(ops.mhc_head_compute_mix(mix.npu(),scale.npu(),base.npu(),1e-6),ref['mhc_head_compute_mix_ref'](mix,scale,base,1e-6))
    raw=torch.randn((1,7,4,4),generator=g);comb=ref['sinkhorn_normalize_ref'](raw,repeat,1e-6);compare(ops.sinkhorn_normalize(raw.npu(),repeat,1e-6),comb)
    post=torch.rand((1,7,4,1),generator=g)
    compare(ops.mhc_post(hidden.npu(),res.npu(),post.npu(),comb.npu()),ref['mhc_post_ref'](hidden,res,post,comb))
