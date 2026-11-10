# SPDX-License-Identifier: Apache-2.0
"""CPU-only equation and state-seam tests, no full model or NPU allocations."""
import unittest
import numpy as np
import torch
from torch import nn
from deepinfer.model_executor.models.glm5_next.decoder import Glm5NextDenseDecoder,GlmDenseMLP
from deepinfer.model_executor.models.glm5_next.mhc import GlmMHC,HCMix
from deepinfer.model_executor.models.glm5_next.kda import Glm5NextKDA


def make_hc(hidden=8):
    return GlmMHC(torch.randn(24,4*hidden)*.05,torch.randn(24)*.05,torch.tensor([.2,.3,.4]),streams=4,hidden_size=hidden)


def make_attention(hidden=8):
    layer=Glm5NextKDA();layer.num_heads=2;layer.head_dim=3
    layer.hidden_size=hidden;layer.conv_width=4;layer.lower_bound=-5.;layer.backend='torch';layer.norm_eps=1e-5
    layer.in_proj_qkvbfg_a=None;layer.chunk_prefill_single=False;layer.reference_sharded_projection=False
    for attr,dim in [('q_proj',(hidden,6)),('k_proj',(hidden,6)),('v_proj',(hidden,6)),('b_proj',(hidden,2)),
                     ('f_a_proj',(hidden,3)),('f_b_proj',(3,6)),('g_a_proj',(hidden,3)),('g_b_proj',(3,6)),('o_proj',(6,hidden))]:
        setattr(layer,attr,nn.Linear(*dim,bias=False))
    for name,value in [('q_conv',torch.randn(6,1,4)),('k_conv',torch.randn(6,1,4)),('v_conv',torch.randn(6,1,4)),
                       ('a_log',torch.randn(2)),('dt_bias',torch.randn(6)),('o_norm_weight',torch.ones(3))]:
        layer.register_buffer(name,value)
    return layer


def make_decoder(layer_id=0):
    return Glm5NextDenseDecoder(layer_id=layer_id,attn_hc=make_hc(),ffn_hc=make_hc(),input_norm=torch.ones(8),
        ffn_norm=torch.ones(8),attention=make_attention(),mlp=GlmDenseMLP(nn.Linear(8,5,bias=False),
        nn.Linear(8,5,bias=False),nn.Linear(5,8,bias=False),hidden_size=8,intermediate_size=5)).eval()


class DecoderContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5302);torch.set_num_threads(2)

    def test_mhc_numpy_oracle(self):
        layer=make_hc(5);residual=torch.randn(3,4,5)
        collapsed,mix=layer.pre(residual)
        r=residual.numpy().astype(np.float64);fn=layer.fn.numpy().astype(np.float64)
        base=layer.base.numpy().astype(np.float64);scale=layer.scale.numpy().astype(np.float64)
        flat=r.reshape(3,20);logits=(flat@fn.T)/np.sqrt((flat*flat).mean(-1,keepdims=True)+1e-5)
        pre=1/(1+np.exp(-(logits[:,:4]*scale[0]+base[:4])))+1e-6
        post=2/(1+np.exp(-(logits[:,4:8]*scale[1]+base[4:8])))
        comb=(logits[:,8:]*scale[2]+base[8:]).reshape(3,4,4)
        comb=np.exp(comb-comb.max(-1,keepdims=True));comb/=comb.sum(-1,keepdims=True);comb+=1e-6
        comb/=comb.sum(-2,keepdims=True)+1e-6
        for _ in range(19):
            comb/=comb.sum(-1,keepdims=True)+1e-6;comb/=comb.sum(-2,keepdims=True)+1e-6
        np.testing.assert_allclose(collapsed.numpy(),(r*pre[...,None]).sum(1),rtol=2e-5,atol=1e-6)
        np.testing.assert_allclose(mix.post.numpy()[...,0],post,rtol=2e-5,atol=1e-6)
        np.testing.assert_allclose(mix.comb.numpy(),comb,rtol=2e-5,atol=1e-6)
        output=torch.randn(3,5);expected=np.empty_like(r)
        for t in range(3):
            for j in range(4):
                expected[t,j]=sum(comb[t,i,j]*r[t,i] for i in range(4))+post[t,j]*output[t].numpy()
        np.testing.assert_allclose(layer.post(output,residual,mix).numpy(),expected,rtol=2e-5,atol=1e-6)

    def test_post_orientation_and_sinkhorn_marginals(self):
        layer=make_hc(2);r=torch.arange(8,dtype=torch.float32).reshape(1,4,2)
        comb=torch.zeros(1,4,4)
        for i in range(4):comb[0,i,(i+1)%4]=1
        out=layer.post(torch.zeros(1,2),r,HCMix(torch.zeros(1,4,1),comb))
        torch.testing.assert_close(out,r[:,[3,0,1,2]])
        _,mix=layer.pre(r)
        torch.testing.assert_close(mix.comb.sum(-1),torch.ones(1,4),atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(mix.comb.sum(-2),torch.ones(1,4),atol=3e-6,rtol=3e-6)

    def test_zero_logits_and_empty(self):
        layer=GlmMHC(torch.zeros(24,32),torch.zeros(24),torch.ones(3),streams=4,hidden_size=8)
        r=torch.ones(2,4,8);collapsed,mix=layer.pre(r)
        torch.testing.assert_close(collapsed,torch.full((2,8),4*(.5+1e-6)))
        torch.testing.assert_close(mix.post,torch.ones(2,4,1))
        empty=torch.empty(0,4,8);collapsed,mix=layer.pre(empty)
        self.assertEqual(tuple(layer.post(collapsed,empty,mix).shape),(0,4,8))

    def test_dense_asymmetric_clamp(self):
        class Constant(nn.Module):
            def __init__(self,value):super().__init__();self.register_buffer('value',torch.tensor(value))
            def forward(self,x):return self.value.expand(x.shape[0],-1)
        mlp=GlmDenseMLP(Constant([-100.,100.,-20.,20.]),Constant([-100.,100.,20.,-20.]),nn.Identity(),hidden_size=4,intermediate_size=4)
        gate=torch.tensor([[-100.,10.,-20.,10.]]);up=torch.tensor([[-10.,10.,10.,-10.]])
        torch.testing.assert_close(mlp(torch.zeros(1,4)),torch.nn.functional.silu(gate)*up)
        self.assertEqual(tuple(mlp(torch.empty(0,4)).shape),(0,4))

    def test_decoder_two_layer_segmented_state_and_isolation(self):
        first,second=make_decoder(),make_decoder(1);x=torch.randn(7,8);saved=x.clone()
        full0,state0=first(x);full1,state1=second(full0)
        a0,s0=first(x[:3]);a1,s1=second(a0)
        b0,s0=first(x[3:],s0);b1,s1=second(b0,s1)
        torch.testing.assert_close(torch.cat((a0,b0)),full0,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(torch.cat((a1,b1)),full1,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(s0.recurrent,state0.recurrent,rtol=1e-4,atol=1e-6)
        torch.testing.assert_close(s1.recurrent,state1.recurrent,rtol=1e-4,atol=1e-6)
        self.assertEqual((s0.tokens_seen,s1.tokens_seen),(7,7))
        fresh,_=first(x);torch.testing.assert_close(fresh,full0,rtol=0,atol=0)
        torch.testing.assert_close(x,saved,rtol=0,atol=0)

    def test_first_layer_expansion_and_next_layer_rejection(self):
        layer=make_decoder();x=torch.randn(2,8)
        out,_=layer(x);expanded,_=layer(layer.attn_hc.expand(x));torch.testing.assert_close(out,expanded)
        with self.assertRaises(ValueError):make_decoder(1)(x)

    def test_invalid_mhc_contracts(self):
        with self.assertRaises(ValueError):GlmMHC(torch.zeros(24,32),torch.zeros(24),torch.ones(3),streams=4,hidden_size=8,sinkhorn_iters=0)
        layer=make_hc()
        with self.assertRaises(ValueError):layer.pre(torch.zeros(2,8))
        with self.assertRaises(ValueError):layer.post(torch.zeros(2,8),torch.zeros(2,4,8),HCMix(torch.zeros(2,4),torch.zeros(2,4,4)))

if __name__=='__main__':unittest.main(verbosity=2)
