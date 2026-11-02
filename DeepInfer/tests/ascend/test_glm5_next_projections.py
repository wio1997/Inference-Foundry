# SPDX-License-Identifier: Apache-2.0
"""Device-free projection seams, independent normalization/rounding equations."""
from dataclasses import replace
import unittest
import numpy as np
import torch
from torch import nn

from deepinfer.hugging_face.modelslim910c import FloatLinear
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,state_from_aligned_prefix
from deepinfer.model_executor.models.glm5_next.projections import GlmMLAProjections,GlmIndexerProjections


def bf16(x):return torch.tensor(np.asarray(x),dtype=torch.float64).bfloat16().float().numpy().astype(np.float64)
def array(x):return x.float().numpy().astype(np.float64)
def linear(x,w):return bf16(x@array(w).T)
def rms(x,w,eps):return bf16(x/np.sqrt((x*x).mean(-1,keepdims=True)+eps)*array(w))


class Counted(nn.Module):
    def __init__(self,width):super().__init__();self.width=width;self.calls=0
    def forward(self,x):self.calls+=1;return x[:,torch.arange(self.width)%x.shape[-1]].clone()


class ProjectionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5318)

    def mla(self):
        dims=(7,2,4,3,5,3);hidden,heads,qk,v,qr,kv=dims
        shapes=[(qr,hidden),(kv,hidden),(heads*qk,qr),(heads*(qk+v),kv),(hidden,heads*v)]
        modules=[FloatLinear(torch.randn(*s).bfloat16()*.125) for s in shapes]
        norms=[(torch.randn(n)*.1+1).bfloat16() for n in (qr,kv)]
        module=GlmMLAProjections(hidden_size=hidden,num_heads=heads,qk_dim=qk,value_dim=v,q_rank=qr,kv_rank=kv,
                                 q_a=modules[0],kv_a=modules[1],q_b=modules[2],kv_b=modules[3],o_proj=modules[4],q_norm=norms[0],kv_norm=norms[1])
        return module,modules,norms

    def indexer(self):
        hidden,qr,heads,width=7,5,2,4;layout=DSACacheLayout(3,width,4,8,8)
        tensors=[torch.randn(width,hidden).bfloat16(),torch.randn(heads,hidden).bfloat16(),torch.randn(width,hidden).bfloat16(),
                 torch.randn(width).bfloat16(),torch.randn(width).bfloat16(),torch.randn(4,width).bfloat16()]
        wq=FloatLinear(torch.randn(heads*width,qr).bfloat16())
        m=GlmIndexerProjections(wq_b=wq,key_weight=tensors[0],head_weight=tensors[1],gate_weight=tensors[2],norm_weight=tensors[3],norm_bias=tensors[4],ape=tensors[5],hidden_size=hidden,q_rank=qr,num_heads=heads,layout=layout)
        return m,tensors

    def test_mla_projection_norm_split_and_finish_numpy(self):
        m,mods,norms=self.mla();x=torch.randn(3,7).bfloat16();saved=x.clone();out=m.project(x)
        qr=rms(linear(array(x),mods[0].weight.detach()),norms[0],1e-5)
        latent=rms(linear(array(x),mods[1].weight.detach()),norms[1],1e-5)
        query=linear(qr,mods[2].weight.detach()).reshape(3,2,4)
        np.testing.assert_array_equal(array(out.query_rank),qr);np.testing.assert_array_equal(array(out.latent),latent);np.testing.assert_array_equal(array(out.query),query)
        key,value=m.expand_kv(out.latent);expanded=linear(latent,mods[3].weight.detach()).reshape(3,2,7)
        np.testing.assert_array_equal(array(key),expanded[...,:4]);np.testing.assert_array_equal(array(value),expanded[...,4:])
        finished=m.finish(value);np.testing.assert_array_equal(array(finished),linear(expanded[...,4:].reshape(3,6),mods[4].weight.detach()))
        torch.testing.assert_close(x,saved,rtol=0,atol=0)

    def test_actual_mla_widths_with_injected_projection_outputs(self):
        # Actual dimensions without allocating actual INT8 matrices on CPU.
        q_a,kv_a,q_b,kv_b,o=[Counted(n) for n in (1536,512,64*256,64*512,4096)]
        m=GlmMLAProjections(hidden_size=4096,num_heads=64,qk_dim=256,value_dim=256,q_rank=1536,kv_rank=512,
                            q_a=q_a,kv_a=kv_a,q_b=q_b,kv_b=kv_b,o_proj=o,q_norm=torch.ones(1536).bfloat16(),kv_norm=torch.ones(512).bfloat16())
        x=torch.randn(2,4096).bfloat16();p=m.project(x);self.assertEqual(p.query.shape,(2,64,256));self.assertEqual(p.latent.shape,(2,512))
        k,v=m.expand_kv(p.latent);self.assertEqual(k.shape,(2,64,256));self.assertEqual(v.shape,(2,64,256));self.assertEqual(m.finish(v).shape,(2,4096))
        self.assertEqual([z.calls for z in (q_a,kv_a,q_b,kv_b,o)],[1]*5)

    def test_mla_empty_skips_modules_and_bad_outputs_reject(self):
        m,_,_=self.mla();m.q_a=Counted(5);m.kv_a=Counted(3);m.q_b=Counted(8);m.kv_b=Counted(14);m.o_proj=Counted(7)
        p=m.project(torch.empty(0,7,dtype=torch.bfloat16));self.assertEqual(p.query.shape,(0,2,4))
        k,v=m.expand_kv(p.latent);self.assertEqual(m.finish(v).shape,(0,7));self.assertEqual(sum(z.calls for z in (m.q_a,m.kv_a,m.q_b,m.kv_b,m.o_proj)),0)
        m.q_a=Counted(4)
        with self.assertRaises(ValueError):m.project(torch.ones(2,7,dtype=torch.bfloat16))
        with self.assertRaises(ValueError):m.project(torch.ones(2,7))
        with self.assertRaises(ValueError):m.finish(torch.zeros(2,2,4,dtype=torch.bfloat16))

    def test_native_mla_cpu_guard_before_reader(self):
        class NoAccess:
            def __getattr__(self,name):raise AssertionError('reader access before native CPU rejection')
        with self.assertRaisesRegex(ValueError,'no CPU dequantization fallback'):GlmMLAProjections.from_checkpoint(NoAccess(),3,'cpu')

    def test_indexer_layernorm_query_gate_and_head_scale_numpy(self):
        m,w=self.indexer();x=torch.randn(5,7).bfloat16();qr=torch.randn(5,5).bfloat16();p=m(x,qr)
        raw=linear(array(x),w[0]);centered=raw-raw.mean(-1,keepdims=True)
        expected=centered/np.sqrt((centered*centered).mean(-1,keepdims=True)+1e-6)*array(w[3])+array(w[4])
        np.testing.assert_allclose(array(p.raw_keys),expected,rtol=2e-6,atol=5e-7)
        self.assertEqual(p.raw_keys.dtype,torch.float32);self.assertFalse(torch.equal(p.raw_keys,p.raw_keys.bfloat16().float()))
        np.testing.assert_array_equal(array(p.gate_scores),linear(array(x),w[2]))
        np.testing.assert_array_equal(array(p.query),linear(array(qr),m.wq_b.weight.detach()).reshape(5,2,4))
        head=bf16(array(x)@array(w[1]).T);expected_weights=bf16(head*((4**-.5)*(2**-.5)))
        np.testing.assert_array_equal(array(p.head_weights),expected_weights)
        self.assertEqual(m.ape.dtype,torch.float32)

    def test_cache_only_skips_query_and_still_updates_raw_history(self):
        m,_=self.indexer();m.wq_b=Counted(8);x=torch.randn(7,7).bfloat16();p=m(x,compute_topk=False)
        self.assertIsNone(p.query);self.assertIsNone(p.head_weights);self.assertEqual(m.wq_b.calls,0)
        latent=torch.randn(7,3).bfloat16();initial=state_from_aligned_prefix(latent[:0],torch.empty(0,4,dtype=torch.bfloat16),m.layout)
        result=m.append_cache(latent,p,initial,start_position=0)
        self.assertEqual((result.tokens_seen,result.compressed.shape),(7,(1,4)))
        for pos in (4,5,6):torch.testing.assert_close(result.tail[0,pos%4],p.raw_keys[pos],rtol=0,atol=0)

    def test_indexer_chunking_request_independence_and_empty(self):
        m,_=self.indexer();x=torch.randn(9,7).bfloat16();qr=torch.randn(9,5).bfloat16();full=m(x,qr)
        parts=[m(x[:3],qr[:3]),m(x[3:4],qr[3:4]),m(x[4:],qr[4:])]
        for name in ('query','head_weights','raw_keys','gate_scores'):
            torch.testing.assert_close(torch.cat([getattr(p,name) for p in parts]),getattr(full,name),rtol=0,atol=0)
        m.wq_b=Counted(8);p=m(x[:0],qr[:0]);self.assertEqual(p.query.shape,(0,2,4));self.assertEqual(m.wq_b.calls,0)
        p=m(x[:0],compute_topk=False);self.assertIsNone(p.query)

    def test_indexer_contract_errors_and_owned_parameter_buffers(self):
        m,w=self.indexer();original=m.ape.clone();w[5].zero_();torch.testing.assert_close(original,m.ape,rtol=0,atol=0)
        x=torch.randn(3,7).bfloat16();qr=torch.randn(3,5).bfloat16()
        for bad in (None,(qr,qr),qr.float(),qr[:2]):
            with self.assertRaises(ValueError):m(x,bad)
        with self.assertRaises(ValueError):m(x.float(),qr)
        m.wq_b=Counted(7)
        with self.assertRaises(ValueError):m(x,qr)

if __name__=='__main__':unittest.main(verbosity=2)
