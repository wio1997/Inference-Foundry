# SPDX-License-Identifier: Apache-2.0
"""Independent row/norm/projection oracles for bounded text boundaries."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from deepinfer.model_executor.models.glm5_next.text_boundary import TextBoundaryManifest,text_boundary_manifest,GlmTextBoundary,GlmLogitsShard


def np64(x):return x.float().numpy().astype(np.float64)
def rounded(x):return torch.tensor(x,dtype=torch.float64).bfloat16()


class Reader:
    def __init__(self):
        self.embed=torch.arange(21).reshape(7,3).bfloat16()/8
        self.head=(torch.randn(7,3)*.5).bfloat16();self.norm=torch.tensor([.75,1.,1.25]).bfloat16();self.calls=[]
        self.quant={n:'FLOAT' for n in ('model.language_model.embed_tokens.weight','model.language_model.norm.weight','lm_head.weight')}
    def descriptor(self,name):
        x=self.embed if 'embed_tokens' in name else self.norm if '.norm.' in name else self.head
        return None,None,{'shape':list(x.shape),'dtype':'BF16'}
    def raw_rows(self,name,ids):self.calls.append(('embed',ids.clone()));return self.embed[ids].clone(),'BF16'
    def row_slice(self,name,start,stop):self.calls.append(('head',start,stop));return self.head[start:stop].clone(),'BF16'


class TextBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(5324);self.reader=Reader()
        self.manifest=TextBoundaryManifest(3,7,2,1e-5,1.3)
        self.boundary=GlmTextBoundary(self.reader,self.manifest,self.reader.norm,max_unique_tokens=3,max_tokens=8)

    def test_unique_embedding_rows_restore_order_and_duplicates(self):
        ids=torch.tensor([6,1,6,0]);out=self.boundary.embed(ids)
        torch.testing.assert_close(out,self.reader.embed[ids],rtol=0,atol=0)
        self.assertEqual(self.reader.calls[0][1].tolist(),[0,1,6])
        with torch.inference_mode():out[0].zero_()
        self.assertFalse(torch.equal(out[0],out[2]))

    def test_embedding_invalid_ids_and_budgets_precede_payload(self):
        for ids in (torch.tensor([-1]),torch.tensor([7]),torch.arange(4),torch.zeros(9,dtype=torch.int64),torch.tensor([1.])):
            with self.assertRaises(ValueError):self.boundary.embed(ids)
        self.assertEqual(self.reader.calls,[])

    def test_final_mean_then_BF16_then_rms_numpy(self):
        streams=torch.randn(5,2,3).bfloat16();saved=streams.clone();out=self.boundary.final_hidden(streams)
        collapsed=np64(rounded(np64(streams).mean(1)))
        expected=rounded(collapsed/np.sqrt((collapsed*collapsed).mean(-1,keepdims=True)+1e-5)*np64(self.reader.norm))
        torch.testing.assert_close(out,expected,rtol=0,atol=0);torch.testing.assert_close(streams,saved,rtol=0,atol=0)
        with self.assertRaises(ValueError):self.boundary.final_hidden(out)
        with self.assertRaises(ValueError):self.boundary.final_hidden(torch.zeros(9,2,3,dtype=torch.bfloat16))

    def test_head_shard_BF16_and_FP32_precision_scale_once(self):
        hidden=torch.randn(5,3).bfloat16();dot=np64(hidden)@np64(self.reader.head[2:6]).T
        bf16=self.boundary.head_shard(2,6,max_rows=4);fp32=self.boundary.head_shard(2,6,max_rows=4,head_dtype=torch.float32)
        self.assertEqual((bf16.vocab_start,bf16.vocab_end),(2,6))
        expected=rounded(np64(rounded(dot))*1.3);torch.testing.assert_close(bf16(hidden),expected,rtol=0,atol=0)
        np.testing.assert_allclose(fp32(hidden).numpy(),dot*1.3,rtol=2e-6,atol=2e-7)
        self.assertFalse(torch.equal(expected,rounded(dot*1.3)))

    def test_shard_bounds_dtype_and_empty_skip_payload_or_compute(self):
        for start,stop,kwargs in ((0,7,{'max_rows':2}),(-1,1,{}),(0,8,{}),(0,2,{'head_dtype':torch.float16})):
            with self.assertRaises(ValueError):self.boundary.head_shard(start,stop,**kwargs)
        self.assertEqual(self.reader.calls,[])
        empty=self.boundary.embed(torch.empty(0,dtype=torch.int64));self.assertEqual(empty.shape,(0,3));self.assertEqual(self.reader.calls,[])
        shard=self.boundary.head_shard(0,2);self.assertEqual(shard(torch.empty(0,3,dtype=torch.bfloat16)).shape,(0,2))
        with self.assertRaises(ValueError):shard(torch.zeros(9,3,dtype=torch.bfloat16))
        with self.assertRaises(ValueError):shard(torch.zeros(2,3))

    def test_metadata_does_not_tie_tables_or_accept_quantized_payload(self):
        config=SimpleNamespace(dtype='bfloat16',hidden_size=3,vocab_size=7,hf_config={'text_config':{'mhc':True,'hc_mult':2,'rms_norm_eps':1e-5,'tie_word_embeddings':False}})
        manifest=text_boundary_manifest(self.reader,config);self.assertEqual(manifest.logit_scale,1.)
        self.assertNotEqual(manifest.embedding_name,manifest.head_name)
        self.reader.quant[manifest.head_name]='W8A8_DYNAMIC'
        with self.assertRaises(ValueError):text_boundary_manifest(self.reader,config)
        self.assertEqual(self.reader.calls,[])

    def test_invalid_factory_bounds_reject_before_reader_access(self):
        class NoAccess:
            def __getattr__(self,name):raise AssertionError('reader accessed before bound rejection')
        with self.assertRaises(ValueError):GlmTextBoundary.from_checkpoint(NoAccess(),max_unique_tokens=0)
        with self.assertRaises(ValueError):GlmTextBoundary.from_checkpoint(NoAccess(),max_tokens=True)

if __name__=='__main__':unittest.main(verbosity=2)
