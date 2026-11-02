# SPDX-License-Identifier: Apache-2.0
"""Bounded synthetic reader and independent rank-local CPU projection checks."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
import torch
from test_glm5_next_sfa_inputs import fixture as packet_fixture
from deepinfer.model_executor.models.glm5_next.sfa_inputs import SFAHeadPartition,NoPESFAContract
from deepinfer.model_executor.models.glm5_next.sfa_weights import rank_kv_b_manifest,load_rank_kv_b_cpu


def config():
    return NS(model_type='glm5_next',mla_dimensions={'num_attention_heads':64,'qk_nope_head_dim':256,'kv_lora_rank':512,'v_head_dim':256,'qk_rope_head_dim':0},
        hf_config={'text_config':{'mla_use_nope':True}},indexer_config={'index_kpool':4,'index_topk':2048},
        layers=[NS(layer_id=i,attention='deepseek_sparse_attention' if i%4==3 else 'linear_attention') for i in range(45)])


class Reader:
    def __init__(self,dtype='BF16'):
        self.quant={f'model.language_model.layers.{i}.self_attn.kv_b_proj.weight':'FLOAT' for i in (3,43)}
        self.max_read_bytes=8*1024**2;self.desc={'dtype':dtype,'shape':[32768,512],'data_offsets':[0,32768*512*(2 if dtype=='BF16' else 4)]};self.calls=[];self.last=None
    def descriptor(self,name):return Path('inert-test-shard.safetensors'),32,self.desc
    def row_slice(self,name,first,last):
        self.calls.append((name,first,last));row=torch.arange(first,last)[:,None];col=torch.arange(512)[None,:]
        self.last=((row.remainder(37)-18)*.003+(col.remainder(29)-14)*.0021).float()
        if self.desc['dtype']=='BF16':self.last=self.last.bfloat16()
        return self.last,self.desc['dtype']
    def tensor(self,*a,**k):raise AssertionError('full matrix read')
    matrix=materialize=load_linear=linear_payload=tensor


def components(rank=0,world=16,layer=3,count=3,dtype='BF16'):
    c=config();p,*_=packet_fixture(start=3,count=count,world=world,rank=rank);p=replace(p,layer_id=layer)
    reader=Reader(dtype);w=load_rank_kv_b_cpu(reader,c,layer,p.heads,p.contract);return reader,c,p,w


def close_bf16(test,value,expected):
    rounded=torch.tensor(expected).bfloat16();f=rounded.float();spacing=torch.maximum((torch.nextafter(rounded,torch.full_like(rounded,float('inf'))).float()-f).abs(),(torch.nextafter(rounded,torch.full_like(rounded,float('-inf'))).float()-f).abs());err=(value.float()-f).abs()
    test.assertTrue(bool((err<=spacing+1e-5).all()));test.assertLessEqual(float(torch.linalg.vector_norm(err)/torch.linalg.vector_norm(f).clamp_min(1e-20)),.001)


class SFAWeightsTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5337)

    def test_manifest_first_and_last_contiguous_rank_rows_and_payload_bounds(self):
        c=config();contract=NoPESFAContract.from_config(c,reference_source_sha256='a'*64)
        for rank,layer,rows in ((0,3,(0,2048)),(15,43,(30720,32768))):
            r=Reader();h=SFAHeadPartition(64,16,rank,tuple(range(rank*4,rank*4+4)));m=rank_kv_b_manifest(r,c,layer,h,contract)
            self.assertEqual((m.first_row,m.stop_row),rows);self.assertEqual((m.checkpoint_bytes,m.runtime_bytes),(2097152,2097152));self.assertEqual(r.calls,[])

    def test_single_slice_FLOAT_BF16_and_F32_rounding_orientation_digests(self):
        for dtype in ('BF16','F32'):
            r,c,p,w=components(rank=15,layer=43,dtype=dtype);expected=r.last.bfloat16().reshape(4,512,512)
            self.assertEqual(r.calls,[(w.manifest.tensor_name,30720,32768)])
            self.assertTrue(torch.equal(w.weights.key_up,expected[:,:256]));self.assertTrue(torch.equal(w.weights.value_up,expected[:,256:].transpose(1,2)))
            raw=r.last.contiguous().view(torch.uint8).numpy().tobytes();runtime=r.last.bfloat16().contiguous().view(torch.uint8).numpy().tobytes()
            self.assertEqual(w.checkpoint_sha256,hashlib.sha256(raw).hexdigest());self.assertEqual(w.runtime_sha256,hashlib.sha256(runtime).hexdigest())
            self.assertEqual(w.manifest.checkpoint_bytes,2097152*(2 if dtype=='F32' else 1))

    def test_unsharded_oversize_and_reader_budget_reject_before_payload(self):
        c=config();contract=NoPESFAContract.from_config(c,reference_source_sha256='a'*64)
        class NoAccess:
            def __getattr__(self,name):raise AssertionError('reader accessed before head guard')
        with self.assertRaisesRegex(ValueError,'at most4'):load_rank_kv_b_cpu(NoAccess(),c,3,SFAHeadPartition(64,1,0,tuple(range(64))),contract)
        h=SFAHeadPartition(64,16,0,(0,1,2,3))
        for kw in ({'max_checkpoint_bytes':1},{'max_runtime_bytes':1},{'max_local_heads':2},{'max_local_heads':64}):
            r=Reader()
            with self.assertRaises(ValueError):load_rank_kv_b_cpu(r,c,3,h,contract,**kw)
            self.assertEqual(r.calls,[])
        r=Reader();r.max_read_bytes=1
        with self.assertRaises(ValueError):load_rank_kv_b_cpu(r,c,3,h,contract)
        self.assertEqual(r.calls,[])

    def test_wrong_recipe_dtype_shape_range_layer_or_value_dim_preflight(self):
        c=config();contract=NoPESFAContract.from_config(c,reference_source_sha256='a'*64);h=SFAHeadPartition(64,16,0,(0,1,2,3))
        for kind in ('recipe','dtype','shape','range','layer','value'):
            r=Reader();cc=config();layer=3
            if kind=='recipe':r.quant[next(iter(r.quant))]='W8A8_DYNAMIC'
            elif kind=='dtype':r.desc['dtype']='I8'
            elif kind=='shape':r.desc['shape'][1]=256
            elif kind=='range':r.desc['data_offsets'][1]-=1
            elif kind=='layer':layer=0
            else:cc.mla_dimensions['v_head_dim']=128
            with self.subTest(kind=kind),self.assertRaises(ValueError):load_rank_kv_b_cpu(r,cc,layer,h,contract)
            self.assertEqual(r.calls,[])

    def test_descriptor_change_or_payload_mismatch_after_read_rejects(self):
        c=config();contract=NoPESFAContract.from_config(c,reference_source_sha256='a'*64);h=SFAHeadPartition(64,16,0,(0,1,2,3))
        for kind in ('descriptor','payload','nonfinite','rounding'):
            class Bad(Reader):
                def row_slice(self,*args):
                    v,d=super().row_slice(*args)
                    if kind=='descriptor':self.desc['data_offsets']=[16,self.desc['data_offsets'][1]+16]
                    elif kind=='payload':return v[:1],d
                    elif kind=='nonfinite':v[0,0]=float('nan')
                    else:v.fill_(3.4e38)
                    return v,d
            r=Bad('F32')
            with self.subTest(kind=kind),self.assertRaises(ValueError):load_rank_kv_b_cpu(r,c,3,h,contract)
            self.assertEqual(len(r.calls),1)

    def test_query_and_value_projection_independent_FP64_oracle(self):
        r,c,p,w=components(dtype='F32');query=(torch.randn(3,4,256)*.125).bfloat16();latent=(torch.randn(3,4,512)*.125).bfloat16()
        q=w.absorb_query_cpu(query,plan=p);v=w.value_up_cpu(latent,plan=p)
        qnp=query.float().numpy().astype(np.float64);lnp=latent.float().numpy().astype(np.float64);weight=r.last.bfloat16().float().numpy().astype(np.float64).reshape(4,512,512)
        close_bf16(self,q,np.einsum('thp,hpl->thl',qnp,weight[:,:256]));close_bf16(self,v,np.einsum('thl,hlv->thv',lnp,weight[:,256:].transpose(0,2,1)))

    def test_layer_partition_source_and_projection_input_guards(self):
        r,c,p,w=components();query=torch.ones(3,4,256,dtype=torch.bfloat16)
        for bad in (replace(p,layer_id=7),replace(p,heads=SFAHeadPartition(64,16,15,(60,61,62,63))),replace(p,contract=replace(p.contract,reference_source_sha256='b'*64))):
            with self.assertRaises(ValueError):w.absorb_query_cpu(query,plan=bad)
        for value in (query[:2],query.float(),torch.empty(query.shape,device='meta',dtype=torch.bfloat16),torch.full_like(query,float('nan'))):
            with self.assertRaises(ValueError):w.absorb_query_cpu(value,plan=p)
        with self.assertRaises(ValueError):w.absorb_query_cpu(query,plan=p,max_output_bytes=1)
        with self.assertRaises(ValueError):w.absorb_query_cpu(query,plan=p,max_tokens=1)

    def test_empty_skips_math_and_returns_owned_empty_shapes(self):
        r,c,p,w=components(count=0)
        with patch('torch.einsum',side_effect=AssertionError('empty contraction')):
            self.assertEqual(w.absorb_query_cpu(torch.empty(0,4,256,dtype=torch.bfloat16),plan=p).shape,(0,4,512))
            self.assertEqual(w.value_up_cpu(torch.empty(0,4,512,dtype=torch.bfloat16),plan=p).shape,(0,4,256))

    def test_reader_payload_mutation_does_not_alias_prepared_weights(self):
        r,c,p,w=components();uk=w.weights.key_up.clone();uv=w.weights.value_up.clone()
        with torch.inference_mode():r.last.zero_()
        self.assertTrue(torch.equal(w.weights.key_up,uk));self.assertTrue(torch.equal(w.weights.value_up,uv))
        q=torch.ones(3,4,256,dtype=torch.bfloat16);saved=q.clone();out=w.absorb_query_cpu(q,plan=p)
        with torch.inference_mode():out.zero_()
        self.assertTrue(torch.equal(q,saved));self.assertTrue(torch.equal(w.weights.key_up,uk))


if __name__=='__main__':unittest.main(verbosity=2)
