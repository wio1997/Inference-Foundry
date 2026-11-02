# SPDX-License-Identifier: Apache-2.0
"""Existing ModelSlim row reader reuse and CPU backend stubs;no INT8 math."""
from dataclasses import replace
from pathlib import Path
import unittest,sys
from unittest.mock import Mock,patch
import torch
from test_glm5_next_sfa_inputs import fixture
from test_glm5_next_sfa_weights import config
from deepinfer.hugging_face.modelslim910c import ModelSlimCheckpoint
from deepinfer.model_executor.models.glm5_next.query_payload import rank_qb_manifest,load_rank_qb_cpu


class Reader:
    linear_payload=ModelSlimCheckpoint.linear_payload
    def __init__(self):
        self.quant={};self.desc={};self.reads=[];self.max_read_bytes=4*1024**2;self.weight=None;self.scale=torch.full((16384,1),.1234);self.offset=torch.zeros(16384,1)
        for layer in (3,43):
            p=f'model.language_model.layers.{layer}.self_attn.q_b_proj'
            for suffix,dtype,shape,size in [('weight','I8',[16384,1536],16384*1536),('weight_scale','F32',[16384,1],16384*4),('weight_offset','F32',[16384,1],16384*4)]:self.quant[p+'.'+suffix]='W8A8_DYNAMIC';self.desc[p+'.'+suffix]={'dtype':dtype,'shape':shape,'data_offsets':[0,size]}
    def descriptor(self,name):return Path('inert-query-shard'),32,self.desc[name]
    def row_slice(self,name,first,last):self.reads.append(('rows',first,last));self.weight=torch.full((last-first,1536),-7,dtype=torch.int8);return self.weight,'I8'
    def tensor(self,name):
        self.reads.append(('tensor',name))
        if name.endswith('weight_scale'):return self.scale,'F32'
        if name.endswith('weight_offset'):return self.offset,'F32'
        raise AssertionError('full q_b read')


def data(rank=0,layer=3,count=3):
    c=config();c.hidden_size=4096;c.mla_dimensions['q_lora_rank']=1536;p,*_=fixture(start=3,count=count,rank=rank);p=replace(p,layer_id=layer);r=Reader();payload=load_rank_qb_cpu(r,c,layer,p.heads,p.contract);return r,c,p,payload


def run(payload,p,x,**extra):
    kw=dict(plan=p,expected_identity=p.identity,backend=lambda a,w,s,**k:torch.zeros(len(a),p.heads.local_heads*256,dtype=torch.bfloat16),input_policy='rms_normalized_qr1536',quantization_policy='full_qr_dynamic_int8',bias_policy='absent');kw.update(extra);return payload.project_cpu(x,**kw)


class QueryPayloadTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_manifest_rows_budgets_and_existing_reader_reuse(self):
        for rank,layer,rows in [(0,3,(0,1024)),(15,43,(15360,16384))]:
            r,c,p,q=data(rank,layer);m=q.manifest;self.assertEqual((m.first_row,m.stop_row),rows);self.assertEqual(m.full_shape,(16384,1536));self.assertEqual((m.weight_bytes,m.checkpoint_bytes,m.runtime_bytes),(1572864,1703936,1574912));self.assertEqual(r.reads[0],('rows',*rows));self.assertEqual(len(r.reads),3);self.assertEqual(q.weight_nk.dtype,torch.int8);self.assertTrue(bool((q.weight_nk==-7).all()));self.assertTrue(torch.equal(q.weight_scale,r.scale[rows[0]:rows[1]].flatten().bfloat16()))
            with torch.inference_mode():r.weight.zero_();r.scale.zero_()
            self.assertTrue(bool((q.weight_nk==-7).all()));self.assertTrue(bool((q.weight_scale>0).all()))

    def test_descriptors_recipes_budget_and_partition_preflight_before_reads(self):
        r,c,p,q=data()
        for kind in ('recipe','shape','range','dtype','checkpoint','runtime','reader'):
            rr=Reader();kw={};name=q.manifest.prefix+'.weight'
            if kind=='recipe':rr.quant[name]='FLOAT'
            elif kind=='shape':rr.desc[name]['shape']=[1024,1536]
            elif kind=='range':rr.desc[name]['data_offsets'][1]-=1
            elif kind=='dtype':rr.desc[name]['dtype']='BF16'
            elif kind=='checkpoint':kw['max_checkpoint_bytes']=1
            elif kind=='runtime':kw['max_runtime_bytes']=1
            else:rr.max_read_bytes=1
            with self.assertRaises(ValueError):load_rank_qb_cpu(rr,c,3,p.heads,p.contract,**kw)
            self.assertEqual(rr.reads,[])
        from deepinfer.model_executor.models.glm5_next.sfa_inputs import SFAHeadPartition
        with self.assertRaises(ValueError):rank_qb_manifest(Reader(),c,3,SFAHeadPartition(64,1,0,tuple(range(64))),p.contract)

    def test_existing_channel_value_guards_and_metadata_drift(self):
        _,c,p,q=data()
        for kind in ('zero','offset','overflow','drift'):
            r=Reader()
            if kind=='zero':r.scale[0]=0
            elif kind=='offset':r.offset[0]=1
            elif kind=='overflow':r.scale[0]=3.4e38
            else:
                original=r.row_slice
                def changed(*args):value=original(*args);r.desc[q.manifest.prefix+'.weight']['data_offsets']=[16,16+16384*1536];return value
                r.row_slice=changed
            with self.assertRaises(ValueError):load_rank_qb_cpu(r,c,3,p.heads,p.contract)

    def test_full_qr_backend_input_local_output_order_identity_and_copies(self):
        for rank,layer in [(0,3),(15,43)]:
            _,_,p,q=data(rank,layer);x=torch.arange(3*1536).reshape(3,1536).bfloat16();saved=x.clone();w=q.weight_nk.clone();s=q.weight_scale.clone();result=torch.arange(3*1024).reshape(3,1024).bfloat16()
            def backend(a,weight,scale,*,spec):
                self.assertTrue(torch.equal(a,x));self.assertEqual((spec.identity,spec.output_rows,spec.rank),(p.identity,(rank*1024,(rank+1)*1024),rank));self.assertEqual((spec.input_policy,spec.quantization_policy),('rms_normalized_qr1536','full_qr_dynamic_int8'));a.zero_();weight.zero_();scale.zero_();return result
            out=run(q,p,x,backend=backend);self.assertEqual(out.identity,p.identity);self.assertTrue(torch.equal(out.query,result.reshape(3,4,256)));self.assertTrue(torch.equal(x,saved));self.assertTrue(torch.equal(q.weight_nk,w));self.assertTrue(torch.equal(q.weight_scale,s))
            with torch.inference_mode():out.query.zero_()
            self.assertFalse(bool((result==0).all()))

    def test_stale_plan_policy_backend_geometry_and_budget_rejects_before_callback(self):
        _,_,p,q=data();x=torch.ones(3,1536,dtype=torch.bfloat16);callback=Mock(side_effect=AssertionError('called'))
        for kw in ({'expected_identity':replace(p,revision=99).identity},{'plan':replace(p,layer_id=7),'expected_identity':replace(p,layer_id=7).identity},{'input_policy':'raw_q_a'},{'quantization_policy':'prepared_input'},{'bias_policy':'rank0'},{'max_input_bytes':1},{'max_output_bytes':1},{'max_payload_bytes':1}):
            with self.assertRaises(ValueError):run(q,p,x,backend=callback,**kw)
        with self.assertRaises(ValueError):run(q,p,x,backend=None)
        callback.assert_not_called()

    def test_bad_input_payload_mutation_output_and_backend_failure(self):
        _,_,p,q=data();x=torch.ones(3,1536,dtype=torch.bfloat16);callback=Mock(side_effect=AssertionError('called'))
        for bad in (x[:2],x.float(),torch.full_like(x,float('nan'))):
            with self.assertRaises(ValueError):run(q,p,bad,backend=callback)
        with torch.inference_mode():q.weight_scale[0]=2
        with self.assertRaisesRegex(ValueError,'digest'):run(q,p,x,backend=callback)
        callback.assert_not_called();_,_,p,q=data()
        for bad in (None,torch.zeros(3,16384,dtype=torch.bfloat16),torch.zeros(3,1024),torch.full((3,1024),float('inf'),dtype=torch.bfloat16)):
            with self.assertRaises(ValueError):run(q,p,x,backend=lambda *a,**k:bad)
        with self.assertRaisesRegex(RuntimeError,'backend failure'):run(q,p,x,backend=Mock(side_effect=RuntimeError('backend failure')))

    def test_empty_skips_payload_inputs_hashing_and_callback(self):
        _,_,p,q=data(count=0);callback=Mock(side_effect=AssertionError('called'))
        with patch('deepinfer.model_executor.models.glm5_next.query_payload._digest',side_effect=AssertionError('empty digest')):out=run(q,p,None,backend=callback)
        self.assertEqual(out.query.shape,(0,4,256));self.assertEqual(out.identity,p.identity);callback.assert_not_called()

    def test_native_modules_absent(self):self.assertFalse(any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))


if __name__=='__main__':unittest.main(verbosity=2)
