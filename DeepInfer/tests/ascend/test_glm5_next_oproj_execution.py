# SPDX-License-Identifier: Apache-2.0
"""CPU callback stubs only; no dynamic quantization/INT8 GEMM/TP collective."""
from dataclasses import replace
import sys
import hashlib
import unittest
from unittest.mock import Mock,patch
import torch
from test_glm5_next_oproj_payload import setup,fake_weight
from deepinfer.model_executor.models.glm5_next.oproj_payload import load_o_proj_columns_cpu
from deepinfer.model_executor.models.glm5_next.oproj_execution import execute_rank_oproj_cpu


def data(rank=0,layer=3,count=3):
    r,c,p=setup(rank,layer,count)
    with patch('deepinfer.model_executor.models.glm5_next.oproj_payload.read_i8_columns_cpu',side_effect=fake_weight):q=load_o_proj_columns_cpu(r,c,layer,p.heads,p.contract)
    x=torch.arange(count*4*256).reshape(count,4,256).bfloat16()
    return q,p,x


def run(q,p,x,**extra):
    kw=dict(plan=p,expected_identity=p.identity,expected_runtime_scale_sha256=hashlib.sha256(q.weight_scale.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),quantization_policy='rank_local_dynamic_int8',bias_policy='absent',rank_apply=lambda a,w,s,**k:torch.zeros((a.shape[0],4096),dtype=torch.bfloat16),reduce_sum=lambda y,**k:y)
    kw.update(extra);return execute_rank_oproj_cpu(q,x,**kw)


class OProjExecutionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_rank_apply_then_full_output_sum_with_bound_spec(self):
        for rank,layer in ((0,3),(15,43)):
            q,p,x=data(rank,layer);events=[];specs=[]
            def apply(a,w,s,*,spec):
                events.append('apply');specs.append(spec)
                self.assertEqual((a.shape,w.shape,s.shape),((3,1024),(4096,1024),(4096,)))
                self.assertTrue(torch.equal(a,x.reshape(3,1024)));self.assertTrue(torch.equal(w,q.weight_nk));self.assertTrue(torch.equal(s,q.weight_scale))
                self.assertEqual((spec.identity,spec.rank,spec.world_size,spec.input_columns),(p.identity,rank,16,(rank*1024,(rank+1)*1024)))
                self.assertEqual((spec.quantization_policy,spec.bias_policy),('rank_local_dynamic_int8','absent'))
                return torch.full((3,4096),rank+1,dtype=torch.bfloat16)
            def reduce(y,*,spec):
                events.append('sum');self.assertIs(spec,specs[0]);self.assertTrue(bool((y==rank+1).all()));return torch.full_like(y,23)
            y=run(q,p,x,rank_apply=apply,reduce_sum=reduce)
            self.assertEqual(events,['apply','sum']);self.assertEqual(y.shape,(3,4096));self.assertTrue(bool((y==23).all()))

    def test_backend_and_reducer_buffers_do_not_alias_inputs_or_result(self):
        q,p,x=data();before=x.clone();weight=q.weight_nk.clone();scale=q.weight_scale.clone();local=torch.ones(3,4096,dtype=torch.bfloat16);reduced=torch.full_like(local,7)
        def apply(a,w,s,**k):a.zero_();w.zero_();s.zero_();return local
        def reduce(y,**k):y.zero_();return reduced
        result=run(q,p,x,rank_apply=apply,reduce_sum=reduce)
        self.assertTrue(torch.equal(x,before));self.assertTrue(torch.equal(q.weight_nk,weight));self.assertTrue(torch.equal(q.weight_scale,scale));self.assertTrue(bool((local==1).all()))
        with torch.inference_mode():result.zero_()
        self.assertTrue(bool((reduced==7).all()))

    def test_request_revision_and_equal_metadata_foreign_owner_reject_before_callbacks(self):
        q,p,x=data();_,foreign,_=data();callback=Mock(side_effect=AssertionError('called'))
        for identity in (replace(p,revision=p.revision+1).identity,replace(p,request_id='other').identity,foreign.identity):
            with self.subTest(identity=identity),self.assertRaisesRegex(ValueError,'identity'):run(q,p,x,expected_identity=identity,rank_apply=callback,reduce_sum=callback)
        callback.assert_not_called()

    def test_no_implicit_quant_bias_or_reducer_policy(self):
        q,p,x=data();callback=Mock(side_effect=AssertionError('called'))
        for kw in ({'quantization_policy':None},{'quantization_policy':'prepared_input'},{'bias_policy':'rank0'},{'rank_apply':None},{'reduce_sum':None}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):run(q,p,x,rank_apply=callback,**kw) if 'rank_apply' not in kw else run(q,p,x,**kw)
        callback.assert_not_called()
        with self.assertRaises(TypeError):execute_rank_oproj_cpu(q,x,plan=p,expected_identity=p.identity)

    def test_plan_geometry_budget_guards_precede_callbacks(self):
        q,p,x=data();callback=Mock(side_effect=AssertionError('called'))
        for kw in ({'plan':replace(p,layer_id=7),'expected_identity':replace(p,layer_id=7).identity},{'max_payload_bytes':1},{'max_input_bytes':1},{'max_output_bytes':1},{'max_output_bytes':True}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):run(q,p,x,rank_apply=callback,reduce_sum=callback,**kw)
        for m in (replace(q.manifest,first_column=1),replace(q.manifest,runtime_bytes=1),replace(q.manifest,prefix='wrong')):
            with self.assertRaises(ValueError):run(replace(q,manifest=m),p,x,rank_apply=callback)
        callback.assert_not_called()

    def test_payload_dtype_digest_and_scale_guards(self):
        q,p,x=data();callback=Mock(side_effect=AssertionError('called'))
        for bad in (replace(q,weight_nk=q.weight_nk.to(torch.uint8)),replace(q,weight_sha256='0'*64),replace(q,source_scale_sha256='bad'),replace(q,weight_scale=q.weight_scale.float()),replace(q,weight_scale=torch.zeros_like(q.weight_scale)),replace(q,weight_scale=torch.full_like(q.weight_scale,float('nan')))):
            with self.assertRaises(ValueError):run(bad,p,x,rank_apply=callback)
        with torch.inference_mode():q.weight_nk[0,0]=3
        with self.assertRaisesRegex(ValueError,'digest changed'):run(q,p,x,rank_apply=callback)
        callback.assert_not_called()

    def test_positive_scale_mutation_requires_expected_digest_before_backend(self):
        q,p,x=data();callback=Mock(side_effect=AssertionError('called'));expected=hashlib.sha256(q.weight_scale.view(torch.uint8).numpy().tobytes()).hexdigest()
        with torch.inference_mode():q.weight_scale[0]=2
        with self.assertRaisesRegex(ValueError,'scale digest changed'):run(q,p,x,rank_apply=callback,expected_runtime_scale_sha256=expected)
        with self.assertRaisesRegex(ValueError,'expected runtime scale digest'):run(q,p,x,rank_apply=callback,expected_runtime_scale_sha256=None)
        callback.assert_not_called()

    def test_invalid_input_rejects_before_backend(self):
        q,p,x=data();callback=Mock(side_effect=AssertionError('called'))
        for bad in (x.float(),x[:2],torch.full_like(x,float('inf'))):
            with self.assertRaises(ValueError):run(q,p,bad,rank_apply=callback)
        callback.assert_not_called()

    def test_bad_backend_output_never_reaches_reducer(self):
        q,p,x=data();reducer=Mock(side_effect=AssertionError('sum called'))
        for bad in (None,torch.zeros(3,1024,dtype=torch.bfloat16),torch.zeros(3,4096),torch.full((3,4096),float('nan'),dtype=torch.bfloat16)):
            with self.assertRaises(ValueError):run(q,p,x,rank_apply=lambda *a,**k:bad,reduce_sum=reducer)
        reducer.assert_not_called()
        with self.assertRaisesRegex(RuntimeError,'backend failed'):run(q,p,x,rank_apply=Mock(side_effect=RuntimeError('backend failed')),reduce_sum=reducer)
        reducer.assert_not_called()
        self.assertEqual(run(q,p,x).shape,(3,4096))  # no request state committed by failed call

    def test_bad_reducer_output_or_failure_propagates_without_publication(self):
        q,p,x=data()
        for bad in (None,torch.zeros(3,1024,dtype=torch.bfloat16),torch.zeros(3,4096),torch.full((3,4096),float('inf'),dtype=torch.bfloat16)):
            with self.assertRaises(ValueError):run(q,p,x,reduce_sum=lambda *a,**k:bad)
        with self.assertRaisesRegex(RuntimeError,'sum failed'):run(q,p,x,reduce_sum=Mock(side_effect=RuntimeError('sum failed')))
        self.assertEqual(run(q,p,x).shape,(3,4096))

    def test_empty_skips_digest_and_all_callbacks_but_keeps_identity_guard(self):
        q,p,x=data(count=0);callback=Mock(side_effect=AssertionError('called'))
        with patch('deepinfer.model_executor.models.glm5_next.oproj_execution._digest',side_effect=AssertionError('empty digest')):
            y=run(q,p,x,rank_apply=callback,reduce_sum=callback)
        self.assertEqual(y.shape,(0,4096));self.assertEqual(y.dtype,torch.bfloat16);callback.assert_not_called()
        with self.assertRaises(ValueError):run(q,p,x,expected_identity=replace(p,revision=99).identity)

    def test_noncontiguous_input_and_backend_output_preserve_order_and_owned_result(self):
        q,p,x=data();values=torch.stack((x,x+1),-1)[...,0];self.assertFalse(values.is_contiguous())
        backing=torch.arange(3*4096*2).reshape(3,4096,2).bfloat16();out=backing[...,0]
        y=run(q,p,values,rank_apply=lambda a,w,s,**k:out)
        self.assertTrue(y.is_contiguous());self.assertTrue(torch.equal(y,out));self.assertNotEqual(y.data_ptr(),out.data_ptr())

    def test_no_native_backend_imports(self):
        self.assertFalse(any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))


if __name__=='__main__':unittest.main(verbosity=2)
