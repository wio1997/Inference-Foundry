# SPDX-License-Identifier: Apache-2.0
"""Synthetic FLOAT CPU chain plus INT8/sum markers; no NPU or INT8 math."""
from dataclasses import replace
from types import SimpleNamespace as NS
import hashlib,sys,unittest
from unittest.mock import Mock,patch
import torch
from torch import nn
from test_glm5_next_sfa_inputs import fixture as packet_fixture
from test_glm5_next_sfa_weights import components
from test_glm5_next_oproj_execution import data as o_data
from deepinfer.model_executor.models.glm5_next.absorbed_mla import absorbed_sparse_mla_cpu
from deepinfer.model_executor.models.glm5_next.private_pages import gather_private_latent_cpu
from deepinfer.model_executor.models.glm5_next.cache import state_from_aligned_prefix
from deepinfer.model_executor.models.glm5_next.hybrid_decoder import PreparedDSAAttention,GlmHybridDecoder,PreparedDecoderStack
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState
from deepinfer.model_executor.models.glm5_next.rank_dsa import execute_rank_dsa_cpu,CPURankAttentionResult,CPUProjectedDSAResult,CPUProjectedDSAOperation


def digest(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def data(rank=0,layer=3,start=3,count=3):
    p,g,raw,stage,_,ids,_=packet_fixture(start=start,count=count,rank=rank);p=replace(p,layer_id=layer)
    _,_,_,kvb=components(rank=rank,layer=layer,count=count);oproj,_,_=o_data(rank,layer,count)
    query=(torch.arange(count*4*256).reshape(count,4,256).float().remainder(17)*.001).bfloat16()
    return p,g,raw,stage,ids,kvb,oproj,query


def attend(packet):
    out=torch.empty_like(packet.query)
    for row in range(len(out)):
        ids=packet.sparse_indices[row,0];ids=ids[ids>=0].long();cache=packet.cache[packet.block_table[0,ids//128],ids%128,0]
        logits=torch.einsum('hl,sl->hs',packet.query[row].float(),cache.float())*packet.scale
        out[row]=torch.einsum('hs,sl->hl',torch.softmax(logits,-1),cache.float()).bfloat16()
    return CPURankAttentionResult(packet.identity,out)


def run(d,**extra):
    p,g,raw,stage,ids,kvb,oproj,q=d
    kw=dict(plan=p,expected_identity=p.identity,kv_b=kvb,o_proj=oproj,expected_runtime_scale_sha256=digest(oproj.weight_scale),attention=attend,rank_apply=lambda x,w,s,**k:torch.ones((len(x),4096),dtype=torch.bfloat16),reduce_sum=lambda y,**k:y,quantization_policy='rank_local_dynamic_int8',bias_policy='absent',index_namespace='logical_token')
    kw.update(extra);return execute_rank_dsa_cpu(q,raw,stage,ids,**kw)


class MLA(nn.Module):
    hidden_size=4096;num_heads=64;qk_dim=256;value_dim=256;q_rank=1536;kv_rank=512
    def __init__(self):super().__init__();self.projects=0;self.finishes=0
    def project(self,x):
        self.projects+=1;return NS(query=torch.full((len(x),64,256),.01,dtype=torch.bfloat16),query_rank=torch.zeros(len(x),1536,dtype=torch.bfloat16),latent=torch.zeros(len(x),512,dtype=torch.bfloat16))
    def finish(self,x):self.finishes+=1;raise AssertionError('double O-projection')


class Indexer(nn.Module):
    hidden_size=4096;q_rank=1536
    def __init__(self,layout):super().__init__();self.layout=layout;self.calls=0
    def forward(self,x,q):self.calls+=1;return None


def decoder_fixture(count=4):
    d=data(start=0,count=count);p,g,raw,stage,ids,kvb,oproj,_=d;mla=MLA();indexer=Indexer(g.layout)
    state=state_from_aligned_prefix(torch.empty(0,512,dtype=torch.bfloat16),torch.empty(0,128,dtype=torch.bfloat16),g.layout)
    next_state=state_from_aligned_prefix(torch.zeros(count,512,dtype=torch.bfloat16),torch.zeros(count//4,128,dtype=torch.bfloat16),g.layout)
    events=[]
    def operation(step,working,*,plan,expected_identity):
        events.append('operation');working.tail.fill_(99)
        head_query=step.mla.query[:,plan.heads.head_ids].contiguous()
        def attention(packet):events.append('attention');return attend(packet)
        def apply(x,w,s,**kw):events.append('rank_apply');return torch.full((len(x),4096),2,dtype=torch.bfloat16)
        def reduce(y,**kw):events.append('sum');return y+1
        output=execute_rank_dsa_cpu(head_query,raw,stage,ids,plan=plan,expected_identity=expected_identity,kv_b=kvb,o_proj=oproj,expected_runtime_scale_sha256=digest(oproj.weight_scale),attention=attention,rank_apply=apply,reduce_sum=reduce,quantization_policy='rank_local_dynamic_int8',bias_policy='absent',index_namespace='logical_token')
        return output,next_state
    wrapped=CPUProjectedDSAOperation(p,p.identity,operation);module=PreparedDSAAttention(mla,indexer,Mock(side_effect=AssertionError('default operation used')))
    return d,module,state,next_state,wrapped,events


class RankDSATests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5340)

    def test_full_rank_chain_matches_separate_absorbed_FLOAT_reference_then_markers(self):
        for rank,layer in ((0,3),(15,43)):
            d=data(rank,layer);p,g,raw,stage,ids,kvb,oproj,q=d;events=[];snapshots=[t.clone() for t in (raw,stage.main,stage.index,stage.tail,q,ids)]
            logical=gather_private_latent_cpu(raw,stage,g,torch.arange(p.candidate.end));oracle=absorbed_sparse_mla_cpu(q,logical,ids,torch.arange(p.candidate.start,p.candidate.end),kvb.weights,sink_policy='absent')
            def attention(packet):
                events.append('attention');self.assertEqual(packet.identity,p.identity);self.assertEqual(packet.scale,1/16);self.assertTrue(torch.equal(packet.query,oracle.query_latent));self.assertIs(packet.source_kwargs_cpu()['key'],packet.cache)
                value=attend(packet);self.assertTrue(torch.equal(value.latent,oracle.attention_latent));return value
            def apply(x,w,s,*,spec):
                events.append('rank_apply');self.assertTrue(torch.equal(x,oracle.values.reshape(3,1024)));self.assertEqual(spec.identity,p.identity);return torch.full((3,4096),rank+1,dtype=torch.bfloat16)
            def reduce(y,**kw):events.append('sum');return y+5
            result=run(d,attention=attention,rank_apply=apply,reduce_sum=reduce)
            self.assertEqual(result.identity,p.identity);self.assertTrue(bool((result.hidden==rank+6).all()));self.assertEqual(events,['attention','rank_apply','sum'])
            for t,saved in zip((raw,stage.main,stage.index,stage.tail,q,ids),snapshots):self.assertTrue(torch.equal(t,saved))

    def test_stale_identity_wrong_weights_and_work_bounds_precede_callbacks(self):
        d=data();p=d[0];callback=Mock(side_effect=AssertionError('called'))
        for kw in ({'expected_identity':replace(p,revision=99).identity},{'kv_b':replace(d[5],manifest=replace(d[5].manifest,layer_id=7))},{'o_proj':replace(d[6],manifest=replace(d[6].manifest,heads=replace(p.heads,rank=1,head_ids=(4,5,6,7))))},{'max_projection_work':1},{'max_weight_bytes':1},{'max_output_bytes':1},{'index_namespace':'pool'},{'quantization_policy':'prepared_input'},{'attention':None}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):run(d,**dict(attention=callback,**kw)) if 'attention' not in kw else run(d,**kw)
        callback.assert_not_called()

    def test_packet_guard_rejects_bad_selection_or_stage_before_attention(self):
        d=data();bad=d[4].clone();bad[0,0,0]=999;callback=Mock(side_effect=AssertionError('called'))
        for changed in ((d[:4]+(bad,)+d[5:]),(d[:3]+(replace(d[3],plan=replace(d[0].candidate,tail_private=4)),)+d[4:])):
            with self.assertRaises(ValueError):run(changed,attention=callback)
        callback.assert_not_called()

    def test_attention_result_identity_shape_dtype_finiteness_rejected_before_rank(self):
        d=data();p=d[0];backend=Mock(side_effect=AssertionError('rank called'));good=torch.zeros(3,4,512,dtype=torch.bfloat16)
        for result in (good,CPURankAttentionResult(replace(p,revision=99).identity,good),CPURankAttentionResult(p.identity,good.float()),CPURankAttentionResult(p.identity,good[:2]),CPURankAttentionResult(p.identity,torch.full_like(good,float('nan')))):
            with self.assertRaises(ValueError):run(d,attention=lambda *a:result,rank_apply=backend)
        backend.assert_not_called()

    def test_attention_and_backend_failure_leave_source_backing_unchanged(self):
        d=data();saved=d[2].clone();saved_stage=d[3].main.clone();reducer=Mock(side_effect=AssertionError('sum called'))
        def bad(packet):packet.cache.zero_();raise RuntimeError('attention failed')
        with self.assertRaisesRegex(RuntimeError,'attention failed'):run(d,attention=bad,reduce_sum=reducer)
        with self.assertRaisesRegex(RuntimeError,'rank failed'):run(d,rank_apply=Mock(side_effect=RuntimeError('rank failed')),reduce_sum=reducer)
        self.assertTrue(torch.equal(d[2],saved));self.assertTrue(torch.equal(d[3].main,saved_stage));reducer.assert_not_called()

    def test_private_UV_snapshot_survives_out_of_band_original_weight_mutation(self):
        d=data();expected=[]
        def attention(packet):
            result=attend(packet);expected.append(d[5].value_up_cpu(result.latent,plan=d[0]))
            with torch.inference_mode():d[5].weights.value_up.zero_()
            return result
        def apply(x,w,s,**kw):self.assertTrue(torch.equal(x,expected[0].reshape(3,1024)));return torch.ones(3,4096,dtype=torch.bfloat16)
        run(d,attention=attention,rank_apply=apply)

    def test_empty_skips_all_tensors_projections_and_callbacks(self):
        p,*_=packet_fixture(start=3,count=0);callback=Mock(side_effect=AssertionError('called'))
        with patch('torch.einsum',side_effect=AssertionError('empty math')):
            out=execute_rank_dsa_cpu(None,None,None,None,plan=p,expected_identity=p.identity,kv_b=None,o_proj=None,expected_runtime_scale_sha256='a'*64,attention=callback,rank_apply=callback,reduce_sum=callback,quantization_policy='rank_local_dynamic_int8',bias_policy='absent',index_namespace='logical_token')
        self.assertEqual(out.hidden.shape,(0,4096));self.assertEqual(out.identity,p.identity);callback.assert_not_called()

    def test_explicit_projected_decoder_runs_chain_once_without_finish_or_source_mutation(self):
        d,m,state,next_state,wrapped,events=decoder_fixture();before=state.clone();x=torch.zeros(4,4096,dtype=torch.bfloat16)
        y,following=m(x,state,operation=wrapped)
        self.assertTrue(bool((y==3).all()));self.assertEqual(following.tokens_seen,4);self.assertEqual((m.mla.projects,m.mla.finishes,m.indexer.calls),(1,0,1));self.assertEqual(events,['operation','attention','rank_apply','sum']);self.assertTrue(torch.equal(state.tail,before.tail));self.assertEqual(state.tokens_seen,0)

    def test_projected_decoder_rejects_candidate_mismatch_before_projections(self):
        d,m,state,_,wrapped,_=decoder_fixture()
        for x in (torch.zeros(3,4096,dtype=torch.bfloat16),torch.empty(0,4096,dtype=torch.bfloat16)):
            with self.assertRaisesRegex(ValueError,'candidate'):m(x,state,operation=wrapped)
        self.assertEqual(m.mla.projects,0)

    def test_projected_geometry_and_returned_state_ownership(self):
        d,m,state,next_state,wrapped,_=decoder_fixture();x=torch.zeros(4,4096,dtype=torch.bfloat16)
        m.mla.q_rank=5
        with self.assertRaisesRegex(ValueError,'geometry'):m(x,state,operation=wrapped)
        m.mla.q_rank=1536;original=m.layout
        with self.assertRaisesRegex(ValueError,'cache layout'):wrapped.validate_input(x,state,m.mla,replace(original,lookahead=1))
        out,following=m(x,state,operation=wrapped)
        with torch.inference_mode():following.latent.fill_(77);out.zero_()
        self.assertTrue(bool((next_state.latent==0).all()));self.assertEqual(m.mla.finishes,0)

    def test_projected_identity_unwrapped_output_and_bad_state_reject_without_finish(self):
        d,m,state,next_state,wrapped,_=decoder_fixture();p=d[0];x=torch.zeros(4,4096,dtype=torch.bfloat16);good=CPUProjectedDSAResult(p.identity,torch.ones_like(x))
        operations=[lambda step,s:(good,next_state),CPUProjectedDSAOperation(p,p.identity,lambda *a,**k:(replace(good,identity=replace(p,revision=99).identity),next_state)),CPUProjectedDSAOperation(p,p.identity,lambda *a,**k:(good,state)),CPUProjectedDSAOperation(p,p.identity,lambda *a,**k:(torch.ones_like(x),next_state))]
        for op in operations:
            with self.assertRaises(ValueError):m(x,state,operation=op)
        self.assertEqual(m.mla.finishes,0);self.assertEqual(state.tokens_seen,0)

    def test_projected_callback_exception_preserves_direct_input_state(self):
        d,m,state,_,wrapped,_=decoder_fixture();saved=state.clone()
        def bad(step,working,**kw):working.tail.fill_(99);raise RuntimeError('projected failure')
        with self.assertRaisesRegex(RuntimeError,'projected failure'):m(torch.zeros(4,4096,dtype=torch.bfloat16),state,operation=replace(wrapped,operation=bad))
        self.assertTrue(torch.equal(state.tail,saved.tail));self.assertEqual(state.tokens_seen,0);self.assertEqual(m.mla.finishes,0)

    def test_projected_empty_plan_validates_and_skips_decoder_components(self):
        d,m,state,_,wrapped,_=decoder_fixture(count=0);out,following=m(torch.empty(0,4096,dtype=torch.bfloat16),state,operation=wrapped)
        self.assertEqual(out.shape,(0,4096));self.assertEqual(following.tokens_seen,0);self.assertEqual((m.mla.projects,m.mla.finishes,m.indexer.calls),(0,0,0))

    def test_hybrid_and_stack_reject_foreign_layer_and_request(self):
        d,m,state,_,wrapped,_=decoder_fixture();p=d[0]
        class HC(nn.Module):streams=4;hidden_size=4096
        layer=GlmHybridDecoder(layer_id=3,attention_kind='deepseek_sparse_attention',mlp_kind='dense',attn_hc=HC(),ffn_hc=HC(),input_norm=torch.ones(4096),ffn_norm=torch.ones(4096),attention=m,mlp=nn.Identity())
        other=replace(p,layer_id=7)
        with self.assertRaisesRegex(ValueError,'layer differs'):layer(torch.zeros(4,4096,dtype=torch.bfloat16),state,dsa_operation=CPUProjectedDSAOperation(other,other.identity,wrapped.operation))
        # Existing stack guard can be checked using an injected small backbone.
        from test_glm5_next_hybrid_decoder import setup_stack
        stack,mixed,*_=setup_stack();foreign=replace(p,layer_id=3,request_id='other')
        with self.assertRaisesRegex(ValueError,'request differs'):stack(torch.zeros(4,6,dtype=torch.bfloat16),mixed,dsa_operations={1:CPUProjectedDSAOperation(foreign,foreign.identity,wrapped.operation)})

    def test_no_native_modules_loaded(self):self.assertFalse(any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))


if __name__=='__main__':unittest.main(verbosity=2)
