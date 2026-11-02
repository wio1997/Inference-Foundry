# SPDX-License-Identifier: Apache-2.0
"""Private compressed-page reads, causal selection and tiny mixed CPU seam."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch
from test_glm5_next_private_pages import fixture as page_fixture
from test_glm5_next_private_request import following
from test_glm5_next_private_text_request import fixture as text_fixture,plans,same_pages
from test_glm5_next_hybrid_decoder import equal_state
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,state_from_aligned_prefix
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry,view_tail,pack_tail_planes
from deepinfer.model_executor.models.glm5_next.private_pages import prepare_private_dsa_cpu,gather_private_index_cpu
from deepinfer.model_executor.models.glm5_next.private_request import PrivateDSARequest
from deepinfer.model_executor.models.glm5_next.private_selection import private_absorbed_mla_cpu
from deepinfer.model_executor.models.glm5_next.absorbed_mla import AbsorbedMLAContract,split_kv_b_cpu,absorbed_sparse_mla_cpu
from deepinfer.model_executor.models.glm5_next.sparse_reference import select_pools_reference


def prepared():
    l=DSACacheLayout(4,2,4,4,128);g=CANNCacheGeometry(l,7*4*4)
    empty=state_from_aligned_prefix(torch.empty(0,4,dtype=torch.bfloat16),torch.empty(0,2,dtype=torch.bfloat16),l)
    before=following(empty,3,g);after=following(before,5,g)
    after=replace(after,compressed=torch.tensor([[-1.,-1.],[10.,10.]]).bfloat16())  # explicit synthetic completed producers
    main=torch.zeros(10*g.main_page_bytes,dtype=torch.uint8);index=torch.zeros(10*g.index_page_bytes,dtype=torch.uint8);tail=torch.zeros(5*g.tail_page_bytes,dtype=torch.uint8)
    main.view(torch.bfloat16).reshape(10,128,4)[1,:3]=before.latent
    rawindex=index.view(torch.bfloat16).reshape(10,32,2);rawindex[1,0]=100;rawindex[1,1]=-100  # old unused rows must not be read
    view_tail(tail,5,g)[0]=pack_tail_planes(before.tail)
    r=PrivateDSARequest('select',before,main,index,tail,(1,),0,geometry=g,num_main_pages=10,num_tail_pages=5,output_width=4,max_context_tokens=32)
    views=[]
    def attend(view):views.append(view);return view.gather(torch.tensor(view.positions))
    branch=r.prepare(after,(8,),tail_private=3,occupied_main=frozenset({1}),occupied_tail=frozenset({0}),expected_stamp=r.stamp,attention=attend)
    return r,branch,views[0],before,after,(main,index,tail)


def bind(r,q):
    seen={};weights={}
    for i in (1,2):
        mla=r.stack.layers[i].attention.mla;l=r.stack.layout
        weights[i]=split_kv_b_cpu(mla.kv_b.weight,AbsorbedMLAContract(mla.num_heads,tuple(range(mla.num_heads)),mla.qk_dim,mla.kv_rank,mla.value_dim,l.pool_size,l.topk_tokens))
        producer=q._producers[i]
        def produce(step,state,producer=producer,i=i):
            value=producer(step,state);seen[i]=value;return value
        def attend(step,view,i=i):
            completed=seen[i];actual=private_absorbed_mla_cpu(view,step.indexer.query,step.indexer.head_weights,step.mla.query,weights[i],sink_policy='absent')
            expected=select_pools_reference(step.indexer.query,step.indexer.head_weights,completed.compressed,step.positions,l)
            assert torch.equal(actual.selection.pool_ids,expected.pool_ids)
            assert torch.equal(view.gather_compressed(torch.arange(len(completed.compressed))),completed.compressed)
            oracle=absorbed_sparse_mla_cpu(step.mla.query,completed.latent,expected.token_indices,step.positions,weights[i],sink_policy='absent')
            assert torch.equal(actual.attention.values,oracle.values)
            return actual.attention.values
        q._producers[i]=produce;q._attentions[i]=attend
    return seen


class PrivateSelectionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5335)

    def test_gather_shared_copied_partial_and_new_index_pages(self):
        g,p,raw,latent,compressed,tail=page_fixture(start=255,tokens=5)
        with torch.inference_mode():raw[1].view(torch.bfloat16).copy_(torch.arange(raw[1].numel()//2).bfloat16())
        saved=raw[1].clone();stage=prepare_private_dsa_cpu(*raw,p,g,latent,compressed,tail)
        ids=torch.tensor([0,62,63,64,-1]);actual=gather_private_index_cpu(raw[1],stage,g,ids);old=raw[1].view(torch.bfloat16).reshape(10,32,2)
        expected=torch.stack((old[1,0],old[4,30],compressed[0],compressed[1],torch.zeros(2,dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(actual,expected));self.assertTrue(torch.equal(raw[1],saved))
        with torch.inference_mode():actual.zero_()
        self.assertTrue(torch.equal(gather_private_index_cpu(raw[1],stage,g,ids),expected))

    def test_index_gather_empty_stage_padding_bounds_and_device_guards(self):
        g,p,raw,l,c,t=page_fixture(tokens=0);stage=prepare_private_dsa_cpu(*raw,p,g,l,c,t)
        self.assertEqual(gather_private_index_cpu(raw[1],stage,g,torch.tensor([],dtype=torch.int32)).shape,(0,2))
        self.assertFalse(bool(gather_private_index_cpu(raw[1],stage,g,torch.tensor([-1])).any()))
        for ids in (torch.tensor([p.end//4]),torch.tensor([-2]),torch.tensor([0.])):
            with self.assertRaises(ValueError):gather_private_index_cpu(raw[1],stage,g,ids)
        with self.assertRaisesRegex(ValueError,'byte budget'):gather_private_index_cpu(raw[1],stage,g,torch.tensor([0]),max_output_bytes=1)
        with self.assertRaisesRegex(ValueError,'CPU-only'):gather_private_index_cpu(torch.empty(raw[1].shape,device='meta',dtype=torch.uint8),stage,g,torch.tensor([0]))
        with self.assertRaises(ValueError):gather_private_index_cpu(raw[1][1:],stage,g,torch.tensor([0]))
        with self.assertRaises(ValueError):gather_private_index_cpu(raw[1],replace(stage,index=torch.empty(1,dtype=torch.uint8)),g,torch.tensor([0]))

    def test_new_pool_visibility_causal_tail_and_old_poison_not_selected(self):
        r,b,v,before,after,raw=prepared();stamp=r.stamp;pages=r.snapshot()
        q=torch.ones(5,2,2,dtype=torch.bfloat16);w=torch.full((5,2),.5,dtype=torch.bfloat16);sel=v.select_pools(q,w)
        self.assertEqual(sel.pool_ids.tolist(),[[0],[0],[0],[0],[1]])
        self.assertEqual(sel.pool_scores[:,0].tolist(),[-2,-2,-2,-2,20])
        self.assertEqual(sel.visible_pools.tolist(),[1,1,1,1,2])
        self.assertEqual(sel.token_indices[3,:7].tolist(),[0,1,2,3,4,5,6]);self.assertEqual(sel.token_indices[4,:4].tolist(),[4,5,6,7])
        stale=select_pools_reference(q,w,raw[1].view(torch.bfloat16).reshape(10,32,2)[1,:2],torch.tensor(v.positions),v.layout)
        self.assertNotEqual(stale.pool_ids[-1].tolist(),sel.pool_ids[-1].tolist())
        self.assertEqual(r.stamp,stamp);self.assertTrue(torch.equal(r.snapshot().index_raw,pages.index_raw));self.assertEqual(before.tokens_seen,3)
        self.assertTrue(torch.equal(v.gather_compressed(torch.tensor([0,1])),after.compressed))

    def test_bridge_matches_dense_oracle_and_stale_index_changes_values(self):
        r,b,v,before,after,raw=prepared();c=AbsorbedMLAContract(2,(0,1),3,4,2,4,4)
        weights=split_kv_b_cpu((torch.randn(10,4)*.125).bfloat16(),c);query=(torch.randn(5,2,3)*.1).bfloat16();iq=torch.ones(5,2,2,dtype=torch.bfloat16);hw=torch.full((5,2),.5,dtype=torch.bfloat16)
        actual=private_absorbed_mla_cpu(v,iq,hw,query,weights,sink_policy='absent');expected=select_pools_reference(iq,hw,after.compressed,torch.tensor(v.positions),v.layout)
        oracle=absorbed_sparse_mla_cpu(query,after.latent,expected.token_indices,torch.tensor(v.positions),weights,sink_policy='absent')
        self.assertTrue(torch.equal(actual.attention.values,oracle.values));self.assertTrue(torch.equal(actual.attention.query_latent,oracle.query_latent))
        stale=select_pools_reference(iq,hw,raw[1].view(torch.bfloat16).reshape(10,32,2)[1,:2],torch.tensor(v.positions),v.layout)
        wrong=absorbed_sparse_mla_cpu(query,after.latent,stale.token_indices,torch.tensor(v.positions),weights,sink_policy='absent')
        self.assertFalse(torch.equal(wrong.values[-1],actual.attention.values[-1]));self.assertEqual(r.stamp.revision,0)

    def test_selection_and_bridge_contracts_reject_before_publication(self):
        r,b,v,*_=prepared();q=torch.ones(5,2,2,dtype=torch.bfloat16);w=torch.ones(5,2,dtype=torch.bfloat16)
        with self.assertRaises(ValueError):v.select_pools(q,w,max_history_pools=1)
        with self.assertRaises(ValueError):v.select_pools(q,w,max_score_elements=1)
        with self.assertRaises(ValueError):v.select_pools(q[:1],w[:1])
        c=AbsorbedMLAContract(2,(0,1),3,4,2,4,4);weights=split_kv_b_cpu(torch.ones(10,4,dtype=torch.bfloat16),c);query=torch.ones(5,2,3,dtype=torch.bfloat16)
        for extra in ({'sink_policy':'smla'},{'max_latent_tokens':7},{'max_tokens':4},{'max_work_elements':1}):
            kw={'sink_policy':'absent',**extra}
            with self.assertRaises(ValueError):private_absorbed_mla_cpu(v,q,w,query,weights,**kw)
        mismatch=split_kv_b_cpu(torch.ones(10,4,dtype=torch.bfloat16),replace(c,topk_tokens=8))
        with self.assertRaises(ValueError):private_absorbed_mla_cpu(v,q,w,query,mismatch,sink_policy='absent')
        with self.assertRaises(ValueError):v.gather_compressed(torch.tensor([2]))
        self.assertEqual(r.stamp.revision,0)

    def test_empty_view_has_no_visible_queries_or_selection(self):
        from deepinfer.model_executor.models.glm5_next.private_request import CandidateDSAReadView
        from types import SimpleNamespace
        g,p,raw,l,c,t=page_fixture(tokens=0);stage=prepare_private_dsa_cpu(*raw,p,g,l,c,t);v=CandidateDSAReadView(SimpleNamespace(main_raw=raw[0],index_raw=raw[1]),stage,g)
        sel=v.select_pools(torch.empty(0,2,2,dtype=torch.bfloat16),torch.empty(0,2,dtype=torch.bfloat16))
        self.assertEqual(sel.pool_ids.shape,(0,2));self.assertEqual(sel.token_indices.shape,(0,11))

    def test_two_DSA_private_text_integrates_candidate_selection_and_absorption(self):
        r,s,q,_,_=text_fixture();seen=bind(r,q);out=q.step(torch.tensor([0,1,2,3,4]),expected_stamp=q.stamp,page_plans=plans(q,5))
        self.assertEqual(out.logits.shape,(5,4));self.assertEqual(q.stamp.revision,1);self.assertEqual(set(seen),{1,2});self.assertEqual({v.tokens_seen for v in q.snapshot().layers.values()},{5})

    def test_late_selected_attention_failure_preserves_all_mixed_frames(self):
        r,s,q,_,_=text_fixture();bind(r,q);q.step(torch.tensor([0,1,2]),expected_stamp=q.stamp,page_plans=plans(q,3));saved=q.snapshot();pages=q.page_snapshot();stamp=q.stamp;original=q._attentions[2]
        def fail(step,view):original(step,view);raise RuntimeError('late candidate selected attention rejection')
        q._attentions[2]=fail
        with self.assertRaisesRegex(RuntimeError,'late candidate'):q.step(torch.tensor([3,4]),expected_stamp=stamp,page_plans=plans(q,2))
        equal_state(saved,q.snapshot());same_pages(pages,q.page_snapshot());self.assertEqual(q.stamp,stamp)

    def test_selected_private_text_whole_and_segmented_match(self):
        r,s,q,b,_=text_fixture();bind(r,q);other=MixedRequestState.capture('B',r.stack.config,r.stack.layout,s.layers)
        q2=r.open_private_request(other,page_bindings=b,producers=q._producers,attentions=q._attentions);bind(r,q2)
        ids=torch.tensor([0,1,2,3,4,5,6]);whole=q.step(ids,expected_stamp=q.stamp,page_plans=plans(q,7));out=[];start=0
        for stop in (1,3,4,7):out.append(q2.step(ids[start:stop],expected_stamp=q2.stamp,page_plans=plans(q2,stop-start)).logits);start=stop
        self.assertTrue(torch.equal(torch.cat(out),whole.logits));equal_state(q.snapshot(),replace(q2.snapshot(),request_id=q.request_id))


from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState
if __name__=='__main__':unittest.main(verbosity=2)
