# SPDX-License-Identifier: Apache-2.0
"""CPU single-layer request publication; no native allocator or model forward."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,state_from_aligned_prefix,append_dsa
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry,view_tail,pack_tail_planes
from deepinfer.model_executor.models.glm5_next.private_request import PrivateDSARequest


def following(state,count,g):
    l=g.layout;start=state.tokens_seen
    latent=(torch.arange(start,start+count)[:,None]+torch.arange(l.latent_dim)[None,:]*.125).bfloat16()
    keys=torch.arange(count*l.index_dim).reshape(count,l.index_dim).float()+start;gates=torch.zeros_like(keys)
    return append_dsa(latent,keys,gates,torch.zeros(l.pool_size,l.index_dim),state,l,start_position=start)


def fixture(request_id='fixture',start=3):
    l=DSACacheLayout(4,2,4,8,128);g=CANNCacheGeometry(l,7*4*4)
    empty=state_from_aligned_prefix(torch.empty(0,4,dtype=torch.bfloat16),torch.empty(0,2,dtype=torch.bfloat16),l);state=following(empty,start,g)
    main=torch.zeros(10*g.main_page_bytes,dtype=torch.uint8);index=torch.zeros(10*g.index_page_bytes,dtype=torch.uint8);tail=torch.full((5*g.tail_page_bytes,),0x55,dtype=torch.uint8)
    table=(1,) if start else ();main.view(torch.bfloat16).reshape(10,128,4)[1,:start]=state.latent;index.view(torch.bfloat16).reshape(10,32,2)[1,:start//4]=state.compressed
    view_tail(tail,5,g)[0]=pack_tail_planes(state.tail)
    r=PrivateDSARequest(request_id,state,main,index,tail,table,0,geometry=g,num_main_pages=10,num_tail_pages=5,output_width=4,max_context_tokens=32)
    return r,g,state,(main,index,tail)


def prepare(r,g,state,count=2,**extra):
    kwargs=dict(tail_private=3,occupied_main=frozenset({1,5}),occupied_tail=frozenset({0,1}),expected_stamp=r.stamp,
                attention=lambda view:view.gather(torch.tensor(view.positions)))
    kwargs.update(extra);return r.prepare(following(state,count,g),(8,) if count else (),**kwargs)


def same(test,a,b):
    test.assertEqual(a.stamp,b.stamp)
    for x,y in zip((a.main_raw,a.index_raw,a.tail_raw,a.state.latent,a.state.compressed,a.state.tail,a.state.tail_positions),
                   (b.main_raw,b.index_raw,b.tail_raw,b.state.latent,b.state.compressed,b.state.tail,b.state.tail_positions)):
        test.assertTrue(torch.equal(x,y))


class PrivateRequestTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_prepare_reads_candidate_without_publishing_then_commits(self):
        r,g,state,raw=fixture();before=r.snapshot();branch=prepare(r,g,state);same(self,before,r.snapshot())
        self.assertEqual(branch.candidate_stamp.logical_table,(8,));self.assertTrue(torch.equal(branch.output,following(state,2,g).latent[3:]))
        result=r.publish(branch,expected_stamp=before.stamp,validate=lambda output:None);after=r.snapshot()
        self.assertEqual((after.stamp.tokens_seen,after.stamp.revision,branch.status),(5,1,'published'));self.assertTrue(torch.equal(result,after.state.latent[3:]))
        self.assertTrue(torch.equal(raw[0],before.main_raw));self.assertTrue(torch.equal(raw[1],before.index_raw));self.assertTrue(torch.equal(raw[2],before.tail_raw))

    def test_attention_and_late_validator_failures_preserve_full_frame(self):
        r,g,state,_=fixture();before=r.snapshot()
        def fail(x):raise RuntimeError('injected failure')
        with self.assertRaisesRegex(RuntimeError,'injected'):prepare(r,g,state,attention=fail)
        same(self,before,r.snapshot());branch=prepare(r,g,state)
        with self.assertRaisesRegex(RuntimeError,'injected'):r.publish(branch,expected_stamp=before.stamp,validate=fail)
        same(self,before,r.snapshot());self.assertEqual(branch.status,'rejected')
        with self.assertRaises(ValueError):r.publish(branch,expected_stamp=before.stamp,validate=lambda x:None)

    def test_two_prepared_branches_only_first_can_publish(self):
        r,g,state,_=fixture();stamp=r.stamp;a=prepare(r,g,state);b=prepare(r,g,state)
        r.publish(a,expected_stamp=stamp,validate=lambda x:None);after=r.snapshot()
        with self.assertRaises(ValueError):r.publish(b,expected_stamp=stamp,validate=lambda x:self.fail('stale validator executed'))
        same(self,after,r.snapshot())

    def test_wrong_position_revision_table_and_foreign_owner_reject_early(self):
        r,g,state,_=fixture();other,_,_,_=fixture();stamp=r.stamp
        for expected in (replace(stamp,tokens_seen=4),replace(stamp,revision=2),replace(stamp,logical_table=(4,)),other.stamp):
            with self.assertRaises(ValueError):prepare(r,g,state,expected_stamp=expected,attention=lambda v:self.fail('attention executed'))

    def test_prefix_rewrite_and_bad_output_reject_without_publication(self):
        r,g,state,_=fixture();before=r.snapshot();next_state=following(state,2,g)
        with torch.inference_mode():next_state.latent[0]=99
        with self.assertRaisesRegex(ValueError,'rewrites'):r.prepare(next_state,(8,),tail_private=3,occupied_main=frozenset({1}),occupied_tail=frozenset({0}),expected_stamp=r.stamp,attention=lambda v:None)
        for value in (torch.zeros(2,4),torch.zeros(3,4,dtype=torch.bfloat16),torch.full((2,4),float('nan'),dtype=torch.bfloat16)):
            with self.assertRaises(ValueError):prepare(r,g,state,attention=lambda v:value)
        same(self,before,r.snapshot())

    def test_retained_payload_output_gather_and_snapshot_copies_are_private(self):
        r,g,state,_=fixture();kept=[]
        def attention(view):
            value=view.gather(torch.tensor(view.positions));kept.append(value);return value
        branch=prepare(r,g,state,attention=attention);expected=branch.output
        with torch.inference_mode():kept[0].zero_();branch.output.zero_()
        self.assertTrue(torch.equal(branch.output,expected));r.publish(branch,expected_stamp=r.stamp,validate=lambda x:x.zero_())
        snapshot=r.snapshot();again=r.snapshot()
        with torch.inference_mode():snapshot.state.latent.zero_();snapshot.main_raw.zero_()
        same(self,again,r.snapshot())

    def test_following_state_mutation_after_prepare_cannot_change_commit(self):
        r,g,state,_=fixture();next_state=following(state,2,g);branch=r.prepare(next_state,(8,),tail_private=3,occupied_main=frozenset({1}),occupied_tail=frozenset({0}),expected_stamp=r.stamp,attention=lambda v:v.gather(torch.tensor(v.positions)))
        with torch.inference_mode():next_state.latent.zero_();next_state.compressed.zero_();next_state.tail.zero_()
        r.publish(branch,expected_stamp=r.stamp,validate=lambda x:None);self.assertFalse(bool((r.snapshot().state.latent==0).all()))

    def test_empty_skips_attention_validator_and_revision_change(self):
        r,g,state,_=fixture();before=r.snapshot();branch=prepare(r,g,state,count=0,tail_private=None,attention=lambda v:self.fail('empty attention'))
        out=r.publish(branch,expected_stamp=r.stamp,validate=lambda v:self.fail('empty validator'));same(self,before,r.snapshot());self.assertEqual(out.shape,(0,4))
        changed=state.clone()
        with torch.inference_mode():changed.tail[0,0]=99
        with self.assertRaisesRegex(ValueError,'empty transition'):r.prepare(changed,(),tail_private=None,occupied_main=frozenset({1}),occupied_tail=frozenset({0}),expected_stamp=r.stamp,attention=lambda v:None)

    def test_reentrant_use_and_closed_owner_fail(self):
        r,g,state,_=fixture()
        with self.assertRaisesRegex(RuntimeError,'busy'):prepare(r,g,state,attention=lambda view:r.snapshot())
        branch=prepare(r,g,state);stamp=r.stamp;r.close();r.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):r.publish(branch,expected_stamp=stamp,validate=lambda x:None)

    def test_frame_budget_native_and_speculation_guards(self):
        r,g,state,raw=fixture()
        with patch.object(torch.Tensor,'clone',side_effect=AssertionError('clone before budget')):
            with self.assertRaisesRegex(ValueError,'byte budget'):PrivateDSARequest('bad',state,*raw,(1,),0,geometry=g,num_main_pages=10,num_tail_pages=5,output_width=4,max_backing_bytes=1)
        with self.assertRaisesRegex(ValueError,'CPU-only'):PrivateDSARequest('bad',state,torch.empty(raw[0].shape,device='meta',dtype=torch.uint8),*raw[1:],(1,),0,geometry=g,num_main_pages=10,num_tail_pages=5,output_width=4)
        speculative=CANNCacheGeometry(DSACacheLayout(4,2,4,8,128,1),7*4*4)
        with self.assertRaisesRegex(ValueError,'speculative'):PrivateDSARequest('bad',state,*raw,(1,),0,geometry=speculative,num_main_pages=10,num_tail_pages=5,output_width=4)


if __name__=='__main__':unittest.main(verbosity=2)
