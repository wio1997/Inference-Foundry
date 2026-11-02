# SPDX-License-Identifier: Apache-2.0
"""Small CPU page-planning/copy/gather invariants; no serving allocator."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry
from deepinfer.model_executor.models.glm5_next.private_pages import plan_private_dsa_pages,prepare_private_dsa_cpu,gather_private_latent_cpu


def fixture(start=127,tokens=5,block=128):
    g=CANNCacheGeometry(DSACacheLayout(4,2,4,8,block),7*4*4)
    prefix=tuple((1,4,2)[i] for i in range((start+block-1)//block));changed=(start+tokens-1)//block-start//block+1 if tokens else 0
    kwargs=dict(geometry=g,num_main_pages=10,num_tail_pages=5,occupied_main=frozenset((*prefix,5)),occupied_tail=frozenset({0,1}),tail_source=0)
    plan=plan_private_dsa_pages(start,tokens,prefix,(8,3)[:changed],tail_private=3 if tokens else None,**kwargs)
    main=torch.arange(10*g.main_page_bytes,dtype=torch.int64).remainder(239).to(torch.uint8);index=torch.arange(10*g.index_page_bytes,dtype=torch.int64).remainder(197).to(torch.uint8);tail=torch.full((5*g.tail_page_bytes,),0x55,dtype=torch.uint8)
    latent=torch.arange(tokens*4).reshape(tokens,4).bfloat16();compressed=torch.arange(plan.new_pools*2).reshape(plan.new_pools,2).bfloat16();after=torch.arange(16).reshape(4,4).float() if tokens else None
    return g,plan,(main,index,tail),latent,compressed,after


class PrivatePageTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_partial_page_and_new_page_copy_as_coupled_pair(self):
        g,p,raw,latent,compressed,tail=fixture();saved=[t.clone() for t in raw];s=prepare_private_dsa_cpu(*raw,p,g,latent,compressed,tail)
        self.assertEqual(p.read_table,(8,3));self.assertEqual(p.copy_sources,(1,None));self.assertEqual(p.new_pools,2)
        before=saved[0].view(torch.bfloat16).reshape(10,128,4);out=s.main.view(torch.bfloat16).reshape(2,128,4)
        self.assertTrue(torch.equal(out[0,:127],before[1,:127]));self.assertTrue(torch.equal(out[0,127],latent[0]));self.assertTrue(torch.equal(out[1,:4],latent[1:]));self.assertTrue(bool((out[1,4:]==0).all()))
        idx=s.index.view(torch.bfloat16).reshape(2,32,2);old=saved[1].view(torch.bfloat16).reshape(10,32,2)
        self.assertTrue(torch.equal(idx[0,:31],old[1,:31]));self.assertTrue(torch.equal(idx[0,31],compressed[0]));self.assertTrue(torch.equal(idx[1,0],compressed[1]))
        for a,b in zip(raw,saved):self.assertTrue(torch.equal(a,b))

    def test_untouched_shared_prefix_and_gather_switch_to_private(self):
        g,p,raw,latent,compressed,tail=fixture(start=255,tokens=4);s=prepare_private_dsa_cpu(*raw,p,g,latent,compressed,tail)
        self.assertEqual(p.read_table,(1,8,3));ids=torch.tensor([0,254,255,256,258,-1]);actual=gather_private_latent_cpu(raw[0],s,g,ids)
        old=raw[0].view(torch.bfloat16).reshape(10,128,4)
        expected=torch.stack((old[1,0],old[4,126],latent[0],latent[1],latent[3],torch.zeros(4,dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(actual,expected))

    def test_larger_logical_pages_expand_to_kernel128_ids(self):
        g,p,*_=fixture(start=255,tokens=4,block=256)
        self.assertEqual(p.read_table,(8,3));self.assertEqual(p.kernel_read_table,(16,17,6,7))

    def test_tail_padding_preserved_and_fresh_tail_zero_padding(self):
        g,p,raw,l,c,tail=fixture();s=prepare_private_dsa_cpu(*raw,p,g,l,c,tail)
        self.assertTrue(torch.equal(s.tail.view(torch.float32).reshape(1,7,4)[0,:4],tail));self.assertTrue(bool((s.tail[0,4*4*4:]==0x55).all()))
        fresh=plan_private_dsa_pages(0,1,(),(8,),geometry=g,num_main_pages=10,num_tail_pages=5,occupied_main=frozenset({1}),occupied_tail=frozenset({0}),tail_private=3)
        out=prepare_private_dsa_cpu(*raw,fresh,g,l[:1],c[:0],tail);self.assertTrue(bool((out.tail[0,4*4*4:]==0).all()))

    def test_empty_reserves_and_copies_nothing(self):
        g,p,raw,l,c,tail=fixture(tokens=0);s=prepare_private_dsa_cpu(*raw,p,g,l,c,tail)
        self.assertEqual(p.read_table,p.committed_table);self.assertEqual(s.main.numel()+s.index.numel()+s.tail.numel(),0)
        self.assertTrue(torch.equal(gather_private_latent_cpu(raw[0],s,g,torch.tensor([0])),raw[0].view(torch.bfloat16).reshape(10,128,4)[1,:1]))

    def test_reservation_collisions_unrelated_occupied_and_missing_history(self):
        g,_,_,_,_,_=fixture();base=dict(geometry=g,num_main_pages=10,num_tail_pages=5,occupied_main=frozenset({1,5}),occupied_tail=frozenset({0,1}),tail_source=0,tail_private=3)
        for kind in ('prefix','other','duplicate','tail','history','short'):
            kw=base.copy();dest=(8,3)
            if kind=='prefix':dest=(1,3)
            elif kind=='other':dest=(5,3)
            elif kind=='duplicate':dest=(8,8)
            elif kind=='tail':kw['tail_private']=1
            elif kind=='history':kw['tail_source']=None
            else:dest=(8,)
            with self.subTest(kind=kind),self.assertRaises(ValueError):plan_private_dsa_pages(127,5,(1,),dest,**kw)

    def test_forged_plan_table_copy_source_and_alias_reject(self):
        _,p,*_=fixture()
        for kw in ({'read_table':(8,2)},{'copy_sources':(4,None)},{'private_ids':(1,3)},{'tail_private':0},{'changed_logical_pages':(0,) }):
            with self.assertRaises(ValueError):replace(p,**kw)

    def test_failure_and_byte_guard_leave_committed_bytes_unchanged(self):
        g,p,raw,l,c,tail=fixture();saved=[v.clone() for v in raw]
        with patch('torch.zeros',side_effect=AssertionError('allocated before budget')):
            with self.assertRaisesRegex(ValueError,'byte budget'):prepare_private_dsa_cpu(*raw,p,g,l,c,tail,max_private_bytes=1)
        bad=l.clone();bad[0,0]=float('nan')
        with self.assertRaises(ValueError):prepare_private_dsa_cpu(*raw,p,g,bad,c,tail)
        for a,b in zip(raw,saved):self.assertTrue(torch.equal(a,b))

    def test_private_stages_are_independent_and_not_published(self):
        g,p,raw,l,c,tail=fixture();one=prepare_private_dsa_cpu(*raw,p,g,l,c,tail);two=prepare_private_dsa_cpu(*raw,p,g,l,c,tail)
        with torch.inference_mode():one.main.zero_();one.index.zero_();one.tail.zero_()
        self.assertFalse(torch.equal(one.main,two.main));self.assertEqual(p.committed_table,(1,));self.assertEqual(p.read_table,(8,3))

    def test_native_incomplete_payload_and_future_gather_reject(self):
        g,p,raw,l,c,tail=fixture();s=prepare_private_dsa_cpu(*raw,p,g,l,c,tail)
        with self.assertRaisesRegex(ValueError,'CPU-only'):prepare_private_dsa_cpu(torch.empty(raw[0].shape,device='meta',dtype=torch.uint8),*raw[1:],p,g,l,c,tail)
        with self.assertRaises(ValueError):prepare_private_dsa_cpu(*raw,p,g,l,c[:0],tail)
        with self.assertRaises(ValueError):gather_private_latent_cpu(raw[0],s,g,torch.tensor([p.end]))
        with self.assertRaisesRegex(ValueError,'byte budget'):gather_private_latent_cpu(raw[0],s,g,torch.tensor([0]),max_output_bytes=1)


if __name__=='__main__':unittest.main(verbosity=2)
