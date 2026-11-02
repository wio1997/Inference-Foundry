# SPDX-License-Identifier: Apache-2.0
"""CPU byte-address/aliasing and independent request-tail metadata checks."""
from dataclasses import replace
import unittest
import torch

from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next.cann_cache import (
    CANNCacheGeometry,view_main_index,view_tail,pack_tail_planes,unpack_tail_rows,
    expand_kernel_block_table,plan_tail,prepare_tail_state,
)


def geometry(block=256,lookahead=3,padded_rows=13):
    return CANNCacheGeometry(DSACacheLayout(512,128,4,2048,block,lookahead),padded_rows*256*4)


class CANNCacheTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5319)

    def test_dense_kernel_views_share_exact_byte_addresses(self):
        g=geometry();pages=4;main_raw=torch.zeros(pages*g.main_page_bytes,dtype=torch.uint8);index_raw=torch.zeros(pages*g.index_page_bytes,dtype=torch.uint8)
        main,index,rope=view_main_index(main_raw,index_raw,pages,g)
        self.assertEqual(main.shape,(8,128,1,512));self.assertEqual(index.shape,(8,32,1,128));self.assertEqual(rope.shape,(8,128,1,0))
        for page,offset,column in [(3,255,511),(1,128,7),(2,0,3)]:
            kernel=page*2+offset//128;row=offset%128;main[kernel,row,0,column]=page+.5
            element=(page*256+offset)*512+column
            self.assertEqual(main_raw.view(torch.bfloat16)[element].item(),page+.5)
        index[7,31,0,127]=42
        self.assertEqual(index_raw.view(torch.bfloat16)[(3*64+63)*128+127].item(),42)
        self.assertEqual(main.untyped_storage().data_ptr(),rope.untyped_storage().data_ptr())

    def test_kernel_table_noncontiguous_pages_padding_and_empty(self):
        g=geometry(block=512);ids=torch.tensor([[7,2,-1],[0,-1,-1]])
        actual=expand_kernel_block_table(ids,g,num_logical_pages=8)
        expected=torch.tensor([[28,29,30,31,8,9,10,11,-1,-1,-1,-1],[0,1,2,3,-1,-1,-1,-1,-1,-1,-1,-1]],dtype=torch.int32)
        torch.testing.assert_close(actual,expected)
        self.assertEqual(expand_kernel_block_table(torch.empty(0,3,dtype=torch.int32),g,num_logical_pages=0).shape,(0,12))
        with self.assertRaises(ValueError):expand_kernel_block_table(torch.tensor([[8]]),g,num_logical_pages=8)
        with self.assertRaises(ValueError):expand_kernel_block_table(torch.tensor([[-2]]),g,num_logical_pages=8)

    def test_padded_tail_stride_planes_roundtrip_and_padding_sentinel(self):
        g=geometry();raw=torch.full((3*g.tail_page_bytes,),0x55,dtype=torch.uint8);tail=view_tail(raw,3,g)
        planes=torch.randn(2,7,128);packed=pack_tail_planes(planes);tail[2]=packed
        torch.testing.assert_close(unpack_tail_rows(tail[2]),planes,rtol=0,atol=0)
        self.assertEqual(tail.stride(),(13*256,256,1));self.assertFalse(tail.is_contiguous())
        begin=2*g.tail_page_bytes+7*256*4
        self.assertTrue(bool((raw[begin:]==0x55).all()))
        planes.zero_();self.assertFalse(torch.equal(packed,pack_tail_planes(planes)))
        restored=unpack_tail_rows(packed);restored.zero_();self.assertFalse(torch.equal(packed,pack_tail_planes(restored)))

    def test_tail_plan_matches_independent_integer_equations(self):
        g=geometry();starts=[0,3,4,5,17,100];lengths=[0,1,2,11,3,0];pages=[6,2,0,5,3,-1]
        plan=plan_tail(torch.tensor(starts),torch.tensor(lengths),torch.tensor(pages),g,num_tail_pages=8)
        history=[];writes=[];state=[]
        for i,(start,length,page) in enumerate(zip(starts,lengths,pages)):
            history.append([max(page,0)*13+((start//4*4+c)%7) for c in range(4)])
            final=start+length;remaining=final%4
            writes.append([page*13+((final-remaining+c)%7) if length and c<remaining and page>=0 else -1 for c in range(4)])
            base=(i+1+(6 if start%4+length>=4 else 0))*4
            state.append([base+c for c in range(4)])
        torch.testing.assert_close(plan.history_indices,torch.tensor(history));torch.testing.assert_close(plan.write_indices,torch.tensor(writes));torch.testing.assert_close(plan.state_indices,torch.tensor(state))
        torch.testing.assert_close(plan.start_offsets,torch.tensor([s%4 for s in starts],dtype=torch.int32))
        self.assertEqual(plan.state_table.shape,(6,4));self.assertEqual(plan.state_table.tolist(),[[i+1]+[i+7]*3 for i in range(6)])

    def test_history_gather_is_private_and_ignores_page_padding(self):
        g=geometry();raw=torch.zeros(8*g.tail_page_bytes,dtype=torch.uint8);tail=view_tail(raw,8,g)
        for page in range(8):
            for row in range(7):tail[page,row].fill_(page*100+row)
        starts=torch.tensor([3,17,100]);lengths=torch.tensor([1,3,0]);pages=torch.tensor([6,2,-1])
        plan=plan_tail(starts,lengths,pages,g,num_tail_pages=8);state=prepare_tail_state(tail,plan,4)
        self.assertEqual(state.shape,(7,4,256));self.assertEqual(state[0].abs().sum().item(),0);self.assertEqual(state[4:].abs().sum().item(),0)
        for request in range(3):
            for c in range(4):
                physical=max(int(pages[request]),0);slot=(int(starts[request])//4*4+c)%7
                self.assertEqual(state[request+1,c,0].item(),physical*100+slot)
        saved=state.clone();tail.zero_();torch.testing.assert_close(state,saved,rtol=0,atol=0)
        broken=replace(plan,page_rows=7)
        with self.assertRaises(ValueError):prepare_tail_state(tail,broken,4)
        bad_indices=plan.history_indices.clone();bad_indices[0,0]=6*13+8
        with self.assertRaises(ValueError):prepare_tail_state(tail,replace(plan,history_indices=bad_indices),4)

    def test_tail_ownership_and_invalid_input_reject(self):
        g=geometry()
        for starts,lengths,pages in [([0,0],[1,1],[2,2]),([0],[1],[-1]),([-1],[0],[2]),([0],[-1],[2]),([0],[0],[8])]:
            with self.assertRaises(ValueError):plan_tail(torch.tensor(starts),torch.tensor(lengths),torch.tensor(pages),g,num_tail_pages=8)
        with self.assertRaises(ValueError):plan_tail(torch.tensor([0.]),torch.tensor([1]),torch.tensor([2]),g,num_tail_pages=8)
        with self.assertRaises(ValueError):plan_tail(torch.tensor([0]),torch.tensor([1,2]),torch.tensor([2]),g,num_tail_pages=8)

    def test_invalid_geometry_and_backing_cannot_bind_silently(self):
        with self.assertRaises(ValueError):geometry(block=16)
        with self.assertRaises(ValueError):geometry(padded_rows=6)
        g=geometry()
        with self.assertRaises(ValueError):replace(g,tail_page_bytes=g.tail_page_bytes+4)
        with self.assertRaises(ValueError):replace(g,kernel_block_size=64)
        with self.assertRaises(ValueError):view_main_index(torch.empty(g.main_page_bytes+1024,dtype=torch.uint8),torch.empty(g.index_page_bytes,dtype=torch.uint8),1,g)
        with self.assertRaises(ValueError):view_tail(torch.zeros(g.tail_page_bytes),1,g)
        with self.assertRaises(ValueError):pack_tail_planes(torch.zeros(2,7,128,dtype=torch.bfloat16))
        with self.assertRaises(ValueError):unpack_tail_rows(torch.zeros(7,255))

    def test_empty_views_and_request_batch_skip_work(self):
        g=geometry();main,index,rope=view_main_index(torch.empty(0,dtype=torch.uint8),torch.empty(0,dtype=torch.uint8),0,g)
        self.assertEqual(main.shape,(0,128,1,512));self.assertEqual(index.shape,(0,32,1,128));self.assertEqual(rope.numel(),0)
        tail=view_tail(torch.empty(0,dtype=torch.uint8),0,g);empty=torch.empty(0,dtype=torch.int64)
        plan=plan_tail(empty,empty,empty,g,num_tail_pages=0);self.assertEqual(plan.state_table.shape,(0,1))
        self.assertEqual(prepare_tail_state(tail,plan,4).shape,(1,4,256))

if __name__=='__main__':unittest.main(verbosity=2)
