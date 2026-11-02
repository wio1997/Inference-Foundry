# SPDX-License-Identifier: Apache-2.0
"""CPU-only compact expert namespace and grouped/serial seam checks."""
from dataclasses import replace
import unittest
import numpy as np
import torch
from torch import nn
from deepinfer.model_executor.models.glm5_next.moe import MoERoutes,pack_routes
from deepinfer.model_executor.models.glm5_next.expert_bank import (
    compact_expert_plan,execute_compact,SelectedExpertBank,load_selected_expert_bank,
)


def sample_routes():
    return MoERoutes(torch.tensor([[287,3],[9,287],[3,9]],dtype=torch.int32),
                     torch.tensor([[.7,1.8],[1.2,1.3],[.4,2.1]]))


class ToyGroupedFFN(nn.Module):
    def __init__(self,ids,hidden=4,intermediate=5):
        super().__init__();self.ids=ids;self.count=len(ids);self.calls=0;self.seen_groups=None
        g=torch.Generator().manual_seed(5316)
        for name,shape in [('gate',(len(ids),intermediate,hidden)),('up',(len(ids),intermediate,hidden)),('down',(len(ids),hidden,intermediate))]:
            self.register_buffer(name,torch.randn(shape,generator=g)*.25)

    def forward_grouped(self,x,groups,*,group_list_type):
        assert group_list_type==0;self.calls+=1;self.seen_groups=groups.clone()
        assert len(groups)==self.count
        output=torch.empty_like(x);start=0
        for i,end in enumerate(groups.tolist()):
            gate=torch.nn.functional.linear(x[start:end],self.gate[i]).clamp_max(10.)
            up=torch.nn.functional.linear(x[start:end],self.up[i]).clamp(-10.,10.)
            activated=torch.nn.functional.silu(gate)*up
            output[start:end]=torch.nn.functional.linear(activated,self.down[i]);start=end
        assert start==len(x)
        return output


class Shared(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x):self.calls+=1;return x*.125


class CompactExpertBankTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(5316);torch.set_num_threads(2)

    def test_sparse_global_ids_to_compact_local_ids(self):
        packed=pack_routes(sample_routes(),num_experts=288);compact=compact_expert_plan(packed)
        torch.testing.assert_close(compact.global_expert_ids,torch.tensor([3,9,287]))
        torch.testing.assert_close(compact.local_expert_ids,torch.tensor([0,0,1,1,2,2]))
        torch.testing.assert_close(compact.group_ends,torch.tensor([2,4,6]))
        torch.testing.assert_close(compact.global_expert_ids[compact.local_expert_ids],packed.expert_ids)

    def test_fixed_bank_keeps_zero_token_experts(self):
        packed=pack_routes(sample_routes(),num_experts=288)
        compact=compact_expert_plan(packed,[0,3,7,9,200,287])
        torch.testing.assert_close(compact.group_ends,torch.tensor([0,2,2,4,4,6]))
        torch.testing.assert_close(compact.local_expert_ids,torch.tensor([1,1,3,3,5,5]))

    def test_grouped_swiglu_and_shared_match_independent_serial_numpy(self):
        ids=[0,3,7,9,200,287];bank=ToyGroupedFFN(ids);shared=Shared()
        wrapper=SelectedExpertBank(ids,bank,num_experts=288,shared=shared)
        routes=sample_routes();x=torch.randn(3,4);saved=x.clone();out=wrapper(x,routes)
        expected=x.numpy().astype(np.float64)*.125
        for t in range(3):
            for expert,weight in zip(routes.expert_ids[t].tolist(),routes.weights[t].tolist()):
                i=ids.index(expert);a=x[t].numpy().astype(np.float64)
                gate=np.minimum(bank.gate[i].numpy().astype(np.float64)@a,10.)
                up=np.clip(bank.up[i].numpy().astype(np.float64)@a,-10.,10.)
                activated=gate/(1+np.exp(-gate))*up
                expected[t]+=weight*(bank.down[i].numpy().astype(np.float64)@activated)
        np.testing.assert_allclose(out.numpy(),expected,atol=2e-7,rtol=2e-5)
        self.assertEqual((bank.calls,shared.calls),(1,1))
        torch.testing.assert_close(bank.seen_groups,torch.tensor([0,2,2,4,4,6]))
        torch.testing.assert_close(x,saved,rtol=0,atol=0)

    def test_route_order_within_token_does_not_change_result(self):
        routes=sample_routes();bank=ToyGroupedFFN([3,9,287]);wrapper=SelectedExpertBank([3,9,287],bank,num_experts=288)
        x=torch.randn(3,4);a=wrapper(x,routes)
        b=wrapper(x,MoERoutes(routes.expert_ids.flip(-1),routes.weights.flip(-1)))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_empty_requests_do_not_call_bank_or_shared(self):
        routes=MoERoutes(torch.empty(0,8,dtype=torch.int32),torch.empty(0,8))
        plan=compact_expert_plan(pack_routes(routes,num_experts=288));self.assertEqual(plan.group_ends.numel(),0)
        self.assertEqual(tuple(execute_compact(torch.empty(0,4),plan,None).shape),(0,4))
        for ids,bank in [([],None),([0,287],ToyGroupedFFN([0,287]))]:
            shared=Shared();wrapper=SelectedExpertBank(ids,bank,num_experts=288,shared=shared)
            self.assertEqual(tuple(wrapper(torch.empty(0,4),routes).shape),(0,4));self.assertEqual(shared.calls,0)
            if bank is not None:self.assertEqual(bank.calls,0)

    def test_missing_active_expert_and_invalid_bank_order_reject(self):
        packed=pack_routes(sample_routes(),num_experts=288)
        for ids in ([3,9],[287,9,3],[3,3,9,287],[-1,3,9,287],[3,9,288],[3.,9.,287.]):
            with self.assertRaises(ValueError):compact_expert_plan(packed,ids)
        with self.assertRaises(ValueError):SelectedExpertBank([3,9,287],ToyGroupedFFN([3,9]),num_experts=288)

    def test_bad_global_counts_rows_or_output_reject(self):
        packed=pack_routes(sample_routes(),num_experts=288)
        for wrong in (replace(packed,group_ends=packed.group_ends+1),replace(packed,expert_ids=packed.expert_ids.flip(0))):
            with self.assertRaises(ValueError):compact_expert_plan(wrong)
        compact=compact_expert_plan(packed)
        with self.assertRaises(ValueError):execute_compact(torch.zeros(3,4),compact,lambda x,g,**kw:torch.zeros(len(x),5))
        wrapper=SelectedExpertBank([3,9,287],ToyGroupedFFN([3,9,287]),num_experts=288)
        with self.assertRaises(ValueError):wrapper(torch.zeros(2,4),sample_routes())

    def test_native_factory_cpu_guard_precedes_reader_access(self):
        class NoReaderAccess:
            def __getattr__(self,name):raise AssertionError('native factory touched reader before CPU rejection')
        with self.assertRaisesRegex(ValueError,'no CPU dequantization fallback'):
            load_selected_expert_bank(NoReaderAccess(),3,[3,9,287],'cpu')

if __name__=='__main__':unittest.main(verbosity=2)
