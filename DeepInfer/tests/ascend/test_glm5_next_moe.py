# SPDX-License-Identifier: Apache-2.0
"""CPU equation/dispatch tests. No expert weights, full model or NPU compute."""
import unittest
import numpy as np
import torch
from deepinfer.model_executor.models.glm5_next.moe import (
    MoERoutes,GlmMoERouter,sigmoid_noaux_routes,pack_routes,dispatch_hidden,combine_packed,
)


def numpy_routes(logits,bias,k,scale=2.5):
    z=np.asarray(logits,dtype=np.float64);b=np.asarray(bias,dtype=np.float64)
    scores=np.exp(-np.logaddexp(0.,-z));adjusted=scores+b
    ids=np.argsort(-adjusted,axis=-1,kind='stable')[:,:k]
    selected=np.take_along_axis(scores,ids,axis=-1)
    return ids,selected/selected.sum(-1,keepdims=True)*scale


class GlmMoERoutingTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(5315);torch.set_num_threads(2)

    def test_actual_dimensions_numpy_sigmoid_oracle(self):
        logits=torch.randn(7,288);bias=torch.randn(288)*.1
        routes=sigmoid_noaux_routes(logits,bias,top_k=8)
        ids,weights=numpy_routes(logits.numpy(),bias.numpy(),8)
        for token in range(7):
            got=dict(zip(routes.expert_ids[token].tolist(),routes.weights[token].tolist()))
            expected=dict(zip(ids[token].tolist(),weights[token].tolist()))
            self.assertEqual(set(got),set(expected))
            for expert in expected:self.assertAlmostEqual(got[expert],expected[expert],places=6)
        torch.testing.assert_close(routes.weights.sum(-1),torch.full((7,),2.5))
        self.assertEqual(routes.expert_ids.dtype,torch.int32)

    def test_bias_selects_but_does_not_weight(self):
        logits=torch.tensor([[0.,1.,2.,3.]]);bias=torch.tensor([10.,0.,0.,0.])
        routes=sigmoid_noaux_routes(logits,bias,top_k=2)
        self.assertEqual(set(routes.expert_ids[0].tolist()),{0,3})
        scores=logits.sigmoid()[0];mass=scores[0]+scores[3]
        for expert,weight in zip(routes.expert_ids[0],routes.weights[0]):
            torch.testing.assert_close(weight,scores[expert]/mass*2.5)

    def test_fp32_router_projection_and_input_preservation(self):
        raw=torch.randn(288,16).bfloat16();bias=torch.randn(288);x=torch.randn(3,16).bfloat16();saved=x.clone()
        module=GlmMoERouter(raw,bias)
        self.assertEqual(module.weight.dtype,torch.float32)
        expected=x.float()@raw.float().T
        torch.testing.assert_close(module.logits(x),expected,rtol=0,atol=0)
        routes=module(x);self.assertEqual(tuple(routes.weights.shape),(3,8))
        torch.testing.assert_close(x,saved,rtol=0,atol=0)

    def test_pack_counts_and_unscaled_shared_expert(self):
        logits=torch.randn(5,288);routes=sigmoid_noaux_routes(logits,torch.zeros(288),top_k=8)
        plan=pack_routes(routes,num_experts=288);x=torch.randn(5,4);saved=x.clone()
        packed=dispatch_hidden(x,plan)
        # Distinct expert functions expose any wrong pack/weight/unpack seam.
        expert_out=packed*(plan.expert_ids+1).unsqueeze(-1).float()+.125
        shared=torch.randn(5,4)
        out=combine_packed(expert_out,plan,shared_output=shared)
        expected=shared.double().numpy().copy()
        for t in range(5):
            for e,w in zip(routes.expert_ids[t].tolist(),routes.weights[t].tolist()):
                expected[t]+=w*(x[t].double().numpy()*(e+1)+.125)
        np.testing.assert_allclose(out.numpy(),expected,rtol=2e-6,atol=1e-4)
        self.assertEqual(int(plan.group_ends[-1]),40)
        counts=torch.bincount(routes.expert_ids.long().flatten(),minlength=288)
        torch.testing.assert_close(plan.group_ends,counts.cumsum(0))
        self.assertTrue(bool((counts==0).any()))
        self.assertTrue(bool((plan.expert_ids[1:]>=plan.expert_ids[:-1]).all()))
        torch.testing.assert_close(x,saved,rtol=0,atol=0)

    def test_combine_rounds_to_bf16_only_at_end(self):
        routes=MoERoutes(torch.tensor([[0,1],[0,1]],dtype=torch.int32),torch.tensor([[.8,1.7],[1.6,.9]]))
        plan=pack_routes(routes,num_experts=3);expert=torch.randn(4,7).bfloat16();shared=torch.randn(2,7).bfloat16()
        expected=torch.zeros(2,7,dtype=torch.float64)
        for row,t in enumerate(plan.token_indices.tolist()):expected[t]+=expert[row].double()*float(plan.weights[row])
        expected+=shared.double()
        torch.testing.assert_close(combine_packed(expert,plan,shared_output=shared),expected.bfloat16(),rtol=0,atol=0)

    def test_single_token_expert_contributions_follow_stable_fp32_order(self):
        routes=MoERoutes(torch.tensor([[7,0,3,1,4,6,5,2]],dtype=torch.int32),torch.ones(1,8,dtype=torch.float32))
        plan=pack_routes(routes,num_experts=8)
        expert=torch.tensor([[1e8],[-1e8],[3.],[4.],[5.],[6.],[7.],[8.]],dtype=torch.float32)
        expected=torch.zeros(1,1,dtype=torch.float32)
        for row in expert:expected=expected+row
        for _ in range(3):torch.testing.assert_close(combine_packed(expert,plan),expected,rtol=0,atol=0)

    def test_empty_routes_and_all_experts(self):
        empty=sigmoid_noaux_routes(torch.empty(0,288),torch.zeros(288),top_k=8)
        plan=pack_routes(empty,num_experts=288)
        self.assertEqual(tuple(plan.group_ends.shape),(288,));self.assertEqual(int(plan.group_ends.sum()),0)
        packed=dispatch_hidden(torch.empty(0,5),plan)
        self.assertEqual(tuple(combine_packed(packed,plan).shape),(0,5))
        routes=sigmoid_noaux_routes(torch.randn(2,8),torch.zeros(8),top_k=8)
        for row in routes.expert_ids:self.assertEqual(set(row.tolist()),set(range(8)))

    def test_ties_have_valid_unique_ids_without_order_promise(self):
        routes=sigmoid_noaux_routes(torch.zeros(3,288),torch.zeros(288),top_k=8)
        for row in routes.expert_ids:self.assertEqual(len(set(row.tolist())),8)
        torch.testing.assert_close(routes.weights,torch.full((3,8),2.5/8))

    def test_zero_mass_and_nonfinite_are_explicit_errors(self):
        with self.assertRaises(ValueError):sigmoid_noaux_routes(torch.full((1,4),-1000.),torch.zeros(4),top_k=2)
        for invalid in (float('nan'),float('inf')):
            logits=torch.zeros(1,4);logits[0,0]=invalid
            with self.assertRaises(ValueError):sigmoid_noaux_routes(logits,torch.zeros(4),top_k=2)
        with self.assertRaises(ValueError):sigmoid_noaux_routes(torch.zeros(1,4),torch.zeros(4),top_k=2,routed_scaling_factor=float('inf'))

    def test_invalid_packing_and_projection_contracts(self):
        for ids in (torch.tensor([[0,0]]),torch.tensor([[-1,1]]),torch.tensor([[0,4]])):
            with self.assertRaises(ValueError):pack_routes(MoERoutes(ids,torch.ones(1,2)),num_experts=4)
        with self.assertRaises(ValueError):sigmoid_noaux_routes(torch.zeros(1,4).bfloat16(),torch.zeros(4),top_k=2)
        with self.assertRaises(ValueError):sigmoid_noaux_routes(torch.zeros(1,4),torch.zeros(4),top_k=5)
        with self.assertRaises(ValueError):GlmMoERouter(torch.zeros(4,5),torch.zeros(3),top_k=2)

if __name__=='__main__':unittest.main(verbosity=2)
