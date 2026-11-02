# SPDX-License-Identifier: Apache-2.0
"""Independent tiny mathematical oracles and explicit bounded CPU contracts."""
import unittest
import numpy as np
import torch
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next.sparse_reference import select_pools_reference,sparse_mla_reference


class SparseReferenceTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5320)

    def test_weighted_query_score_excludes_per_head_relu(self):
        layout=DSACacheLayout(3,1,4,4,128)
        query=torch.tensor([[[2.],[-3.]]]).bfloat16();weights=torch.ones(1,2).bfloat16()
        keys=torch.tensor([[1.],[2.]]).bfloat16()
        result=select_pools_reference(query,weights,keys,torch.tensor([7]),layout)
        self.assertEqual(result.pool_ids.tolist(),[[0]])
        self.assertEqual(result.pool_scores.tolist(),[[-1.]])
        self.assertEqual(result.token_indices.tolist(),[[0,1,2,3,-1,-1,-1]])
        relu_scores=[max(2*k,0)+max(-3*k,0) for k in (1,2)]
        self.assertEqual(int(np.argmax(relu_scores)),1)

    def test_pool_selection_numpy_causality_and_tail_pack(self):
        layout=DSACacheLayout(3,4,4,8,128);q=torch.randn(4,3,4).bfloat16();w=torch.randn(4,3).bfloat16();keys=torch.randn(6,4).bfloat16();positions=torch.tensor([2,7,14,23])
        result=select_pools_reference(q,w,keys,positions,layout)
        qbar=(q.float().numpy().astype(np.float64)*w.float().numpy().astype(np.float64)[...,None]).sum(1)
        scores=qbar@keys.float().numpy().astype(np.float64).T
        for row,pos in enumerate(positions.tolist()):
            n=(pos+1)//4;ids=sorted(range(n),key=lambda i:(-scores[row,i],i))[:2]
            self.assertEqual(result.pool_ids[row,:len(ids)].tolist(),ids)
            expected=[p for pool in ids for p in range(pool*4,pool*4+4)]+list(range(n*4,pos+1))
            self.assertEqual(result.token_indices[row,:len(expected)].tolist(),expected)
            self.assertTrue(bool((result.token_indices[row,len(expected):]==-1).all()))
        original=select_pools_reference(q[:2],w[:2],keys,positions[:2],layout)
        keys[2:]=float('nan')
        changed=select_pools_reference(q[:2],w[:2],keys,positions[:2],layout)
        torch.testing.assert_close(original.token_indices,changed.token_indices,rtol=0,atol=0)

    def test_cpu_ties_are_stable_and_padded_rows_do_no_work(self):
        layout=DSACacheLayout(3,4,4,8,128);q=torch.ones(3,2,4).bfloat16();w=torch.ones(3,2).bfloat16();keys=torch.zeros(4,4).bfloat16()
        q[1]=float('nan');w[1]=float('nan')
        r=select_pools_reference(q,w,keys,torch.tensor([15,999,-1]),layout,active_rows=torch.tensor([True,False,False]))
        self.assertEqual(r.pool_ids[0].tolist(),[0,1]);self.assertTrue(bool((r.token_indices[1:]==-1).all()))
        with self.assertRaises(ValueError):select_pools_reference(q[:1],w[:1],keys[:1],torch.tensor([15]),layout)
        with self.assertRaises(ValueError):select_pools_reference(q[:1],w[:1],keys,torch.tensor([15]),layout,max_score_elements=1)

    def test_sparse_attention_numpy_per_head_softmax_and_value_layout(self):
        q=torch.randn(3,2,4).bfloat16();k=torch.randn(7,2,4).bfloat16();v=torch.randn(7,2,3).bfloat16()
        ids=torch.tensor([[2,0,-1,-1],[4,1,3,-1],[-1,-1,-1,-1]],dtype=torch.int32);positions=torch.tensor([2,4,-1])
        out,lse=sparse_mla_reference(q,k,v,ids,positions)
        expected=np.zeros((3,2,3),dtype=np.float64);expected_lse=np.full((3,2),-np.inf)
        for row in (0,1):
            chosen=ids[row][ids[row]>=0].tolist()
            for head in range(2):
                logits=np.array([np.dot(q[row,head].float().numpy().astype(np.float64),k[i,head].float().numpy().astype(np.float64))*.5 for i in chosen])
                exp=np.exp(logits-logits.max());prob=exp/exp.sum()
                expected[row,head]=sum(p*v[i,head].float().numpy().astype(np.float64) for p,i in zip(prob,chosen))
                expected_lse[row,head]=logits.max()+np.log(exp.sum())
        torch.testing.assert_close(out,torch.tensor(expected).bfloat16(),rtol=0,atol=0)
        np.testing.assert_allclose(lse.numpy(),expected_lse,rtol=2e-6,atol=2e-6)
        # Unselected future cache entries are allowed to be uninitialized.
        k[5:]=float('nan');v[5:]=float('nan')
        again,_=sparse_mla_reference(q,k,v,ids,positions);torch.testing.assert_close(again,out,rtol=0,atol=0)

    def test_invalid_sparse_ids_and_budget_reject(self):
        q=torch.ones(1,2,4).bfloat16();k=torch.ones(5,2,4).bfloat16();v=torch.ones(5,2,3).bfloat16()
        for ids in ([[0,-1,1]],[[0,0,-1]],[[4,-1,-1]],[[-2,-1,-1]],[[5,-1,-1]]):
            with self.assertRaises(ValueError):sparse_mla_reference(q,k,v,torch.tensor(ids),torch.tensor([3]))
        with self.assertRaises(ValueError):sparse_mla_reference(q,k,v,torch.tensor([[0,1]]),torch.tensor([3]),max_work_elements=1)
        with self.assertRaises(ValueError):sparse_mla_reference(q.float(),k,v,torch.tensor([[0]]),torch.tensor([3]))

    def test_empty_query_and_empty_selected_context(self):
        layout=DSACacheLayout(3,4,4,8,128)
        r=select_pools_reference(torch.empty(0,2,4,dtype=torch.bfloat16),torch.empty(0,2,dtype=torch.bfloat16),torch.empty(0,4,dtype=torch.bfloat16),torch.empty(0,dtype=torch.int64),layout)
        self.assertEqual(r.token_indices.shape,(0,11))
        out,lse=sparse_mla_reference(torch.empty(0,2,4,dtype=torch.bfloat16),torch.empty(0,2,4,dtype=torch.bfloat16),torch.empty(0,2,3,dtype=torch.bfloat16),torch.empty(0,0,dtype=torch.int32),torch.empty(0,dtype=torch.int64))
        self.assertEqual(out.shape,(0,2,3));self.assertEqual(lse.shape,(0,2))

if __name__=='__main__':unittest.main(verbosity=2)
