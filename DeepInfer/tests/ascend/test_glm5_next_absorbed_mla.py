# SPDX-License-Identifier: Apache-2.0
"""Small independent NumPy oracles; no native imports or cache writes."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from deepinfer.model_executor.models.glm5_next.absorbed_mla import AbsorbedMLAContract,split_kv_b_cpu,absorbed_sparse_mla_cpu


def fixture():
    torch.manual_seed(5331);c=AbsorbedMLAContract(5,(4,1),3,4,2,4,8)
    weight=(torch.randn(10,4)*.25).bfloat16();query=(torch.randn(4,2,3)*.5).bfloat16();latent=torch.randn(7,4).bfloat16()
    indices=torch.tensor([[0,1,-1,-1],[3,1,2,-1],[6,0,4,2],[-1]*4],dtype=torch.int32);positions=torch.tensor([1,3,6,-999])
    return c,weight,query,latent,indices,positions


def np64(x):return x.float().numpy().astype(np.float64)
def rounded(x):return torch.tensor(x).bfloat16().float().numpy().astype(np.float64)


class AbsorbedMLATests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_head_mapping_and_private_orientation(self):
        c,w,*_=fixture();original=w.clone();weights=split_kv_b_cpu(w,c);blocks=w.reshape(2,5,4)
        self.assertTrue(torch.equal(weights.key_up,blocks[:,:3]));self.assertTrue(torch.equal(weights.value_up,blocks[:,3:].transpose(1,2)))
        with torch.inference_mode():weights.key_up.zero_()
        self.assertTrue(torch.equal(w,original));self.assertTrue(torch.equal(weights.value_up,blocks[:,3:].transpose(1,2)))

    def test_independent_stage_rounding_and_lse_oracle(self):
        c,w,q,latent,ids,pos=fixture();weights=split_kv_b_cpu(w,c);out=absorbed_sparse_mla_cpu(q,latent,ids,pos,weights,sink_policy='absent')
        block=np64(w).reshape(2,5,4);key=block[:,:3];value=block[:,3:].transpose(0,2,1);ql=np.zeros((4,2,4));attn=np.zeros_like(ql);lse=np.full((4,2),-np.inf)
        ql[:3]=rounded(np.einsum('thp,hpl->thl',np64(q[:3]),key))
        for row in range(3):
            selected=[int(x) for x in ids[row].tolist() if x>=0];x=np64(latent)[selected];logits=ql[row]@x.T*c.scale
            m=logits.max(axis=1,keepdims=True);exp=np.exp(logits-m);probs=exp/exp.sum(axis=1,keepdims=True)
            attn[row]=rounded(probs@x);lse[row]=(m+np.log(exp.sum(axis=1,keepdims=True)))[:,0]
        values=rounded(np.einsum('thl,hlv->thv',attn,value))
        np.testing.assert_array_equal(np64(out.query_latent),ql);np.testing.assert_array_equal(np64(out.attention_latent),attn)
        np.testing.assert_array_equal(np64(out.values),values);np.testing.assert_allclose(np64(out.lse),lse,atol=2e-7,rtol=1e-6)

    def test_original_qk_scale_survives_latent_absorption(self):
        c=AbsorbedMLAContract(1,(0,),1,2,1,4,4);w=torch.tensor([[1.,0.],[1.,1.]]).bfloat16();weights=split_kv_b_cpu(w,c)
        out=absorbed_sparse_mla_cpu(torch.ones(1,1,1,dtype=torch.bfloat16),torch.tensor([[0.,0.],[2.,0.]]).bfloat16(),torch.tensor([[0,1]]),torch.tensor([1]),weights,sink_policy='absent')
        logits=np.array([0.,2.]);p=np.exp(logits-logits.max());p/=p.sum();expected=rounded(np.array([2*p[1]]))[0]
        self.assertEqual(out.values.item(),expected);self.assertEqual(c.scale,1.)
        self.assertNotEqual(out.values.item(),rounded(np.array([2/(1+np.exp(-2/np.sqrt(2)))]))[0])

    def test_3d_ranked_indices_and_permutation_preserve_selected_set(self):
        c,w,q,l,ids,p=fixture();weights=split_kv_b_cpu(w,c);a=absorbed_sparse_mla_cpu(q,l,ids,p,weights,sink_policy='absent')
        b=absorbed_sparse_mla_cpu(q,l,ids[:,None],p,weights,sink_policy='absent')
        self.assertTrue(torch.equal(a.values,b.values));reordered=ids.clone();reordered[2]=ids[2].flip(0)
        other=absorbed_sparse_mla_cpu(q,l,reordered,p,weights,sink_policy='absent');torch.testing.assert_close(a.values,other.values,atol=0,rtol=0)

    def test_empty_padding_and_unselected_nonfinite_are_ignored(self):
        c,w,q,l,ids,p=fixture();q[-1]=float('nan');l[5]=float('nan');weights=split_kv_b_cpu(w,c)
        out=absorbed_sparse_mla_cpu(q,l,ids,p,weights,sink_policy='absent');self.assertTrue(bool((out.values[-1]==0).all()));self.assertTrue(bool(torch.isneginf(out.lse[-1]).all()))
        empty=absorbed_sparse_mla_cpu(q[:0],l[:0],ids[:0],p[:0],weights,sink_policy='absent');self.assertEqual(empty.values.shape,(0,2,2))

    def test_invalid_selected_ids_and_live_nonfinite_reject(self):
        for kind in ('duplicate','future','hole','negative','nonfinite_query','nonfinite_latent'):
            c,w,q,l,ids,p=fixture();weights=split_kv_b_cpu(w,c)
            if kind=='duplicate':ids[0,1]=0
            elif kind=='future':ids[0,1]=2
            elif kind=='hole':ids[0]=torch.tensor([0,-1,1,-1])
            elif kind=='negative':ids[0,0]=-2
            elif kind=='nonfinite_query':q[0,0,0]=float('inf')
            else:l[0,0]=float('nan')
            with self.subTest(kind=kind),self.assertRaises(ValueError):absorbed_sparse_mla_cpu(q,l,ids,p,weights,sink_policy='absent')

    def test_native_and_expanded_kv_or_unresolved_sink_reject(self):
        c,w,q,l,ids,p=fixture();weights=split_kv_b_cpu(w,c)
        for kind in ('meta','expanded','sink'):
            query=q;latent=l;policy='absent'
            if kind=='meta':query=torch.empty(q.shape,dtype=q.dtype,device='meta')
            elif kind=='expanded':latent=l[:,None]
            else:policy='SMLA_default_1'
            with self.assertRaises(ValueError):absorbed_sparse_mla_cpu(query,latent,ids,p,weights,sink_policy=policy)

    def test_budgets_reject_before_output_allocations(self):
        c,w,q,l,ids,p=fixture()
        with patch.object(torch.Tensor,'clone',side_effect=AssertionError('cloned before budget')):
            with self.assertRaisesRegex(ValueError,'byte budget'):split_kv_b_cpu(w,c,max_weight_bytes=1)
        weights=split_kv_b_cpu(w,c)
        with patch('torch.zeros',side_effect=AssertionError('allocated before budget')):
            with self.assertRaisesRegex(ValueError,'work exceeds'):absorbed_sparse_mla_cpu(q,l,ids,p,weights,sink_policy='absent',max_work_elements=1)

    def test_explicit_head_identity_and_dimension_guards(self):
        for ids in ((1,1),(5,),(),tuple(range(5))):
            with self.assertRaises(ValueError):AbsorbedMLAContract(5,ids,3,4,2,4,8)
        with self.assertRaises(ValueError):AbsorbedMLAContract(5,(1,),0,4,2)


if __name__=='__main__':unittest.main(verbosity=2)
