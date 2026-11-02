# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU mechanism tests; run on target, no NPU allocations."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from deepinfer.model_executor.models.glm5_next.kda import causal_conv, recurrent_kda


class KDAContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(53)
        torch.set_num_threads(2)

    def test_conv_orientation_silu_and_history(self):
        x=torch.arange(21,dtype=torch.float32).reshape(7,3)/10
        w=torch.tensor([[1.,2.,3.,4.],[4.,3.,2.,1.],[-1.,2.,-3.,4.]])
        h=torch.zeros(3,3)
        expected=F.silu(F.conv1d(F.pad(x.T.unsqueeze(0),(3,0)),w.unsqueeze(1),groups=3)).squeeze(0).T
        actual,tail=causal_conv(x,w,h)
        torch.testing.assert_close(actual,expected)
        first,state=causal_conv(x[:2],w,h)
        second,final=causal_conv(x[2:],w,state)
        torch.testing.assert_close(torch.cat((first,second)),expected)
        torch.testing.assert_close(final,x[-3:])
        self.assertTrue(torch.equal(h,torch.zeros_like(h)))

    def test_numpy_oracle_non_square_vk_and_nonzero_initial_state(self):
        # Independent NumPy row-vector transition: S' = S D (I-beta kk^T)+beta vk^T.
        t,h,kdim,vdim=5,3,7,5
        q,k,g=[torch.randn(t,h,kdim) for _ in range(3)]
        v=torch.randn(t,h,vdim); beta=torch.randn(t,h)
        initial=torch.randn(h,vdim,kdim); a=torch.randn(h); bias=torch.randn(h*kdim)
        actual,final=recurrent_kda(q,k,v,g,beta,initial,a,bias)
        qn=q.numpy().astype(np.float64);kn=k.numpy().astype(np.float64)
        qn/=np.sqrt((qn*qn).sum(-1,keepdims=True)+1e-6)*np.sqrt(kdim)
        kn/=np.sqrt((kn*kn).sum(-1,keepdims=True)+1e-6)
        gate=-5/(1+np.exp(-np.exp(a.numpy().astype(np.float64))[None,:,None]*(g.numpy()+bias.numpy().reshape(h,kdim))))
        bn=1/(1+np.exp(-beta.numpy())); s=initial.numpy().astype(np.float64); outs=[]
        for i in range(t):
            for head in range(h):
                key=kn[i,head];b=bn[i,head]
                transition=np.diag(np.exp(gate[i,head]))@(np.eye(kdim)-b*np.outer(key,key))
                s[head]=s[head]@transition+b*np.outer(v[i,head].numpy(),key)
            outs.append(np.einsum('hvk,hk->hv',s,qn[i]))
        np.testing.assert_allclose(actual.numpy(),np.stack(outs),atol=2e-6,rtol=2e-5)
        np.testing.assert_allclose(final.numpy(),s,atol=2e-6,rtol=2e-5)

    def test_real_glm_dimensions_segmented_continuation_and_input_immutability(self):
        t,h,d=11,64,128
        q,k,v,g=[torch.randn(t,h,d).bfloat16() for _ in range(4)]
        b=torch.randn(t,h).bfloat16();a=torch.randn(h);bias=torch.randn(h*d)
        initial=torch.randn(h,d,d)*0.01; saved=initial.clone()
        full,final=recurrent_kda(q,k,v,g,b,initial,a,bias)
        state=initial;outputs=[]
        for first,last in [(0,3),(3,10),(10,11)]:
            output,state=recurrent_kda(q[first:last],k[first:last],v[first:last],g[first:last],b[first:last],state,a,bias)
            outputs.append(output)
        torch.testing.assert_close(torch.cat(outputs),full,rtol=0,atol=0)
        torch.testing.assert_close(state,final,rtol=0,atol=0)
        torch.testing.assert_close(initial,saved,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(full).all())

    def test_empty_and_zero_qk(self):
        q=torch.zeros(2,2,3);v=torch.ones(2,2,4);s=torch.ones(2,4,3)
        out,state=recurrent_kda(q,q,v,q,torch.zeros(2,2),s,torch.zeros(2),torch.zeros(6))
        torch.testing.assert_close(out,torch.zeros_like(out))
        torch.testing.assert_close(state,s*np.exp(-5),atol=1e-7,rtol=1e-6)
        out,state=recurrent_kda(q[:0],q[:0],v[:0],q[:0],torch.zeros(0,2),s,torch.zeros(2),torch.zeros(6))
        self.assertEqual(tuple(out.shape),(0,2,4));torch.testing.assert_close(state,s)

    def test_bad_gate_and_state_rejected(self):
        q=torch.ones(1,2,3);b=torch.zeros(1,2);a=torch.zeros(2);bias=torch.zeros(6);s=torch.zeros(2,3,3)
        for lower in (0,1,-6,float('nan')):
            with self.assertRaises(ValueError):recurrent_kda(q,q,q,q,b,s,a,bias,lower_bound=lower)
        with self.assertRaises(ValueError):recurrent_kda(q,q,q,q,b,s.bfloat16(),a,bias)

if __name__=='__main__':unittest.main(verbosity=2)
