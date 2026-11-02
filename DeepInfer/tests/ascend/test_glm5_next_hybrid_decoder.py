# SPDX-License-Identifier: Apache-2.0
"""Small injected hybrid composition; no model-weight/INT8 or NPU execution."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from deepinfer.hugging_face.modelslim910c import FloatLinear
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,state_from_aligned_prefix
from deepinfer.model_executor.models.glm5_next.kda import KDAState
from deepinfer.model_executor.models.glm5_next.mhc import GlmMHC
from deepinfer.model_executor.models.glm5_next.moe import GlmMoERouter
from deepinfer.model_executor.models.glm5_next.expert_bank import SelectedExpertBank
from deepinfer.model_executor.models.glm5_next.projections import GlmMLAProjections,GlmIndexerProjections
from deepinfer.model_executor.models.glm5_next.hybrid_decoder import GlmMoEMLP,PreparedDSAAttention,GlmHybridDecoder,PreparedDecoderStack
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState
from deepinfer.model_executor.models.glm5_next.sparse_reference import select_pools_reference,sparse_mla_reference


class FFN(nn.Module):
    def forward(self,x):return (x.float()*.25).bfloat16()


class Bank(nn.Module):
    count=5
    def __init__(self):super().__init__();self.calls=0;self.fail=False
    def forward_grouped(self,x,groups,*,group_list_type):
        self.calls+=1
        if self.fail:raise RuntimeError('injected FFN failure')
        out=torch.empty_like(x);begin=0
        for expert,end in enumerate(groups.tolist()):out[begin:end]=(x[begin:end].float()*(expert+1)*.125).bfloat16();begin=end
        return out


class Shared(nn.Module):
    def __init__(self):super().__init__();self.calls=0
    def forward(self,x):self.calls+=1;return (x.float()*.125).bfloat16()


class KDAStub(nn.Module):
    """Causal stateful test double, not KDA arithmetic evidence."""
    def forward(self,x,state):
        recurrent=state.recurrent.clone();out=torch.empty_like(x)
        for row in range(len(x)):
            recurrent[0,0,0]+=x[row,0].float()
            out[row]=(x[row].float()*.25+recurrent[0,0,0]*.001).bfloat16()
        return out,KDAState(tuple(t.clone() for t in state.conv),recurrent,state.tokens_seen+len(x))


class ReferenceDSAOp(nn.Module):
    def __init__(self,mla,indexer):super().__init__();self.mla,self.indexer=mla,indexer;self.calls=0;self.mutate_input=False;self.bad_count=False
    def forward(self,step,state):
        self.calls+=1
        if self.mutate_input:state.tail.fill_(99)
        next_state=self.indexer.append_cache(step.mla.latent,step.indexer,state,start_position=state.tokens_seen)
        selected=select_pools_reference(step.indexer.query,step.indexer.head_weights,next_state.compressed,step.positions,self.indexer.layout)
        k,v=self.mla.expand_kv(next_state.latent)
        values,_=sparse_mla_reference(step.mla.query,k,v,selected.token_indices,step.positions,scale=step.scale)
        if self.bad_count:next_state=replace(next_state,tokens_seen=next_state.tokens_seen-1)
        return values,next_state


def hc():return GlmMHC(torch.randn(8,12)*.02,torch.zeros(8),torch.ones(3),streams=2,hidden_size=6,sinkhorn_iters=5)

def dsa():
    layout=DSACacheLayout(3,4,4,8,128)
    modules=[FloatLinear((torch.randn(*s)*.125).bfloat16()) for s in ((5,6),(3,6),(8,5),(14,3),(6,6))]
    mla=GlmMLAProjections(hidden_size=6,num_heads=2,qk_dim=4,value_dim=3,q_rank=5,kv_rank=3,q_a=modules[0],kv_a=modules[1],q_b=modules[2],kv_b=modules[3],o_proj=modules[4],q_norm=torch.ones(5).bfloat16(),kv_norm=torch.ones(3).bfloat16())
    indexer=GlmIndexerProjections(wq_b=FloatLinear((torch.randn(8,5)*.125).bfloat16()),key_weight=(torch.randn(4,6)*.125).bfloat16(),head_weight=(torch.randn(2,6)*.125).bfloat16(),gate_weight=(torch.randn(4,6)*.125).bfloat16(),norm_weight=torch.ones(4),norm_bias=torch.zeros(4),ape=torch.randn(4,4)*.01,hidden_size=6,q_rank=5,num_heads=2,layout=layout)
    op=ReferenceDSAOp(mla,indexer)
    return PreparedDSAAttention(mla,indexer,op),op


def empty_dsa(layout):return state_from_aligned_prefix(torch.empty(0,3,dtype=torch.bfloat16),torch.empty(0,4,dtype=torch.bfloat16),layout)
def empty_kda():return KDAState(tuple(torch.zeros(3,6,dtype=torch.bfloat16) for _ in range(3)),torch.zeros(2,3,3),0)


def setup_stack():
    attention,op=dsa();bank,shared=Bank(),Shared()
    router=GlmMoERouter((torch.randn(5,6)*.1).bfloat16(),torch.zeros(5),top_k=2)
    experts=SelectedExpertBank(range(5),bank,num_experts=5,shared=shared)
    moe=GlmMoEMLP(router,experts,hidden_size=6,num_experts=5)
    config=SimpleNamespace(layers=[SimpleNamespace(layer_id=0,attention='linear_attention',mlp='dense'),SimpleNamespace(layer_id=1,attention='deepseek_sparse_attention',mlp='sparse')],linear_attn_config={'num_heads':2,'head_dim':3,'short_conv_kernel_size':4})
    layers={}
    for i,(kind,mlp_kind,attn,ffn) in enumerate([('linear_attention','dense',KDAStub(),FFN()),('deepseek_sparse_attention','sparse',attention,moe)]):
        layers[i]=GlmHybridDecoder(layer_id=i,attention_kind=kind,mlp_kind=mlp_kind,attn_hc=hc(),ffn_hc=hc(),input_norm=torch.ones(6).bfloat16(),ffn_norm=torch.ones(6).bfloat16(),attention=attn,mlp=ffn)
    stack=PreparedDecoderStack(config,attention.layout,layers)
    state=MixedRequestState.capture('req-A',config,attention.layout,{0:empty_kda(),1:empty_dsa(attention.layout)})
    return stack,state,op,bank,shared


def equal_state(a,b):
    assert a.tokens_seen==b.tokens_seen and a.request_id==b.request_id
    for layer,s in a.layers.items():
        t=b.layers[layer];assert s.tokens_seen==t.tokens_seen
        names=('recurrent',) if isinstance(s,KDAState) else ('latent','compressed','tail','tail_positions')
        for name in names:torch.testing.assert_close(getattr(s,name),getattr(t,name),rtol=0,atol=0)
        if isinstance(s,KDAState):
            for x,y in zip(s.conv,t.conv):torch.testing.assert_close(x,y,rtol=0,atol=0)


class HybridDecoderTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5321)

    def test_two_layer_segmented_continuation_and_request_isolation(self):
        stack,state,op,bank,shared=setup_stack();x=torch.randn(7,6).bfloat16();saved=x.clone();whole,final=stack(x,state)
        partial=state;outputs=[];start=0
        for end in (1,3,4,7):
            out,partial=stack(x[start:end],partial);outputs.append(out);start=end
        torch.testing.assert_close(torch.cat(outputs),whole,rtol=0,atol=0);equal_state(partial,final)
        self.assertEqual((final.tokens_seen,final.layers[0].tokens_seen,final.layers[1].tokens_seen),(7,7,7))
        self.assertEqual((op.calls,bank.calls,shared.calls),(5,5,5));torch.testing.assert_close(x,saved,rtol=0,atol=0)
        self.assertEqual(state.tokens_seen,0);self.assertEqual(state.layers[1].latent.numel(),0)
        other=MixedRequestState.capture('req-B',stack.config,stack.layout,state.layers);out,other_final=stack(x,other)
        torch.testing.assert_close(out,whole,rtol=0,atol=0);self.assertEqual(other_final.request_id,'req-B')

    def test_failed_second_ffn_does_not_publish_mutated_attention_state(self):
        stack,state,op,bank,_=setup_stack();x=torch.randn(4,6).bfloat16();saved=MixedRequestState.capture(state.request_id,stack.config,stack.layout,state.layers);hidden=x.clone()
        op.mutate_input=True;bank.fail=True
        with self.assertRaisesRegex(RuntimeError,'injected FFN failure'):stack(x,state)
        equal_state(state,saved);torch.testing.assert_close(x,hidden,rtol=0,atol=0)
        self.assertEqual(state.tokens_seen,0)

    def test_bad_attention_counter_and_wrong_kind_reject(self):
        stack,state,op,_,_=setup_stack();op.bad_count=True
        with self.assertRaises(ValueError):stack(torch.randn(4,6).bfloat16(),state)
        self.assertEqual(state.tokens_seen,0)
        with self.assertRaises(ValueError):stack.layers[1](torch.randn(2,2,6).bfloat16(),empty_kda())
        with self.assertRaises(ValueError):stack.layers[0](torch.randn(2,6),state.layers[0])

    def test_empty_stack_step_skips_dsa_router_experts_and_shared(self):
        stack,state,op,bank,shared=setup_stack();out,next_state=stack(torch.empty(0,6,dtype=torch.bfloat16),state)
        self.assertEqual(out.shape,(0,2,6));equal_state(next_state,state);self.assertEqual((op.calls,bank.calls,shared.calls),(0,0,0))

    def test_moe_routes_once_and_namespaces_reject(self):
        stack,_,_,bank,shared=setup_stack();moe=stack.layers[1].mlp;calls=[]
        handle=moe.router.register_forward_hook(lambda *args:calls.append(1))
        out=moe(torch.randn(3,6).bfloat16());handle.remove()
        self.assertEqual(out.shape,(3,6));self.assertEqual((len(calls),bank.calls,shared.calls),(1,1,1))
        with self.assertRaises(ValueError):GlmMoEMLP(moe.router,moe.experts,hidden_size=6,num_experts=4)
        with self.assertRaises(ValueError):moe(torch.randn(3,6))

    def test_explicit_components_and_complete_stack_required(self):
        stack,state,_,_,_=setup_stack()
        with self.assertRaises(ValueError):PreparedDSAAttention(stack.layers[1].attention.mla,stack.layers[1].attention.indexer,None)
        with self.assertRaises(ValueError):PreparedDecoderStack(stack.config,stack.layout,{0:stack.layers[0]})
        with self.assertRaises(ValueError):stack(torch.randn(1,6).bfloat16(),replace(state,tokens_seen=3))
        class NoReader:
            def __getattr__(self,name):raise AssertionError('implicit checkpoint factory accessed reader')
        with self.assertRaisesRegex(ValueError,'explicit attention/FFN'):GlmHybridDecoder.from_checkpoint(NoReader(),3,'cpu')

if __name__=='__main__':unittest.main(verbosity=2)
