# SPDX-License-Identifier: Apache-2.0
"""Bounded synthetic CPU text transactions; no model/INT8/NPU computation."""
from dataclasses import replace
from types import SimpleNamespace as NS
import sys,unittest
from unittest.mock import patch
import torch
from torch import nn
from test_glm5_next_rank_dsa import MLA,attend,digest
from test_glm5_next_sfa_weights import components,config
from test_glm5_next_oproj_execution import data as o_data
from test_glm5_next_text_request import Head
from test_glm5_next_hybrid_decoder import equal_state
from test_glm5_next_private_text_request import same_pages
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout,state_from_aligned_prefix,append_dsa
from deepinfer.model_executor.models.glm5_next.kda import KDAState
from deepinfer.model_executor.models.glm5_next.sfa_inputs import SFAHeadPartition,NoPESFAContract
from deepinfer.model_executor.models.glm5_next.rank_dsa import CPUProjectedDSAResult
from deepinfer.model_executor.models.glm5_next.projected_reader import CPUProjectedDSAReader
from deepinfer.model_executor.models.glm5_next.hybrid_decoder import PreparedDSAAttention,GlmHybridDecoder,PreparedDecoderStack
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry
from deepinfer.model_executor.models.glm5_next.deferred_pages import CPUPageBinding
from deepinfer.model_executor.models.glm5_next.private_request import PrivateDSARequest
from deepinfer.model_executor.models.glm5_next.text_boundary import TextBoundaryManifest,GlmTextBoundary
from deepinfer.model_executor.models.glm5_next.text_request import PreparedTextRunner


class HC(nn.Module):
    streams=4;hidden_size=4096
    def expand(self,x):return x[:,None,:].expand(-1,4,-1).clone()
    def pre(self,x):return x.float().mean(1).bfloat16(),None
    def post(self,y,x,mix):return x+y[:,None,:]


class FFN(nn.Module):
    fail=False
    def forward(self,x):
        if self.fail:raise RuntimeError('later FFN failure')
        return torch.zeros_like(x)


class KDA(nn.Module):
    def forward(self,x,state):return torch.zeros_like(x),KDAState(tuple(t.clone() for t in state.conv),state.recurrent+len(x),state.tokens_seen+len(x))


class Indexer(nn.Module):
    hidden_size=4096;q_rank=1536
    def __init__(self,layout):super().__init__();self.layout=layout
    def forward(self,x,q):return NS(query=torch.zeros(len(x),32,128,dtype=torch.bfloat16),head_weights=torch.full((len(x),32),1/32,dtype=torch.bfloat16))


class Embedding:
    def __init__(self):self.calls=0;self.weight=(torch.arange(8*4096).reshape(8,4096).float().remainder(31)*.001).bfloat16()
    def raw_rows(self,name,ids):self.calls+=1;return self.weight[ids].clone(),'BF16'


def fixture(*,source_contract=None,max_requests=2,rank_query=False):
    c=config();c.hidden_size=4096;c.vocab_size=8;c.mla_dimensions['q_lora_rank']=1536;c.hf_config['text_config'].update(hc_mult=4,rms_norm_eps=1e-5)
    c.layers=[NS(layer_id=i,attention='deepseek_sparse_attention' if i in (3,7) else 'linear_attention',mlp='dense') for i in range(8)]
    c.linear_attn_config={'num_heads':1,'head_dim':1,'short_conv_kernel_size':4}  # KDA control-flow stub only
    layout=DSACacheLayout(512,128,4,2048,128);g=CANNCacheGeometry(layout,9*256*4);layers={};states={};bindings={};producers={};readers={};calls=[];kept=[]
    contract=source_contract or NoPESFAContract.from_config(c,reference_source_sha256='a'*64)
    for i in range(8):
        if i not in (3,7):
            attention=KDA();state=KDAState(tuple(torch.zeros(3,1,dtype=torch.bfloat16) for _ in range(3)),torch.zeros(1,1,1),0)
        else:
            mla=MLA();attention=PreparedDSAAttention(mla,Indexer(layout),lambda *a:(_ for _ in ()).throw(AssertionError('default DSA used')))
            state=state_from_aligned_prefix(torch.empty(0,512,dtype=torch.bfloat16),torch.empty(0,128,dtype=torch.bfloat16),layout)
            bindings[i]=CPUPageBinding(torch.zeros(8*g.main_page_bytes,dtype=torch.uint8),torch.zeros(8*g.index_page_bytes,dtype=torch.uint8),torch.zeros(4*g.tail_page_bytes,dtype=torch.uint8),g,8,4,torch.empty(1,0,dtype=torch.int32),-1)
            def producer(step,state,layer=i):
                calls.append(('producer',layer,state.tokens_seen));n=len(step.positions);latent=(step.positions[:,None].expand(n,512).float()*.001+layer*.01).bfloat16()
                return append_dsa(latent,torch.zeros(n,128),torch.zeros(n,128),torch.zeros(4,128),state,layout,start_position=state.tokens_seen)
            producers[i]=producer;rank=0 if i==3 else 15;anchor=3 if i==3 else 43
            _,_,_,kvb=components(rank=rank,layer=anchor);oproj,_,_=o_data(rank,anchor)
            kvb=replace(kvb,manifest=replace(kvb.manifest,layer_id=i,contract=contract,tensor_name=f'model.language_model.layers.{i}.self_attn.kv_b_proj.weight'))
            oproj=replace(oproj,manifest=replace(oproj.manifest,layer_id=i,contract=contract,prefix=f'model.language_model.layers.{i}.self_attn.o_proj'))
            def operation(step,view,*,plan,expected_identity,kvb=kvb,oproj=oproj,layer=i):
                calls.append(('reader',layer,plan.candidate.start));kept.append(plan)
                selected=view.select_pools(step.indexer.query,step.indexer.head_weights)
                indices=selected.token_indices.to(torch.int32).unsqueeze(1)
                def attention(packet):calls.append(('attention',layer,plan.candidate.start));return attend(packet)
                def apply(x,w,s,*,spec):
                    calls.append(('rank_apply',layer,plan.candidate.start));assert x.shape==(plan.candidate.tokens,1024) and spec.identity==expected_identity and w.dtype==torch.int8
                    marker=(torch.arange(plan.candidate.start,plan.candidate.end).float()*.01+layer*.001).bfloat16();return marker[:,None].expand(-1,4096).clone()
                def reduce(y,**kw):calls.append(('sum_stub',layer,plan.candidate.start));return y
                return view.execute_rank_dsa_cpu((step.mla.query if rank_query else step.mla.query[:,plan.heads.head_ids]).contiguous(),indices,plan=plan,expected_identity=expected_identity,kv_b=kvb,o_proj=oproj,expected_runtime_scale_sha256=digest(oproj.weight_scale),attention=attention,rank_apply=apply,reduce_sum=reduce,quantization_policy='rank_local_dynamic_int8',bias_policy='absent',index_namespace='logical_token')
            heads=SFAHeadPartition(64,16,rank,tuple(range(rank*4,rank*4+4)));readers[i]=CPUProjectedDSAReader(i,heads,contract,operation)
            if rank_query:
                from test_glm5_next_query_payload import Reader
                from deepinfer.model_executor.models.glm5_next.query_payload import load_rank_qb_cpu
                from deepinfer.model_executor.models.glm5_next.projections import GlmMLAProjections
                rr=Reader()
                if i==7:
                    for name in list(rr.quant):
                        if '.43.' in name:
                            alias=name.replace('.43.','.7.');rr.quant[alias]=rr.quant[name];rr.desc[alias]=rr.desc[name]
                payload=load_rank_qb_cpu(rr,c,i,heads,contract)
                # Bind each layer by value, including A projection instrumentation.
                def a(width,label,layer=i):
                    class BoundA(nn.Module):
                        def forward(self,x):calls.append((label,layer,len(x)));return x[:,:width].clone()
                    return BoundA()
                def forbidden(*a,**k):raise AssertionError('full-head projection/finish used')
                local=GlmMLAProjections(hidden_size=4096,num_heads=64,qk_dim=256,value_dim=256,q_rank=1536,kv_rank=512,q_a=a(1536,'q_a'),kv_a=a(512,'kv_a'),q_b=forbidden,kv_b=forbidden,o_proj=forbidden,q_norm=torch.ones(1536,dtype=torch.bfloat16),kv_norm=torch.ones(512,dtype=torch.bfloat16))
                local.project=forbidden;local.finish=forbidden;attention.mla=local
                def backend(x,w,scale,*,spec,layer=i):
                    calls.append(('q_b_local',layer,len(x)));assert x.shape[1]==1536 and spec.rank==(0 if layer==3 else 15)
                    # Independent normalized q_a oracle: nonconstant qr, not raw A.
                    assert torch.allclose(x.float().square().mean(-1),torch.ones(len(x)),atol=.02)
                    return torch.full((len(x),1024),.01,dtype=torch.bfloat16)
                original=attention.indexer.forward
                def index(x,qr,original=original,layer=i):
                    calls.append(('indexer_full_qr',layer,len(x)));assert qr.shape==(len(x),1536);return original(x,qr)
                attention.indexer.forward=index
                readers[i]=CPUProjectedDSAReader(i,heads,contract,operation,payload,backend)
        states[i]=state;layers[i]=GlmHybridDecoder(layer_id=i,attention_kind=c.layers[i].attention,mlp_kind='dense',attn_hc=HC(),ffn_hc=HC(),input_norm=torch.ones(4096,dtype=torch.bfloat16),ffn_norm=torch.ones(4096,dtype=torch.bfloat16),attention=attention,mlp=FFN())
    stack=PreparedDecoderStack(c,layout,layers);state=MixedRequestState.capture('projected-A',c,layout,states);embedding=Embedding()
    boundary=GlmTextBoundary(embedding,TextBoundaryManifest(4096,8,4,1e-5,1.0),torch.ones(4096,dtype=torch.bfloat16),max_unique_tokens=8,max_tokens=8)
    head=Head(torch.full((4,4096),.001,dtype=torch.bfloat16),vocab_start=1,scale=1.0,max_tokens=8)
    r=PreparedTextRunner(stack,boundary,head,max_context_tokens=32,max_requests=max_requests)
    q=r.open_private_request(state,page_bindings=bindings,producers=producers,attentions=readers)
    return r,state,q,bindings,calls,kept,embedding


def plans(q,count):
    return {i:dict(private_ids=(next(p for p in range(8) if p not in table),),tail_private=next(p for p in range(4) if p!=tail),occupied_main=frozenset(table),occupied_tail=frozenset({tail}) if tail>=0 else frozenset()) for i,table,tail in q.stamp.pages}


class ProjectedPrivateTextTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5341)

    def test_full_projected_chain_publishes_one_frame_without_finish_or_layer_publish(self):
        r,s,q,bindings,calls,kept,_=fixture();before=q.page_snapshot()
        with patch.object(PrivateDSARequest,'publish',side_effect=AssertionError('layer publish')):
            out=q.step(torch.tensor([1,2,3]),expected_stamp=q.stamp,page_plans=plans(q,3))
        self.assertEqual((q.stamp.tokens_seen,q.stamp.revision,out.logits.shape),(3,1,(3,4)));self.assertEqual({v.tokens_seen for v in q.snapshot().layers.values()},{3})
        self.assertEqual([v[:2] for v in calls],[('producer',3),('reader',3),('attention',3),('rank_apply',3),('sum_stub',3),('producer',7),('reader',7),('attention',7),('rank_apply',7),('sum_stub',7)])
        self.assertTrue(all(r.stack.layers[i].attention.mla.finishes==0 for i in (3,7)));self.assertEqual([p.layer_id for p in kept],[3,7]);self.assertEqual({p.request_id for p in kept},{q.request_id});self.assertTrue(all(p.revision==0 for p in kept));self.assertEqual(before[3].stamp.tokens_seen,0)
        for b in bindings.values():self.assertFalse(bool(b.main_raw.any()));self.assertFalse(bool(b.index_raw.any()));self.assertFalse(bool(b.tail_raw.any()))

    def test_later_reader_FFN_head_failures_leave_all_layers_pages_unpublished(self):
        for failure in ('reader','ffn','head','nan'):
            r,s,q,_,calls,_,_=fixture();stamp=q.stamp;pages=q.page_snapshot();saved=q.snapshot()
            if failure=='reader':q._attentions[7]=replace(q._attentions[7],operation=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('later reader failure')))
            elif failure=='ffn':r.stack.layers[7].mlp.fail=True
            elif failure=='head':r.head.fail=True
            else:r.head.forward=lambda x:torch.full((len(x),4),float('nan'))
            with self.assertRaises((RuntimeError,ValueError)):q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=plans(q,2))
            self.assertIn(('sum_stub',3,0),calls);self.assertEqual(q.stamp,stamp);same_pages(pages,q.page_snapshot());equal_state(saved,q.snapshot())

    def test_abandoned_attempt_result_cannot_be_reused_on_same_revision_retry(self):
        r,s,q,_,calls,kept,_=fixture();reader=q._attentions[3];outputs=[]
        def remember(*args,**kwargs):value=reader.operation(*args,**kwargs);outputs.append(value);return value
        q._attentions[3]=replace(reader,operation=remember);stamp=q.stamp;r.head.fail=True
        with self.assertRaises(RuntimeError):q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=plans(q,2))
        first=kept[0].identity;r.head.fail=False;q._attentions[3]=replace(reader,operation=lambda *a,**k:outputs[0])
        with self.assertRaisesRegex(ValueError,'result identity'):q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=plans(q,2))
        self.assertEqual(q.stamp,stamp);q._attentions[3]=reader;q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=plans(q,2))
        self.assertNotEqual(first,kept[-2].identity);self.assertEqual(q.stamp.revision,1);self.assertEqual({v.tokens_seen for v in q.snapshot().layers.values()},{2})

    def test_segmented_continuation_and_other_request_match(self):
        r,s,q,b,_,_,_=fixture();other=MixedRequestState.capture('projected-B',r.stack.config,r.stack.layout,s.layers);q2=r.open_private_request(other,page_bindings=b,producers=q._producers,attentions=q._attentions)
        ids=torch.tensor([0,1,2,3,4,5,6]);whole=q.step(ids,expected_stamp=q.stamp,page_plans=plans(q,7));outputs=[];start=0
        for stop in (1,3,4,7):outputs.append(q2.step(ids[start:stop],expected_stamp=q2.stamp,page_plans=plans(q2,stop-start)).logits);start=stop
        self.assertTrue(torch.equal(torch.cat(outputs),whole.logits));equal_state(q.snapshot(),replace(q2.snapshot(),request_id=q.request_id));self.assertEqual(q.stamp.revision,1);self.assertEqual(q2.stamp.revision,4)

    def test_stale_stamp_changed_partition_and_reservation_fail_before_embedding(self):
        r,s,q,_,calls,_,embedding=fixture();stamp=q.stamp
        for bad in (replace(stamp,revision=9),replace(stamp,_owner=object()),replace(stamp,request_id='foreign')):
            with self.assertRaises(ValueError):q.step(torch.tensor([1]),expected_stamp=bad,page_plans=plans(q,1))
        reader=q._attentions[3];q._attentions[3]=replace(reader,heads=SFAHeadPartition(64,16,1,(4,5,6,7)))
        with self.assertRaisesRegex(ValueError,'specification'):q.step(torch.tensor([1]),expected_stamp=stamp,page_plans=plans(q,1))
        q._attentions[3]=reader;bad=plans(q,1);bad[7]['occupied_main']=frozenset(bad[7]['private_ids'])
        with self.assertRaises(ValueError):q.step(torch.tensor([1]),expected_stamp=stamp,page_plans=bad)
        self.assertEqual(embedding.calls,0);self.assertEqual(calls,[]);self.assertEqual(q.stamp,stamp)

    def test_reader_wrong_identity_type_shape_and_nonfinite_rejected(self):
        for kind in ('identity','type','shape','dtype','nan'):
            r,s,q,_,_,_,_=fixture();reader=q._attentions[3];stamp=q.stamp;saved=q.page_snapshot()
            def bad(step,view,*,plan,expected_identity):
                value=torch.zeros(len(step.positions),4096,dtype=torch.bfloat16)
                if kind=='identity':return CPUProjectedDSAResult(replace(plan,revision=99).identity,value)
                if kind=='type':return value
                if kind=='shape':value=value[:,:1024]
                if kind=='dtype':value=value.float()
                if kind=='nan':value.fill_(float('nan'))
                return CPUProjectedDSAResult(expected_identity,value)
            q._attentions[3]=replace(reader,operation=bad)
            with self.assertRaises(ValueError):q.step(torch.tensor([1]),expected_stamp=stamp,page_plans=plans(q,1))
            self.assertEqual(q.stamp,stamp);same_pages(saved,q.page_snapshot())

    def test_actual_view_rejects_wrong_candidate_before_execution(self):
        r,s,q,_,_,_,_=fixture();reader=q._attentions[3]
        def bad(step,view,*,plan,expected_identity):
            changed=replace(plan,candidate=replace(plan.candidate,tail_private=(plan.candidate.tail_private+1)%4))
            with self.assertRaisesRegex(ValueError,'candidate'):view.require_sfa_plan_cpu(changed,expected_identity=changed.identity)
            return reader.operation(step,view,plan=plan,expected_identity=expected_identity)
        q._attentions[3]=replace(reader,operation=bad);q.step(torch.tensor([1,2]),expected_stamp=q.stamp,page_plans=plans(q,2))

    def test_empty_skips_producers_readers_embedding_and_revision(self):
        r,s,q,_,calls,kept,embedding=fixture();stamp=q.stamp;r.head.fail=True
        out=q.step(torch.empty(0,dtype=torch.int64),expected_stamp=stamp,page_plans={})
        self.assertEqual(out.logits.shape,(0,4));self.assertEqual(q.stamp,stamp);self.assertEqual((calls,kept,embedding.calls),([],[],0))

    def test_returned_output_snapshots_and_callback_buffers_do_not_alias_commit(self):
        r,s,q,_,_,_,_=fixture();reader=q._attentions[3];kept=[]
        def remember(*a,**k):value=reader.operation(*a,**k);kept.append(value);return value
        q._attentions[3]=replace(reader,operation=remember);out=q.step(torch.tensor([1,2]),expected_stamp=q.stamp,page_plans=plans(q,2));saved=q.snapshot();pages=q.page_snapshot();snap=q.page_snapshot()
        with torch.inference_mode():kept[0].hidden.fill_(99);out.hidden.zero_();out.logits.zero_();snap[3].main_raw.zero_();snap[7].state.latent.zero_()
        equal_state(saved,q.snapshot());same_pages(pages,q.page_snapshot())

    def test_reentrant_reader_and_closed_request_preserve_ownership(self):
        r,s,q,_,_,_,_=fixture();reader=q._attentions[3];stamp=q.stamp
        q._attentions[3]=replace(reader,operation=lambda *a,**k:q.snapshot())
        with self.assertRaisesRegex(RuntimeError,'busy'):q.step(torch.tensor([1]),expected_stamp=stamp,page_plans=plans(q,1))
        self.assertEqual(q.stamp,stamp);q.close()
        with self.assertRaises(RuntimeError):q.snapshot()

    def test_constructor_reader_metadata_guards(self):
        r,s,q,b,_,_,_=fixture();readers=dict(q._attentions);producers=q._producers;q.close();readers[3]=replace(readers[3],layer_id=7)
        with self.assertRaises(ValueError):r.open_private_request(s,page_bindings=b,producers=producers,attentions=readers)
        self.assertEqual(r._requests,{})

    def test_rank_query_reaches_private_chain_without_full_heads_or_duplicate_norm(self):
        from deepinfer.model_executor.models.glm5_next import projections
        r,s,q,_,calls,_,_=fixture(rank_query=True)
        original=projections.rms_norm;norms=[]
        def norm(x,w,eps):norms.append(tuple(x.shape));return original(x,w,eps)
        with patch.object(projections,'rms_norm',side_effect=norm):
            result=q.step(torch.tensor([1,2]),expected_stamp=q.stamp,page_plans=plans(q,2))
        self.assertEqual(result.logits.shape,(2,4));self.assertEqual(norms,[(2,1536),(2,512)]*2)
        for layer in (3,7):
            self.assertEqual([v[0] for v in calls if v[1]==layer],['q_a','kv_a','q_b_local','indexer_full_qr','producer','reader','attention','rank_apply','sum_stub'])
        self.assertEqual(q.stamp.tokens_seen,2)

    def test_rank_query_failure_and_empty_leave_frame_unchanged(self):
        r,s,q,_,calls,_,_=fixture(rank_query=True);stamp=q.stamp;pages=q.page_snapshot()
        q.step(torch.empty(0,dtype=torch.int64),expected_stamp=stamp,page_plans={});self.assertEqual(calls,[])
        reader=q._attentions[7]
        q._attentions[7]=replace(reader,query_backend=lambda *a,**k:torch.full((2,1024),float('nan'),dtype=torch.bfloat16))
        with self.assertRaises(ValueError):q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=plans(q,2))
        self.assertIn(('sum_stub',3,0),calls);self.assertEqual(q.stamp,stamp);same_pages(pages,q.page_snapshot());equal_state(s,q.snapshot())

    def test_no_native_modules_imported(self):self.assertFalse(any(n in sys.modules for n in ('torch_npu','fla_npu','_C_ascend')))


if __name__=='__main__':unittest.main(verbosity=2)
