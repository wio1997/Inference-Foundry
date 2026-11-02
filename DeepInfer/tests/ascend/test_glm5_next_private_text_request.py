# SPDX-License-Identifier: Apache-2.0
"""Tiny injected KDA/DSA/DSA text stack; no actual model/NPU simulation."""
from dataclasses import replace
import copy
import unittest
from unittest.mock import patch
import torch
from test_glm5_next_text_request import fixture as text_fixture
from test_glm5_next_hybrid_decoder import dsa,hc,FFN,empty_dsa,equal_state
from deepinfer.model_executor.models.glm5_next.hybrid_decoder import GlmHybridDecoder,PreparedDecoderStack
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry
from deepinfer.model_executor.models.glm5_next.deferred_pages import CPUPageBinding
from deepinfer.model_executor.models.glm5_next.private_request import PrivateDSARequest
from deepinfer.model_executor.models.glm5_next.sparse_reference import sparse_mla_reference


def fixture(max_requests=2):
    runner,initial,*_=text_fixture(max_requests=max_requests,max_context_tokens=32)
    c=runner.stack.config;c.layers.append(type(c.layers[0])(layer_id=2,attention='deepseek_sparse_attention',mlp='dense'))
    attention,_=dsa();last=GlmHybridDecoder(layer_id=2,attention_kind='deepseek_sparse_attention',mlp_kind='dense',attn_hc=hc(),ffn_hc=hc(),input_norm=torch.ones(6).bfloat16(),ffn_norm=torch.ones(6).bfloat16(),attention=attention,mlp=FFN())
    runner.stack=PreparedDecoderStack(c,runner.stack.layout,{0:runner.stack.layers[0],1:runner.stack.layers[1],2:last})
    state=MixedRequestState.capture(initial.request_id,c,runner.stack.layout,{**initial.layers,2:empty_dsa(runner.stack.layout)})
    geometry=CANNCacheGeometry(runner.stack.layout,7*8*4);bindings={};producers={};attentions={};calls=[]
    for layer in (1,2):
        b=CPUPageBinding(torch.zeros(8*geometry.main_page_bytes,dtype=torch.uint8),torch.zeros(8*geometry.index_page_bytes,dtype=torch.uint8),torch.zeros(4*geometry.tail_page_bytes,dtype=torch.uint8),geometry,8,4,torch.empty((1,0),dtype=torch.int32),-1);bindings[layer]=b
        module=runner.stack.layers[layer].attention
        def producer(step,state,module=module,layer=layer):
            calls.append(('producer',layer,state.tokens_seen));return module.indexer.append_cache(step.mla.latent,step.indexer,state,start_position=state.tokens_seen)
        def attention(step,view,module=module,layer=layer):
            calls.append(('attention',layer,view.tokens_seen));latent=view.gather(torch.arange(view.tokens_seen));k,v=module.mla.expand_kv(latent)
            ids=torch.arange(view.tokens_seen).expand(len(step.positions),-1).clone()
            ids[ids>step.positions[:,None]]=-1
            return sparse_mla_reference(step.mla.query,k,v,ids,step.positions,scale=step.scale)[0]
        producers[layer]=producer;attentions[layer]=attention
    q=runner.open_private_request(state,page_bindings=bindings,producers=producers,attentions=attentions)
    return runner,state,q,bindings,calls


def plans(q,count):
    out={}
    for layer,table,tail in q.stamp.pages:
        occupied=frozenset(table);page=next(i for i in range(8) if i not in occupied)
        target=next(i for i in range(4) if i!=tail)
        out[layer]=dict(private_ids=(page,),tail_private=target,occupied_main=occupied,occupied_tail=frozenset({tail}) if tail>=0 else frozenset())
    return out


def same_pages(a,b):
    assert set(a)==set(b)
    for i in a:
        x,y=a[i],b[i];assert x.stamp==y.stamp
        for n in ('main_raw','index_raw','tail_raw'):assert torch.equal(getattr(x,n),getattr(y,n))
        for n in ('latent','compressed','tail','tail_positions'):assert torch.equal(getattr(x.state,n),getattr(y.state,n))


class PrivateTextTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5334)

    def test_complete_stack_publishes_one_frame_without_layer_publish(self):
        r,s,q,raw,calls=fixture();stamp=q.stamp;before=q.page_snapshot()
        with patch.object(PrivateDSARequest,'publish',side_effect=AssertionError('sequential layer publication forbidden')):
            out=q.step(torch.tensor([1,2,3]),expected_stamp=stamp,page_plans=plans(q,3))
        self.assertEqual((q.stamp.tokens_seen,q.stamp.revision,out.logits.shape),(3,1,(3,4)))
        self.assertEqual([x[:2] for x in calls],[('producer',1),('attention',1),('producer',2),('attention',2)])
        after=q.page_snapshot();self.assertEqual({f.state.tokens_seen for f in after.values()},{3});self.assertEqual(q.snapshot().layers[0].tokens_seen,3)
        for b in raw.values():self.assertFalse(bool(b.main_raw.any()));self.assertFalse(bool(b.index_raw.any()));self.assertFalse(bool(b.tail_raw.any()))
        self.assertEqual(q.stamp.pages,tuple((i,f.stamp.logical_table,f.stamp.tail_page_id) for i,f in sorted(after.items())))
        self.assertEqual(before[1].state.tokens_seen,0)

    def test_later_layer_failure_keeps_earlier_candidate_and_kda_unpublished(self):
        r,s,q,_,calls=fixture();stamp=q.stamp;pages=q.page_snapshot();saved=q.snapshot()
        def fail(step,view):raise RuntimeError('second DSA attention failure')
        q._attentions[2]=fail
        with self.assertRaisesRegex(RuntimeError,'second DSA'):q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=plans(q,2))
        self.assertIn(('attention',1,2),calls);self.assertEqual(q.stamp,stamp);same_pages(pages,q.page_snapshot());equal_state(saved,q.snapshot())

    def test_last_ffn_head_and_bad_logits_leave_every_frame_stable(self):
        for failure in ('ffn','head','nonfinite'):
            r,s,q,_,_=fixture();stamp=q.stamp;pages=q.page_snapshot();saved=q.snapshot()
            if failure=='ffn':
                def fail(x):raise RuntimeError('last FFN failure')
                r.stack.layers[2].mlp.forward=fail
            elif failure=='head':r.head.fail=True
            else:r.head.forward=lambda x:torch.full((len(x),4),float('nan'))
            with self.assertRaises((RuntimeError,ValueError)):q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=plans(q,2))
            self.assertEqual(q.stamp,stamp);same_pages(pages,q.page_snapshot());equal_state(saved,q.snapshot())

    def test_returned_mixed_state_must_match_prepared_dsa_before_head(self):
        for failure in ('cache','request','count'):
            r,s,q,_,_=fixture();stamp=q.stamp;pages=q.page_snapshot();saved=q.snapshot();original=r.stack.forward
            def corrupt(*args,**kwargs):
                streams,next_state=original(*args,**kwargs)
                if failure=='cache':
                    next_state.layers[1].latent[0,0]+=1
                elif failure=='request':next_state=replace(next_state,request_id='wrong')
                else:next_state=replace(next_state,tokens_seen=99)
                return streams,next_state
            r.stack.forward=corrupt
            with patch.object(r.head,'forward',side_effect=AssertionError('head before state validation')):
                with self.assertRaises(ValueError):q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=plans(q,2))
            self.assertEqual(q.stamp,stamp);same_pages(pages,q.page_snapshot());equal_state(saved,q.snapshot())

    def test_step_scoped_operations_preserve_default_component_bindings(self):
        r,s,q,_,_=fixture();operations={i:r.stack.layers[i].attention.operation for i in (1,2)}
        with patch.object(operations[1],'forward',side_effect=AssertionError('default attention operation used')):
            q.step(torch.tensor([1,2]),expected_stamp=q.stamp,page_plans=plans(q,2))
        self.assertTrue(all(r.stack.layers[i].attention.operation is v for i,v in operations.items()))

    def test_retry_after_head_failure_advances_all_layers_once(self):
        r,s,q,_,_=fixture();stamp=q.stamp;r.head.fail=True
        with self.assertRaises(RuntimeError):q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=plans(q,2))
        r.head.fail=False;q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=plans(q,2))
        self.assertEqual({v.tokens_seen for v in q.snapshot().layers.values()},{2});self.assertEqual(q.stamp.revision,1)

    def test_segmented_matches_whole_text_and_state(self):
        r,s,q,b,c=fixture();other=MixedRequestState.capture('B',r.stack.config,r.stack.layout,s.layers)
        q2=r.open_private_request(other,page_bindings=b,producers=q._producers,attentions=q._attentions)
        ids=torch.tensor([0,1,2,3,4,5,6]);full=q.step(ids,expected_stamp=q.stamp,page_plans=plans(q,7));outs=[]
        start=0
        for stop in (1,3,4,7):outs.append(q2.step(ids[start:stop],expected_stamp=q2.stamp,page_plans=plans(q2,stop-start)).logits);start=stop
        torch.testing.assert_close(torch.cat(outs),full.logits,rtol=0,atol=0);equal_state(q.snapshot(),replace(q2.snapshot(),request_id=q.request_id))

    def test_stale_owner_revision_position_table_fail_before_producers(self):
        r,s,q,_,calls=fixture();stamp=q.stamp
        for bad in (replace(stamp,revision=1),replace(stamp,tokens_seen=1),replace(stamp,pages=()),replace(stamp,_owner=object())):
            with self.assertRaisesRegex(ValueError,'stale/foreign'):q.step(torch.tensor([1]),expected_stamp=bad,page_plans=plans(q,1))
        self.assertEqual(calls,[])
        q.step(torch.tensor([1]),expected_stamp=stamp,page_plans=plans(q,1));saved=q.snapshot()
        with self.assertRaises(ValueError):q.step(torch.tensor([2]),expected_stamp=stamp,page_plans=plans(q,1))
        equal_state(saved,q.snapshot())

    def test_wrong_producer_count_or_page_plan_never_commits(self):
        for failure in ('count','reservation','missing'):
            r,s,q,_,_=fixture();stamp=q.stamp;saved=q.page_snapshot();p=plans(q,2)
            if failure=='count':
                produce=q._producers[2];q._producers[2]=lambda step,state:replace(produce(step,state),tokens_seen=1)
            elif failure=='reservation':p[2]['occupied_main']=frozenset({p[2]['private_ids'][0]})
            else:del p[2]
            with self.assertRaises(ValueError):q.step(torch.tensor([1,2]),expected_stamp=stamp,page_plans=p)
            self.assertEqual(q.stamp,stamp);same_pages(saved,q.page_snapshot())

    def test_snapshot_callback_and_output_buffers_do_not_alias_committed(self):
        r,s,q,_,_=fixture();kept=[];original=q._producers[1]
        def producer(step,state):
            value=original(step,state);kept.append(value);return value
        q._producers[1]=producer;out=q.step(torch.tensor([1,2]),expected_stamp=q.stamp,page_plans=plans(q,2));saved=q.snapshot();pages=q.page_snapshot();snap=q.page_snapshot()
        with torch.inference_mode():
            kept[0].latent.zero_();kept[0].tail.zero_();out.logits.zero_();out.hidden.zero_();snap[1].main_raw.zero_();snap[1].state.latent.zero_()
        equal_state(saved,q.snapshot());same_pages(pages,q.page_snapshot())

    def test_empty_reentrant_close_and_shared_runner_ownership(self):
        r,s,q,b,calls=fixture();stamp=q.stamp;r.head.fail=True
        out=q.step(torch.empty(0,dtype=torch.int64),expected_stamp=stamp,page_plans={});self.assertEqual(out.logits.shape,(0,4));self.assertEqual(q.stamp,stamp);self.assertEqual(calls,[])
        with self.assertRaises(ValueError):r.open_request(s)
        r.head.fail=False;original=q._attentions[1];q._attentions[1]=lambda step,view:q.snapshot()
        with self.assertRaisesRegex(RuntimeError,'busy'):q.step(torch.tensor([1]),expected_stamp=stamp,page_plans=plans(q,1))
        self.assertEqual(q.stamp,stamp);q.close();q.close()
        with self.assertRaises(RuntimeError):q.snapshot()
        with self.assertRaises(RuntimeError):q.step(torch.tensor([1]),expected_stamp=stamp,page_plans={})
        self.assertEqual(r.open_request(s).snapshot().tokens_seen,0)

    def test_incomplete_maps_budget_and_cpu_guards(self):
        r,s,q,b,_=fixture();q.close()
        for kwargs in (dict(producers={},attentions={}),dict(producers=q._producers,attentions=q._attentions,max_backing_bytes=1)):
            with self.assertRaises(ValueError):r.open_private_request(s,page_bindings=b,**kwargs)
        self.assertFalse(r._requests)
        q=r.open_private_request(s,page_bindings=b,producers=q._producers,attentions=q._attentions)
        for ids in (torch.tensor([1.]),torch.tensor([11]),torch.tensor([True]),torch.ones((1,1),dtype=torch.int64)):
            with self.assertRaises(ValueError):q.step(ids,expected_stamp=q.stamp,page_plans=plans(q,1))
        self.assertEqual(q.stamp.tokens_seen,0)


if __name__=='__main__':unittest.main(verbosity=2)
