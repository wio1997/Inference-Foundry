# SPDX-License-Identifier: Apache-2.0
"""Injected tiny two-layer request lifecycle; no actual model/native execution."""
from dataclasses import replace
import unittest
import torch
from test_glm5_next_hybrid_decoder import setup_stack,equal_state
from deepinfer.model_executor.models.glm5_next.text_boundary import TextBoundaryManifest,GlmTextBoundary,GlmLogitsShard
from deepinfer.model_executor.models.glm5_next.text_request import PreparedTextRunner
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState


class Reader:
    def __init__(self):self.weight=(torch.randn(11,6)*.125).bfloat16();self.calls=0
    def raw_rows(self,name,ids):self.calls+=1;return self.weight[ids].clone(),'BF16'


class Head(GlmLogitsShard):
    fail=False
    def forward(self,hidden):
        if self.fail:raise RuntimeError('injected head failure')
        return super().forward(hidden)


def fixture(**kwargs):
    stack,state,op,bank,shared=setup_stack();stack.config.hidden_size=6;stack.config.vocab_size=11
    stack.config.hf_config={'text_config':{'hc_mult':2,'rms_norm_eps':1e-5}}
    reader=Reader();boundary=GlmTextBoundary(reader,TextBoundaryManifest(6,11,2,1e-5,1.25),torch.ones(6).bfloat16(),max_unique_tokens=8,max_tokens=8)
    head=Head((torch.randn(4,6)*.125).bfloat16(),vocab_start=3,scale=1.25,max_tokens=8)
    runner=PreparedTextRunner(stack,boundary,head,**kwargs)
    return runner,state,op,bank,shared,reader


class TextRequestTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5326)

    def test_complete_step_matches_explicit_component_chain(self):
        r,s,*_=fixture();ids=torch.tensor([0,3,0,8]);q=r.open_request(s)
        with torch.inference_mode():
            streams,expected_state=r.stack(r.boundary.embed(ids),s)
            hidden=r.boundary.final_hidden(streams);logits=r.head(hidden)
        out=q.step(ids,expected_tokens_seen=0)
        self.assertEqual((out.token_start,out.token_stop,out.vocab_start,out.vocab_end),(0,4,3,7))
        torch.testing.assert_close(out.hidden,hidden,rtol=0,atol=0);torch.testing.assert_close(out.logits,logits,rtol=0,atol=0);equal_state(q.snapshot(),expected_state)

    def test_segmented_steps_match_whole_and_independent_request(self):
        r,s,*_=fixture(max_requests=2);whole=r.open_request(s)
        other=MixedRequestState.capture('req-B',r.stack.config,r.stack.layout,s.layers);q=r.open_request(other)
        ids=torch.tensor([0,1,2,3,4,5,6]);expected=whole.step(ids,expected_tokens_seen=0);outputs=[]
        start=0
        for stop in (1,3,4,7):outputs.append(q.step(ids[start:stop],expected_tokens_seen=start).logits);start=stop
        torch.testing.assert_close(torch.cat(outputs),expected.logits,rtol=0,atol=0)
        a,b=whole.snapshot(),q.snapshot();equal_state(a,replace(b,request_id=a.request_id))

    def test_token_major_mode_preserves_segmented_output_and_atomic_failure(self):
        r,s,*_=fixture(max_requests=2,native_tokenwise=True);q=r.open_request(s)
        other=r.open_request(MixedRequestState.capture('token-major-B',r.stack.config,r.stack.layout,s.layers))
        ids=torch.tensor([0,1,2]);whole=q.step(ids,expected_tokens_seen=0)
        left=other.step(ids[:1],expected_tokens_seen=0);right=other.step(ids[1:],expected_tokens_seen=1)
        torch.testing.assert_close(whole.logits,torch.cat((left.logits,right.logits)),rtol=0,atol=0)
        equal_state(q.snapshot(),replace(other.snapshot(),request_id=q.request_id))
        q.close();other.close();fresh=r.open_request(s);saved=fresh.snapshot();original=r.head.forward;calls=[]
        def fail_second(hidden):
            calls.append(1)
            if len(calls)==2:raise RuntimeError('second token head failure')
            return original(hidden)
        r.head.forward=fail_second
        with self.assertRaisesRegex(RuntimeError,'second token head failure'):fresh.step(ids,expected_tokens_seen=0)
        equal_state(saved,fresh.snapshot())

    def test_head_failure_does_not_publish_then_retry_advances_once(self):
        r,s,*_=fixture();q=r.open_request(s);saved=q.snapshot();r.head.fail=True
        with self.assertRaisesRegex(RuntimeError,'head failure'):q.step(torch.tensor([1,2]),expected_tokens_seen=0)
        equal_state(q.snapshot(),saved);r.head.fail=False;q.step(torch.tensor([1,2]),expected_tokens_seen=0)
        self.assertEqual(q.snapshot().tokens_seen,2)

    def test_decoder_failure_cannot_mutate_owned_or_other_request(self):
        r,s,op,bank,*_=fixture(max_requests=2);q=r.open_request(s)
        b=r.open_request(MixedRequestState.capture('req-B',r.stack.config,r.stack.layout,s.layers));saved=q.snapshot();other=b.snapshot()
        op.mutate_input=True;bank.fail=True
        with self.assertRaises(RuntimeError):q.step(torch.tensor([1]),expected_tokens_seen=0)
        equal_state(saved,q.snapshot());equal_state(other,b.snapshot())

    def test_stale_bounds_and_bad_ids_reject_without_payload_or_commit(self):
        r,s,*rest=fixture(max_context_tokens=2);reader=rest[-1];q=r.open_request(s)
        for ids,position in ((torch.tensor([1]),1),(torch.tensor([1]),True),(torch.tensor([1,2,3]),0),(torch.tensor([11]),0),(torch.tensor([1.]),0)):
            before=reader.calls
            with self.assertRaises(ValueError):q.step(ids,expected_tokens_seen=position)
            self.assertEqual(reader.calls,before);self.assertEqual(q.snapshot().tokens_seen,0)
        q.step(torch.tensor([1,2]),expected_tokens_seen=0)
        with self.assertRaises(ValueError):q.step(torch.tensor([1]),expected_tokens_seen=2)
        with self.assertRaises(ValueError):q.step(torch.tensor([1]),expected_tokens_seen=0)

    def test_input_snapshots_and_outputs_do_not_alias_request_state(self):
        r,s,*_=fixture();q=r.open_request(s);owned=q.snapshot()
        with torch.inference_mode():s.layers[0].recurrent.fill_(99);owned.layers[1].tail.fill_(99)
        self.assertFalse(bool((q.snapshot().layers[0].recurrent==99).any()));self.assertFalse(bool((q.snapshot().layers[1].tail==99).any()))
        out=q.step(torch.tensor([1]),expected_tokens_seen=0);saved=q.snapshot()
        with torch.inference_mode():out.hidden.zero_();out.logits.zero_()
        equal_state(saved,q.snapshot())

    def test_close_reopen_unique_owner_and_busy_rejection(self):
        r,s,*_=fixture();q=r.open_request(s)
        with self.assertRaises(ValueError):r.open_request(s)
        with self.assertRaises(ValueError):r.open_request(replace(s,request_id='other'))
        with self.assertRaises(AttributeError):q.request_id='changed'
        r._lock.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError,'busy'):q.snapshot()
        finally:r._lock.release()
        q.close();q.close()
        with self.assertRaises(RuntimeError):q.snapshot()
        with self.assertRaises(RuntimeError):q.step(torch.tensor([1]),expected_tokens_seen=0)
        self.assertEqual(r.open_request(s).snapshot().tokens_seen,0)

    def test_empty_skips_decoder_head_and_payload(self):
        r,s,op,bank,shared,reader=fixture();q=r.open_request(s);r.head.fail=True
        out=q.step(torch.empty(0,dtype=torch.int64),expected_tokens_seen=0)
        self.assertEqual((op.calls,bank.calls,shared.calls,reader.calls),(0,0,0,0));self.assertEqual(out.logits.shape,(0,4));self.assertEqual(out.hidden.shape,(0,6));equal_state(q.snapshot(),s)

    def test_geometry_and_explicit_bindings_reject(self):
        r,s,*_=fixture()
        with self.assertRaises(ValueError):PreparedTextRunner(None,r.boundary,r.head)
        with self.assertRaises(ValueError):PreparedTextRunner(r.stack,r.boundary,r.head,max_requests=True)
        r.stack.config.hidden_size=7
        with self.assertRaises(ValueError):PreparedTextRunner(r.stack,r.boundary,r.head)
        r.stack.config.hidden_size=6;r.head.scale=2.
        with self.assertRaises(ValueError):PreparedTextRunner(r.stack,r.boundary,r.head)

    def test_incomplete_and_inconsistent_snapshot_reject_before_owner(self):
        r,s,*_=fixture()
        with self.assertRaises(ValueError):r.open_request(replace(s,layers={0:s.layers[0]}))
        with self.assertRaises(ValueError):r.open_request(replace(s,tokens_seen=1))
        self.assertEqual(r.open_request(s).snapshot().tokens_seen,0)


if __name__=='__main__':unittest.main(verbosity=2)
