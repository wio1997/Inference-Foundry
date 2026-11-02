# SPDX-License-Identifier: Apache-2.0
"""Explicit three-layer injected request/page commit, CPU only."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from test_glm5_next_text_request import fixture
from test_glm5_next_hybrid_decoder import empty_dsa,equal_state
from test_glm5_next_paged_commit import fixture as write_fixture,raw_bytes
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry
from deepinfer.model_executor.models.glm5_next.deferred_pages import CPUPageBinding
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState
import deepinfer.model_executor.models.glm5_next.paged_commit as commit


def make_bindings(runner):
    layout=runner.stack.layout;g=CANNCacheGeometry(layout,(layout.pool_size+2)*2*layout.index_dim*4)
    return {s.layer_id:CPUPageBinding(torch.full((2*g.main_page_bytes,),0x5a,dtype=torch.uint8),torch.full((2*g.index_page_bytes,),0x5a,dtype=torch.uint8),torch.full((g.tail_page_bytes,),0x5a,dtype=torch.uint8),g,2,1,torch.tensor([[0,1]]),0)
            for s in runner.stack.config.layers if s.attention=='deepseek_sparse_attention'}


def setup(**kwargs):
    r,s,op,bank,shared,reader=fixture(**kwargs)
    layer=copy.deepcopy(r.stack.layers[1]);layer.layer_id=2;r.stack.layers.append(layer)
    r.stack.config.layers.append(SimpleNamespace(layer_id=2,attention='deepseek_sparse_attention',mlp='sparse'))
    states=dict(s.layers);states[2]=empty_dsa(r.stack.layout)
    state=MixedRequestState.capture(s.request_id,r.stack.config,r.stack.layout,states)
    return r,state,make_bindings(r)


def buffers(bindings):return {i:tuple(raw_bytes(t) for t in (b.main_raw,b.index_raw,b.tail_raw)) for i,b in bindings.items()}


def expected_bytes(binding,before,following,original):
    out=[bytearray(x) for x in original];g=binding.geometry;l=g.layout;B=l.logical_block_size;p=l.pool_size
    for position in range(before.tokens_seen,following.tokens_seen):
        page=int(binding.logical_table[0,position//B]);offset=(page*B+position%B)*l.latent_dim*2
        data=raw_bytes(following.latent[position]);out[0][offset:offset+len(data)]=data
        if (position+1)%p==0:
            offset=(page*(B//p)+(position%B)//p)*l.index_dim*2;data=raw_bytes(following.compressed[position//p]);out[1][offset:offset+len(data)]=data
    if following.tokens_seen>before.tokens_seen:
        for position in range(following.tokens_seen//p*p,following.tokens_seen):
            slot=position%p;offset=binding.tail_page_id*g.tail_page_bytes+slot*2*l.index_dim*4
            data=raw_bytes(following.tail[0,slot])+raw_bytes(following.tail[1,slot]);out[2][offset:offset+len(data)]=data
    return tuple(bytes(x) for x in out)


class DeferredPagesTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(5328)

    def test_two_DSA_layers_and_request_metadata_commit_after_output(self):
        r,s,bindings=setup();q=r.open_request(s,page_bindings=bindings)
        for start,ids in ((0,torch.tensor([1,2,3])),(3,torch.tensor([4,5,6,7]))):
            before=q.snapshot();raw=buffers(bindings);out=q.step(ids,expected_tokens_seen=start);after=q.snapshot()
            self.assertEqual(out.token_stop,after.tokens_seen)
            for layer,binding in bindings.items():self.assertEqual(buffers(bindings)[layer],expected_bytes(binding,before.layers[layer],after.layers[layer],raw[layer]))

    def test_head_failure_creates_no_page_writes_then_retry_advances_once(self):
        r,s,b=setup();q=r.open_request(s,page_bindings=b);saved=buffers(b);r.head.fail=True
        with patch.object(CPUPageBinding,'stage',side_effect=AssertionError('staged before head success')):
            with self.assertRaisesRegex(RuntimeError,'head failure'):q.step(torch.arange(7),expected_tokens_seen=0)
        self.assertEqual(buffers(b),saved);equal_state(q.snapshot(),s)
        r.head.fail=False;q.step(torch.arange(7),expected_tokens_seen=0);self.assertEqual(q.snapshot().tokens_seen,7)

    def test_later_decoder_failure_preserves_earlier_pages_and_metadata(self):
        r,s,b=setup();q=r.open_request(s,page_bindings=b);saved=buffers(b);r.stack.layers[2].mlp.experts.bank.fail=True
        with self.assertRaises(RuntimeError):q.step(torch.arange(7),expected_tokens_seen=0)
        self.assertEqual(buffers(b),saved);equal_state(q.snapshot(),s)

    def test_failure_after_sixth_write_rolls_back_both_layers_before_publish(self):
        r,s,b=setup();q=r.open_request(s,page_bindings=b);saved=buffers(b);calls=[]
        def fail_last(target,ids,values):
            target.index_copy_(0,ids,values);calls.append(len(ids))
            if len(calls)==6:raise RuntimeError('second layer tail write failed')
        with patch.object(commit,'_write_rows',side_effect=fail_last):
            with self.assertRaisesRegex(RuntimeError,'second layer'):q.step(torch.arange(7),expected_tokens_seen=0)
        self.assertEqual(len(calls),6);self.assertEqual(buffers(b),saved);equal_state(q.snapshot(),s)
        q.step(torch.arange(7),expected_tokens_seen=0);self.assertEqual(q.snapshot().tokens_seen,7)

    def test_late_stale_layer_rejects_before_first_layer_write(self):
        r,s,b=setup();q=r.open_request(s,page_bindings=b);saved=buffers(b);original=CPUPageBinding.stage
        def stale(binding,previous,following):
            batch=original(binding,previous,following)
            if binding is b[2]:binding.main_raw[0]^=1
            return batch
        with patch.object(CPUPageBinding,'stage',new=stale),patch.object(commit,'_write_rows',side_effect=AssertionError('wrote before all-layer validation')):
            with self.assertRaisesRegex(ValueError,'changed after staging'):q.step(torch.arange(7),expected_tokens_seen=0)
        self.assertEqual(buffers(b)[1],saved[1]);self.assertNotEqual(buffers(b)[2],saved[2]);equal_state(q.snapshot(),s)

    def test_close_resume_requires_matching_page_history(self):
        r,s,b=setup();q=r.open_request(s,page_bindings=b);q.step(torch.arange(7),expected_tokens_seen=0);saved=q.snapshot();q.close()
        b[1].main_raw[0]^=1
        with self.assertRaisesRegex(ValueError,'history disagrees'):r.open_request(saved,page_bindings=b)
        b[1].main_raw[0]^=1;q=r.open_request(saved,page_bindings=b);q.step(torch.tensor([8]),expected_tokens_seen=7)
        self.assertEqual(q.snapshot().tokens_seen,8)

    def test_whole_backing_ownership_and_copied_table_metadata(self):
        r,s,b=setup(max_requests=2);q=r.open_request(s,page_bindings=b);other=replace(s,request_id='other')
        with self.assertRaisesRegex(ValueError,'exclusive'):r.open_request(other,page_bindings=dict(b))
        separate=make_bindings(r);r.open_request(other,page_bindings=separate)
        b[1].logical_table[0,1]=0
        with self.assertRaisesRegex(ValueError,'unique'):q.step(torch.tensor([1]),expected_tokens_seen=0)
        self.assertEqual(q.snapshot().tokens_seen,0)

    def test_binding_coverage_duplicates_and_non_cpu_reject(self):
        r,s,b=setup()
        with self.assertRaises(ValueError):r.open_request(s,page_bindings={1:b[1]})
        x=b[1]
        with self.assertRaisesRegex(ValueError,'unique'):CPUPageBinding(x.main_raw,x.index_raw,x.tail_raw,x.geometry,2,1,torch.tensor([[0,0]]),0)
        with self.assertRaisesRegex(ValueError,'CPU-only'):CPUPageBinding(torch.empty(1,device='meta',dtype=torch.uint8),x.index_raw,x.tail_raw,x.geometry,2,1,x.logical_table,0)
        table=torch.tensor([[0,1]]);new=CPUPageBinding(x.main_raw,x.index_raw,x.tail_raw,x.geometry,2,1,table,0);table.fill_(-1)
        self.assertEqual(new.logical_table.tolist(),[[0,1]])

    def test_empty_input_skips_staging_and_leaves_pages_unchanged(self):
        r,s,b=setup();q=r.open_request(s,page_bindings=b);saved=buffers(b)
        with patch.object(CPUPageBinding,'stage',side_effect=AssertionError('empty staging called')):q.step(torch.empty(0,dtype=torch.int64),expected_tokens_seen=0)
        self.assertEqual(buffers(b),saved);equal_state(q.snapshot(),s)

    def test_group_bounds_duplicate_overlap_and_empty_contracts(self):
        self.assertEqual(commit.commit_dsa_batches(()),())
        g,raw,a=write_fixture();one=commit.stage_dsa_writes(*raw,g,**a);saved=[raw_bytes(t) for t in raw]
        for batches,options in (((one,one),{}),((one,),{'max_payload_bytes':1}),((one,),{'max_batches':True})):
            with self.assertRaises(ValueError):commit.commit_dsa_batches(batches,**options)
        two=commit.stage_dsa_writes(*raw,g,**a)
        with self.assertRaisesRegex(ValueError,'overlap'):commit.commit_dsa_batches((one,two))
        self.assertEqual([raw_bytes(t) for t in raw],saved)


if __name__=='__main__':unittest.main(verbosity=2)
