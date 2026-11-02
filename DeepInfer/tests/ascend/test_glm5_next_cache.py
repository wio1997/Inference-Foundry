# SPDX-License-Identifier: Apache-2.0
"""CPU state mechanics; independent pool/slot/tail oracles, no device kernels."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from deepinfer.model_executor.models.glm5_next.cache import (
    DSACacheLayout,compressed_slot_mapping,expand_pool_indices,
    state_from_aligned_prefix,append_dsa,validate_dsa_state,
)
from deepinfer.model_executor.models.glm5_next.kda import KDAState
from deepinfer.model_executor.models.glm5_next.request_state import MixedRequestState


def empty(layout):
    return state_from_aligned_prefix(torch.empty(0,layout.latent_dim,dtype=torch.bfloat16),
                                     torch.empty(0,layout.index_dim,dtype=torch.bfloat16),layout)


def assert_state_equal(a,b):
    assert a.tokens_seen==b.tokens_seen
    for name in ('latent','compressed','tail','tail_positions'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)


class CacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(5317)
        self.layout=DSACacheLayout(512,128,4,2048,16)

    def inputs(self,n):
        return (torch.randn(n,512).bfloat16(),torch.randn(n,128).bfloat16(),
                torch.randn(n,128)*4,torch.randn(4,128))

    def test_actual_width_layout_and_independent_namespaces(self):
        self.assertEqual(self.layout.storage_shapes(3,2),{'main':(3,16,512),'index':(3,4,128),'tail':(2,2,4,128)})
        self.assertEqual(replace(self.layout,lookahead=3).tail_capacity,7)
        for kwargs in ({'logical_block_size':15},{'pool_size':1},{'topk_tokens':2047},{'lookahead':-1},{'lookahead':True}):
            with self.assertRaises(ValueError):replace(self.layout,**kwargs)

    def test_noncontiguous_pages_and_padding_match_integer_slot_oracle(self):
        positions=torch.arange(45);blocks=[9,2,41]
        slots=torch.tensor([blocks[p//16]*16+p%16 if p%7 else -1 for p in positions.tolist()])
        expected=[]
        for slot,p in zip(slots.tolist(),positions.tolist()):
            expected.append(-1 if slot<0 or (p+1)%4 else slot//16*4+(slot%16)//4)
        torch.testing.assert_close(compressed_slot_mapping(slots,positions,self.layout),torch.tensor(expected))
        with self.assertRaises(ValueError):compressed_slot_mapping(slots.float(),positions,self.layout)
        with self.assertRaises(ValueError):compressed_slot_mapping(slots,-positions,self.layout)

    def test_pool_expansion_tail_compaction_short_long_and_empty(self):
        layout=replace(self.layout,topk_tokens=8);positions=torch.tensor([0,2,3,4,7,8,11,14,19])
        selected=torch.full((len(positions),2),-1,dtype=torch.int32)
        expected=torch.full((len(positions),11),-1,dtype=torch.int32)
        for row,pos in enumerate(positions.tolist()):
            complete=(pos+1)//4;ids=list(reversed(range(complete)))[:2];selected[row,:len(ids)]=torch.tensor(ids,dtype=torch.int32)
            wanted=[p for pool in ids for p in range(pool*4,pool*4+4)]+list(range(complete*4,pos+1))
            expected[row,:len(wanted)]=torch.tensor(wanted,dtype=torch.int32)
            self.assertTrue(all(0<=p<=pos for p in wanted))
        torch.testing.assert_close(expand_pool_indices(selected,positions,layout),expected)
        self.assertEqual(expand_pool_indices(torch.empty(0,2,dtype=torch.int32),torch.empty(0,dtype=torch.int64),layout).shape,(0,11))
        for ids,pos in [([0,0],7),([0,2],7),([-1,0],3),([0,1],0)]:
            with self.assertRaises(ValueError):expand_pool_indices(torch.tensor([ids]),torch.tensor([pos]),layout)

    def test_channel_softmax_pool_matches_independent_numpy(self):
        latent,keys,gates,ape=self.inputs(19);saved=empty(self.layout);before=saved.clone()
        result=append_dsa(latent,keys,gates,ape,saved,self.layout,start_position=0)
        expected=[]
        # Per-channel FP64 independent equation, then explicit BF16 storage.
        for start in range(0,16,4):
            logits=gates[start:start+4].double().numpy()+ape.double().numpy()
            exp=np.exp(logits-logits.max(axis=0));weights=exp/exp.sum(axis=0)
            expected.append((weights*keys[start:start+4].float().numpy()).sum(axis=0))
        torch.testing.assert_close(result.compressed,torch.tensor(np.array(expected)).bfloat16(),rtol=0,atol=0)
        assert_state_equal(saved,before)
        for pos in range(15,19):
            slot=pos%4;self.assertEqual(result.tail_positions[slot].item(),pos)
            torch.testing.assert_close(result.tail[0,slot],keys[pos].float(),rtol=0,atol=0)
            torch.testing.assert_close(result.tail[1,slot],gates[pos],rtol=0,atol=0)

    def test_chunking_boundary_and_request_isolation(self):
        latent,keys,gates,ape=self.inputs(19);whole=append_dsa(latent,keys,gates,ape,empty(self.layout),self.layout,start_position=0)
        for cuts in ([1,3,4,5,12,19],[3,8,9,16,19],list(range(1,20))):
            state=empty(self.layout);start=0
            for end in cuts:
                state=append_dsa(latent[start:end],keys[start:end],gates[start:end],ape,state,self.layout,start_position=start);start=end
            assert_state_equal(state,whole)
        a=empty(self.layout);b=empty(self.layout);a.tail[0,0,0]=99
        self.assertEqual(b.tail[0,0,0].item(),0)
        state=append_dsa(latent[:3],keys[:3],gates[:3],ape,b,self.layout,start_position=0);copy=state.clone()
        append_dsa(latent[3:4],keys[3:4],gates[3:4],ape,state,self.layout,start_position=3)
        assert_state_equal(copy,state)
        assert_state_equal(append_dsa(latent[:0],keys[:0],gates[:0],ape,state,self.layout,start_position=3),state)

    def test_prefix_restore_only_on_aligned_boundary(self):
        latent,keys,gates,ape=self.inputs(11)
        all_state=append_dsa(latent,keys,gates,ape,empty(self.layout),self.layout,start_position=0)
        prefix=append_dsa(latent[:8],keys[:8],gates[:8],ape,empty(self.layout),self.layout,start_position=0)
        restored=state_from_aligned_prefix(prefix.latent,prefix.compressed,self.layout)
        self.assertTrue(bool((restored.tail_positions==-1).all()))
        continued=append_dsa(latent[8:],keys[8:],gates[8:],ape,restored,self.layout,start_position=8)
        torch.testing.assert_close(continued.latent,all_state.latent,rtol=0,atol=0)
        torch.testing.assert_close(continued.compressed,all_state.compressed,rtol=0,atol=0)
        for pos in range(8,11):torch.testing.assert_close(continued.tail[:,pos%4],all_state.tail[:,pos%4],rtol=0,atol=0)
        with self.assertRaises(ValueError):state_from_aligned_prefix(latent[:7],prefix.compressed[:1],self.layout)
        with torch.inference_mode():prefix.latent.zero_()
        self.assertFalse(torch.equal(prefix.latent,restored.latent))

    def test_invalid_continuation_does_not_mutate_source(self):
        latent,keys,gates,ape=self.inputs(3);state=append_dsa(latent,keys,gates,ape,empty(self.layout),self.layout,start_position=0);saved=state.clone()
        with self.assertRaises(ValueError):append_dsa(latent,keys,gates,ape,state,self.layout,start_position=2)
        with self.assertRaises(ValueError):append_dsa(latent,keys,gates.half(),ape,state,self.layout,start_position=3)
        bad=gates.clone();bad[0,0]=float('nan')
        with self.assertRaises(ValueError):append_dsa(latent,keys,bad,ape,state,self.layout,start_position=3)
        broken=state.clone();broken.tail_positions[1]=-1
        with self.assertRaises(ValueError):validate_dsa_state(broken,self.layout)
        assert_state_equal(state,saved)

    def test_lookahead_capacity_changes_storage_not_pool_arithmetic(self):
        latent,keys,gates,ape=self.inputs(19)
        a=append_dsa(latent,keys,gates,ape,empty(self.layout),self.layout,start_position=0)
        layout=replace(self.layout,lookahead=3);b=append_dsa(latent,keys,gates,ape,empty(layout),layout,start_position=0)
        torch.testing.assert_close(a.compressed,b.compressed,rtol=0,atol=0)
        self.assertEqual(b.tail.shape,(2,7,128))

    def test_mixed_bundle_complete_atomic_update_and_owned_snapshots(self):
        config=SimpleNamespace(layers=[SimpleNamespace(layer_id=0,attention='linear_attention'),SimpleNamespace(layer_id=1,attention='deepseek_sparse_attention')],
                               linear_attn_config={'num_heads':2,'head_dim':3,'short_conv_kernel_size':4})
        def kda(n):return KDAState(tuple(torch.zeros(3,6,dtype=torch.bfloat16) for _ in range(3)),torch.zeros(2,3,3),n)
        initial={0:kda(0),1:empty(self.layout)};bundle=MixedRequestState.capture('req-A',config,self.layout,initial)
        initial[0].recurrent.fill_(99);self.assertEqual(bundle.layers[0].recurrent.max().item(),0)
        latent,keys,gates,ape=self.inputs(5);dsa=append_dsa(latent,keys,gates,ape,bundle.layers[1],self.layout,start_position=0)
        updates={0:kda(5),1:dsa};next_state=bundle.advance(config,self.layout,updates,num_tokens=5)
        updates[0].conv[0].fill_(7);self.assertEqual(next_state.layers[0].conv[0].max().item(),0)
        self.assertEqual((bundle.tokens_seen,next_state.tokens_seen),(0,5))
        for wrong in ({0:kda(5)},{0:kda(4),1:dsa},{0:dsa,1:kda(5)}):
            with self.assertRaises(ValueError):bundle.advance(config,self.layout,wrong,num_tokens=5)
        with self.assertRaises(ValueError):bundle.advance(config,self.layout,{0:kda(5),1:dsa},num_tokens=4)
        with self.assertRaises(TypeError):next_state.layers[3]=kda(5)
        self.assertEqual(bundle.tokens_seen,0)

if __name__=='__main__':unittest.main(verbosity=2)
