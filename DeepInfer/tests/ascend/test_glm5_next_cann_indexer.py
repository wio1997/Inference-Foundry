# SPDX-License-Identifier: Apache-2.0
"""Sentinel/causal packing references, independent scalar expected values."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
from deepinfer.model_executor.models.glm5_next.cann_indexer import CANNIndexerContract,prepare_nonempty_pki_cpu,pack_cann_token_indices_cpu,validate_indexer_source_calls


SOURCE='''
class Backend:
 def _init_cann_ops(self):
  self._key_pool=torch.ops.cann_ops_transformer.key_pool
  self._pool_key_indexer=partial(adapters.pool_key_indexer_nonempty,torch.ops.cann_ops_transformer.pool_key_indexer)
 def _forward_cann(self):
  state=self._key_pool(hidden_states,self._key_weight,self._gate_weight,self.index_kpool_compress_ape,state,indexer_metadata.keypool_state_table,indexer_metadata.keypool_start_offsets,norm_weight=self._norm_weight_f32,norm_bias=self._norm_bias_f32,cu_seqlens=indexer_metadata.query_start_loc,cmp_ratio=self.index_kpool,norm_eps=norm_eps)
  indices,_=self._pool_key_indexer(query,indexer_cache,weights.to(torch.bfloat16),query_metadata.pool_tail,actual_seq_q=query_metadata.cum_query_lens,actual_seq_k=query_metadata.seq_lens,block_table=indexer_metadata.block_table,layout_q="TND",layout_k="PA_BBND",topk=self.topk_tokens,pool_size=self.index_kpool,mask_mode=3)
'''
ADAPTER='''
def pool_key_indexer_nonempty():
 return prepare_nonempty_keypool_inputs(query,weights,actual_seq_q,actual_seq_k,pool_tail_k,block_table)
'''


def inputs():
    c=CANNIndexerContract(2,3,4,8)
    q=torch.arange(5*2*6).reshape(5,2,6).bfloat16()[:,:,::2]
    w=-torch.arange(20).reshape(5,4).bfloat16()[:,::2]
    ends=torch.tensor([2,2,4],dtype=torch.int64);lengths=torch.tensor([0,1,32],dtype=torch.int64);tails=torch.tensor([3,0,1],dtype=torch.int64)
    table=torch.tensor([[0,9,1,9],[1,9,-1,9],[0,9,-1,9]],dtype=torch.int32)[:,::2]
    return c,[q,w,ends,lengths,tails,table]


class CANNIndexerTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_sentinel_inputs_match_scalar_copy_and_clamp(self):
        c,a=inputs();saved=[t.clone() for t in a];out=prepare_nonempty_pki_cpu(*a,c,num_kernel_blocks=2)
        expected=[a[0].tolist()+[[[0.]*3 for _ in range(2)]],a[1].tolist()+[[0.,0.]],a[2].tolist()+[6],[max(x,1) for x in a[3].tolist()]+[1],a[4].tolist()+[0],a[5].tolist()+[[0,0]]]
        for got,values,old,now in zip(out,expected,saved,a):
            self.assertEqual(got.tolist(),values);self.assertTrue(got.is_contiguous());self.assertEqual(got.dtype,now.dtype);self.assertTrue(torch.equal(old,now))

    def test_all_padding_and_zero_requests_keep_safe_dummy(self):
        c=CANNIndexerContract(2,3,4,8)
        for tokens,n in ((0,0),(3,0),(3,2)):
            args=[torch.zeros(tokens,2,3,dtype=torch.bfloat16),torch.zeros(tokens,2,dtype=torch.bfloat16),torch.zeros(n,dtype=torch.int64),torch.zeros(n,dtype=torch.int64),torch.zeros(n,dtype=torch.int64),torch.zeros(n,1,dtype=torch.int32)]
            out=prepare_nonempty_pki_cpu(*args,c,num_kernel_blocks=1)
            self.assertEqual(out[2].tolist(),[0]*n+[tokens+1]);self.assertEqual(out[3].tolist(),[1]*(n+1));self.assertEqual(out[5].shape,(n+1,1))

    def test_outputs_are_private_across_calls_and_layers(self):
        c,a=inputs();one=prepare_nonempty_pki_cpu(*a,c,num_kernel_blocks=2);two=prepare_nonempty_pki_cpu(*a,c,num_kernel_blocks=2)
        with torch.inference_mode():one[0].zero_();one[2].zero_()
        self.assertFalse(torch.equal(one[0],two[0]));self.assertEqual(two[2].tolist(),[2,2,4,6]);self.assertEqual(a[2].tolist(),[2,2,4])

    def test_metadata_zero_pool_backing_and_bounds_reject(self):
        for bad in ('missing_pool','short_table','negative','tail','nonmonotonic','wrong_rows','no_blocks','budget'):
            c,a=inputs();kw={'num_kernel_blocks':2}
            if bad=='missing_pool':a[5][0,0]=-1
            elif bad=='short_table':a[3][0]=65
            elif bad=='negative':a[3][0]=-1
            elif bad=='tail':a[4][0]=4
            elif bad=='nonmonotonic':a[2][1]=1
            elif bad=='wrong_rows':a[5]=a[5][:1]
            elif bad=='no_blocks':kw['num_kernel_blocks']=0
            else:kw['max_output_bytes']=1
            with self.assertRaises(ValueError):prepare_nonempty_pki_cpu(*a,c,**kw)

    def test_budget_rejects_before_copies_and_native_device_rejects(self):
        c,a=inputs()
        with patch('torch.cat',side_effect=AssertionError('allocated before budget')):
            with self.assertRaisesRegex(ValueError,'byte bound'):prepare_nonempty_pki_cpu(*a,c,num_kernel_blocks=2,max_output_bytes=1)
        a[0]=torch.empty((5,2,3),device='meta',dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'CPU-only'):prepare_nonempty_pki_cpu(*a,c,num_kernel_blocks=2)

    def test_padded_nonfinite_rows_are_ignored_but_live_rows_reject(self):
        c,a=inputs();a[0][4].fill_(float('nan'));out=prepare_nonempty_pki_cpu(*a,c,num_kernel_blocks=2)
        self.assertTrue(bool(torch.isnan(out[0][4]).all()))
        a[0][0].fill_(float('nan'))
        with self.assertRaises(ValueError):prepare_nonempty_pki_cpu(*a,c,num_kernel_blocks=2)

    def test_pack_expanded_tokens_matches_scalar_causal_tail(self):
        c=CANNIndexerContract(2,3,4,8);positions=torch.tensor([0,2,3,7,8,-999]);ends=torch.tensor([2,2,5]);indices=torch.full((6,8),12345,dtype=torch.int32)
        indices[2,:4]=torch.arange(4,dtype=torch.int32);indices[3]=torch.arange(8,dtype=torch.int32);indices[4]=torch.tensor([4,5,6,7,0,1,2,3],dtype=torch.int32)
        out=pack_cann_token_indices_cpu(indices,positions,ends,c,index_namespace='expanded_token');expected=[]
        for row,pos in enumerate(positions.tolist()):
            values=[]
            if row<5:
                complete=(pos+1)//4*4;values=indices[row,:min(complete,8)].tolist()+list(range(complete,pos+1))
            expected.append([values+[-1]*(11-len(values))])
        self.assertEqual(out.tolist(),expected);self.assertEqual(out.dtype,torch.int32)

    def test_empty_pack_and_invalid_namespace_candidates_or_budget(self):
        c=CANNIndexerContract(2,3,4,8)
        for n in (0,3):
            out=pack_cann_token_indices_cpu(torch.empty(n,8,dtype=torch.int32),torch.full((n,),-999),torch.empty(0,dtype=torch.int64),c,index_namespace='expanded_token')
            self.assertEqual(out.shape,(n,1,11));self.assertTrue(bool((out==-1).all()))
        for kind in ('pool','duplicate','future','short','budget'):
            values=torch.arange(8,dtype=torch.int32).reshape(1,8);kw={'index_namespace':'expanded_token'}
            if kind=='pool':kw['index_namespace']='pool_id'
            elif kind=='duplicate':values[0,1]=0
            elif kind=='future':values[0,0]=8
            elif kind=='short':values=values[:,:7]
            else:kw['max_output_bytes']=1
            with self.assertRaises(ValueError):pack_cann_token_indices_cpu(values,torch.tensor([7]),torch.tensor([1]),c,**kw)

    def test_config_contract_and_source_calls_are_not_binary_abi(self):
        cfg=NS(mla_dimensions={'qk_rope_head_dim':0,'kv_lora_rank':512},indexer_config={'index_n_heads':32,'index_head_dim':128,'index_kpool':4,'index_topk':2048,'index_kpool_compress':True,'index_kpool_always_select_tail':True},hf_config={'text_config':{'mla_use_nope':True}})
        c=CANNIndexerContract.from_config(cfg);self.assertEqual(c.output_width,2051);self.assertFalse(c.to_dict()['native_ABI_validated'])
        report=validate_indexer_source_calls(SOURCE,ADAPTER);self.assertFalse(report['native_ABI_validated']);self.assertTrue(report['source_only'])

    def test_source_argument_role_layout_and_missing_dummy_drift_reject(self):
        for source,adapter in ((SOURCE.replace('mask_mode=3','mask_mode=2'),ADAPTER),(SOURCE.replace('block_table=indexer_metadata.block_table','block_table=query_metadata.block_table'),ADAPTER),(SOURCE.replace('self._key_weight,self._gate_weight','self._gate_weight,self._key_weight'),ADAPTER),(SOURCE,ADAPTER.replace('prepare_nonempty_keypool_inputs','unsafe_inputs'))):
            with self.assertRaises(ValueError):validate_indexer_source_calls(source,adapter)


if __name__=='__main__':unittest.main(verbosity=2)
