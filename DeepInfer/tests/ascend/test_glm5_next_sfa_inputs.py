# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU input-role checks;no binary/operator or allocation ownership."""
from dataclasses import replace
import unittest
from unittest.mock import patch
import torch
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry
from deepinfer.model_executor.models.glm5_next.private_pages import plan_private_dsa_pages,prepare_private_dsa_cpu,gather_private_latent_cpu
from deepinfer.model_executor.models.glm5_next.sfa_inputs import NoPESFAContract,SFAHeadPartition,NoPESFAInputPlan,prepare_sfa_packet_cpu


def fixture(start=127,count=5,block=128,world=16,rank=0):
    g=CANNCacheGeometry(DSACacheLayout(512,128,4,2048,block),9*256*4);n=24
    table=tuple(range(1,(start+block-1)//block+1));occupied=frozenset(table)
    touched=(start+count-1)//block-start//block+1 if count else 0;ids=tuple(i for i in reversed(range(n)) if i not in occupied)[:touched]
    p=plan_private_dsa_pages(start,count,table,ids,geometry=g,num_main_pages=n,num_tail_pages=5,occupied_main=occupied,occupied_tail=frozenset({0}),tail_source=0,tail_private=3 if count else None)
    main=torch.zeros(n*g.main_page_bytes,dtype=torch.uint8);index=torch.zeros(n*g.index_page_bytes,dtype=torch.uint8);tail=torch.zeros(5*g.tail_page_bytes,dtype=torch.uint8)
    mv=main.view(torch.bfloat16).reshape(n,block,512)
    for logical,page in enumerate(table):mv[page]=(logical*block+torch.arange(block)[:,None]).expand(block,512).bfloat16()/256
    appended=(torch.arange(start,start+count)[:,None].expand(count,512).float()/256+1).bfloat16()
    stage=prepare_private_dsa_cpu(main,index,tail,p,g,appended,torch.zeros(p.new_pools,128,dtype=torch.bfloat16),torch.zeros(4,256) if count else None)
    heads=SFAHeadPartition(64,world,rank,tuple(range(rank*(64//world),(rank+1)*(64//world))))
    plan=NoPESFAInputPlan('fixture',3,2,p,heads,NoPESFAContract(64,256,512,4,2048,'a'*64))
    query=torch.full(plan.query_shape,.125,dtype=torch.bfloat16);indices=torch.full(plan.indices_shape,-1,dtype=torch.int32)
    for row,pos in enumerate(range(start,start+count)):
        tail_start=(pos+1)//4*4;selected=list(range(min(tail_start,2048)))+list(range(tail_start,pos+1));indices[row,0,:len(selected)]=torch.tensor(selected,dtype=torch.int32)
    return plan,g,main,stage,query,indices,appended


def prepare(plan,main,stage,query,indices,**extra):
    args=dict(expected_identity=plan.identity,index_namespace='logical_token');args.update(extra)
    return prepare_sfa_packet_cpu(plan,main,stage,query,indices,**args)


class SFAInputTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_source_roles_alias_cache_and_raw_lengths_without_leading_zero(self):
        p,g,raw,s,q,ids,_=fixture();before=raw.clone();packet=prepare(p,raw,s,q,ids);kw=packet.source_kwargs_cpu()
        self.assertIs(kw['key'],kw['value']);self.assertIs(kw['key'],packet.cache);self.assertFalse(packet.native_ABI_validated)
        self.assertEqual(kw['actual_seq_lengths_query'].tolist(),[5]);self.assertEqual(kw['actual_seq_lengths_kv'].tolist(),[132])
        self.assertEqual((kw['layout_query'],kw['layout_kv'],kw['sparse_mode'],kw['attention_mode']),('TND','PA_BSND',3,2))
        self.assertEqual(packet.scale,1/16);self.assertNotEqual(packet.scale,512**-.5)
        self.assertEqual(packet.cache.shape,(24,128,1,512));self.assertTrue(torch.equal(raw,before))
        self.assertEqual(packet.block_table.tolist(),[list(p.candidate.kernel_read_table)])
        self.assertTrue(all(kw[n].dtype==torch.int32 for n in ('sparse_indices','block_table','actual_seq_lengths_query','actual_seq_lengths_kv')))

    def test_candidate_installation_and_kernel256_page_expansion(self):
        p,g,raw,s,q,ids,append=fixture(start=255,count=4,block=256,rank=15);packet=prepare(p,raw,s,q,ids)
        self.assertEqual(packet.query.shape,(4,4,512));self.assertEqual(p.heads.head_ids,(60,61,62,63));self.assertEqual(packet.cache.shape,(48,128,1,512))
        chosen=torch.tensor([0,254,255,256,258]);expected=gather_private_latent_cpu(raw,s,g,chosen);actual=[]
        for token in chosen.tolist():actual.append(packet.cache[packet.block_table[0,token//128],token%128,0])
        self.assertTrue(torch.equal(torch.stack(actual),expected));self.assertTrue(torch.equal(expected[2:],append[[0,1,3]]))
        self.assertEqual(packet.block_table.tolist(),[[i for page in p.candidate.read_table for i in (page*2,page*2+1)]])

    def test_complete_partitions_reject_arbitrary_samples_and_wrong_rank(self):
        for kwargs in ((64,16,0,(0,63)),(64,16,0,(4,5,6,7)),(64,3,0,(0,1)),(64,16,16,(64,65,66,67)),(64,32,0,(True,1))):
            with self.assertRaises(ValueError):SFAHeadPartition(*kwargs)
        self.assertEqual(SFAHeadPartition(64,1,0,tuple(range(64))).local_heads,64)
        self.assertEqual(SFAHeadPartition(64,32,31,(62,63)).local_heads,2)

    def test_stale_request_layer_revision_table_partition_and_foreign_owner(self):
        p,g,raw,s,q,ids,_=fixture();token=p.identity
        changed=replace(p.candidate,private_ids=(22,21),read_table=(22,21))
        other=NoPESFAInputPlan(p.request_id,p.layer_id,p.revision,p.candidate,p.heads,p.contract)
        wrong=(replace(token,revision=3),replace(token,request_id='B'),replace(token,layer_id=7),replace(p,candidate=changed).identity,replace(p,heads=SFAHeadPartition(64,16,1,(4,5,6,7))).identity,other.identity)
        with patch.object(torch.Tensor,'clone',side_effect=AssertionError('copied stale packet')):
            for stamp in wrong:
                with self.assertRaisesRegex(ValueError,'stale/foreign'):prepare(p,raw,s,q,ids,expected_identity=stamp)

    def test_stage_plan_or_query_original_QK_and_indices_shape_reject(self):
        p,g,raw,s,q,ids,_=fixture();other=replace(p.candidate,private_ids=(22,21),read_table=(22,21))
        for st,query,ix in ((replace(s,plan=other),q,ids),(s,q[:,:,:256],ids),(s,q,ids[:,0]),(s,q,ids.long()),(s,q.float(),ids)):
            with self.assertRaises(ValueError):prepare(p,raw,st,query,ix)
        for namespace in ('pool','physical','expanded_token'):
            with self.assertRaises(ValueError):prepare(p,raw,s,q,ids,index_namespace=namespace)

    def test_token_causality_packed_unique_count_and_tail_policy(self):
        p,g,raw,s,q,ids,_=fixture(start=3,count=2)
        for kind in ('future','duplicate','hole','tail','short','negative'):
            bad=ids.clone()
            if kind=='future':bad[0,0,0]=4
            elif kind=='duplicate':bad[0,0,1]=0
            elif kind=='hole':bad[0,0,1]=-1
            elif kind=='tail':bad[1,0,4]=3
            elif kind=='short':bad[0,0,3]=-1
            else:bad[0,0,4]=-2
            with self.subTest(kind=kind),self.assertRaises(ValueError):prepare(p,raw,s,q,bad)

    def test_byte_budget_before_clone_and_cpu_guard(self):
        p,g,raw,s,q,ids,_=fixture()
        with patch.object(torch.Tensor,'clone',side_effect=AssertionError('clone before byte guard')):
            for kw in ({'max_cache_bytes':1},{'max_output_bytes':1}):
                with self.assertRaisesRegex(ValueError,'byte budget'):prepare(p,raw,s,q,ids,**kw)
        with self.assertRaisesRegex(ValueError,'CPU-only'):prepare(p,raw,s,torch.empty(q.shape,device='meta',dtype=torch.bfloat16),ids)
        with self.assertRaises(ValueError):prepare(p,raw[1:],s,q,ids)

    def test_unselected_nan_allowed_selected_nan_and_query_nan_reject(self):
        p,g,raw,s,q,ids,_=fixture();live=raw.clone()
        with torch.inference_mode():live.view(torch.bfloat16).reshape(24,128,512)[20].fill_(float('nan'))
        packet=prepare(p,live,s,q,ids);self.assertTrue(bool(torch.isnan(packet.cache[20]).all()))
        bad=s.main.clone()
        with torch.inference_mode():bad.view(torch.bfloat16).reshape(len(p.candidate.private_ids),128,512)[0,127]=float('nan')
        with self.assertRaisesRegex(ValueError,'selected candidate'):prepare(p,raw,replace(s,main=bad),q,ids)
        with torch.inference_mode():q[0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'query'):prepare(p,raw,s,q,ids)

    def test_returned_packet_owns_source_query_indices_cache_and_metadata(self):
        p,g,raw,s,q,ids,_=fixture();packet=prepare(p,raw,s,q,ids);saved=packet.cache.clone();saved_q=packet.query.clone();saved_ids=packet.sparse_indices.clone()
        with torch.inference_mode():raw.zero_();s.main.zero_();q.zero_();ids.fill_(-1)
        self.assertTrue(torch.equal(packet.cache,saved));self.assertTrue(torch.equal(packet.query,saved_q));self.assertTrue(torch.equal(packet.sparse_indices,saved_ids))
        self.assertEqual(packet.identity,p.identity)

    def test_empty_skips_all_tensor_reads_and_clone(self):
        p,*_=fixture(start=3,count=0)
        with patch.object(torch.Tensor,'clone',side_effect=AssertionError('empty clone')):
            self.assertIsNone(prepare(p,None,None,None,None))
        self.assertTrue(p.skip_operator)

    def test_profile_sink_dimension_evidence_and_int32_plan_guards(self):
        p,*_=fixture()
        for kw in ({'profile':'smla'},{'sink_policy':'1.0'},{'latent_dim':256},{'reference_source_sha256':'missing'}):
            with self.assertRaises(ValueError):replace(p.contract,**kw)
        for kw in ({'layer_id':0},{'revision':True},{'revision':-1},{'request_id':''}):
            with self.assertRaises(ValueError):replace(p,**kw)
        self.assertEqual(p.query_shape,(5,4,512));self.assertEqual(p.indices_shape,(5,1,2051))


if __name__=='__main__':unittest.main(verbosity=2)
