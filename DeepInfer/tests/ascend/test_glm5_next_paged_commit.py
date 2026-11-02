# SPDX-License-Identifier: Apache-2.0
"""Raw-byte scalar oracles for bounded synchronous CPU cache writes."""
import unittest
from unittest.mock import patch
import torch
from deepinfer.model_executor.models.glm5_next.cache import DSACacheLayout
from deepinfer.model_executor.models.glm5_next.cann_cache import CANNCacheGeometry
import deepinfer.model_executor.models.glm5_next.paged_commit as commit


def fixture(starts=(3,127),lengths=(3,5),block=128,padding=2):
    layout=DSACacheLayout(3,2,4,8,block);g=CANNCacheGeometry(layout,(4+padding)*2*2*4)
    pages,tail_pages=6,3;n=len(starts);tables=torch.tensor([[2,1,0],[3,4,5]][:n],dtype=torch.int32)
    if not n:tables=torch.empty((0,3),dtype=torch.int32)
    ids=torch.tensor([2,0][:n],dtype=torch.int64)
    raw=[torch.full((pages*g.main_page_bytes,),0x5a,dtype=torch.uint8),torch.full((pages*g.index_page_bytes,),0x5a,dtype=torch.uint8),torch.full((tail_pages*g.tail_page_bytes,),0x5a,dtype=torch.uint8)]
    total=sum(lengths);complete=sum((s+l)//4-s//4 for s,l in zip(starts,lengths))
    latent=(torch.arange(total*3).reshape(total,3).float()/8).bfloat16()
    compressed=(torch.arange(complete*2).reshape(complete,2).float()/16+1).bfloat16()
    scratch=torch.arange((1+2*n)*4*4).reshape(1+2*n,4,4).float()+100
    args=dict(num_main_pages=pages,num_tail_pages=tail_pages,logical_table=tables,starts=torch.tensor(starts,dtype=torch.int64),lengths=torch.tensor(lengths,dtype=torch.int64),tail_page_ids=ids,latent=latent,compressed=compressed,tail_state=scratch)
    return g,raw,args


def raw_bytes(t):return t.contiguous().view(torch.uint8).numpy().tobytes()


def oracle(g,raw,a):
    expected=[bytearray(raw_bytes(t)) for t in raw];packed=pool_row=0;n=len(a['starts']);B=g.layout.logical_block_size;p=g.layout.pool_size
    for request,(start,length) in enumerate(zip(a['starts'].tolist(),a['lengths'].tolist())):
        for position in range(start,start+length):
            page=int(a['logical_table'][request,position//B]);offset=(page*B+position%B)*g.layout.latent_dim*2
            data=raw_bytes(a['latent'][packed]);expected[0][offset:offset+len(data)]=data;packed+=1
            if (position+1)%p==0:
                offset=(page*(B//p)+(position%B)//p)*g.layout.index_dim*2;data=raw_bytes(a['compressed'][pool_row]);expected[1][offset:offset+len(data)]=data;pool_row+=1
        end=start+length
        if length:
            group=request+1+(n if start%p+length>=p else 0)
            for position in range(end//p*p,end):
                offset=int(a['tail_page_ids'][request])*g.tail_page_bytes+(position%p)*2*g.layout.index_dim*4
                data=raw_bytes(a['tail_state'][group,position%p]);expected[2][offset:offset+len(data)]=data
    return [bytes(x) for x in expected]


class PagedCommitTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_three_buffers_match_independent_raw_byte_scatter(self):
        for B in (128,256):
            g,raw,a=fixture(block=B);before=[t.clone() for t in raw];expected=oracle(g,raw,a)
            batch=commit.stage_dsa_writes(*raw,g,**a)
            for x,y in zip(raw,before):self.assertTrue(torch.equal(x,y))
            result=batch.commit();self.assertEqual(result['rows'],{'main':8,'index':3,'tail':2})
            self.assertEqual([raw_bytes(x) for x in raw],expected)

    def test_all_pool_offsets_empty_complete_and_block_boundaries(self):
        for start in (0,1,2,3,4,5,126,127,128,255):
            for length in (0,1,3,4,7):
                g,raw,a=fixture((start,),(length,),block=128);expected=oracle(g,raw,a)
                commit.stage_dsa_writes(*raw,g,**a).commit();self.assertEqual([raw_bytes(t) for t in raw],expected)

    def test_late_synchronous_failure_restores_all_attempted_rows(self):
        g,raw,a=fixture();saved=[raw_bytes(t) for t in raw];batch=commit.stage_dsa_writes(*raw,g,**a);calls=[]
        def fail_after_write(target,ids,values):
            target.index_copy_(0,ids,values);calls.append(len(ids))
            if len(calls)==2:raise RuntimeError('injected after index cache write')
        with patch.object(commit,'_write_rows',side_effect=fail_after_write):
            with self.assertRaisesRegex(RuntimeError,'injected'):batch.commit()
        self.assertEqual(calls,[8,3]);self.assertEqual([raw_bytes(t) for t in raw],saved);self.assertEqual(batch.summary()['state'],'aborted')
        with self.assertRaises(ValueError):batch.commit()

    def test_stale_destination_rejects_before_any_other_write(self):
        g,raw,a=fixture();batch=commit.stage_dsa_writes(*raw,g,**a)
        raw[2][int(a['tail_page_ids'][0])*g.tail_page_bytes]^=1;saved=[raw_bytes(t) for t in raw]
        with patch.object(commit,'_write_rows',side_effect=AssertionError('must reject before write')):
            with self.assertRaisesRegex(ValueError,'changed after staging'):batch.commit()
        self.assertEqual([raw_bytes(t) for t in raw],saved)

    def test_staged_payloads_are_private_and_terminal_abort_does_not_write(self):
        g,raw,a=fixture();expected=oracle(g,raw,a);batch=commit.stage_dsa_writes(*raw,g,**a)
        a['latent'].zero_();a['compressed'].zero_();a['tail_state'].zero_();a['logical_table'].zero_();a['tail_page_ids'].zero_()
        batch.commit();self.assertEqual([raw_bytes(t) for t in raw],expected)
        with self.assertRaises(ValueError):batch.commit()
        g,raw,a=fixture();saved=[raw_bytes(t) for t in raw];batch=commit.stage_dsa_writes(*raw,g,**a);batch.abort()
        with self.assertRaises(ValueError):batch.commit()
        self.assertEqual([raw_bytes(t) for t in raw],saved)

    def test_zero_requests_and_unassigned_inactive_request_are_noops(self):
        for starts,lengths in (((),()),((17,),(0,))):
            g,raw,a=fixture(starts,lengths);a['tail_page_ids'].fill_(-1);a['logical_table'].fill_(-1);saved=[raw_bytes(t) for t in raw]
            batch=commit.stage_dsa_writes(*raw,g,**a)
            with patch.object(commit,'_write_rows',side_effect=AssertionError('empty write dispatched')):batch.commit()
            self.assertEqual([raw_bytes(t) for t in raw],saved);self.assertEqual(batch.summary()['payload_bytes'],0)

    def test_alias_missing_page_and_out_of_range_ids_reject_without_write(self):
        for mutation in ('alias','missing','range','tail_alias'):
            g,raw,a=fixture();saved=[raw_bytes(t) for t in raw]
            if mutation=='alias':a['logical_table'][1,0]=2
            elif mutation=='missing':a['logical_table'][0,0]=-1
            elif mutation=='range':a['logical_table'][0,0]=6
            else:a['tail_page_ids'][1]=2
            with self.assertRaises(ValueError):commit.stage_dsa_writes(*raw,g,**a)
            self.assertEqual([raw_bytes(t) for t in raw],saved)

    def test_payload_dtype_shape_nan_and_metadata_budgets_reject(self):
        for mutation in ('dtype','shape','nan','tokens','bytes','table','overflow','lookahead'):
            g,raw,a=fixture();saved=[raw_bytes(t) for t in raw]
            if mutation=='dtype':a['latent']=a['latent'].float()
            elif mutation=='shape':a['compressed']=a['compressed'][:1]
            elif mutation=='nan':a['tail_state'][3,0,0]=float('nan')
            elif mutation=='tokens':a['max_tokens']=7
            elif mutation=='bytes':a['max_payload_bytes']=1
            elif mutation=='table':a['max_table_entries']=1
            elif mutation=='overflow':a['starts'][0]=2**31-1
            else:
                g=CANNCacheGeometry(DSACacheLayout(3,2,4,8,128,1),96)
            with self.assertRaises(ValueError):commit.stage_dsa_writes(*raw,g,**a)
            self.assertEqual([raw_bytes(t) for t in raw],saved)

    def test_nan_before_images_and_unused_scratch_preserve_exact_bits(self):
        g,raw,a=fixture()
        for t in raw:t.fill_(255)
        # Request1 ends on a complete pool, so all its output tail is unused.
        a['tail_state'][4].fill_(float('nan'));expected=oracle(g,raw,a)
        commit.stage_dsa_writes(*raw,g,**a).commit();self.assertEqual([raw_bytes(t) for t in raw],expected)

    def test_overlapping_backing_rejects_before_mutation(self):
        g,raw,a=fixture();raw[1]=raw[0][:len(raw[1])];saved=[raw_bytes(t) for t in raw]
        with self.assertRaisesRegex(ValueError,'must not overlap'):commit.stage_dsa_writes(*raw,g,**a)
        self.assertEqual([raw_bytes(t) for t in raw],saved)

    def test_native_or_non_cpu_backing_rejects_before_any_device_action(self):
        g,raw,a=fixture();raw[0]=torch.empty(1,device='meta',dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError,'CPU-only'):commit.stage_dsa_writes(*raw,g,**a)


if __name__=='__main__':unittest.main(verbosity=2)
