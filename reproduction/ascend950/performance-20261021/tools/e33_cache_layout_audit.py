#!/usr/bin/env python3
"""Reconcile cache backing allocations independently from tensor view sizes."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    input_path=ROOT/'evidence/E33-memory-inputs.json.gz'
    inputs=json.loads(gzip.decompress(input_path.read_bytes()))['records'];rows=[];flat=[]
    for item in inputs:
        s=item['snapshot'];tag=item['framework'];common=dict(framework=tag,run_id=item['run_id'],rank=item['rank'],boundary=item['boundary'])
        if tag=='di':
            cfg=s['cache_layout'];tensors=[t for t in s['tensors'] if t['category'] in ['kv_cache','state_cache']]
            assert len(tensors)==2
            nblocks=cfg['num_gpu_blocks'];page=cfg['cache_config.bytes_per_block'];state_slots=cfg['cache_config.num_state_seqs'];state_page=cfg['cache_config.bytes_of_state_per_seqs']
            kv=next(t for t in tensors if t['category']=='kv_cache');state=next(t for t in tensors if t['category']=='state_cache')
            payload=sum(x[2] for x in cfg['kvcache_layout']);assert payload==3*64*(288+68)+128*(288+68)
            assert page==payload+128 and kv['storage_nbytes']==nblocks*page and state['storage_nbytes']==state_slots*state_page
            detail=dict(backing_storage_count=2,tensor_views=2,kv_blocks=nblocks,kv_page_bytes=page,kv_payload_bytes_per_page=payload,
                        final_page_padding_bytes=page-payload,kv_backing_bytes=kv['storage_nbytes'],state_slots=state_slots,state_slot_bytes=state_page,state_backing_bytes=state['storage_nbytes'],
                        total_cache_backing_bytes=kv['storage_nbytes']+state['storage_nbytes'])
            for t in tensors:flat.append(dict(**common,storage_address=t['storage_ptr'],storage_bytes=t['storage_nbytes'],view_name=t['name'],shape=json.dumps(t['shape']),stride=json.dumps(t['stride']),dtype=t['dtype'],data_offset_bytes=0))
        else:
            nblocks=s['kv_cache_config']['num_blocks']
            groups=collections.defaultdict(list)
            for t in s['kv']:groups[(t['storage_data_ptr'],t['storage_nbytes'])].append(t)
            assert len(groups)==47 and len(s['kv'])==59
            pools=[];rings=[]
            for (ptr,size),ts in sorted(groups.items()):
                record=dict(storage_address=ptr,storage_bytes=size,view_count=len(ts),views=[])
                for t in ts:
                    view=dict(name=t['name'],shape=t['shape'],stride=t['stride'],dtype=t['dtype'],element_size=t['element_size'],data_offset_bytes=t['data_ptr']-ptr)
                    record['views'].append(view)
                    flat.append(dict(**common,storage_address=ptr,storage_bytes=size,view_name=t['name'],shape=json.dumps(t['shape']),stride=json.dumps(t['stride']),dtype=t['dtype'],data_offset_bytes=t['data_ptr']-ptr))
                if len(ts)==1:
                    assert ts[0]['shape']==[129,128,1,544] and size==129*128*544;rings.append(record)
                else:
                    assert len(ts)==4 and {t['shape'][0] for t in ts}=={nblocks}
                    page=ts[0]['stride'][0]*ts[0]['element_size'];assert size==nblocks*page
                    record.update(pages=nblocks,page_bytes=page);pools.append(record)
            assert len(pools)==4 and len(rings)==43
            assert collections.Counter(p['page_bytes'] for p in pools)=={65536:3,58368:1}
            pooled_bytes=sum(x['storage_bytes'] for x in pools);ring_bytes=sum(x['storage_bytes'] for x in rings)
            assert pooled_bytes==nblocks*(3*65536+58368) and ring_bytes==43*129*128*544
            detail=dict(backing_storage_count=47,tensor_views=59,shared_pool_blocks=nblocks,shared_page_bytes=3*65536+58368,
                        shared_pool_backing_bytes=pooled_bytes,swa_ring_backing_bytes=ring_bytes,total_cache_backing_bytes=pooled_bytes+ring_bytes,shared_pools=pools,swa_ring_count=43,swa_ring_shape=[129,128,1,544])
        rows.append(dict(**common,**detail))
    result=dict(input_path=str(input_path.relative_to(ROOT)),input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),records=rows,checks=dict(all64_snapshot_formula_checks=True),
                format_evidence=dict(di_swa_bytes_per_token=528,v_swa_bytes_per_token=544,di_compressed_kv_bytes_per_token=288,v_compressed_kv_bytes_per_token=320,
                                     di_source='evidence/E33-di-installed-source/deepinfer/model_executor/layers/tilelang_ops/cache_common.py:117',
                                     v_source='evidence/source-official-ds41/ascend-source/vllm_ascend/core/deepseek_v41.py:244; observed cache widths and allocation formula crosschecked in all64 snapshots.',
                                     di_swa_scale_format='UE8M0',di_compressed_kv_scale_format='E4M3',v_swa_and_compressed_kv_scale_format='BF16'),
                limits=['Allocated backing capacity is not active logical content or measured memory traffic.',
                        'Shared views can describe the same storage for different cache resource types; their tensor nbytes must not be summed.',
                        'Block counts differ and refer to heterogeneous resource layouts; they are not directly comparable request capacities.',
                        'The installed worker imports core.deepseek_v41; an older models/deepseek_v41/cache_config.py copy was not used as proof of current allocation policy.',
                        'Storage-format byte widths differ; no precision-equivalence or speedup percentage is inferred.'])
    (ROOT/'data/E33-cache-layout-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'data/E33-cache-layout-views.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    for tag in ['di','vllm']:
        r=next(x for x in rows if x['framework']==tag and x['rank']==0 and '-NORMAL-' in x['run_id'] and 'start' in x['boundary'])
        print(tag,{k:v for k,v in r.items() if 'bytes' in k and isinstance(v,int)})
    print('64 real snapshots reconciled')


if __name__=='__main__':main()
