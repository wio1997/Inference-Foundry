#!/usr/bin/env python3
"""Disjoint virtual-storage and memory-counter accounting for frozen snapshots."""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
from storage_intervals import partition_intervals

ROOT=Path(__file__).resolve().parents[1]
ORDER=['native_kv','native_state','native_ep_comm','shared_main_draft_torch',
       'kv_torch','main_weights_torch','main_buffers_torch','draft_torch',
       'vision_torch','other_torch']


def translated(item):
    s=item['snapshot'];records=[]
    if item['framework']=='di':
        for t in s['tensors']:
            if not t['device'].startswith('npu:'):continue
            cat={'kv_cache':'native_kv','state_cache':'native_state','comm_symm_buffer':'native_ep_comm'}.get(t['category'])
            if cat is None:
                assert t['category'] in ['weights','scales','model_buffers','engram_lookup'],t['category']
                cat=('draft_torch' if t['name'].startswith('MTP_module.') else
                     'vision_torch' if t['name'].startswith('vision_language_tuner.') else
                     'main_weights_torch' if t['category']=='weights' else 'main_buffers_torch')
            records.append(dict(label=t['name'],category=cat,address=t['storage_ptr'],size_bytes=t['storage_nbytes']))
        dm=s['device_memory'];allocated=dm['torch_allocated_bytes'];reserved=dm['torch_reserved_bytes']
        free=dm['free_bytes'];total=dm['total_bytes'];cpu=s['cpu_memory']
        notes=' '.join(s['notes'])
        recv={int(n) for n in re.findall(r'num_gpu_bytes=(\d+)',notes)}
        rdma={int(n) for n in re.findall(r'num_rdma_storage_bytes=(\d+)',notes)}
        assert len(recv)==len(rdma)==1
        native_extra=dict(native_engram_receive_requested_capacity=recv.pop(),native_engram_storage_requested_capacity=rdma.pop())
        bounded=s['tensor_metadata_bounded'];errors=s['errors']
        cpu_rss=cpu['rss_bytes'];affinity=cpu['affinity'];peak=dm.get('torch_peak_allocated_bytes')
    else:
        raw=[]
        for kind in ['model','draft']:
            for t in s[kind]['entries']:
                if not t['device'].startswith('npu:'):continue
                cat=('draft_torch' if kind=='draft' else 'main_weights_torch' if t['kind']=='param' else 'main_buffers_torch')
                raw.append(dict(label=kind+':'+t['name'],category=cat,address=t['storage_data_ptr'],size_bytes=t['storage_nbytes']))
        for kind,cat in [('kv','kv_torch'),('graph_comm','other_torch'),('other_tensors','other_torch')]:
            for t in s[kind]:
                if t['device'].startswith('npu:'):
                    raw.append(dict(label=kind+':'+t['name'],category=cat,address=t['storage_data_ptr'],size_bytes=t['storage_nbytes']))
        cats=collections.defaultdict(set)
        for t in raw:cats[(t['address'],t['size_bytes'])].add(t['category'])
        for t in raw:
            seen=cats[(t['address'],t['size_bytes'])]
            if 'draft_torch' in seen and any(v.startswith('main_') for v in seen):t['category']='shared_main_draft_torch'
        records=raw
        dm=s['torch'];allocated=dm['allocated'];reserved=dm['reserved'];free,total=dm['mem_get_info'];cpu=s['cpu']
        native_extra={};bounded={'truncated':s['summary']['truncated']};errors=s['errors']
        rss=cpu.get('proc_status',{}).get('VmRSS')
        cpu_rss=int(rss.split()[0])*1024 if rss is not None else cpu.get('rss_bytes')
        affinity=cpu.get('affinity',[]);peak=dm.get('max_allocated')
    partition=partition_intervals(records,ORDER)
    used=total-free
    categories=partition['bytes_by_category']
    visible_torch=sum(v for k,v in categories.items() if k.endswith('_torch'))
    known_native=sum(v for k,v in categories.items() if k.startswith('native_'))+sum(native_extra.values())
    components={k:v for k,v in categories.items() if v}
    components.update(native_extra)
    components.update(unnamed_torch_allocated=allocated-visible_torch,
                      torch_reserved_unallocated=reserved-allocated,
                      native_driver_unaccounted=used-reserved-known_native)
    assert sum(components.values())==used
    checks=dict(no_negative_components=all(v>=0 for v in components.values()),
                visible_torch_within_allocator=visible_torch<=allocated,
                no_truncation=not bounded.get('truncated'),no_snapshot_errors=not errors,
                sum_conserved=sum(components.values())==used)
    return dict(framework=item['framework'],run_id=item['run_id'],rank=item['rank'],boundary=item['boundary'],
                source=item['source'],source_sha256=item.get('source_sha256'),source_line=item.get('source_line'),
                source_line_sha256=item.get('source_line_sha256'),checks=checks,
                total_bytes=total,free_bytes=free,device_used_bytes=used,
                torch_allocated_bytes=allocated,torch_reserved_bytes=reserved,torch_peak_allocated_bytes=peak,
                reachable_storage_union_bytes=partition['union_bytes'],visible_torch_bytes=visible_torch,
                components=components,aliases=partition,records=records,
                cpu_rss_bytes=cpu_rss,cpu_affinity=affinity,cpu_metadata=cpu,
                limits=['Storage addresses are process-local virtual allocations; no physical HBM-bank mapping is available.',
                        'Shared main/draft ownership is identified only when both names are present with the same storage interval.',
                        'Unnamed Torch allocations include graph/workspace/temporary allocations that are not separately measured.',
                        'Native requested capacities are source-backed allocation requests, not exact allocator or driver overhead.',
                        'The native/driver remainder includes padding, HCCL/runtime/driver overhead and any other device users.',
                        'CPU RSS does not measure Engram host/offload capacity, pinned memory, or all native mappings.'])


def main():
    p=ROOT/'evidence/E33-memory-inputs.json.gz'
    x=json.loads(gzip.decompress(p.read_bytes()));rows=[translated(v) for v in x['records']]
    assert len(rows)==64
    checks=all(all(v['checks'].values()) for v in rows)
    result=dict(input_path=str(p),input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                snapshot_count=len(rows),all_reconciliation_checks_passed=checks,records=rows,
                source_notes=dict(di_native_kv_state='evidence/E33-di-installed-source/csrc/asc_utils.cpp:88; cache_engine.py _alloc_gpu_uint8',
                                  di_native_engram='evidence/E33-di-installed-source/vendor/DeepEP-Ascend/csrc/buffers/engram.hpp:34',
                                  di_native_ep='evidence/E33-di-installed-source/deepinfer/model_executor/parallel_utils/userbuffer.py:183',
                                  v_engram_offload='evidence/E33-feature-source/feature-source/ascend-models/models/deepseek_v41/engram_offload.py:676'),
                scope='Quiescent boundary allocation ledger. Conservation and explicit remainder do not prove complete allocation attribution or causality.')
    target=ROOT/'data/E33-memory-ledger.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    flat=[]
    for v in rows:
        for k,n in v['components'].items():flat.append(dict(framework=v['framework'],run_id=v['run_id'],rank=v['rank'],boundary=v['boundary'],component=k,bytes=n,GiB=n/2**30))
    with (ROOT/'data/E33-memory-ledger.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(dict(snapshot_count=len(rows),all_checks_passed=checks)))
    for run in sorted({v['run_id'] for v in rows}):
        selected=[v for v in rows if v['run_id']==run and 'start' in v['boundary']]
        assert len(selected)==8
        print(run,'used_GiB',[round(v['device_used_bytes']/2**30,3) for v in selected],
              'remainder_GiB',[round(v['components']['native_driver_unaccounted']/2**30,3) for v in selected],
              'partial_overlap_pairs',[len(v['aliases']['partial_overlap_pairs']) for v in selected])
    assert checks


if __name__=='__main__':main()
