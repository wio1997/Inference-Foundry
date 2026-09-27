#!/usr/bin/env python3
"""Reduce Run576 to conditional route/operand census, never compulsory traffic."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from loop080_postpark_validate_run576 import validate


ROOT=Path('/data/wio/Inference_Foundry')
ARM=ROOT/'evidence/20260928_loop080_bound/run576/live/b'
OUT=ROOT/'evidence/20260928_loop080_bound/run576/operand_census.json'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def require(ok,why):
    if not ok:raise ValueError(why)


def main():
    capture_paths=[ARM/'capture'/f'rank{rank}_cohort5.json' for rank in range(8)]
    require(all(p.exists() for p in capture_paths),'missing rank capture')
    rows=[json.loads(p.read_text()) for p in capture_paths]
    admission=validate(rows)
    joined=json.loads((ARM/'joined_admission.json').read_text())
    capture=json.loads((ARM/'capture_admission.json').read_text())
    client=json.loads((ARM/'client_admission.json').read_text())
    require(joined['status']=='diagnostic_capture_client_runtime_join_pass' and
            joined['cycle']==admission['cycle']==capture['cycle'] and
            joined['run_tag']==admission['run_tag']==capture['run_tag'] and
            joined['request_ids']==admission['request_ids']==capture['request_ids'] and
            admission['total_cycles']==capture['total_cycles'],
            'client/runtime/capture identity join')
    require(client['status']=='client_two_phase_admitted' and
            client['measured_summary']['n']==48 and
            client['measured_summary']['success']==48 and
            client['measured_summary']['fail']==0,
            'measured client admission')
    matched=[]
    for req in admission['request_ids']:
        hit=[row for row in client['request_index'] if
             req==row['response_id'] or req.startswith(row['response_id']+'-')]
        require(len(hit)==1 and hit[0]['phase']=='measured',
                'client measured request identity')
        matched.append(hit[0])
    require({row['dataset_index'] for row in matched}==set(range(12)) and
            len({row['response_id'] for row in matched})==12,
            'first measured cohort client set')
    runtime_paths=[ARM/'runtime'/f'rank{rank}_cohort5.json' for rank in range(8)]
    require(all(p.exists() for p in runtime_paths),'all8 Runtime rows missing')
    for rank,p in enumerate(runtime_paths):
        runtime=json.loads(p.read_text())
        require(runtime['rank']==rank and runtime['cohort']==5 and
                runtime['req_ids']==admission['request_ids'] and
                runtime['cycles']==admission['total_cycles'] and
                runtime['host_mirror_exact'] is True and
                runtime['target_graph_mode']=='FULL' and
                runtime['generated_output_counts']==[1024]*12,
                'all8 Runtime identity/output')
    cleanup=(ARM.parent/'cleanup_status.txt').read_text()
    require(all(f'{name}=0' in cleanup.splitlines() for name in (
        'run_exit','stop_exit','stop_verify_exit','restore_exit','source_sha_exit',
        'source_compare_exit','script_sha_exit','script_compare_exit','final_exit')),
        'controller cleanup not all zero')
    for name in ('source','scripts'):
        require((ARM.parent/f'{name}_before.sha256').read_bytes()==
                (ARM.parent/f'{name}_after.sha256').read_bytes(),
                f'{name} before/after SHA manifest differs')
    cycle=str(admission['cycle'])
    records=[r['records'][cycle] for r in rows]
    result=dict(status='scoped_conditional_operand_census',
                provenance={str(p.relative_to(ROOT)):digest(p) for p in
                    [*capture_paths,*runtime_paths,ARM/'capture_admission.json',
                     ARM/'joined_admission.json',ARM/'client_admission.json',
                     ARM.parent/'cleanup_status.txt',ARM.parent/'patch_check.json',
                     ARM.parent/'install.json',ARM.parent/'restore.json',
                     ARM.parent/'source_before.sha256',ARM.parent/'source_after.sha256',
                     ARM.parent/'scripts_before.sha256',ARM.parent/'scripts_after.sha256']},
                cohort=5,selected_cycle=admission['cycle'],total_cycles=admission['total_cycles'],
                parked_slots=admission['parked_slots'],
                algorithm_configuration_edit=False,
                trajectory_noninterference_proven=False,
                current_formal_tps=571.681,
                formal_W0_transfer=False,latency_or_tps_inference=False,
                strict_numeric_bounds_unchanged=True,models={})
    for model in ('target','draft'):
        per_rank=[]
        all_routes=sorted(records[0][model],key=lambda x:x['ordinal'])
        layers=len(all_routes);rows_per_layer=len(all_routes[0]['ids'])
        unique_global=[len(set(e for row in layer['ids'] for e in row))
                       for layer in all_routes]
        for rank,record in enumerate(records):
            layer_rows=sorted(record[model],key=lambda x:x['ordinal'])
            allocated={};selected_slice_estimate=0;owned_counts=[]
            for layer in layer_rows:
                owned=sum(int(count>0) for count in layer['group'])
                owned_counts.append(owned)
                for name,values in layer['weights'].items():
                    if values is None:continue
                    for value in values:
                        ptr=value['storage_ptr'];nbytes=value['storage_nbytes']
                        key=str(ptr)
                        if key in allocated:
                            require(allocated[key]['bytes']==nbytes,'aliased operand size mismatch')
                        else:
                            allocated[key]=dict(bytes=nbytes,npu_format=value['npu_format'],
                                                dtype=value['dtype'],operand=name,
                                                shape=value['shape'])
                        # Pure arithmetic selected-expert slice estimate for
                        # array-like first dimension; not a verified native
                        # physical read or compulsory HBM traffic.
                        require(value['shape'][0]==32 and nbytes%32==0,
                                '32-local-expert storage geometry')
                        selected_slice_estimate+=owned*(nbytes//32)
            per_rank.append(dict(rank=rank,
                routed_incidence=sum(sum(layer['group']) for layer in layer_rows),
                nonempty_local_expert_layer_pairs=sum(owned_counts),
                local_experts_per_layer=owned_counts,
                unique_operand_storage_bytes=sum(x['bytes'] for x in allocated.values()),
                unique_operand_storages=len(allocated),
                selected_expert_slice_arithmetic_bytes=selected_slice_estimate,
                operand_format_histogram=dict(Counter(
                    f"{x['operand']}:{x['dtype']}:{x['npu_format']}" for x in allocated.values()))))
        result['models'][model]=dict(layers=layers,rows_per_layer=rows_per_layer,
            topk=6,global_route_incidence_per_cycle=layers*rows_per_layer*6,
            distinct_global_experts_per_layer=unique_global,
            all8_owned_incidence=sum(x['routed_incidence'] for x in per_rank),
            all8_nonempty_local_expert_layer_pairs=sum(
                x['nonempty_local_expert_layer_pairs'] for x in per_rank),
            per_rank=per_rank)
        require(result['models'][model]['all8_owned_incidence']==
                result['models'][model]['global_route_incidence_per_cycle'],
                f'{model} owner incidence conservation')
    ratio=Counter(a['ratio'] for a in records[0]['attention'])
    require(ratio==Counter({0:2,128:20,4:21}), 'frozen Target DSA ratio census')
    result['attention']=dict(local_q_shape_by_rank=[r['attention'][0]['q']['shape'] for r in records],
        compressor_ratio_layer_counts=dict(sorted(ratio.items())),
        all_full_gather_false=all(not a['full_gather'] for r in records for a in r['attention']),
        capture_reference_only=True)
    result['limits']=[
        'First-post-park cohort5 cycle is a diagnostic W0, not Run99 formal W0.',
        'Selected-expert slice arithmetic divides resident storage by 32; NC1HWC0/NZ physical expert contiguity and actual GMM reads are unproven.',
        'Captured Graph parameter references were read after Target completion; actual native replay argument identity remains conditional.',
        'No full Target→router semantic row map, compulsory HBM/communication, strict C+/B or scheduling makespan follows.',
        'Collector does not intentionally edit the algorithm configuration; trajectory noninterference is unproven.',
    ]
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],cycle=result['selected_cycle'],
         target_incidence=result['models']['target']['global_route_incidence_per_cycle'],
         draft_incidence=result['models']['draft']['global_route_incidence_per_cycle'])))


if __name__=='__main__':main()
