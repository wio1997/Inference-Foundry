#!/usr/bin/env python3
"""Admit only a scoped all8 terminal real-weight GMM service observation."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/data/wio/Inference_Foundry')
CENSUS_SHA='615aeb42086b035903867261e45642a6390a13ba396d91ec31d7cb62320ca1e8'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def read(path, provenance):
    data = path.read_bytes()
    provenance[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def build(arm):
    provenance = {}
    census_path=ROOT/'evidence/20260928_loop080_bound/run576/operand_census.json'
    census=read(census_path,provenance)
    ledger_path=ROOT/'evidence/20260927_loop077_bound/run378/ledger.json'
    read(ledger_path,provenance)
    need(provenance[str(ledger_path.relative_to(ROOT))]==
         'c5bbe55c0853b011188cd0f2e3e6e0a673951b8e95c5b6fd625ca80e45827211',
         'Run378 ledger SHA drift')
    need(provenance[str(census_path.relative_to(ROOT))]==CENSUS_SHA,
         'Run576 census SHA drift')
    client = read(arm/'client_admission.json', provenance)
    need(client['status']=='client_two_phase_admitted' and client['request_count']==96 and
         client['warmup_summary']['success']==client['measured_summary']['success']==48 and
         client['warmup_summary']['fail']==client['measured_summary']['fail']==0 and
         client['warmup_summary']['concurrency']==client['measured_summary']['concurrency']==12 and
         client['warmup_summary']['max_tokens']==client['measured_summary']['max_tokens']==1024,
         'client contract')
    need(len(client['request_index'])==96 and
         [x['phase'] for x in client['request_index']]==['warmup']*48+['measured']*48,
         'client phase order')
    runtime_ids = None
    rows=[]
    for rank in range(8):
        runtime = read(arm/'runtime'/f'rank{rank}_cohort8.json', provenance)
        result = read(arm/'bench'/f'rank{rank}.json', provenance)
        fixture = read(ROOT/f'evidence/20260928_loop080_bound/run576/live/b/capture/rank{rank}_cohort5.json', provenance)
        fixture_rel=f'evidence/20260928_loop080_bound/run576/live/b/capture/rank{rank}_cohort5.json'
        need(provenance[fixture_rel]==census['provenance'].get(fixture_rel) and
             result['fixture_sha256']==provenance[fixture_rel] and
             result['census_sha256']==CENSUS_SHA and
             result['synthetic_activation_seed']==577000+rank,
             f'rank{rank} pinned fixture/seed')
        contract=result['operator_contract']
        need(contract['gmm1']=='grouped_matmul_swiglu_quant_v2' and
             contract['gmm1_dequant_mode']==0 and contract['group_list_type']==1 and
             contract['gmm2']=='npu_grouped_matmul' and
             contract['gmm2_split_item']==2 and contract['gmm2_group_type']==0 and
             contract['gmm2_output_dtype']=='torch.bfloat16' and
             len(contract['swiglu_limits'])==43 and
             all(x==10.0 for x in contract['swiglu_limits']),
             f'rank{rank} native invocation contract')
        if runtime_ids is None:
            runtime_ids = runtime['req_ids']
            need(len(runtime_ids)==len(set(runtime_ids))==12,'final cohort request IDs')
            matched=[]
            for req in runtime_ids:
                options=[row for row in client['request_index']
                         if req==row['response_id'] or req.startswith(row['response_id']+'-')]
                need(len(options)==1 and options[0]['phase']=='measured',
                     'final cohort client identity ambiguous')
                matched.append(options[0])
            need({row['dataset_index'] for row in matched}==set(range(36,48)),
                 'final cohort is not last measured request set')
        need(runtime['rank']==result['rank']==rank and runtime['cohort']==result['cohort']==8 and
             runtime['pass'] is True and runtime['generated_output_counts']==[1024]*12 and
             runtime['req_ids']==runtime_ids, f'rank{rank} Runtime/client join')
        need(result['status']=='terminal_real_weight_GMM_x_full_TP8_HCCL_mixed_service' and
             result['run_tag']=='LOOP080-RUN580-B' and result['target_layers']==43 and
             result['fixture']=='Run576 cohort5 cycle170' and
             result['eager_graph_valid_rows_close'] is True and
             len(result['actual_weight_storage_ptrs'])==43 and
             result['mixed']['status']=='all8_terminal_mixed_resource_service' and
             result['mixed']['ledger_sha256']=='c5bbe55c0853b011188cd0f2e3e6e0a673951b8e95c5b6fd625ca80e45827211' and
             result['mixed']['collective_count']==265 and
             result['mixed']['kinds']=={'hcom_allGather':135,'hcom_reduceScatter':87,'hcom_alltoall':43} and
             result['mixed']['distinct_input_output_buffers_per_collective'] is True and
             result['mixed']['host_end_ns']>result['mixed']['host_start_ns'] and
             result['free_hbm_before_bytes']>=4*1024**3,
             f'rank{rank} service admission')
        expected_incidence=sum(sum(x['group']) for x in fixture['records']['170']['target'])
        need(result['route_incidence']==expected_incidence and
             result['group_counts_per_layer']==[x['group'] for x in fixture['records']['170']['target']] and
             result['free_hbm_after_bytes']>0 and
             result['max_memory_reserved_bytes']>=result['max_memory_allocated_bytes']>0,
             f'rank{rank} route fixture join')
        mixed=result['mixed']
        checkpoint=read(arm/'bench'/f'rank{rank}_unprofiled_checkpoint.json',provenance)
        need(checkpoint['status']=='unprofiled_four_arm_checkpoint' and
             checkpoint['run_tag']=='LOOP080-RUN580-B' and checkpoint['rank']==rank and
             checkpoint['arm_order']==mixed['arm_order'] and
             checkpoint['samples']==mixed['samples'] and
             checkpoint['rank_median_joint_ms']==mixed['rank_median_joint_ms'] and
             checkpoint['host_start_ns']==mixed['host_start_ns'] and
             checkpoint['host_end_ns']==mixed['host_end_ns'],
             f'rank{rank} pre-profiler four-arm checkpoint parity')
        need([x['generation'] for x in mixed['collective_correctness']]==[0,1,2] and
             all(x['all265_outputs_exact'] is True and x['mismatch']==0
                 for x in mixed['collective_correctness']),
             f'rank{rank} fresh all265 collective outputs')
        mc=mixed['mixed_correctness']
        need(mc['generation']==3 and mc['gmm_valid_rows_close'] is True and
             mc['gmm_mismatch']==0 and mc['all265_outputs_exact'] is True and
             mc['hccl_mismatch']==0 and mc['joint_ms']>0 and
             0<mc['gmm_branch_done_ms']<=mc['joint_ms']+0.02 and
             0<mc['hccl_branch_done_ms']<=mc['joint_ms']+0.02,
             f'rank{rank} fresh concurrent two-branch correctness/join')
        need(mixed['profile_status']=='raw_timeline_only_no_online_analysis' and
             mixed['profile_root']==str((arm/'bench'/'profile'/f'rank{rank}').relative_to(ROOT)) and
             mixed['profile_raw_file_count']>=3 and mixed['profile_raw_bytes']>=1000 and
             [s['label'] for s in mixed['profile_steps']]==
             ['warmup_serial','active_serial','active_concurrent'] and
             [s['arm'] for s in mixed['profile_steps']]==
             ['serial','serial','concurrent'] and
             all(0<s['event_sample']['joint_ms']<1000 for s in mixed['profile_steps']),
             f'rank{rank} raw short-window profile identity')
        need(mixed['arm_order']==['gmm','hccl','serial','concurrent',
                                  'concurrent','serial','hccl','gmm'] and
             set(mixed['samples'])=={'gmm','hccl','serial','concurrent'},
             f'rank{rank} balanced mixed arms')
        for arm_name,samples in mixed['samples'].items():
            need(len(samples)==20 and all(0<x['joint_ms']<1000 and
                 0<x['host_submit_wait_ms']<1000 for x in samples),
                 f'rank{rank} {arm_name} samples')
            if arm_name!='hccl':
                need(all(0<x['gmm_branch_done_ms']<1000 for x in samples),
                     f'rank{rank} {arm_name} GMM branch')
                need(all(x['joint_ms']+0.02>=x['gmm_branch_done_ms'] for x in samples),
                     f'rank{rank} {arm_name} GMM join')
            if arm_name!='gmm':
                need(all(0<x['hccl_branch_done_ms']<1000 for x in samples),
                     f'rank{rank} {arm_name} HCCL branch')
                need(all(x['joint_ms']+0.02>=x['hccl_branch_done_ms'] for x in samples),
                     f'rank{rank} {arm_name} HCCL join')
            if arm_name=='serial':
                need(all(x['hccl_branch_done_ms']+0.02>=x['gmm_branch_done_ms']
                         for x in samples),f'rank{rank} serial order')
            need(abs(mixed['rank_median_joint_ms'][arm_name]-
                     statistics.median(x['joint_ms'] for x in samples))<1e-9,
                 f'rank{rank} {arm_name} median')
        rows.append(result)
    need(len({row['run_tag'] for row in rows})==1 and
         max(row['mixed']['host_start_ns'] for row in rows)<
         min(row['mixed']['host_end_ns'] for row in rows),
         'all8 coordinated Host window')
    runtime_paths=sorted((arm/'runtime').glob('rank*_cohort*.json'))
    need(len(runtime_paths)==64 and
         sorted((json.loads(path.read_text())['rank'],json.loads(path.read_text())['cohort'])
                for path in runtime_paths)==[(rank,cohort)
                                            for rank in range(8) for cohort in range(1,9)],
         'Run580 exactly eight cohorts on all8 ranks')
    need(int((arm/'server_post_count.txt').read_text())==96,
         'Run580 exactly96 POSTs before stop')
    arm_medians={arm:[row['mixed']['rank_median_joint_ms'][arm] for row in rows]
                 for arm in ('gmm','hccl','serial','concurrent')}
    return dict(status='scoped_all8_attained_service',
                measurement_class='terminal_independent_ready_real_weight_GMM_x_full265_TP8_HCCL',
                contract='48 warmup + 48 measured c12/1024; final cohort8; no later service',
                run_tag='LOOP080-RUN580-B',
                provenance=provenance,
                arm_rank_median_joint_ms=arm_medians,
                arm_slowest_rank_median_joint_ms={arm:max(values) for arm,values in arm_medians.items()},
                all8_host_window_overlap_ms=(min(x['mixed']['host_end_ns'] for x in rows)-
                                             max(x['mixed']['host_start_ns'] for x in rows))/1e6,
                strict_capacity_upper=None,
                resource_hardware_endpoint_s=None,
                scheduling_execution_endpoint_s=None,
                product_e2e_tps_interval=None,
                numeric_current_to_limit_distance=None,
                limits=['Production resident W4A8 weight format/values, Run576 route groups, private nonzero synthetic activation and scale.',
                        'Independent already-ready current-order 265 TP8 HCCL calls with per-call private buffers and synthetic values.',
                        'Attained mixed-resource service is not formal Run99 W0, compulsory communication, strict C+ or legal Product schedule.',
                        'Overlapping Host windows do not certify cross-rank device timestamp alignment or physical overlap.',
                        'Terminal diagnostic delays last measured API cohort and cannot be used as formal TPS.'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    result=build(a.arm.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in
                      ('status','arm_slowest_rank_median_joint_ms','all8_host_window_overlap_ms')}))


if __name__=='__main__': main()
