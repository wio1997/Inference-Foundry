#!/usr/bin/env python3
"""Admit only a scoped all8 terminal real-weight GMM service observation."""
from __future__ import annotations

import argparse
import csv
import math
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



METRICS = tuple(f'{core}_{metric}(KB)' for core in ('aic','aiv') for metric in (
    'read_main_memory_datas','write_main_memory_datas','GM_to_L1_datas',
    'L0C_to_L1_datas','L0C_to_GM_datas','GM_to_UB_datas','UB_to_GM_datas'))
TYPES=('GroupedMatmulSwigluQuantV2','GroupedMatmul')


def profile_window(result, arm, rank, provenance):
    need(result.get('profiler_level')=='Level1' and
         result.get('profiler_aic_metrics')=='MemoryAccess' and
         result.get('profile_active_replays')==2 and
         result.get('profile_boundary_synchronize') is True,
         f'rank{rank} profiler protocol')
    relative=result.get('profile_csv')
    need(isinstance(relative,str) and relative.endswith('/kernel_details.csv'),
         f'rank{rank} profile path')
    path=ROOT/relative
    need(path.resolve().is_relative_to((arm/'bench'/'profile'/f'rank{rank}').resolve()),
         f'rank{rank} profile path escape')
    raw=path.read_bytes()
    need(hashlib.sha256(raw).hexdigest()==result.get('profile_csv_sha256'),
         f'rank{rank} profile SHA')
    provenance[str(path.relative_to(ROOT))]=hashlib.sha256(raw).hexdigest()
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream)
        headers=set(reader.fieldnames or ())
        need({'Model ID','Task ID','Stream ID','Type','Name','Device_id'}|set(METRICS)<=headers,
             f'rank{rank} profile columns')
        records=list(reader)
    need(len(records)==result.get('profile_native_rows_total') and len(records)>=172,
         f'rank{rank} profile row count')
    # The profiler may export a partial warmup prefix. The admitted active
    # suffix must have precisely two complete 86-task Graph replays.
    suffix=records[-172:]
    need([int(row['Task ID']) for row in suffix]==list(range(86))*2,
         f'rank{rank} incomplete Graph replay suffix')
    model_ids={row['Model ID'] for row in suffix}
    streams={row['Stream ID'] for row in suffix}
    need(len(model_ids)==len(streams)==1 and
         all(int(row['Device_id'])==rank for row in suffix),
         f'rank{rank} mixed Graph/device/stream')
    need(all(row['Type']==TYPES[i%2] for i,row in enumerate(suffix)),
         f'rank{rank} GMM1/GMM2 order')
    need(all(('GroupedMatmulSwigluQuant' in row['Name'] if i%2==0 else
              'GroupedMatmulWeightNz' in row['Name'])
             for i,row in enumerate(suffix)),
         f'rank{rank} GMM native identity')
    replays=[]
    for replay in range(2):
        pair=suffix[86*replay:86*(replay+1)]
        by_type={}
        for parity,label in enumerate(('gmm1','gmm2')):
            selected=pair[parity::2]
            totals={}
            for field in METRICS:
                values=[float(row[field]) for row in selected]
                need(all(math.isfinite(v) and v>=0 for v in values),
                     f'rank{rank} invalid {field}')
                totals[field]=sum(values)
            by_type[label]=dict(task_count=len(selected),counter_KB=totals)
        replays.append(by_type)
    return dict(raw_rows=len(records),ignored_prefix_rows=len(records)-172,
                model_id=next(iter(model_ids)),stream_id=next(iter(streams)),
                complete_replays=replays)


def build(arm):
    provenance = {}
    census_path=ROOT/'evidence/20260928_loop080_bound/run576/operand_census.json'
    census=read(census_path,provenance)
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
    profile_rows=[]
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
        need(result['status']=='isolated_real_weight_synthetic_activation_service' and
             result['run_tag']=='LOOP080-RUN578-B' and result['target_layers']==43 and
             result['fixture']=='Run576 cohort5 cycle170' and
             result['eager_graph_valid_rows_close'] is True and
             len(result['actual_weight_storage_ptrs'])==43 and
             len(result['samples_ms'])==len(result['samples_after_ms'])==20 and
             all(isinstance(x,(int,float)) and 0<x<1000 for x in result['samples_ms']+result['samples_after_ms']) and
             result['host_end_ns']>result['host_start_ns'] and
             result['host_after_end_ns']>result['host_after_start_ns'] and
             result['free_hbm_before_bytes']>=4*1024**3,
             f'rank{rank} service admission')
        expected_incidence=sum(sum(x['group']) for x in fixture['records']['170']['target'])
        need(result['route_incidence']==expected_incidence and
             result['group_counts_per_layer']==[x['group'] for x in fixture['records']['170']['target']] and
             result['free_hbm_after_bytes']>0 and
             result['max_memory_reserved_bytes']>=result['max_memory_allocated_bytes']>0,
             f'rank{rank} route fixture join')
        need(abs(result['median_ms']-statistics.median(result['samples_ms']))<1e-9,
             f'rank{rank} median')
        need(abs(result['median_after_ms']-statistics.median(result['samples_after_ms']))<1e-9,
             f'rank{rank} after median')
        profile_rows.append(profile_window(result,arm,rank,provenance))
        rows.append(result)
    need(len({row['run_tag'] for row in rows})==1 and
         max(row['host_start_ns'] for row in rows)<min(row['host_end_ns'] for row in rows),
         'all8 coordinated Host window')
    runtime_paths=sorted((arm/'runtime').glob('rank*_cohort*.json'))
    need(len(runtime_paths)==64 and
         sorted((json.loads(path.read_text())['rank'],json.loads(path.read_text())['cohort'])
                for path in runtime_paths)==[(rank,cohort)
                                            for rank in range(8) for cohort in range(1,9)],
         'Run578 exactly eight cohorts on all8 ranks')
    need(int((arm/'server_post_count.txt').read_text())==96,
         'Run578 exactly96 POSTs before stop')
    medians=[row['median_ms'] for row in rows]
    medians_after=[row['median_after_ms'] for row in rows]
    return dict(status='scoped_all8_attained_service',
                measurement_class='terminal_isolated_real_weight_synthetic_activation_cross_layer_GMM_graph_and_MemoryAccess',
                contract='48 warmup + 48 measured c12/1024; final cohort8; no later service',
                run_tag='LOOP080-RUN578-B',
                provenance=provenance,
                rank_median_ms=medians,
                rank_median_after_ms=medians_after,
                slowest_rank_median_after_ms=max(medians_after),
                profile_per_rank=profile_rows,
                counter_semantics='Level1 MemoryAccess reported main-memory and GM transfer counters; physical HBM attribution unverified',
                slowest_rank_median_ms=max(medians),
                all8_host_window_overlap_ms=(min(x['host_end_ns'] for x in rows)-
                                             max(x['host_start_ns'] for x in rows))/1e6,
                strict_capacity_upper=None,
                resource_hardware_endpoint_s=None,
                scheduling_execution_endpoint_s=None,
                product_e2e_tps_interval=None,
                numeric_current_to_limit_distance=None,
                limits=['Production resident W4A8 weight format/values, Run576 route groups, private nonzero synthetic activation and scale.',
                        'Attained isolated GMM Graph service, not formal Run99 W0, compulsory traffic, C+, mixed Target/DSpark/HCCL service or legal Product schedule.',
                        'Overlapping Host windows do not certify cross-rank device timestamp alignment or physical overlap.',
                        'Profiler counter traffic is not certified physical HBM bytes; the two complete Graph replay suffixes exclude a possible partial warmup prefix.',
                        'A0/A1 are unprofiled continuous Graph replays; profiler activity occurs between them and is not used as service timing.',
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
                      ('status','slowest_rank_median_ms','all8_host_window_overlap_ms')}))


if __name__=='__main__': main()
