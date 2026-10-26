#!/usr/bin/env python3
"""CPU-only admission negatives using synthetic service rows, never NPU evidence."""
import json
import csv
import hashlib
import shutil
import tempfile
from pathlib import Path

from loop080_realweight_reduce_run578 import ROOT, build, METRICS

source=ROOT/'evidence/20260928_loop080_bound/run576/live/b'
with tempfile.TemporaryDirectory(prefix='run578_gate_',dir=ROOT/'evidence/20260928_loop080_bound') as td:
    arm=Path(td)
    (arm/'runtime').mkdir();(arm/'bench').mkdir()
    shutil.copy2(source/'client_admission.json',arm/'client_admission.json')
    (arm/'server_post_count.txt').write_text('96\n')
    for path in (source/'runtime').glob('rank*_cohort*.json'):
        shutil.copy2(path,arm/'runtime'/path.name)
    for rank in range(8):
        fixture=ROOT/f'evidence/20260928_loop080_bound/run576/live/b/capture/rank{rank}_cohort5.json'
        f=json.loads(fixture.read_text())
        group=[x['group'] for x in f['records']['170']['target']]
        row=dict(status='isolated_real_weight_synthetic_activation_service',
                 run_tag='LOOP080-RUN578-B',rank=rank,cohort=8,
                 fixture='Run576 cohort5 cycle170',target_layers=43,
                 fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                 census_sha256='615aeb42086b035903867261e45642a6390a13ba396d91ec31d7cb62320ca1e8',
                 synthetic_activation_seed=577000+rank,
                 group_counts_per_layer=group,route_incidence=sum(map(sum,group)),
                 operator_contract=dict(gmm1='grouped_matmul_swiglu_quant_v2',
                     gmm1_dequant_mode=0,group_list_type=1,gmm2='npu_grouped_matmul',
                     gmm2_split_item=2,gmm2_group_type=0,
                     gmm2_output_dtype='torch.bfloat16',swiglu_limits=[10.0]*43),
                 actual_weight_storage_ptrs=[{'w1':1,'w2':2}]*43,
                 free_hbm_before_bytes=5*1024**3,free_hbm_after_bytes=4*1024**3,
                 max_memory_allocated_bytes=1,max_memory_reserved_bytes=2,
                 eager_graph_valid_rows_close=True,samples_ms=[1.0]*20,
                 median_ms=1.0,host_start_ns=100,host_end_ns=200,
                 samples_after_ms=[1.1]*20,median_after_ms=1.1,
                 host_after_start_ns=300,host_after_end_ns=400,
                 profiler_level='Level1',profiler_aic_metrics='MemoryAccess',
                 profile_active_replays=2,profile_boundary_synchronize=True)
        profile=arm/'bench'/'profile'/f'rank{rank}'/'mock'/'kernel_details.csv'
        profile.parent.mkdir(parents=True)
        fields=['Model ID','Task ID','Stream ID','Type','Name','Device_id',*METRICS]
        with profile.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for i in list(range(86))*2:
                kind='GroupedMatmulSwigluQuantV2' if i%2==0 else 'GroupedMatmul'
                name='aclnnGroupedMatmulSwigluQuantWeightNzV2' if i%2==0 else 'aclnnGroupedMatmulWeightNz'
                writer.writerow({'Model ID':'42','Task ID':i,'Stream ID':'13',
                                 'Type':kind,'Name':name,'Device_id':rank,
                                 **{field:'1.0' for field in METRICS}})
        row.update(profile_csv=str(profile.relative_to(ROOT)),
                   profile_csv_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
                   profile_native_rows_total=172)
        (arm/'bench'/f'rank{rank}.json').write_text(json.dumps(row))
    assert build(arm)['status']=='scoped_all8_attained_service'
    cases=[('wrong_profile_sha',0,'profile_csv_sha256','bad'),
           ('wrong_fixture_sha',0,'fixture_sha256','bad'),
           ('wrong_seed',0,'synthetic_activation_seed',0),
           ('wrong_group',0,'group_counts_per_layer',[[0]*32]*43),
           ('wrong_cohort',0,'cohort',9)]
    for name,rank,key,value in cases:
        path=arm/'bench'/f'rank{rank}.json'; row=json.loads(path.read_text())
        original=row[key];row[key]=value;path.write_text(json.dumps(row))
        try: build(arm)
        except ValueError: pass
        else: raise AssertionError(name+' passed unexpectedly')
        row[key]=original;path.write_text(json.dumps(row))
    profile=arm/'bench/profile/rank0/mock/kernel_details.csv'
    original=profile.read_text()
    profile.write_text(original.replace(',0,13,GroupedMatmulSwigluQuantV2,',
                                        ',1,13,GroupedMatmulSwigluQuantV2,',1))
    try: build(arm)
    except ValueError: pass
    else: raise AssertionError('corrupt profile passed')
    profile.write_text(original)
    runtime=arm/'runtime/rank0_cohort8.json'
    row=json.loads(runtime.read_text());old=row['req_ids'];row['req_ids']=['wrong']*12
    runtime.write_text(json.dumps(row))
    try: build(arm)
    except ValueError: pass
    else: raise AssertionError('wrong request IDs passed')
print(json.dumps({'status':'cpu_only_gate_pass','negative_cases':len(cases)+2}))
