#!/usr/bin/env python3
"""CPU-only admission negatives using synthetic service rows, never NPU evidence."""
import json
import shutil
import tempfile
from pathlib import Path

from loop080_mixed_reduce_run580 import ROOT, build

source=ROOT/'evidence/20260928_loop080_bound/run576/live/b'
with tempfile.TemporaryDirectory(prefix='run580_gate_',dir=ROOT/'evidence/20260928_loop080_bound') as td:
    arm=Path(td)
    (arm/'runtime').mkdir();(arm/'bench').mkdir()
    shutil.copy2(source/'client_admission.json',arm/'client_admission.json')
    (arm/'server_post_count.txt').write_text('96\n')
    for path in (source/'runtime').glob('rank*_cohort*.json'):
        shutil.copy2(path,arm/'runtime'/path.name)
    for rank in range(8):
        fixture=ROOT/f'evidence/20260928_loop080_bound/run576/live/b/capture/rank{rank}_cohort5.json'
        import hashlib
        f=json.loads(fixture.read_text())
        group=[x['group'] for x in f['records']['170']['target']]
        row=dict(status='terminal_real_weight_GMM_x_full_TP8_HCCL_mixed_service',
                 run_tag='LOOP080-RUN580-B',rank=rank,cohort=8,
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
                 eager_graph_valid_rows_close=True,
                 mixed=dict(status='all8_terminal_mixed_resource_service',
                     ledger_sha256='c5bbe55c0853b011188cd0f2e3e6e0a673951b8e95c5b6fd625ca80e45827211',
                     collective_count=265,
                     kinds={'hcom_allGather':135,'hcom_reduceScatter':87,'hcom_alltoall':43},
                     distinct_input_output_buffers_per_collective=True,
                     collective_correctness=[{'generation':g,'all265_outputs_exact':True,'mismatch':0}
                                             for g in (0,1,2)],
                     mixed_correctness=dict(generation=3,gmm_valid_rows_close=True,
                         gmm_mismatch=0,all265_outputs_exact=True,hccl_mismatch=0,
                         joint_ms=1.0,gmm_branch_done_ms=0.8,hccl_branch_done_ms=0.9),
                     profile_status='raw_timeline_only_no_online_analysis',
                     profile_root=str((arm/'bench'/'profile'/f'rank{rank}').relative_to(ROOT)),
                     profile_raw_file_count=3,profile_raw_bytes=1000,
                     profile_steps=[{'label':label,'arm':test_arm,
                                     'event_sample':{'joint_ms':1.0}}
                                    for label,test_arm in (('warmup_serial','serial'),
                                        ('active_serial','serial'),
                                        ('active_concurrent','concurrent'))],
                     arm_order=['gmm','hccl','serial','concurrent',
                                'concurrent','serial','hccl','gmm'],
                     samples={arm:[dict(joint_ms=1.0,host_submit_wait_ms=2.0,
                                       **({} if arm=='hccl' else {'gmm_branch_done_ms':0.8}),
                                       **({} if arm=='gmm' else {'hccl_branch_done_ms':0.9}))
                                   for _ in range(20)]
                              for arm in ('gmm','hccl','serial','concurrent')},
                     rank_median_joint_ms={arm:1.0 for arm in ('gmm','hccl','serial','concurrent')},
                     host_start_ns=100,host_end_ns=200))
        (arm/'bench'/f'rank{rank}.json').write_text(json.dumps(row))
        m=row['mixed']
        (arm/'bench'/f'rank{rank}_unprofiled_checkpoint.json').write_text(
            json.dumps(dict(status='unprofiled_four_arm_checkpoint',
                            run_tag='LOOP080-RUN580-B',rank=rank,
                            arm_order=m['arm_order'],samples=m['samples'],
                            rank_median_joint_ms=m['rank_median_joint_ms'],
                            host_start_ns=m['host_start_ns'],host_end_ns=m['host_end_ns'])))
    assert build(arm)['status']=='scoped_all8_attained_service'
    cases=[('wrong_fixture_sha',0,'fixture_sha256','bad'),
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
    path=arm/'bench/rank0.json';row=json.loads(path.read_text())
    original=row['mixed']['collective_correctness'][1]['mismatch']
    row['mixed']['collective_correctness'][1]['mismatch']=1
    path.write_text(json.dumps(row))
    try:build(arm)
    except ValueError:pass
    else:raise AssertionError('bad full-chain correctness passed')
    row['mixed']['collective_correctness'][1]['mismatch']=original
    path.write_text(json.dumps(row))
    row['mixed']['mixed_correctness']['gmm_mismatch']=1
    path.write_text(json.dumps(row))
    try:build(arm)
    except ValueError:pass
    else:raise AssertionError('bad fresh concurrent GMM passed')
    row['mixed']['mixed_correctness']['gmm_mismatch']=0
    path.write_text(json.dumps(row))
    row['mixed']['samples']['serial'][0]['hccl_branch_done_ms']=0.1
    path.write_text(json.dumps(row))
    try:build(arm)
    except ValueError:pass
    else:raise AssertionError('invalid serial event order passed')
    row['mixed']['samples']['serial'][0]['hccl_branch_done_ms']=0.9
    path.write_text(json.dumps(row))
    runtime=arm/'runtime/rank0_cohort8.json'
    row=json.loads(runtime.read_text());old=row['req_ids'];row['req_ids']=['wrong']*12
    runtime.write_text(json.dumps(row))
    try: build(arm)
    except ValueError: pass
    else: raise AssertionError('wrong request IDs passed')
print(json.dumps({'status':'cpu_only_gate_pass','negative_cases':len(cases)+4}))
