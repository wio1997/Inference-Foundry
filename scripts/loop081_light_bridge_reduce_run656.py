#!/usr/bin/env python3
"""Read-only same-ON-W0 stage census; never interpret Events as intrinsic service."""
from __future__ import annotations
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry/evidence/20260928_loop081_bound/run656/live')
OUT=ROOT.parent/'on_stage_summary.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def summary(rows):
    xs=sorted(rows)
    return {'n':len(xs),'min':xs[0],'median':statistics.median(xs),
            'p95':xs[min(len(xs)-1,math.ceil(.95*len(xs))-1)],'max':xs[-1]}


def main():
    admission=read(ROOT/'admission.json')
    assert admission['status']=='all_arms_individually_admitted'
    assert admission['cross_arm_fixed_w0'] is False
    assert admission['cross_arm_timing_transfer_admitted'] is False
    arm=admission['arms']['on']
    groups=defaultdict(lambda:defaultdict(list))
    rank_sums={}
    pins={}
    for rank in range(8):
        sums={'target_ms':0.,'proposer_ms':0.,'cycle_ms':0.,'paired_outside_ms':0.,
              'runtime_wall_s':0.}
        for cohort in range(5,9):
            name=f'rank{rank}_cohort{cohort}.json'
            packet_path=ROOT/'on'/'packet'/name
            runtime_path=ROOT/'on'/'runtime'/name
            packet=read(packet_path)
            runtime=read(runtime_path)
            assert admission['arms']['on']['raw_sha256'][str(packet_path)]==sha(packet_path)
            assert admission['arms']['on']['raw_sha256'][str(runtime_path)]==sha(runtime_path)
            pins[str(packet_path)]=sha(packet_path)
            pins[str(runtime_path)]=sha(runtime_path)
            C=packet['identity']['cycles'];events=packet['events'];classes=packet['class_rows']
            assert len(events)==5*C+1 and len(classes)==C
            sums['runtime_wall_s']+=runtime['wall_seconds']
            for i in range(C):
                start=events[5*i]['elapsed_ms']
                end=events[5*(i+1)]['elapsed_ms'] if i+1<C else events[-1]['elapsed_ms']
                target=events[5*i+2]['elapsed_ms']-events[5*i+1]['elapsed_ms']
                draft=events[5*i+4]['elapsed_ms']-events[5*i+3]['elapsed_ms']
                cycle=end-start
                outside=cycle-target-draft
                assert min(target,draft,cycle,outside)>=-1e-6
                kind='startup_0_7' if i<8 else ('parked' if classes[i]['parked_before'] else 'steady_unparked')
                for class_name in ('all',kind):
                    for key,value in (('target_ms',target),('proposer_ms',draft),
                                      ('cycle_ms',cycle),('paired_outside_ms',outside)):
                        groups[class_name][key].append(value)
                for key,value in (('target_ms',target),('proposer_ms',draft),
                                  ('cycle_ms',cycle),('paired_outside_ms',outside)):
                    sums[key]+=value
        sums['event_cycle_span_s']=sums['cycle_ms']/1000
        sums['runtime_minus_event_span_s']=sums['runtime_wall_s']-sums['event_cycle_span_s']
        assert sums['runtime_minus_event_span_s']>=0
        rank_sums[str(rank)]=sums
    assert len(groups['all']['cycle_ms'])==arm['stage_rank_cycle_count']==9600
    result={'status':'same_on_w0_stage_census_only',
            'input_admission_sha256':sha(ROOT/'admission.json'),
            'input_sha256':pins,
            'product_wall_s_diagnostic':arm['client_wall_s'],
            'runtime_sum_rank0_s':rank_sums['0']['runtime_wall_s'],
            'product_minus_runtime_rank0_arithmetic_s':
                arm['client_wall_s']-rank_sums['0']['runtime_wall_s'],
            'rank_sums':rank_sums,
            'stage_class_summary':{kind:{key:summary(values) for key,values in fields.items()}
                                   for kind,fields in groups.items()},
            'cross_arm_timing_transfer_admitted':False,
            'intrinsic_primitive_service_admitted':False,
            'full_product_framework_only_tps_ceiling':None,
            'limits':'Current-stream Event intervals include queue/wait and do not certify native/HCCL completion or legal schedule savings. Stage class distributions and rank sums are same ON W0 only. Arithmetic Product/Runtime and Runtime/Event differences are not disjoint removable terms. OFF arms have different trajectories.'}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'rank_cycles':len(groups['all']['cycle_ms']),
                      'product_wall_s':result['product_wall_s_diagnostic'],
                      'target_median_ms':result['stage_class_summary']['all']['target_ms']['median'],
                      'proposer_median_ms':result['stage_class_summary']['all']['proposer_ms']['median'],
                      'cycle_median_ms':result['stage_class_summary']['all']['cycle_ms']['median']}))

if __name__=='__main__':main()
