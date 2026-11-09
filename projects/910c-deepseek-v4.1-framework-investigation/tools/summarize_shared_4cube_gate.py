#!/usr/bin/env python3
"""Validate one bounded primitive receipt, without promoting it to model evidence."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'handoff/evidence'


def main():
    d=json.loads((ROOT/'data/native-shared-4cube-a2-evidence.json').read_text())
    r=d['result'];rows=next(iter(d['kernel_details'].values()))
    assert d['exit_code']==0 and d['health_before']==d['health_after']=='200'
    assert d['source_sha256']==hashlib.sha256((ROOT/'data/native-shared-4cube-a2-probe.py').read_bytes()).hexdigest()
    assert r['all_tested_bit_equal'] and r['NZ_default_complete']
    assert len(r['checks'])==8 and all(v['byte_differences']==0 and v['finite'] for v in r['checks'].values())
    assert [(int(v['Block Num']),int(v['Mix Block Num'])) for v in rows]==[(8,0),(4,0),(8,0),(4,4)]
    assert rows[1]['Input Data Types']=='INT8;INT8;DT_BF16'
    assert rows[3]['Input Data Types']=='INT8;INT8;DT_BF16;FLOAT'
    assert 'BF16 -> F32' in r['scale_semantics']
    points=[dict(name=v['Name'],block_num=int(v['Block Num']),mix_block_num=int(v['Mix Block Num']),us=float(v['Duration(us)']),input_shapes=v['Input Shapes'],input_dtypes=v['Input Data Types'],input_formats=v['Input Formats']) for v in rows]
    out=dict(status='LIMITED_PRIMITIVE_ADMISSION_CONFIRMED_WHOLE_RESIDENCY_PENDING',case=dict(M=416,layer=0,device=0,input='synthetic BF16 seed20261109',weight='frozen real shared expert',scale=r['scale_semantics']),checks=r['checks'],kernel_details=points,isolated_pipeline_event_us=r['isolated_pipeline_timing'],profile_kernel_sum_us=sum(p['us'] for p in points),profile_kernel_wall_us=max(float(v['Start Time(us)'])+float(v['Duration(us)']) for v in rows)-min(float(v['Start Time(us)']) for v in rows),invalid_first_attempt='data/native-shared-4cube-a1-correction.json',model_requests=0,valid_NPU_attempts=1,invalid_for_target_signature_NPU_attempts=1,pending=['Physical co-residency with actual DFC20/40 and nonexclusive scheduler placement, original DFC slowdown and HBM/L2 contention','All relevant M and forty layer weights/real inputs under per-replacement bit contract','Integrated private-format setting and graph/stream synchronization without changing other primitives','All-rank EP arrival and request critical path; diagnostic v168 reference remains observer-perturbed'],evidence_sha256=hashlib.sha256((ROOT/'data/native-shared-4cube-a2-evidence.json').read_bytes()).hexdigest(),whole_chain_complete=False,profile_field_reference='https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_1149.html')
    (ROOT/'shared-4cube-gate-summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=out['status'],kernel_details=points,isolated_medians=out['isolated_pipeline_event_us']['medians'])))


if __name__=='__main__':main()
