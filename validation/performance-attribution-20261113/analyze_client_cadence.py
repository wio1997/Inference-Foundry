"""从已发布192条客户端记录定位墙钟差异；不执行模型或模拟设备。"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import statistics


def merged(intervals):
    result=[]
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([start,end])
    return result


def overlap(intervals,other):
    return sum(max(0,min(b,y)-max(a,x)) for a,b in intervals for x,y in other)


def analyze(base):
    path=base/'request-timings-138-140.jsonl'
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert len(rows)==192
    expected=json.loads((base/'measured-results-138-140.json').read_text())
    arms={};input_sets={}
    for generation in (138,139,140):
        for concurrency in (4,8):
            key=f'{generation}/C{concurrency}'
            group=[r for r in rows if r['generation']==generation and r['concurrency']==concurrency]
            assert len(group)==32 and len({r['data_id'] for r in group})==32
            assert all(r['success'] and r['input_tokens']==1024 and r['output_tokens']==256 for r in group)
            input_sets[key]={r['data_id']:r['input_sha256'] for r in group}
            points=[r['time_points'] for r in group]
            assert all(len(t)>=3 and all(map(math.isfinite,t)) and all(a<=b for a,b in zip(t,t[1:])) for t in points)
            start=min(t[0] for t in points);end=max(t[-1] for t in points)
            duration=end-start
            assert abs(duration-expected[key]['benchmark_seconds'])<1e-9
            pending=merged([(t[0],t[1]) for t in points])
            pending_time=sum(b-a for a,b in pending)
            events=sorted(v for t in points for v in t[1:])
            gaps=[(a,b) for a,b in zip(events,events[1:])]
            positive=[b-a for t in points for a,b in zip(t[1:],t[2:]) if b-a>0.001]
            thresholds={}
            for ms in (50,100,200):
                threshold=ms/1000
                long=[(a,b) for a,b in gaps if b-a>threshold]
                total=sum(b-a for a,b in long)
                capped=sum(min(b-a,threshold) for a,b in gaps)
                excess=sum(max(0,b-a-threshold) for a,b in gaps)
                initial=events[0]-start
                assert abs(initial+capped+excess-duration)<1e-7
                thresholds[str(ms)]={'count':len(long),'total_seconds':total,
                    'overlap_any_request_waiting_first_record_seconds':overlap(long,pending),
                    'initial_seconds':initial,'capped_gap_seconds':capped,'gap_excess_seconds':excess}
            arms[key]={'requests':32,'duration_seconds':duration,
                'any_request_waiting_first_record_union_seconds':pending_time,
                'no_request_waiting_first_record_seconds':duration-pending_time,
                'timestamp_count_histogram':dict(Counter(map(len,points))),
                'per_request_record_gap_above_1ms_median_ms':statistics.median(positive)*1000,
                'global_record_silence_threshold_ms':thresholds}
    assert all(v==next(iter(input_sets.values())) for v in input_sets.values())
    differences={}
    for c in (4,8):
        ref,on,off=(arms[f'{g}/C{c}'] for g in (138,139,140))
        keys=('duration_seconds','any_request_waiting_first_record_union_seconds','no_request_waiting_first_record_seconds')
        differences[f'C{c}']={'reference_minus_native_on':{k:ref[k]-on[k] for k in keys},
            'native_off_minus_native_on':{k:off[k]-on[k] for k in keys}}
    return {'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':192,
        'same_32_inputs_verified_all_six_arms':True,'arms':arms,'differences':differences,
        'definitions':{'pending_first_record':'union of client [request_start, first response record] intervals;includes queue,admission,prefill,transport',
            'global_silence':'between successive timestamps across all32 requests after start markers are removed;includes response/control/end records',
            'positive_gap_median':'per-request consecutive response-record gaps strictly above1ms;excludes most same-burst records;not model-step latency'},
        'limitations':['client records are not device timestamps','union partitions differ in request composition;they are not isolated stage costs',
            'different SSE framing prevents treating one record as one token or one model forward',
            'silence overlaps a pending-first-record interval but does not identify CPU,NPU,communication or admission as its cause',
            '50/100/200ms are descriptive sensitivity thresholds,not benchmark pass thresholds',
            'single batches;no noise calibration;no causal cohort intervention'],
        'new_npu_runs':0,'goal_complete':False}


if __name__=='__main__':
    base=Path(__file__).resolve().parent
    result=analyze(base)
    (base/'client-cadence-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'records':result['records'],'differences':result['differences'],'new_npu_runs':0},ensure_ascii=False))
