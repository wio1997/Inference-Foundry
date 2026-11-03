#!/usr/bin/env python3
"""Audit existing same-scope FP32/BF16 projection-shaped tasks; no NPU."""
import argparse, collections, hashlib, json, statistics
from decimal import Decimal as D
from pathlib import Path

root=Path(__file__).resolve().parents[2]
p=root/'diagnostic-20261029/target-layer-all-kernels-16rank.jsonl'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4'
by_m=collections.defaultdict(list); spans=collections.defaultdict(list); seen=set(); public_pairs=[]
for line in p.open():
    item=json.loads(line)
    if item.get('kind')!='rank': continue
    for scope in item['data']['layers']:
        if scope['layer'] not in (2,8,14): continue
        key=(scope['ep'],scope['layer'],scope['scheduler_digest']); assert key not in seen; seen.add(key)
        candidates=collections.defaultdict(list); quant=[]
        for group in scope['operator_groups']:
            for task in group['tasks']:
                c=task['csv']; ins=c['Input Shapes'].strip('"[]'); m=ins.split(',')[0]
                if c['Type']=='_quantize_indexer_query_kernel': quant.append(task)
                if c['Type']=='MatMulV2' and m in ('4760','5144','8180') and ins==f'{m},5120;512,5120' and c['Output Shapes'].strip('"[]')==f'{m},512':
                    candidates[(int(m),c['Input Data Types'],c['Output Data Types'])].append(task)
        assert sum(map(len,candidates.values()))==3
        matches=[]
        for m in (4760,5144,8180):
            bf=candidates.get((m,'DT_BF16;DT_BF16','DT_BF16'),[])
            fp=candidates.get((m,'FLOAT;FLOAT','FLOAT'),[])
            if bf or fp:
                assert len(bf)==1 and len(fp)==2
                matches.append(m)
                for task in bf+fp:
                    c=task['csv']
                    assert (c['Name'],c['Input Formats'],c['Output Formats'],c['Accelerator Core'],c['Block Num'])==('aclnnMatmul_MatMulCommon_MatMulV2','ND;ND','ND','AI_CORE','24')
                fp.sort(key=lambda t:D(t['device_start_us']))
                assert fp[0]['csv']['Stream ID']==fp[1]['csv']['Stream ID'] != bf[0]['csv']['Stream ID']
                assert (fp[0]['device_track_pid'],fp[0]['device_track_tid'])==(fp[1]['device_track_pid'],fp[1]['device_track_tid'])
                assert D(fp[0]['device_start_us'])+D(fp[0]['trace_duration_us'])<=D(fp[1]['device_start_us'])
                assert int(fp[0]['csv']['Task ID'])+1==int(fp[1]['csv']['Task ID'])
                assert len(quant)==1
                q=quant[0]
                assert all(t['device_track_pid']==q['device_track_pid'] and t['device_track_tid']!=q['device_track_tid'] and t['csv']['Stream ID']!=q['csv']['Stream ID'] for t in fp)
                def interval(t):
                    start=D(t['device_start_us']); return start,start+D(t['trace_duration_us'])
                def overlap(t,u):
                    x,y=interval(t);z,w=interval(u);return max(D(0),min(y,w)-max(x,z))
                gap=interval(fp[1])[0]-interval(fp[0])[1];assert gap>=0
                q_first,q_second=overlap(fp[0],q),overlap(fp[1],q)
                t0=D(bf[0]['trace_duration_us']); assert t0>0
                spans[(m,'bf16')].append(t0)
                public_pairs.append({'physical_rows':m,'bf16_us':str(t0),'float_us':[str(D(t['trace_duration_us'])) for t in fp],'float_gap_us':str(gap),'quant_duration_us':q['trace_duration_us'],'quant_first_overlap_us':str(q_first),'quant_second_overlap_us':str(q_second),'quant_before_first':interval(q)[1]<=interval(fp[0])[0]})
                for task in fp:
                    t=D(task['trace_duration_us']); assert t>0
                    spans[(m,'float')].append(t); by_m[m].append(t/t0)
        assert len(matches)==1,(key,matches)
assert len(seen)==96 and [len(by_m[m]) for m in (4760,5144,8180)]==[48,48,96]
full=[x for x in public_pairs if D(x['quant_first_overlap_us'])==D(x['quant_duration_us'])]
long_gaps=[x for x in public_pairs if D(x['float_gap_us'])>100]
assert len(full)==87 and len(long_gaps)==9 and all(x['quant_before_first'] for x in long_gaps)
assert all(D(x['quant_second_overlap_us'])==0 for x in public_pairs)
def summary(values):
    return [float(min(values)),float(statistics.median(values)),float(max(values))]
result={str(m):{'float_over_bf16_n':len(by_m[m]),'float_over_bf16_min_median_max':summary(by_m[m]),'bf16_duration_us_min_median_max':summary(spans[(m,'bf16')]),'float_duration_us_min_median_max':summary(spans[(m,'float')])} for m in (4760,5144,8180)}
parser=argparse.ArgumentParser();parser.add_argument('--public-json',type=Path);args=parser.parse_args()
if args.public_json:
    public_pairs.sort(key=lambda x:(x['physical_rows'],D(x['bf16_us']),tuple(map(D,x['float_us']))))
    payload={'scope':'96 selected anonymous same-rank/layer/digest scopes; one BF16 and two FLOAT records per scope','duration_unit':'microseconds','quant_definition':'One selected _quantize_indexer_query_kernel per scope, on a different track from both FLOAT tasks. Overlap and gaps are exported interval arithmetic, not proof of physical parallelism.','pairing_and_capture_limits':'Same physical shape but different streams, inputs, weights and execution states; profiled record, not causal or equal-work evidence. Dataset omits rank, layer, identifiers and timestamps.','source_aggregate_sha256':'0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4','pairs':public_pairs}
    args.public_json.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'scope':'96 selected rank/layer/digest scopes, two FLOAT vs one BF16 record each','results':result,'quant_fully_overlaps_first':len(full),'float_gap_gt_100_us':len(long_gaps),'limits':'Different streams, inputs, weights and execution states; profiler records do not establish causality or tensor identity.'},indent=2))
