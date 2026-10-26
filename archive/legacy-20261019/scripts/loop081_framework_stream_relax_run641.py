#!/usr/bin/env python3
"""Fixed-observed-cost, fixed-stream-affinity relaxation for Run611 Model45 windows."""
from __future__ import annotations
import hashlib, json, bisect
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

ROOT=Path('/data/wio/Inference_Foundry')
BASE=ROOT/'evidence/20260928_loop081_bound/run611'
OUT=ROOT/'evidence/20260928_loop081_bound/run641/fixed_stream_relaxation.json'
PHYSICAL={'KERNEL_AICORE','KERNEL_MIX_AIC','KERNEL_AIVEC','KERNEL_MIX_AIV','KERNEL_AICPU','MEMCPY_ASYNC','SDMA_SQE'}
MANIFEST_SHA='b51eac01dcdc9f183544d51d20040f371f5ffb23a686ded7372b2cb366c84e83'
REPLAY_SHA='9518ee739ece657c8cff467ff476d92e7cc939048945abccc1e9f7e5a009336d'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def need(x,s):
    if not x: raise AssertionError(s)
def dec(x):return Decimal(str(x).strip())
def merge(items):
    out=[]
    for a,b in sorted(items):
        if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(out[-1][1],b))
        else:out.append((a,b))
    return out
def intersection_us(left,right):
    l,r=merge(left),merge(right);i=j=0;total=Decimal(0)
    while i<len(l) and j<len(r):
        total+=max(Decimal(0),min(l[i][1],r[j][1])-max(l[i][0],r[j][0]))
        if l[i][1]<=r[j][1]:i+=1
        else:j+=1
    return total

def main():
    manifest=BASE/'offline_parse/manifest.json'; replay=BASE/'native_graph_replay.json'
    need(sha(manifest)==MANIFEST_SHA and sha(replay)==REPLAY_SHA,'manifest/replay identity')
    m=json.loads(manifest.read_text()); r=json.loads(replay.read_text())
    need(len(m['parsed'])==8 and len(r['rows'])==24,'all8 x3')
    need({(v['rank'],v['ordinal_in_profile']) for v in r['rows']}=={(rank,ordinal) for rank in range(8) for ordinal in range(3)},'rank/ordinal bijection')
    allrows=[]
    for rank,source in enumerate(m['parsed']):
        need(source['rank']==rank,'rank order')
        path=ROOT/source['trace_view_json'];need(sha(path)==source['trace_view_sha256'],'trace SHA')
        data=json.loads(path.read_text())
        model=[x for x in data if x.get('ph')=='X' and x.get('args',{}).get('Model Id')==45]
        need(len(model)==3*5412,'model task population')
        key=lambda x:(x['args']['Physic Stream Id'],x['args']['Task Id'],x['args']['Batch Id'],x['args']['Subtask Id'])
        static_counts=Counter(key(x) for x in model)
        need(len(static_counts)==5412 and set(static_counts.values())=={3},'static key triple occurrence')
        anchors=sorted(dec(x['ts']) for x in model if key(x)[:2]==(1,0))
        need(len(anchors)==3,'replay anchors')
        buckets=[[],[],[],[]]
        for x in model:buckets[bisect.bisect_right(anchors,dec(x['ts']))].append(x)
        need(not buckets[0] and [len(v) for v in buckets[1:]]==[5412]*3,'replay partition')
        need(all({key(x) for x in v}==set(static_counts) for v in buckets[1:]),'replay static key identity')
        for row in sorted((v for v in r['rows'] if v['rank']==rank),key=lambda v:v['ordinal_in_profile']):
            begin,end=dec(row['first_task_start_us']),dec(row['last_task_end_us'])
            ordinal=row['ordinal_in_profile']
            selected=buckets[ordinal+1]
            need(abs(dec(selected[0]['ts'])-begin)<Decimal('1'),'graph start identity')
            need(abs(max(dec(x['ts'])+dec(x['dur']) for x in selected)-end)<Decimal('1'),'graph end identity')
            need(len(selected)==5412,'exact graph task count')
            service=defaultdict(Decimal); physical=Counter(); hcom=defaultdict(Decimal)
            hcom_intervals=[];main_physical_intervals=[];main_wait_intervals=[]
            for x in selected:
                args=x['args'];kind=args.get('Task Type');stream=int(args['Physic Stream Id'])
                if kind in PHYSICAL:
                    duration=dec(x['dur']);need(duration>=0,'duration')
                    service[stream]+=duration;physical[kind]+=1
                    interval=(dec(x['ts']),dec(x['ts'])+duration)
                    if stream==1:main_physical_intervals.append(interval)
                    if 'hcom' in x.get('name','').lower():
                        hcom[stream]+=duration;hcom_intervals.append(interval)
                elif kind in {'EVENT_WAIT','NOTIFY_WAIT'} and stream==1:
                    main_wait_intervals.append((dec(x['ts']),dec(x['ts'])+dec(x['dur'])))
            need(service and sum(physical.values())>2500,'physical population')
            peak_stream,peak=max(service.items(),key=lambda v:v[1])
            span=end-begin
            need(span>=peak,'relaxed must not exceed span')
            allrows.append({'rank':rank,'ordinal':row['ordinal_in_profile'],
                'instrumented_graph_span_us':str(span),
                'fixed_stream_physical_service_max_us':str(peak),
                'max_service_stream_id':peak_stream,
                'conditional_cross_stream_relaxation_us':str(span-peak),
                'physical_service_sum_all_streams_us':str(sum(service.values())),
                'hcom_named_duration_sum_us':str(sum(hcom.values())),
                'hcom_named_duration_by_stream_us':{str(k):str(v) for k,v in sorted(hcom.items())},
                'hcom_overlap_main_physical_us':str(intersection_us(hcom_intervals,main_physical_intervals)),
                'hcom_overlap_main_wait_us':str(intersection_us(hcom_intervals,main_wait_intervals)),
                'per_stream_physical_service_us':{str(k):str(v) for k,v in sorted(service.items())},
                'physical_task_count':dict(physical),
                'graph_task_count':len(selected),
                'trace_view_sha256':source['trace_view_sha256']})
    result={'status':'conditional_fixed_observed_cost_stream_affinity_relaxation_only',
        'scope':'One Run611 observer-perturbed W0, 24 Model45 FULL Graphs. Preserve measured physical task duration and stream assignment; erase all cross-stream waits/edges and ignore resource contention. This is an optimistic restricted subproblem, not a proven legal or intrinsic-service schedule. HCCL durations can include peer wait. No full Product/FW TPS transfer.',
        'manifest_sha256':sha(manifest),'native_graph_replay_sha256':sha(replay),
        'rows':allrows,'whole_product_framework_only_tps_bound':None}
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(result,indent=2)+'\n')
    from statistics import median
    print(json.dumps({'status':result['status'],'rows':len(allrows),
        'span_us_range':[float(min(dec(x['instrumented_graph_span_us']) for x in allrows)),float(max(dec(x['instrumented_graph_span_us']) for x in allrows))],
        'max_stream_service_us_range':[float(min(dec(x['fixed_stream_physical_service_max_us']) for x in allrows)),float(max(dec(x['fixed_stream_physical_service_max_us']) for x in allrows))],
        'relaxation_us_median':median(float(x['conditional_cross_stream_relaxation_us']) for x in allrows)}))
if __name__=='__main__':main()
