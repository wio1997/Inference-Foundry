#!/usr/bin/env python3
"""Read-only same-thread host interval coverage; overlap is not exclusive CPU work."""
import sqlite3,pathlib,json,collections
p=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp0_pp0_tp3_dcp0_ep3_rank3_2128_20261029023525650_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_3.db')
c=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True); c.row_factory=sqlite3.Row
tid=9139690412736
ranges={'short_L9':(1793241328964248040,1793241328965935410),'long_L30':(1793241329712702210,1793241329713134370)}
def union(intervals):
    intervals=sorted(intervals); total=0; cur=None
    for a,b in intervals:
        if b<=a: continue
        if cur is None: cur=[a,b]
        elif a<=cur[1]: cur[1]=max(cur[1],b)
        else: total+=cur[1]-cur[0]; cur=[a,b]
    return total+(cur[1]-cur[0] if cur else 0)
out={}
for tag,(lo,hi) in ranges.items():
    item={'gap_ns':hi-lo,'tables':{}}
    for table in ('PYTORCH_API','CANN_API'):
        q=f'''select a.startNs,a.endNs,a.globalTid,a.connectionId,a.type,s.value name'''+(',a.callchainId' if table=='PYTORCH_API' else '')+f''' from {table} a left join STRING_IDS s on a.name=s.id where a.globalTid=? and cast(a.startNs as integer)<? and cast(a.endNs as integer)>?'''
        rows=[dict(r) for r in c.execute(q,(tid,hi,lo))]
        grouped=collections.defaultdict(list); clipped=[]
        for r in rows:
            a=max(int(r['startNs']),lo); b=min(int(r['endNs']),hi); clipped.append((a,b)); grouped[(r['type'],r['name'])].append((a,b))
        arr=[]
        for (typ,name),ints in grouped.items():
            arr.append({'type':typ,'name':name,'count':len(ints),'overlap_union_ns':union(ints)})
        arr.sort(key=lambda x:-x['overlap_union_ns'])
        item['tables'][table]={'row_count':len(rows),'union_ns':union(clipped),'by_name':arr,'selected_rows':[{'name':r['name'],'type':r['type'],'start_offset_ns':int(r['startNs'])-lo,'end_offset_ns':int(r['endNs'])-lo,'callchainId':r.get('callchainId')} for r in rows]}
    out[tag]=item
print(json.dumps({'tid':tid,'results':out,'limits':'same globalTid host timestamps, clipped interval unions; overlapping/nested API scope is not exclusive CPU execution or waiting; no host-device latency calculation; unrecorded work is unknown'},indent=2))
