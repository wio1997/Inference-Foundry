#!/usr/bin/env python3
"""Summarize existing EP8 selected flow X intervals; no raw trace or NPU access."""
from pathlib import Path
from decimal import Decimal
import hashlib,json
from collections import Counter
D=Path(__file__).resolve().parent
p=D/'ep8-l14-l39-flow-intervals.json'
b=p.read_bytes(); src=json.loads(b)
result={'source_file':p.name,'source_sha256':hashlib.sha256(b).hexdigest(),'trace_sha256':src['trace_sha256'],'scope':'EP8, selected host-layer flows only, matching conditions in source; separate layer summaries, no cross-rank time','layers':{}}
for layer,z in src['layers'].items():
    duration=Decimal(z['mc2_duration_us']); events={}
    for r in z['rows']:
        if r['name'] not in ('EVENT RECORD','EVENT WAIT'): continue
        # The source matches X uniquely by (pid, tid, timestamp); relative timestamps
        # use the same MC2 anchor within this scope and preserve uniqueness.
        key=(r['track_pid'],r['track_tid'],r['relative_start_us'])
        fields=(r['name'],r['duration_us'],r['task'],r['stream'])
        if key in events:
            assert events[key][0]==fields, ('conflicting X',layer,key)
            events[key][1]+=1
        else: events[key]=[fields,1,r]
    counts=Counter(); waits=[]; same_counts=Counter(); alias=0
    for (pid,tid,start),(_,n,r) in events.items():
        alias+=n-1
        s=Decimal(start); e=s+Decimal(r['duration_us'])
        # zero length belongs before if its timestamp precedes start.
        relation='before' if e<=0 else 'after' if s>=duration else 'overlap'
        counts[(r['name'],relation)]+=1
        if tid=='47' and pid=='755144': same_counts[(r['name'],relation)]+=1
        if r['name']=='EVENT WAIT': waits.append({'track_pid':pid,'track_tid':tid,'task':r['task'],'start_relative_us':str(s),'end_relative_us':str(e),'duration_us':r['duration_us'],'relation':relation,'same_mc2_track':tid=='47' and pid=='755144'})
    waits.sort(key=lambda r:Decimal(r['duration_us']),reverse=True)
    result['layers'][layer]={'scope':z['scope'],'mc2_duration_us':str(duration),'selected_sync_flow_rows':sum(v[1] for v in events.values()),'unique_sync_x':len(events),'shared_x_extra_flow_rows':alias,'counts':{f'{a} {b}':n for (a,b),n in sorted(counts.items())},'same_mc2_track_counts':{f'{a} {b}':n for (a,b),n in sorted(same_counts.items())},'longest_waits':waits[:4]}
print(json.dumps(result,indent=2,ensure_ascii=False))
