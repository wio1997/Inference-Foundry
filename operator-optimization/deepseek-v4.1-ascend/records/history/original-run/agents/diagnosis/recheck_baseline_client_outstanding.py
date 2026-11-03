#!/usr/bin/env python3
"""Describe client POST-to-last-recorded-SSE intervals, not engine execution."""
import hashlib,json
from pathlib import Path
p=Path(__file__).with_name('baseline-latency-anonymized.json'); raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha=='0819ff3140d285cac881f2e48141548bd08b734e516a1e91687e2108a6dce4e9';data=json.loads(raw);result=[]
for phase in ('measure1','measure2'):
 records=data['phases'][phase]['records'];assert len(records)==8
 edges=[]
 for x in records:
  start=x['start_offset_ms'];first=x['first_response_offset_ms'];last=x['last_response_offset_ms'];assert 0<=start<=first<=last
  edges.extend([(start,1),(last,-1)])
 edges.sort(key=lambda p:(p[0],p[1]));start=edges[0][0];end=edges[-1][0];window=end-start;assert window>0
 bins=[0.0]*9;active=0;last=start
 for time,delta in edges:
  bins[active]+=time-last;active+=delta;assert 0<=active<=8;last=time
 assert active==0 and abs(sum(bins)-window)<1e-6
 total=sum(i*t for i,t in enumerate(bins)); assert abs(total-sum(x['last_response_offset_ms']-x['start_offset_ms'] for x in records))<1e-6
 result.append({'round':phase,'request_count':len(records),'window_ms':window,'outstanding_duration_ms_by_count':bins,'all_8_fraction':bins[8]/window,'mean_outstanding':total/window,'mean_post_to_last_ms':total/len(records),'tokens':8192,'window_output_tps':8192*1000/window})
out={'input_sha256':sha,'definition':'Each request interval spans client POST start to its last recorded non-DONE SSE data, not HTTP completion or engine execution.','limits':['Client-recorded boundaries only; no inference of service admission, prefill, decode, device activity or server-side concurrency.','Two finite benchmark rounds are not a steady-state guarantee.','Source times are within one client clock domain; one SSE line is not necessarily one token.'],'rounds':result}
t=Path(__file__).with_name('baseline-client-outstanding.json');t.write_text(json.dumps(out,indent=2)+'\n')
for x in result:print(x['round'],'all8_fraction',round(x['all_8_fraction'],6),'mean_outstanding',round(x['mean_outstanding'],6),'tps',round(x['window_output_tps'],3))
print('sha256',hashlib.sha256(t.read_bytes()).hexdigest())
