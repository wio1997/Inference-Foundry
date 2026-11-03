#!/usr/bin/env python3
"""Aggregate saved diagnostic client timestamps; never emit IDs, prompts or outputs."""
import hashlib,json,pathlib,statistics
R=pathlib.Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
p=R/'diagnostic-20261029/runtime/profile-client-results/measure1-responses.json';raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha=='ca431b9300f4907a4a05978019903d0bf58045fec1b9957b3637f24200701c06'
r=json.loads(raw);assert len(r)==8;vals=[];first_chunks_nonempty=0
for x in r:
 v=[x.get(k) for k in ('start_monotonic_ns','first_sse_data_monotonic_ns','first_nonempty_text_monotonic_ns','end_monotonic_ns')];assert all(isinstance(a,int) for a in v) and v[0]<=v[1]<=v[2]<=v[3];vals.append(v)
 cs=x.get('chunks',[]);assert cs and cs[0]['at_monotonic_ns']==v[1];first_chunks_nonempty+=int(cs[0]['nonempty_text'] is True)
def stat(a):return {'min':min(a),'median':statistics.median(a),'max':max(a)}
print(json.dumps({'input_sha256':sha,'n':len(vals),'first_chunk_nonempty_count':first_chunks_nonempty,'start_span_ms':(max(a[0] for a in vals)-min(a[0] for a in vals))/1e6,'start_to_first_data_ms':stat([(a[1]-a[0])/1e6 for a in vals]),'start_to_first_nonempty_text_ms':stat([(a[2]-a[0])/1e6 for a in vals]),'start_to_http_end_ms':stat([(a[3]-a[0])/1e6 for a in vals]),'first_to_nonempty_ms':stat([(a[2]-a[1])/1e6 for a in vals]),'total_http_window_ms':(max(a[3] for a in vals)-min(a[0] for a in vals))/1e6,'limits':['Timestamp recorded after JSON parsing; nonempty text does not count tokens or establish server first-token timestamp.','Different client and capture process, no request-to-device identity or unconfounded profiler overhead.']},indent=2))
