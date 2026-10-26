#!/usr/bin/env python3
"""Real 80K/600 stream with cache usage and source-layer audit, outside AIS timing."""
from bench_profile import PROFILE
from bench_profile import benchmark_locks as acquire_benchmark_locks
import argparse
import hashlib
import fcntl
import json
from pathlib import Path
import time
import urllib.request
from backend_source_audit import verify_backend_source

p=argparse.ArgumentParser()
p.add_argument('--root',default='/work/aisbench-glm53-run')
p.add_argument('--label',default='long80k-warm93')
p.add_argument('--cc',default='smoke600')
p.add_argument('--output',type=int,default=600)
p.add_argument('--expected-key')
a=p.parse_args();r=Path(a.root);(r/'logs').mkdir(exist_ok=True)
locks=[]
locks=acquire_benchmark_locks(r)
assert not (r/f'logs/{a.label}.json').exists() and not (r/f'logs/{a.label}.sse').exists(),'Existing smoke trace requires audit'
row=json.loads((r/f'datasets/cc{a.cc}/test.jsonl').read_text())
body=dict(model=PROFILE['model_name'],messages=[dict(role='user',content=row['question'])],stream=True,
          stream_options=dict(include_usage=True,continuous_usage_stats=True),temperature=0,max_tokens=a.output,ignore_eos=True,
          return_cached_tokens_details=True,chat_template_kwargs=dict(enable_thinking=False))
req=urllib.request.Request(PROFILE['router_url'].rstrip('/')+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
start=time.perf_counter();first=None;usage=None;cached_details=None;content=[];done=False;usage_events=[]
with (r/f'logs/{a.label}.sse').open('wb')as log,urllib.request.urlopen(req,timeout=7200)as response:
    for line in response:
        log.write(line);log.flush()
        if not line.startswith(b'data:'):continue
        data=line[5:].strip()
        if data==b'[DONE]':done=True;break
        chunk=json.loads(data)
        assert not chunk.get('error'),chunk
        if chunk.get('usage'):
            usage=chunk['usage']
            usage_events.append(dict(t=time.perf_counter(),completion_tokens=usage.get('completion_tokens')))
        cd=(chunk.get('sglext')or{}).get('cached_tokens_details')
        if cd is not None:cached_details=cd
        for choice in chunk.get('choices',[]):
            d=choice.get('delta',{});value=d.get('content')or d.get('reasoning_content')or d.get('reasoning')
            if value:
                if first is None:first=time.perf_counter()
                content.append(value)
record=dict(usage_events=usage_events,done=done,e2e_s=time.perf_counter()-start,ttft_s=None if first is None else first-start,
            usage=usage,cached_tokens_details=cached_details,prompt_sha256=hashlib.sha256(row['question'].encode()).hexdigest(),
            output_text=''.join(content),nonempty_generation=bool(content))
errors=[]
counts=[e['completion_tokens']for e in usage_events]
if not counts or any(not isinstance(n,int)for n in counts)or counts!=sorted(counts)or len(set(counts)-{0})<2 or counts[-1]!=a.output:errors.append('Progressive streaming usage missing or invalid')
if not done or not content:errors.append('Incomplete or empty stream')
if not usage or usage.get('prompt_tokens')!=80000 or usage.get('completion_tokens')!=a.output:errors.append('Actual server token counts differ')
if ((usage or{}).get('prompt_tokens_details')or{}).get('cached_tokens')!=74368:errors.append('Actual cached token count differs from74368')
if cached_details is None:errors.append('Missing actual layer breakdown')
else:
    try:record['backend_source_proof']=verify_backend_source(r,a.cc,cached_details)
    except AssertionError as e:errors.append('Actual L3 source audit failed: '+str(e))
if a.expected_key and a.expected_key not in ''.join(content):errors.append('Known verification key not recovered')
record.update(valid=not errors,errors=errors)
(r/f'logs/{a.label}.json').write_text(json.dumps(record,indent=2)+'\n')
assert not errors,record
print(json.dumps({k:v for k,v in record.items()if k!='output_text'}),flush=True)
