#!/usr/bin/env python3
"""Compare engine-tag histogram deltas from frozen snapshots; not request binding."""
import contextlib, io, json, re, runpy, sys
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

root=Path(__file__).resolve().parents[2]
saved=sys.argv
try:
    sys.argv=[str(root/'agents/diagnosis/recheck_baseline_phase_histograms.py')]
    with contextlib.redirect_stdout(io.StringIO()):
        data=runpy.run_path(sys.argv[0],run_name='baseline_histogram_engine_check')
finally:
    sys.argv=saved
snap=data['snapshots']; metric_names=('time_to_first_token_seconds','request_prefill_time_seconds','request_decode_time_seconds','request_generation_tokens')
out={}
for phase in ('measure1','measure2'):
    before,after=snap[(phase,'before')],snap[(phase,'after')]
    means={}
    for name in metric_names:
        means[name]=[]
        for engine in ('0','1'):
            count=after[(name,'count',engine)]-before[(name,'count',engine)]
            amount=after[(name,'sum',engine)]-before[(name,'sum',engine)]
            assert count==4 and amount>=0
            means[name].append(str(amount/count))
    buckets={};extras={}
    for when in ('before','after'):
        path=data['root']/phase/f'metrics-{when}.txt'
        raw=path.read_bytes();assert sha256(raw).hexdigest()==data['expected'][(phase,when)]
        parsed={};extra={}
        for line in raw.decode().splitlines():
            x=re.fullmatch(r'vllm:(prefix_cache_(?:queries|hits)_total|prompt_tokens_by_source_total|request_prompt_tokens_(?:count|sum)|request_prefill_kv_computed_tokens_(?:count|sum))\{([^}]*)\}\s+([^\s]+)',line)
            if x:
                e=re.search(r'(?:^|,)engine="([01])"(?:,|$)',x[2]);source=re.search(r'(?:^|,)source="([^"]+)"(?:,|$)',x[2]);assert e
                key=(x[1],e[1],source[1] if source else None);assert key not in extra;extra[key]=Decimal(x[3])
            m=re.fullmatch(r'vllm:time_to_first_token_seconds_bucket\{([^}]*)\}\s+([^\s]+)',line)
            if not m: continue
            engine=re.search(r'(?:^|,)engine="([01])"(?:,|$)',m[1]);limit=re.search(r'(?:^|,)le="([^"]+)"(?:,|$)',m[1])
            assert engine and limit
            key=(engine[1],limit[1]);assert key not in parsed;parsed[key]=Decimal(m[2])
        buckets[when]=parsed;extras[when]=extra
    delta={limit:[int(buckets['after'][(e,limit)]-buckets['before'][(e,limit)]) for e in ('0','1')] for limit in ('0.5','0.75','1.0','2.5')}
    assert delta=={'0.5':[0,2],'0.75':[1,2],'1.0':[3,4],'2.5':[4,4]}
    ed={}
    for name,source in [('request_prompt_tokens_count',None),('request_prompt_tokens_sum',None),('request_prefill_kv_computed_tokens_count',None),('request_prefill_kv_computed_tokens_sum',None),('prefix_cache_queries_total',None),('prefix_cache_hits_total',None),('prompt_tokens_by_source_total','local_compute'),('prompt_tokens_by_source_total','local_cache_hit'),('prompt_tokens_by_source_total','external_kv_transfer')]:
        ed[name+(':'+source if source else '')]=[int(extras['after'][(name,e,source)]-extras['before'][(name,e,source)]) for e in ('0','1')]
    assert ed['request_prompt_tokens_count']==ed['request_prefill_kv_computed_tokens_count']==[4,4]
    assert ed['request_prompt_tokens_sum']==ed['prefix_cache_queries_total']==[131072,131072]
    assert ed['request_prefill_kv_computed_tokens_sum']==ed['prompt_tokens_by_source_total:local_compute']==[13056,13184]
    assert ed['prefix_cache_hits_total']==ed['prompt_tokens_by_source_total:local_cache_hit']==[118016,117888]
    assert ed['prompt_tokens_by_source_total:external_kv_transfer']==[0,0]
    out[phase]={'counts_per_engine':[4,4],'means':means,'ttft_bucket_deltas':delta,'prompt_cache_counter_deltas':ed}
print(json.dumps({'rounds':out,'units':'time seconds; generation tokens per observation','limits':'Two scrape windows; four aggregate observations per label; per-request identity, request work, historical runtime mapping and causality are not established.'},indent=2))
