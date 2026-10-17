#!/usr/bin/env python3
"""Evict only D local residency using ordinary native requests, then re-GET same bad pool keys."""
import json
from pathlib import Path
import socket
import time
import urllib.request
import uuid


def main():
    assert socket.gethostname() == 'B4-33U-Atlas900RCKA3-32'
    base = Path('/data/wio/glm5-3-vllm-execution/20261017-pd-baseline')
    out = base/'tests/same-key-external-reread-v1'
    out.mkdir(exist_ok=False)
    previous = base/'tests/known-failure-key-replay-v1'
    bad = json.loads((previous/'wave-0-pool.json').read_text())
    bad_local = json.loads((previous/'wave-0-D-pool-local.json').read_text())
    fixture = json.loads((base/'tests/baseline-c1-v1/fixture-4096-128.json').read_text())
    deadline = time.monotonic()+420
    report = {'completed':False,'eviction_requests':41,'D_blocks':1289,'eviction_full_blocks':41*32,
              'scope':'Existing reset API is absent in live OpenAPI.41 ordinary C1 cold D4096/1 requests with unique salts exceed D1289 block capacity and evict old local prefix; consumer PUT disabled, source P idle, so pool objects are unchanged. Then same bad salt/key external D128 and local D8. No new API, RPC, model reload, tensor observation, kernel or precision change. Extra external GET must be496 keys, initial local prefix hit0. Output/logprobs do not prove all KV bytes.'}
    def rpc(payload):
        request = urllib.request.Request('http://172.16.12.33:29482/v1/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=min(90,max(1,deadline-time.monotonic()))) as response:
            return json.load(response)
    def save(name,payload,value):
        (out/(name+'.json')).write_text(json.dumps({'request':payload,'response':value},indent=2)+'\n')
    def metrics(name):
        with urllib.request.urlopen('http://172.16.12.33:29482/metrics',timeout=10) as response: raw=response.read().decode()
        (out/(name+'.txt')).write_text(raw)
        values={}
        for line in raw.splitlines():
            if not line or line.startswith('#'):continue
            key,value=line.rsplit(' ',1);key=key.split('{',1)[0];values[key]=values.get(key,0)+float(value)
        return values
    try:
        initial = metrics('before-eviction')
        for key,value in initial.items():
            if key in ('vllm:num_requests_running','vllm:num_requests_waiting'):assert value==0
        for i in range(41):
            assert deadline-time.monotonic()>45
            payload=dict(bad['request'],max_tokens=1,cache_salt='evict-D-native-'+uuid.uuid4().hex)
            payload.pop('request_id',None)
            value=rpc(payload);save(f'evict-{i:02d}',payload,value)
            assert value['usage']['prompt_tokens']==4096
            if i%10==0:print('Native eviction requests completed:',i+1,flush=True)
        evicted=metrics('after-eviction')
        report['eviction_sdk_get_keys_delta']=evicted.get('vllm:ascend_store_load_get_keys_total',0)-initial.get('vllm:ascend_store_load_get_keys_total',0)
        report['eviction_native_prefix_hits_delta']=evicted.get('vllm:prefix_cache_hits_total',0)-initial.get('vllm:prefix_cache_hits_total',0)
        assert report['eviction_sdk_get_keys_delta']==report['eviction_native_prefix_hits_delta']==0
        payload=dict(bad['request'],request_id='glm53-same-key-reread-'+uuid.uuid4().hex)
        value=rpc(payload);save('same-key-external-reread',payload,value)
        after=metrics('after-external-reread')
        report['reread_consumer_id']=value['id']
        report['reread_sdk_get_keys_delta']=after.get('vllm:ascend_store_load_get_keys_total',0)-evicted.get('vllm:ascend_store_load_get_keys_total',0)
        report['reread_native_prefix_hits_delta']=after.get('vllm:prefix_cache_hits_total',0)-evicted.get('vllm:prefix_cache_hits_total',0)
        assert report['reread_sdk_get_keys_delta']==496 and report['reread_native_prefix_hits_delta']==0
        report['reread_128_matches_original_bad']=value['choices'][0]['token_ids']==bad['response']['choices'][0]['token_ids']
        report['reread_128_matches_original_native']=value['choices'][0]['token_ids']==fixture['reference_ids']
        local_payload=dict(payload,max_tokens=8,logprobs=20,request_id='glm53-same-key-local-'+uuid.uuid4().hex)
        local=rpc(local_payload);save('same-key-local',local_payload,local)
        final=metrics('after-local-reuse')
        report['local_sdk_get_keys_delta']=final.get('vllm:ascend_store_load_get_keys_total',0)-after.get('vllm:ascend_store_load_get_keys_total',0)
        report['local_native_prefix_hits_delta']=final.get('vllm:prefix_cache_hits_total',0)-after.get('vllm:prefix_cache_hits_total',0)
        assert report['local_sdk_get_keys_delta']==0 and report['local_native_prefix_hits_delta']==3968
        a=local['choices'][0];b=bad_local['response']['choices'][0]
        report['local_8_matches_original_bad']=a['token_ids']==b['token_ids']
        report['local_top20_exactly_matches_original_bad']=a['logprobs']['top_logprobs']==b['logprobs']['top_logprobs'] and a['logprobs']['token_logprobs']==b['logprobs']['token_logprobs']
        report['completed']=True
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report,indent=2)+'\nGLM53_RESULT_JSON_END',flush=True)
    assert report['completed'],report.get('failure')


if __name__ == '__main__':
    main()
