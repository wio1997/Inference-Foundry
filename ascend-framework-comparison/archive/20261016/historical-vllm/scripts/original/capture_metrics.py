"""Read-only endpoint metrics snapshot or bounded-lifetime sampler."""
import concurrent.futures,json,os,pathlib,time,urllib.request

ENDPOINTS = {f'P{i}': f'http://172.16.10.{node}:9200/metrics'
             for i,node in enumerate((168,169,166,167))}
ENDPOINTS['D'] = 'http://172.16.10.170:9100/metrics'
def fetch(item):
    role,url=item
    try:
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url,timeout=8) as r: body=r.read().decode()
        return role,{'ok':True,'text':body}
    except Exception as e:return role,{'ok':False,'error':str(e)}
def snapshot():
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        return {'time_ns':time.time_ns(),'endpoints':dict(pool.map(fetch,ENDPOINTS.items()))}
out=pathlib.Path(os.environ['SNAPSHOT_OUT']);out.parent.mkdir(parents=True,exist_ok=True)
stop=os.environ.get('METRICS_STOP_FILE')
if not stop:
    out.write_text(json.dumps(snapshot())+'\n')
else:
    prefixes=('vllm:num_requests_','vllm:kv_cache_usage','vllm:gpu_cache_usage',
              'vllm:prefix_cache_','vllm:external_prefix_cache_',
              'vllm:generation_tokens_total','vllm:prompt_tokens_total',
              'vllm:spec_decode_','vllm:request_success_total',
              'vllm:request_prefill_time_seconds_', 'vllm:request_queue_time_seconds_',
              'vllm:request_decode_time_seconds_', 'vllm:time_to_first_token_seconds_')
    with out.open('a',buffering=1) as f:
        while not pathlib.Path(stop).exists():
            s=snapshot()
            for value in s['endpoints'].values():
                if value.get('ok'):
                    value['text']='\n'.join(line for line in value['text'].splitlines()
                                             if line.startswith(prefixes))
            f.write(json.dumps(s)+'\n');time.sleep(2)
