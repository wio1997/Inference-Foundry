"""One bounded recapture of the frozen C8/P90/32K/1K case, not a speed score."""
from pathlib import Path
import concurrent.futures
import datetime
import hashlib
import hmac
import json
import os
import threading
import time
import traceback
import urllib.request

APP = Path('/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash')
INPUTS = Path('/bench-inputs/workload/frozen')
OUT = APP/'profile-client-results'
OUT.mkdir(exist_ok=False)
GATE = APP/'metadata/PROFILE_ACTIVE'
URL = 'http://127.0.0.1:8172'
state = dict(state='STARTING', not_a_benchmark_score=True, profile_max_iterations=12,
             case=dict(prompt_tokens=32768, output_tokens=1024, concurrency=8, prefix_hit=0.9))
def save(label):
    state.update(state=label,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (OUT/'progress.json').write_text(json.dumps(state,indent=2)+'\n')
    print(label,flush=True)
def request(path, body=None, headers=None, timeout=900):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req=urllib.request.Request(URL+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json',**(headers or {})})
    return opener.open(req,timeout=timeout)
def metrics():
    with request('/metrics',timeout=15) as response:return response.read().decode()
def metric(raw, name):
    return sum(float(row.rsplit(' ',1)[1]) for row in raw.splitlines() if row.startswith('vllm:'+name+'{') or row.startswith('vllm:'+name+' '))
def complete(row, phase, index, stream=False, barrier=None):
    rid='wio-diag169-20261029-'+phase+'-'+str(index)
    body=dict(model='dsv41',prompt=row['prompt'],max_tokens=row['max_tokens'],temperature=0,ignore_eos=True,stream=stream,request_id=rid)
    if stream:body['stream_options']={'include_usage':True}
    headers={'X-data-parallel-rank':str(row['dp_rank'])} if row.get('dp_rank') is not None else {}
    if barrier:barrier.wait()
    output=dict(request_id=rid,expected_engine_request_id='cmpl-'+rid+'-0',
                start_epoch_ns=time.time_ns(),start_monotonic_ns=time.monotonic_ns(),chunks=[],prediction='')
    key=os.environ.get('WIO_DIAG_HMAC_KEY','').encode()
    output['expected_engine_request_hmac_sha256']=hmac.new(key,output['expected_engine_request_id'].encode(),hashlib.sha256).hexdigest()
    with request('/v1/completions',body,headers) as response:
        if not stream:
            data=json.load(response);output.update(usage=data['usage'],prediction=data['choices'][0]['text'],response_id=data.get('id'))
        else:
            raw=[]
            for line in response:
                text=line.decode();raw.append(text)
                if not text.startswith('data:'):continue
                text=text[5:].strip()
                if text=='[DONE]':continue
                data=json.loads(text)
                now=time.monotonic_ns()
                visible=any(choice.get('text','') for choice in data.get('choices',[]))
                output['chunks'].append(dict(at_monotonic_ns=now,choices=len(data.get('choices',[])),nonempty_text=visible))
                if 'first_sse_data_monotonic_ns' not in output:output['first_sse_data_monotonic_ns']=now
                if visible and 'first_nonempty_text_monotonic_ns' not in output:output['first_nonempty_text_monotonic_ns']=now
                output['response_id']=data.get('id',output.get('response_id'))
                for choice in data.get('choices',[]):output['prediction']+=choice.get('text','')
                if data.get('usage'):output['usage']=data['usage']
            output['raw_sse']=''.join(raw)
    output.update(end_monotonic_ns=time.monotonic_ns(),end_epoch_ns=time.time_ns())
    return output
def parallel(rows, phase, stream=False):
    barrier=threading.Barrier(len(rows))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
        return list(pool.map(lambda pair:complete(pair[1],phase,pair[0],stream,barrier),enumerate(rows)))
def phase(name, profile=False):
    folder=INPUTS/name
    seeds=json.loads((folder/'warmup.json').read_text())
    save(name+'_SEEDING')
    seeded=parallel(seeds,name+'-seed')
    (OUT/(name+'-seeds.json')).write_text(json.dumps(seeded,ensure_ascii=False))
    assert all(item['usage']['prompt_tokens']==29696 and item['usage']['completion_tokens']==1 for item in seeded)
    rows=[json.loads(line) for line in next(folder.rglob('*.requests.jsonl')).read_text().splitlines()]
    state[name+'_prompt_sha256']=hashlib.sha256(json.dumps([row['question'] for row in rows],ensure_ascii=False).encode()).hexdigest()
    before=metrics();(OUT/(name+'-metrics-before.txt')).write_text(before)
    started=False
    if profile:
        GATE.write_text(json.dumps({'time_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'phase':name}))
        with request('/start_profile',{},timeout=120) as response:
            state['profile_start_http']=response.status;started=True
    save(name+'_REQUESTS_RUNNING')
    try:
        responses=parallel([dict(prompt=row['question'],max_tokens=1024) for row in rows],name,stream=profile)
    finally:
        try:
            if started:
                with request('/stop_profile',{},timeout=600) as response:state['profile_stop_http']=response.status
        finally:
            GATE.unlink(missing_ok=True)
    after=metrics()
    (OUT/(name+'-responses.json')).write_text(json.dumps(responses,ensure_ascii=False))
    (OUT/(name+'-metrics-after.txt')).write_text(after)
    names=['prefix_cache_queries_total','prefix_cache_hits_total','prompt_tokens_total','generation_tokens_total','request_success_total','num_preemptions_total','spec_decode_num_draft_tokens_total','spec_decode_num_accepted_tokens_total']
    delta={key:metric(after,key)-metric(before,key) for key in names};state[name+'_counter_deltas']=delta
    assert len(responses)==8 and all(item['usage']['prompt_tokens']==32768 and item['usage']['completion_tokens']==1024 for item in responses)
    assert delta['prompt_tokens_total']==262144 and delta['generation_tokens_total']==8192 and delta['request_success_total']==8,delta
    state[name+'_case_valid']=(delta['prefix_cache_queries_total']==262144 and delta['prefix_cache_hits_total']==235904 and delta['num_preemptions_total']==0)
    if not state[name+'_case_valid']:state[name+'_case_mismatch']=delta
    if profile:
        refs={str(row['data_id']):row['prediction'] for row in map(json.loads,(folder/'frozen-vllm-details.jsonl').read_text().splitlines())}
        matches=[dict(data_id=i,exact_utf8_match=value['prediction'].encode()==refs[str(i)].encode()) for i,value in enumerate(responses)]
        (OUT/'prediction-comparison.json').write_text(json.dumps(matches,indent=2))
        state['exact_frozen_prediction_matches']=sum(row['exact_utf8_match'] for row in matches)
try:
    with request('/health',timeout=10) as response:assert response.status==200
    phase('warmup')
    phase('measure1',profile=True)
    save('REQUESTS_COMPLETE_EXPORT_PENDING')
except BaseException as exc:
    state.update(error=repr(exc),traceback=traceback.format_exc())
    GATE.unlink(missing_ok=True)
    save('FAILED')
    raise
