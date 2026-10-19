"""One C1 and one concurrent C4 diagnostic for actual added-copy transport."""
from pathlib import Path
import concurrent.futures
import datetime
import json
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import uuid

assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live2-20261019/node33')
cid='a4db690c1cf2ae583cdcbf84464e1393a24ab4dd3f9df4b192be64c3e2777011'
plan=json.loads((root/'plan.json').read_text())
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id']
left=plan['container']['lifetime_seconds']-(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(d['State']['StartedAt'][:26]+'+00:00')).total_seconds()
assert left>2400 and shutil.disk_usage(root).free>16*1024**3
probe=root/'indexer-k-graph-overlap-probe'
marker, labels=probe/'enable',probe/'profile-copy-labels'
assert (probe/'disabled').exists() and not marker.exists() and not labels.exists()
assert len(list(probe.glob('rank-*.jsonl')))==16
cases=json.loads((root/'prefill-padding-cases-v1.json').read_text())['mixed']
assert len(cases)==4
out=root/'indexer-k-graph-four-way-functional-v2'
out.mkdir(exist_ok=False)
report=dict(completed=False,rows=[],profiles=[],attempts=[],CID=cid,
            scope='One adaptive barrier-C4 functional cohort with current-native bitwise observer on,32 tokens and original PD. Resolve actual N24 producer/consumer shape coverage before service timing; no profiler or performance claim.')


def post(node,path,body,rid=None):
    request=urllib.request.Request('http://172.16.12.%d:%d%s'%(node,29481 if node==32 else 29482,path),json.dumps(body).encode(),
                                   {'Content-Type':'application/json',**({'X-Request-Id':rid} if rid else {})})
    with urllib.request.urlopen(request,timeout=180) as f:
        data=f.read()
    return json.loads(data) if data else None


def idle():
    for _ in range(121):
        values=[]
        for node,port in [(32,29481),(33,29482)]:
            with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,port),timeout=5) as f:
                text=f.read().decode()
            rows=[z for z in text.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
            assert len(rows)>=3
            values.extend(float(z.rsplit(' ',1)[1]) for z in rows)
        if all(x==0 for x in values):
            return
        time.sleep(.25)
    raise AssertionError(values)


def prepare(case,condition,phase):
    rid='indexer-copy-'+uuid.uuid4().hex
    body=dict(model='glm5',prompt=case['prompt_token_ids'],max_tokens=32,min_tokens=1,ignore_eos=True,
              temperature=0,top_p=1,seed=1024,return_token_ids=True,cache_salt=rid,stream=False)
    producer=dict(body,max_tokens=1,kv_transfer_params=dict(do_remote_decode=True,do_remote_prefill=False,
                  remote_engine_id=None,remote_block_ids=None,remote_host=None,remote_port=None))
    response=post(32,'/v1/completions',producer,rid)
    params=response['kv_transfer_params']
    assert params['do_remote_prefill'] and params['remote_num_prompt_tokens']==len(body['prompt'])
    body['kv_transfer_params']=params
    row=dict(condition=condition,phase=phase,case_id=case['case_id'],expected=case['expected'],request_id=rid,
             P_request=producer,P_response=response,D_request=body)
    report['rows'].append(row)
    return body,row


def consume(prepared,barrier):
    body,row=prepared
    barrier.wait(timeout=20)
    row['D_sent_monotonic_ns']=time.monotonic_ns()
    result=post(33,'/v1/completions',body,row['request_id'])
    row['D_response']=result
    tokens=result['choices'][0]['token_ids']
    number=re.search(r'[-+]?\d+(?:\.\d+)?',result['choices'][0]['text'])
    row['numeric_pass']=bool(number and float(number.group())==row['expected'])
    row['budget_pass']=len(tokens)==result['usage']['completion_tokens']==32
    assert row['numeric_pass'] and row['budget_pass']


def group(count,phase,profiling):
    idle()
    condition='C'+str(count)
    marker.touch()
    chosen=cases[:count]
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        prepared=list(pool.map(lambda c:prepare(c,condition,phase),chosen))
    before={p.name for p in (root/'profile').glob('*_ascend_pt')}
    attempt=dict(condition=condition,phase=phase)
    started=False
    try:
        if profiling:
            labels.touch()
            report['attempts'].append(attempt)
            started=True
            attempt['start_profile']=post(33,'/start_profile',{})
        barrier=threading.Barrier(count)
        with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
            list(pool.map(lambda p:consume(p,barrier),prepared))
        idle()
    finally:
        if started:
            attempt['stop_profile']=post(33,'/stop_profile',{})
            attempt['profile_stopped']=True
        labels.unlink(missing_ok=True)
    if profiling:
        dirs=sorted(p for p in (root/'profile').glob('*_ascend_pt') if p.name not in before)
        attempt['dirs']=[str(p) for p in dirs]
        attempt['ranks']=sorted(int(re.search(r'_rank(\d+)_',p.name).group(1)) for p in dirs)
        attempt['raw_bytes']=sum(p.stat().st_size for directory in dirs for p in directory.rglob('*') if p.is_file())
        assert attempt['ranks']==list(range(16)) and attempt['raw_bytes']<4*1024**3
        report['profiles'].append(attempt)
    (out/'progress.json').write_text(json.dumps(report)+'\n')


try:
    (probe/'disabled').unlink()
    group(4,'actual_four_way_functional',False)
    idle()
    report['completed']=True
except Exception as exc:
    report['failure']=type(exc).__name__+': '+str(exc)
finally:
    marker.unlink(missing_ok=True)
    labels.unlink(missing_ok=True)
    (probe/'disabled').touch()
    for row in report['rows']:
        if 'D_sent_monotonic_ns' not in row:
            try:
                row['cleanup_D_response']=post(33,'/v1/completions',row['D_request'],row['request_id'])
                row['unsent_own_KV_drained']=True
            except Exception as exc:
                row['cleanup_failure']=str(exc)
    for attempt in report['attempts']:
        if not attempt.get('profile_stopped'):
            try:
                attempt['cleanup_stop_profile']=post(33,'/stop_profile',{})
                attempt['profile_stopped']=True
            except Exception as exc:
                attempt['stop_failure']=str(exc)
report['candidate_restored_native']=not marker.exists()
report['observer_off']=(probe/'disabled').exists()
(out/'result.json').write_text(json.dumps(report)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['completed'],report.get('failure')
