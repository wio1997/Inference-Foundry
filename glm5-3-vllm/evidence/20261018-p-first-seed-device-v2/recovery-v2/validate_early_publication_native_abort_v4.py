"""One nonce-scoped native backend abort, three complete peers, then cold reuse."""
from pathlib import Path
import concurrent.futures,json,re,socket,subprocess,threading,time,urllib.error,urllib.request,uuid
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live2-20261019/node33')
cid='bfac6522d752e0595df01f537e2f74bbe4c4d00cf9334668283e6885b91e910d'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id']
probe=root/'early-sample-publication-probe';marker=probe/'enable';command=probe/'abort-test-once.json'
assert not marker.exists() and not(probe/'disabled').exists() and not command.exists()
assert json.loads((root/'early-sample-publication-c4-functional-v1/result.json').read_text())['numeric_all_pass']
out=root/'early-sample-publication-native-abort-v1';out.mkdir(exist_ok=False)
cases=json.loads((root/'prefill-padding-cases-v1.json').read_text());report=dict(completed=False,rows=[],scope='One native Scheduler.finish_requests FINISHED_ABORTED call selected by exact task nonce before applying early retirement ACK; three complete peers and two cold64 reuses. This differs from client-close propagation and does not locate the physical conflict window. No performance claim.')
def post(node,body,rid,timeout=120):
    req=urllib.request.Request('http://172.16.12.%d:%d/v1/completions'%(node,29481 if node==32 else 29482),json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
    with urllib.request.urlopen(req,timeout=timeout) as f:return json.load(f)
def idle():
    for i in range(121):
        all_rows=[]
        for n,p in [(32,29481),(33,29482)]:
            with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,p),timeout=5) as f:text=f.read().decode()
            rows=[z for z in text.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
            assert len(rows)>=3;all_rows+=rows
        if all(float(z.rsplit(' ',1)[1])==0 for z in all_rows):return all_rows
        time.sleep(.25)
    raise AssertionError(all_rows)
def prepare(case,rid,budget):
    body=dict(model='glm5',prompt=case['prompt_token_ids'],max_tokens=budget,min_tokens=1,ignore_eos=True,temperature=0,top_p=1,seed=1024,return_token_ids=True,cache_salt=rid,stream=False)
    pb=dict(body,max_tokens=1,kv_transfer_params=dict(do_remote_decode=True,do_remote_prefill=False,remote_engine_id=None,remote_block_ids=None,remote_host=None,remote_port=None))
    response=post(32,pb,rid);body['kv_transfer_params']=response['kv_transfer_params']
    row=dict(request_id=rid,case=case['case_id'],expected=case['expected'],budget=budget,P_response=response,D_request=body);report['rows'].append(row)
    return row
def consume(row,barrier=None,abort=False):
    if barrier:barrier.wait(timeout=20)
    row['D_sent_monotonic_ns']=time.monotonic_ns()
    try:
        response=post(33,row['D_request'],row['request_id'],8 if abort else 120);row['D_response']=response
        if abort:
            assert response.get('usage',{}).get('completion_tokens',0)<row['budget'],'Abort request completed full budget'
            row['aborted_frontend_ended']=True
        else:
            choice=response['choices'][0];tokens=choice['token_ids'];match=re.search(r'[-+]?\d+(?:\.\d+)?',choice['text'])
            assert match and float(match.group())==row['expected']
            assert len(tokens)==response['usage']['completion_tokens']==row['budget'];row['full_response_pass']=True
    except (TimeoutError,socket.timeout) as exc:
        assert abort;row['frontend_timeout_after_internal_abort']=str(exc)
    except urllib.error.HTTPError as exc:
        row['HTTP_error']=dict(status=exc.code,body=exc.read(2048).decode(errors='replace'))
        raise
    except urllib.error.URLError as exc:
        assert abort and isinstance(exc.reason,(TimeoutError,socket.timeout));row['frontend_timeout_after_internal_abort']=str(exc)
try:
    report['before']=idle();marker.touch();needle='earlypubabort-'+uuid.uuid4().hex
    ids=[needle]+['earlypubpeer-'+uuid.uuid4().hex for _ in range(3)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:prepared=list(pool.map(lambda pair:prepare(pair[0],pair[1],128),zip(cases['mixed'],ids)))
    # Keep cold reuse KV ready before the abort; submit D immediately upon
    # the native receipt instead of waiting eight seconds for the front end.
    reuse=[prepare(case,'earlypubreuse-'+uuid.uuid4().hex,64) for case in cases['fixed']]
    command.write_text(json.dumps(dict(client_request_id=needle))+'\n');barrier=threading.Barrier(4)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs=[pool.submit(consume,row,barrier,index==0) for index,row in enumerate(prepared)]
        receipt_path=probe/('abort-test-receipt-'+needle+'.json');started=time.monotonic()
        while True:
            raw=receipt_path.read_bytes() if receipt_path.exists() else b''
            if raw.endswith(b'\n'):
                receipt=json.loads(raw);break
            assert time.monotonic()-started<30,'No native abort receipt'
            time.sleep(.005)
        report['native_receipt_observed_monotonic_ns']=time.monotonic_ns()
        jobs += [pool.submit(consume,row) for row in reuse]
        for job in jobs:job.result()
    assert receipt['client_request_id']==needle and receipt['native_finish_requests_called'] and receipt['deferred_after']>receipt['deferred_before']
    assert receipt['retired_in_reply']<receipt['output_step'] and receipt['processed_before']<receipt['last_request_step']
    report['native_abort_receipt']=receipt
    report['cold_reuse_dispatch_after_receipt_us']=[(z['D_sent_monotonic_ns']-report['native_receipt_observed_monotonic_ns'])/1000 for z in reuse]
    report['after_reuse']=idle();report['completed']=True
except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
finally:
    marker.unlink(missing_ok=True);command.unlink(missing_ok=True)
    for row in report['rows']:
        if 'D_sent_monotonic_ns' not in row:
            try:row['cleanup_unsent_D_response']=post(33,row['D_request'],row['request_id']);row['unsent_own_KV_drained']=True
            except Exception as exc:row['cleanup_failure']=str(exc)
(out/'result.json').write_text(json.dumps(report)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['completed'],report.get('failure')
