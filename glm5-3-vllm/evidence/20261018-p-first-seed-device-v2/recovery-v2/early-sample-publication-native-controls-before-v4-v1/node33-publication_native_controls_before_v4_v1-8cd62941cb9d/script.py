"""Retain two native64 controls per differing case; never enable old publication."""
from pathlib import Path
import copy,json,re,socket,subprocess,time,urllib.request,uuid
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live1-20261019/node33')
cid='8d81520a203ef0baaf71860185d06fcf1d0d60e907ba1e0029852c7cb8f62cfd'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id']
probe=root/'early-sample-publication-probe';assert not(probe/'enable').exists() and (probe/'disabled').exists()
quality=json.loads((root/'early-sample-publication-quality-v1/result.json').read_text())
assert quality['completed'] and quality['numeric_all_pass']
names=['quality-0','quality-1'];assert sorted(k for k,v in quality['sequence_comparisons'].items() if not v['all_same'])==names
out=root/'early-sample-publication-native-controls-before-v4-v1';out.mkdir(exist_ok=False)
report=dict(completed=False,rows=[],scope='Four native-only controls of existing2 differing full64 cases, all retained. Old publication stays off. No performance rerun; no implication that a sequence difference is harmless or caused by the admission window.')
def idle():
    for _ in range(121):
        values=[]
        for node,port in [(32,29481),(33,29482)]:
            with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(node,port),timeout=5) as f:s=f.read().decode()
            rows=[z for z in s.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
            assert len(rows)>=3;values.extend(float(z.rsplit(' ',1)[1]) for z in rows)
        if all(v==0 for v in values):return
        time.sleep(.25)
    raise AssertionError(values)
def post(node,body,rid):
    req=urllib.request.Request('http://172.16.12.%d:%d/v1/completions'%(node,29481 if node==32 else 29482),json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
    with urllib.request.urlopen(req,timeout=120) as f:return json.load(f)
try:
    idle()
    for arm in ['native-A','native-B']:
        for name in names:
            initial=next(z for z in quality['rows'] if z['case_id']==name and z['mode']=='native')
            rid='pubnativecontrol-'+uuid.uuid4().hex;body=copy.deepcopy(initial['request']);body.pop('stream_options',None);body.pop('kv_transfer_params',None);body.update(stream=False,cache_salt=rid)
            assert body['max_tokens']==64
            pb=dict(body,max_tokens=1,kv_transfer_params=dict(do_remote_decode=True,do_remote_prefill=False,remote_engine_id=None,remote_block_ids=None,remote_host=None,remote_port=None))
            row=dict(case_id=name,mode=arm,request_id=rid,P_request=pb,D_request=body,expected=initial['expected']);report['rows'].append(row)
            pr=post(32,pb,rid);row['P_response']=pr;body['kv_transfer_params']=pr['kv_transfer_params'];response=post(33,body,rid);row['D_response']=response
            tokens=response['choices'][0]['token_ids'];match=re.search(r'[-+]?\d+(?:\.\d+)?',response['choices'][0]['text'])
            assert len(tokens)==response['usage']['completion_tokens']==64 and match and float(match.group())==initial['expected']
            row['token_ids']=tokens;row['numeric_pass']=True;idle();(out/'progress.json').write_text(json.dumps(report)+'\n')
    report['membership']={}
    for name in names:
        native=[z['token_ids'] for z in quality['rows'] if z['case_id']==name and z['mode']=='native']+[z['token_ids'] for z in report['rows'] if z['case_id']==name]
        candidate=[z['token_ids'] for z in quality['rows'] if z['case_id']==name and z['mode']=='adaptive']
        report['membership'][name]=dict(native_unique=len({tuple(z) for z in native}),initial_candidate_in_observed_native_set=all(z in native for z in candidate))
    balances={}
    for path in sorted(probe.glob('rank-*.jsonl')):
        balance=peak=0
        for line in path.read_text().splitlines():
            z=json.loads(line)
            if z['kind']=='native_late_kept':balance+=1
            elif z['kind']=='late_retired':balance-=1
            assert balance>=0;peak=max(peak,balance)
        balances[path.name]=dict(peak_retained_late_objects=peak,final_balance=balance)
    report['retained_object_balances']=balances
    report['balance_scope']='Count of retained objects between logged append/prune, not simultaneous GPU kernels or event-ready state at every instant. It does not attribute the full64 differences.'
    report['completed']=True
except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
(out/'result.json').write_text(json.dumps(report)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['completed'],report.get('failure')
