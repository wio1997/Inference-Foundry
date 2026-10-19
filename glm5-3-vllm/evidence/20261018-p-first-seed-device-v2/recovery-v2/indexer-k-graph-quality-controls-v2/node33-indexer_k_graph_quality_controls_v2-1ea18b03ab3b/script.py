"""Finite streaming native/adaptive/native full64 controls for this exact reuse build."""
from pathlib import Path
import copy,json,re,socket,subprocess,time,urllib.request,uuid
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live2-20261019/node33')
cid='a4db690c1cf2ae583cdcbf84464e1393a24ab4dd3f9df4b192be64c3e2777011'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id']
probe=root/'indexer-k-graph-overlap-probe';assert not(probe/'enable').exists() and (probe/'disabled').exists()
quality=json.loads((root/'indexer-k-graph-overlap-quality-v2/result.json').read_text())
assert quality['completed'] and quality['numeric_all_pass']
names=sorted(k for k,v in quality['sequence_comparisons'].items() if not v['all_same']);assert names
out=root/'indexer-k-graph-overlap-quality-controls-v2';out.mkdir(exist_ok=False)
report=dict(completed=False,rows=[],scope='One native/adaptive/native streaming triple per differing full64 case, using the original fixed inference parameters. Match initial streaming request format and all inference parameters; fresh cache salt. All rows retained. No general QA or harmless-rounding conclusion; no formal performance yet.')
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
def stream(body,rid,row):
    req=urllib.request.Request('http://172.16.12.33:29482/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
    chunks=[];tokens=[];text='';usage=None
    with urllib.request.urlopen(req,timeout=120) as response:
        assert response.status==200
        for line in response:
            if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
            z=json.loads(line[6:]);assert not z.get('error'),z;chunks.append(z)
            if z.get('usage'):usage=z['usage']
            for choice in z.get('choices',[]):tokens+=choice.get('token_ids') or [];text+=choice.get('text','')
    row['chunks']=chunks;return dict(choices=[dict(token_ids=tokens,text=text)],usage=usage)
try:
    idle()
    for arm in ['native-A','adaptive','native-B']:
        idle()
        if arm=='adaptive':(probe/'enable').touch()
        else:(probe/'enable').unlink(missing_ok=True)
        for name in names:
            initial=next(z for z in quality['rows'] if z['case_id']==name and z['mode']=='native')
            rid='indexercontrol-'+uuid.uuid4().hex;body=copy.deepcopy(initial['request']);body.pop('stream_options',None);body.pop('kv_transfer_params',None);body.update(stream=True,stream_options=dict(include_usage=True),cache_salt=rid)
            assert body['max_tokens']==64
            pb=dict(body,max_tokens=1,stream=False,kv_transfer_params=dict(do_remote_decode=True,do_remote_prefill=False,remote_engine_id=None,remote_block_ids=None,remote_host=None,remote_port=None))
            pb.pop('stream_options',None)
            row=dict(case_id=name,mode=arm,request_id=rid,P_request=pb,D_request=body,expected=initial['expected']);report['rows'].append(row)
            pr=post(32,pb,rid);row['P_response']=pr;body['kv_transfer_params']=pr['kv_transfer_params'];response=stream(body,rid,row);row['D_response']=response
            tokens=response['choices'][0]['token_ids'];match=re.search(r'[-+]?\d+(?:\.\d+)?',response['choices'][0]['text'])
            assert len(tokens)==response['usage']['completion_tokens']==64 and match and float(match.group())==initial['expected']
            row['token_ids']=tokens;row['numeric_pass']=True;idle();(out/'progress.json').write_text(json.dumps(report)+'\n')
    report['membership']={}
    for name in names:
        native=[z['token_ids'] for z in quality['rows'] if z['case_id']==name and z['mode']=='native']+[z['token_ids'] for z in report['rows'] if z['case_id']==name and z['mode'].startswith('native')]
        candidate=[z['token_ids'] for z in quality['rows'] if z['case_id']==name and z['mode']=='adaptive']+[z['token_ids'] for z in report['rows'] if z['case_id']==name and z['mode']=='adaptive']
        report['membership'][name]=dict(native_unique=len({tuple(z) for z in native}),both_initial_and_repeat_candidate_in_observed_native_set=all(z in native for z in candidate))
    report['completed']=True
except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
finally:(probe/'enable').unlink(missing_ok=True)
(out/'result.json').write_text(json.dumps(report)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['completed'],report.get('failure')
