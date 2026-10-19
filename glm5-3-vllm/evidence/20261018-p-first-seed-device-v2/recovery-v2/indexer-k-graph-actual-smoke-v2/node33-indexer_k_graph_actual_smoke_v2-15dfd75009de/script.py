"""Small actual-selection test before the larger finite correctness cohort."""
from pathlib import Path
import json
import re
import socket
import subprocess
import time
import urllib.request
import uuid

assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live2-20261019/node33')
cid='a4db690c1cf2ae583cdcbf84464e1393a24ab4dd3f9df4b192be64c3e2777011'
d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id']
probe=root/'indexer-k-graph-overlap-probe'
assert not(probe/'enable').exists() and not(probe/'disabled').exists()
case=json.loads((root/'prefill-padding-cases-v1.json').read_text())['mixed'][0]
out=root/'indexer-k-graph-actual-smoke-v2'
out.mkdir(exist_ok=False)
report=dict(completed=False,rows=[],D_CID=cid,scope='One native and one adaptive direct PD8 before full functional cohort. Actual all16 selected/native-comparison log gate follows; numerical pass alone is insufficient.')


def post(node,body,rid):
    q=urllib.request.Request('http://172.16.12.%d:%d/v1/completions'%(node,29481 if node==32 else 29482),
                            json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
    with urllib.request.urlopen(q,timeout=180) as f:
        return json.load(f)


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


try:
    for mode in ['native','adaptive']:
        idle()
        if mode=='adaptive':
            (probe/'enable').touch()
        else:
            (probe/'enable').unlink(missing_ok=True)
        rid='indexer-actual-'+uuid.uuid4().hex
        body=dict(model='glm5',prompt=case['prompt_token_ids'],max_tokens=8,min_tokens=1,ignore_eos=True,
                  temperature=0,top_p=1,seed=1024,return_token_ids=True,cache_salt=rid,stream=False)
        pb=dict(body,max_tokens=1,kv_transfer_params=dict(do_remote_decode=True,do_remote_prefill=False,
                remote_engine_id=None,remote_block_ids=None,remote_host=None,remote_port=None))
        row=dict(mode=mode,P_request=pb,D_request=body,request_id=rid)
        report['rows'].append(row)
        row['P_response']=post(32,pb,rid)
        body['kv_transfer_params']=row['P_response']['kv_transfer_params']
        row['D_response']=post(33,body,rid)
        response=row['D_response']
        number=re.search(r'[-+]?\d+(?:\.\d+)?',response['choices'][0]['text'])
        row['numeric_pass']=bool(number and float(number.group())==case['expected'])
        row['budget_pass']=len(response['choices'][0]['token_ids'])==response['usage']['completion_tokens']==8
        assert row['numeric_pass'] and row['budget_pass']
    idle()
    report['completed']=True
except Exception as exc:
    report['failure']=type(exc).__name__+': '+str(exc)
finally:
    (probe/'enable').unlink(missing_ok=True)
(out/'result.json').write_text(json.dumps(report)+'\n')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
assert report['completed'],report.get('failure')
