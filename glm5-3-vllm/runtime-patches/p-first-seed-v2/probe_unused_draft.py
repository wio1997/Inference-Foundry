#!/usr/bin/env python3
"""Same-instance ABBA test of native versus skipped unused max1 drafting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.request
import uuid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    args=parser.parse_args()
    assert args.execute and socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
    root=args.run_dir.resolve()
    assert str(root).startswith('/data/wio/glm5-3-vllm-execution/20261018-state-tail/unused-draft-live1-20261018/node33')
    plan=json.loads((root/'plan.json').read_text())
    launch=json.loads((root/'launch.json').read_text())
    actual=json.loads(subprocess.check_output(['docker','inspect',launch['container_id']]))[0]
    assert actual['Id']==launch['container_id'] and actual['Name']=='/'+plan['container']['name']
    assert actual['Image']==plan['image']['id'] and actual['State']['Running']
    assert plan['env']['GLM53_UNUSED_SINGLE_TOKEN_DRAFT_PROBE']=='1'
    cases=json.loads(args.cases.read_text())['cases']
    allowed=set(json.loads(plan['env']['GLM53_DIVERGENCE_HISTORY_SHA256']))
    assert {case['history_sha256'] for case in cases}==allowed
    out=root/'unused-draft-requests-v1';out.mkdir(exist_ok=False)
    marker=root/'divergence-probe/skip-unused-single-token-draft'
    report={'completed':False,'rows':[],'order':['draft-native','draft-skip','draft-skip','draft-native'],
            'scope':'Native D only, same loaded weights/image/TP16/K5 and shared-overlap on. Select original native skip branch only for whole max1 batches, between joined cohorts. Existing target sample precedes the decision; no state-cause or performance claim from this diagnostic.'}

    def one(case,mode,cohort,phase,barrier=None,focus=True):
        ids=case['prompt_ids'] if focus else case['prompt_ids'][:8]
        rid='unused-draft-'+uuid.uuid4().hex
        body={'model':'glm5','prompt':ids,'max_tokens':1,'min_tokens':1,
              'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,
              'return_token_ids':True,'return_tokens_as_token_ids':True,
              'logprobs':5,'cache_salt':rid,'stream':False}
        if barrier:barrier.wait()
        row={'case_id':case['case_id'],'focus':focus,'cohort':cohort,
             'phase':phase,'diagnostic_mode':mode,'header_request_id':rid,'request':body,
             'history_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
             'sent_ns':time.monotonic_ns()}
        report['rows'].append(row)
        req=urllib.request.Request('http://172.16.12.33:29482/v1/completions',json.dumps(body).encode(),
                                  {'Content-Type':'application/json','X-Request-Id':rid})
        with urllib.request.urlopen(req,timeout=120) as response:row['response']=json.load(response)
        row['received_ns']=time.monotonic_ns()
        choice=row['response']['choices'][0]
        assert len(choice['token_ids'])==1 and choice['logprobs']['tokens']==['token_id:'+str(choice['token_ids'][0])]
        assert row['response']['usage']['prompt_tokens']==len(ids)
        row['pass']=True

    try:
        for phase,mode in enumerate(report['order']):
            if mode=='draft-skip':marker.touch()
            else:marker.unlink(missing_ok=True)
            for case in cases:
                one(case,mode,'C1-before',phase)
                barrier=threading.Barrier(4)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(lambda i:one(case,mode,'C4',phase,barrier,focus=i in (0,2)),range(4)))
                one(case,mode,'C1-after',phase)
            (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        assert len(report['rows'])==48 and sum(row['focus'] for row in report['rows'])==32
        report['completed']=True
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    finally:marker.unlink(missing_ok=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
    assert report['completed'],report.get('failure')


if __name__=='__main__':main()
