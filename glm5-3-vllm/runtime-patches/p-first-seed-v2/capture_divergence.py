#!/usr/bin/env python3
"""Bounded native-D probes matching the tensor snapshot history whitelist."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request
import uuid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert args.execute
    base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32')
    assert args.output.is_relative_to(base)
    args.output.mkdir(exist_ok=False)
    cases=json.loads((base/'performance-cases-v1/cases.json').read_text())['cases']
    report={'completed':False,'rows':[],
            'scope':'Native D, two C1 then four cold-cohort focus requests per shared history. Unique per-request cache salts. One target token, Top5. Actual tensor/input/dispatch checks required; no performance or general quality acceptance.'}

    def one(case,cohort,barrier=None,focus=True):
        ids=case['prompt_token_ids']+case['expected_token_ids'][:2] if focus else case['prompt_token_ids'][:8]
        rid='tensor-d-'+uuid.uuid4().hex
        body={'model':'glm5','prompt':ids,'max_tokens':1,'min_tokens':1,
              'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,
              'return_token_ids':True,'return_tokens_as_token_ids':True,
              'logprobs':5,'cache_salt':rid,'stream':False}
        if barrier:barrier.wait()
        row={'case_id':case['case_id'],'focus':focus,'cohort':cohort,
             'header_request_id':rid,'request':body,
             'history_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
             'sent_ns':time.monotonic_ns()}
        report['rows'].append(row)
        req=urllib.request.Request('http://172.16.12.33:29482/v1/completions',
            json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
        with urllib.request.urlopen(req,timeout=120) as response:row['response']=json.load(response)
        row['received_ns']=time.monotonic_ns()
        choice=row['response']['choices'][0]
        assert len(choice['token_ids'])==1
        assert choice['logprobs']['tokens']==['token_id:'+str(choice['token_ids'][0])]
        assert row['response']['usage']['prompt_tokens']==len(ids)
        row['pass']=True

    try:
        for case in cases:
            for repeat in range(2):one(case,'C1-'+str(repeat))
        for case in cases:
            for repeat in range(2):
                barrier=threading.Barrier(4)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(lambda i:one(case,'C4-'+str(repeat),barrier,focus=i in (0,2)),range(4)))
        assert sum(row['focus'] for row in report['rows'])==12
        report['completed']=True
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
    assert report['completed'],report.get('failure')


if __name__=='__main__':main()
