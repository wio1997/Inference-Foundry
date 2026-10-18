#!/usr/bin/env python3
"""One-token native logits at the shared pre-divergence history, C1 versus C4."""
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
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert a.execute
    base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32');assert a.output.is_relative_to(base);a.output.mkdir(exist_ok=False);cases=json.loads((base/'performance-cases-v1/cases.json').read_text())['cases'];report={'completed':False,'rows':[],'scope':'Native D cold C4, two identical focus histories plus two8-token fillers, all separate unique cache salts and total<=1024; one target sample. Actual no-cache/no-chunk/no-spec state requires trace check. Not original decode-state equivalence or quality/performance acceptance.'}
    def one(case,concurrency,barrier=None,focus=True):
        ids=(case['prompt_token_ids']+case['expected_token_ids'][:2]) if focus else case['prompt_token_ids'][:8];rid='conditional-d-'+uuid.uuid4().hex;body={'model':'glm5','prompt':ids,'max_tokens':1,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'return_tokens_as_token_ids':True,'logprobs':5,'cache_salt':rid,'stream':False}
        if barrier:barrier.wait()
        row={'case_id':case['case_id'],'focus':focus,'original_N':case['input_tokens'],'C':concurrency,'header_request_id':rid,'request':body,'history_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'sent_ns':time.monotonic_ns()};report['rows'].append(row)
        req=urllib.request.Request('http://172.16.12.33:29482/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
        with urllib.request.urlopen(req,timeout=120) as response:row['response']=json.load(response)
        row['received_ns']=time.monotonic_ns();choice=row['response']['choices'][0];assert len(choice['token_ids'])==1 and choice['logprobs']['tokens']==['token_id:'+str(choice['token_ids'][0])] and row['response']['usage']['prompt_tokens']==len(ids);row['pass']=True;return row
    try:
        for case in cases:
            for _ in range(4):
                barrier=threading.Barrier(4)
                with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda i:one(case,4,barrier,focus=i in (0,2)),range(4)))
        report['completed']=True
    except Exception as ex:report['failure']=type(ex).__name__+': '+str(ex)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['completed'],report.get('failure')


if __name__=='__main__':main()
