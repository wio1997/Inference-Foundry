#!/usr/bin/env python3
"""Native D-only C4 repeat, same model/precision/MTP5, no PD transfer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.request
import uuid


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert a.execute
    base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32');assert a.output.is_relative_to(base);a.output.mkdir(exist_ok=False);cases=json.loads((base/'performance-cases-v1/cases.json').read_text())['cases'];report={'completed':False,'bursts':[],'scope':'C4 same D native full-prefill+decode repeat, without P/PD; logprobs diagnosis only, not timing or full-quality acceptance.'}
    def one(i,barrier,bucket):
        case=cases[i%2];rid='native-d-c4-'+uuid.uuid4().hex;body={'model':'glm5','prompt':case['prompt_token_ids'],'max_tokens':64,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'logprobs':5,'return_tokens_as_token_ids':True,'cache_salt':rid,'stream':True,'stream_options':{'include_usage':True}};barrier.wait();row={'header_request_id':rid,'input_tokens':case['input_tokens'],'case_id':case['case_id'],'expected':case['expected'],'request':body,'sent_ns':time.monotonic_ns(),'chunks':[],'token_ids':[]};bucket.append(row)
        req=urllib.request.Request('http://172.16.12.33:29482/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
        with urllib.request.urlopen(req,timeout=120) as response:
            for line in response:
                if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
                chunk=json.loads(line[6:]);assert not chunk.get('error'),chunk;row['chunks'].append(chunk)
                for c in chunk.get('choices',[]):row['token_ids'].extend(c.get('token_ids') or [])
        row['received_ns']=time.monotonic_ns();row['text']=''.join(c.get('text','') for ch in row['chunks'] for c in ch.get('choices',[]));usage=next(ch['usage'] for ch in reversed(row['chunks']) if ch.get('usage'));assert usage['completion_tokens']==64 and len(row['token_ids'])==64
        match=re.search(r'[-+]?\d+(?:\.\d+)?',row['text']);row['numeric_answer_correct']=bool(match and float(match.group())==case['expected']);row['C1_exact']=row['token_ids']==case['expected_token_ids'];row['first_C1_difference']=next((i for i,(x,y) in enumerate(zip(row['token_ids'],case['expected_token_ids'])) if x!=y),None);row['functional_pass']=True;return row
    try:
        for _ in range(4):
            burst={'rows':[]};report['bursts'].append(burst);barrier=threading.Barrier(4)
            with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda i:one(i,barrier,burst['rows']),range(4)))
            (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        report['completed']=True
    except Exception as ex:report['failure']=type(ex).__name__+': '+str(ex)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['completed'],report.get('failure')


if __name__=='__main__':main()
