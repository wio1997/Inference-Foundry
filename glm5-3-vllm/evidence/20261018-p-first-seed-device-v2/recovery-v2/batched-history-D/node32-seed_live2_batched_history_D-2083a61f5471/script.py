#!/usr/bin/env python3
"""Native batched API: duplicate shared histories plus short fillers fit1024."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--role',choices=('P','D'),default='D');a=p.parse_args();assert a.execute
    base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32');assert a.output.is_relative_to(base);a.output.mkdir(exist_ok=False);cases=json.loads((base/'performance-cases-v1/cases.json').read_text())['cases'];report={'completed':False,'role':a.role,'rows':[],'scope':'Native non-PD one-output batched API with identical focus histories at slots0/2 and two8-token fillers. Total<=1024 avoids required chunking; actual D first-dispatch layout must be checked from trace. No full-quality or performance claim.'}
    try:
        for case in cases:
            focus=case['prompt_token_ids']+case['expected_token_ids'][:2];filler=case['prompt_token_ids'][:8];prompts=[focus,filler,focus,filler];assert sum(map(len,prompts))<=1024
            for _ in range(4):
                rid='batch-history-'+uuid.uuid4().hex;body={'model':'glm5','prompt':prompts,'max_tokens':1,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'return_tokens_as_token_ids':True,'logprobs':5,'cache_salt':rid,'stream':False};row={'original_N':case['input_tokens'],'header_request_id':rid,'history_sha256':hashlib.sha256(json.dumps(focus,separators=(',',':')).encode()).hexdigest(),'request':body,'sent_ns':time.monotonic_ns()};report['rows'].append(row)
                url='http://172.16.12.%d:%d/v1/completions'%((32,29481) if a.role=='P' else (33,29482));req=urllib.request.Request(url,json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
                with urllib.request.urlopen(req,timeout=120) as response:row['response']=json.load(response)
                row['received_ns']=time.monotonic_ns();choices={c['index']:c for c in row['response']['choices']};assert set(choices)==set(range(4)) and all(len(c['token_ids'])==1 for c in choices.values()) and row['response']['usage']['prompt_tokens']==sum(map(len,prompts)) and row['response']['usage']['completion_tokens']==4
                row['duplicate_focus_equal']=choices[0]['token_ids']==choices[2]['token_ids'];row['pass']=True
        report['completed']=True
    except Exception as ex:report['failure']=type(ex).__name__+': '+str(ex)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['completed'],report.get('failure')


if __name__=='__main__':main()
