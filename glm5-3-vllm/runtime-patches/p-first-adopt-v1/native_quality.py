#!/usr/bin/env python3
"""Collect both full native paths for the frozen 24-case diagnostic matrix."""
from decimal import Decimal
import json
from pathlib import Path
import re
import time
import urllib.request
import uuid


def main():
    base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node32')
    pair_file=base/'client-adopt-quality24-collect-v2/pairs.json';deadline=time.monotonic()+450
    while not pair_file.exists():
        assert time.monotonic()<deadline;time.sleep(2)
    paired=json.loads(pair_file.read_text());assert paired['completed']
    cases=json.loads((base/'adopt-cases-v1/cases.json').read_text())['cases']
    refs={(r['prompt_sha256'],r['mode']):r['token_ids'] for r in paired['rows']}
    out=base/'native-quality24-v1';out.mkdir(exist_ok=False)
    report={'completed':False,'rows':[],'scope':'Frozen 24 numerical/extraction tasks, complete native P and D paths compared to PD control/adopt; no quality gate is silently changed'}
    try:
        for case in cases:
            for node in (32,33):
                rid='native-quality-'+uuid.uuid4().hex
                body={'model':'glm5','prompt':case['prompt_token_ids'],'max_tokens':64,'min_tokens':1,'temperature':0,'seed':1024,'ignore_eos':True,'return_token_ids':True,'stream':False,'cache_salt':rid}
                req=urllib.request.Request('http://172.16.12.%d:%d/v1/completions'%(node,29481 if node==32 else 29482),json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
                with urllib.request.urlopen(req,timeout=90) as response:r=json.load(response)
                ids=r['choices'][0]['token_ids'];text=r['choices'][0]['text'];match=re.search(r'[-+]?\d+(?:\.\d+)?',text)
                assert len(ids)==64 and r['usage']['completion_tokens']==64
                report['rows'].append({'node':node,'case_id':case['case_id'],'prompt_sha256':case['prompt_sha256'],'expected':case['expected'],'numeric_pass':bool(match and Decimal(match.group())==case['expected']),'matches_PD_control':ids==refs[(case['prompt_sha256'],'control')],'matches_PD_adopt':ids==refs[(case['prompt_sha256'],'adopt')],'request':body,'response':r})
                (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        report['completed']=True
    except Exception as ex:report['failure']=type(ex).__name__+': '+str(ex)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report,indent=2)+'\nGLM53_RESULT_JSON_END')
    return 0 if report['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
