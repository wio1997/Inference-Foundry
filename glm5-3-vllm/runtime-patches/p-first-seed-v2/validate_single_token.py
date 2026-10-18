#!/usr/bin/env python3
"""Whole-answer max1 checks, including a native-D chunked prefill."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import threading
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
    plan=json.loads((root/'plan.json').read_text());launch=json.loads((root/'launch.json').read_text())
    actual=json.loads(subprocess.check_output(['docker','inspect',launch['container_id']]))[0]
    assert actual['State']['Running'] and actual['Image']==plan['image']['id'] and actual['Name']=='/'+plan['container']['name']
    assert plan['env']['GLM53_UNUSED_SINGLE_TOKEN_DRAFT_PROBE']=='1'
    cases=json.loads(args.cases.read_text())['cases']
    out=root/'single-token-quality-v1';out.mkdir(exist_ok=False)
    marker=root/'divergence-probe/skip-unused-single-token-draft'
    report={'completed':False,'rows':[],'scope':'Native D only. Entire one-token numeric answer,8 previously frozen short cases plus2 arithmetic long cases, and cold C4 short cohorts. Long1536/1537 must actually show chunking in trace; this is not long-PD or general model accuracy acceptance.'}

    def one(case,mode,cohort,barrier=None):
        rid='single-answer-'+uuid.uuid4().hex
        body={'model':'glm5','prompt':case['prompt_ids'],'max_tokens':1,'min_tokens':1,
              'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,
              'return_token_ids':True,'cache_salt':rid,'stream':False}
        if barrier:barrier.wait()
        row={'case_id':case['case_id'],'N':len(case['prompt_ids']),'mode':mode,
             'cohort':cohort,'header_request_id':rid,'request':body,
             'expected':case['expected'],'expected_token_id':case['expected_token_id']}
        report['rows'].append(row)
        req=urllib.request.Request('http://172.16.12.33:29482/v1/completions',json.dumps(body).encode(),
                                  {'Content-Type':'application/json','X-Request-Id':rid})
        with urllib.request.urlopen(req,timeout=120) as response:row['response']=json.load(response)
        choice=row['response']['choices'][0]
        assert choice['token_ids']==[case['expected_token_id']]
        assert choice['text'].strip()==case['expected']
        assert row['response']['usage']['completion_tokens']==1 and row['response']['usage']['prompt_tokens']==row['N']
        row['whole_answer_pass']=True

    def mode(value):
        if value=='draft-skip':marker.touch()
        else:marker.unlink(missing_ok=True)

    try:
        for index,case in enumerate(cases):
            order=['draft-native','draft-skip'] if index%2==0 else ['draft-skip','draft-native']
            for value in order:mode(value);one(case,value,'C1')
            (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        for n in (256,257):
            case=next(case for case in cases if len(case['prompt_ids'])==n)
            for value in ['draft-native','draft-skip']:
                mode(value);barrier=threading.Barrier(4)
                with ThreadPoolExecutor(max_workers=4) as pool:
                    list(pool.map(lambda index:one(case,value,'C4',barrier),range(4)))
            (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        report['completed']=True
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    finally:marker.unlink(missing_ok=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
    assert report['completed'],report.get('failure')


if __name__=='__main__':main()
