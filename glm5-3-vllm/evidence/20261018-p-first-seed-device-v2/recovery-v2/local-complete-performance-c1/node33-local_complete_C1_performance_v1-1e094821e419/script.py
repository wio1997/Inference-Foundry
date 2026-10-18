#!/usr/bin/env python3
"""Paired max1 wire latency on the original official PD proxy."""
import argparse
import json
import math
from pathlib import Path
import socket
import statistics
import subprocess
import time
import urllib.request
import uuid


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--functional-only',action='store_true')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--cases',type=Path,required=True)
    args=parser.parse_args()
    assert args.execute and socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
    root=args.run_dir.resolve()
    assert str(root).startswith('/data/wio/glm5-3-vllm-execution/20261018-state-tail/unused-draft-live1-20261018/node33')
    plan=json.loads((root/'plan.json').read_text());launch=json.loads((root/'launch.json').read_text())
    actual=json.loads(subprocess.check_output(['docker','inspect',launch['container_id']]))[0]
    assert actual['State']['Running'] and actual['Name']=='/'+plan['container']['name'] and actual['Image']==plan['image']['id']
    assert plan['env']['GLM53_UNUSED_SINGLE_TOKEN_DRAFT_PROBE']=='1'
    probe=root/'divergence-probe';marker=probe/'skip-unused-single-token-draft'
    assert not (probe/'shared-overlap-disabled').exists()
    if not args.functional_only:
        assert all(path.exists() for path in [probe/'disabled',probe/'overlap-probe-disabled',probe/'unused-draft-trace-disabled',root/'adopt-trace/disabled'])
    else:assert not (probe/'unused-draft-trace-disabled').exists()
    with urllib.request.urlopen('http://172.16.12.33:29482/v1/models',timeout=5) as response:
        assert any(model['id']=='glm5' for model in json.load(response)['data'])
    cases=json.loads(args.cases.read_text())['cases']
    answers={'sum-three-4':'84','sum-three-1':'79'}
    out=root/('single-token-wire-smoke-v1' if args.functional_only else 'local-complete-performance-c1-v1');out.mkdir(exist_ok=False)
    report={'completed':False,'rows':[],'formal_order':['pd-skip','p-complete','p-complete','pd-skip']*3,
            'scope':'Same patched official CPU29882, native PD with validated D max1 skip vs P native local completion; same loaded P/D and resources. Cold C1 two prior correctness-fixed whole answers, one output, TTFT=last. No general64-output or maximum capacity claim.'}

    def one(case,mode,phase):
        marker.touch()  # Best prior PD max1 skip enabled for both routes.
        ids=case['prompt_ids'][:-2];expected=case['prompt_ids'][-2]
        rid='single-wire-'+uuid.uuid4().hex
        body={'model':'glm5','prompt':ids,'max_tokens':1,'min_tokens':1,
              'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,
              'return_token_ids':True,'cache_salt':rid,'stream':True,
              'stream_options':{'include_usage':True}}
        req=urllib.request.Request('http://172.16.12.32:29882/v1/completions',json.dumps(body).encode(),
                                  {'Content-Type':'application/json','X-Request-Id':rid,'X-GLM53-Route':'p-complete' if mode=='p-complete' else 'native'})
        row={'case_id':case['case_id'],'N':len(ids),'mode':mode,'phase':phase,'request':body,
             'expected_token_id':expected,'expected_answer':answers[case['case_id']],
             'chunks':[],'token_ids':[],'arrival_ns':[]}
        report['rows'].append(row);row['sent_ns']=time.monotonic_ns()
        with urllib.request.urlopen(req,timeout=120) as response:
            row['HTTP_status']=response.status
            row['route_header']=response.headers.get('X-GLM53-Route')
            assert row['route_header']==('p-local-complete' if mode=='p-complete' else None)
            for line in response:
                if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
                chunk=json.loads(line[6:]);assert not chunk.get('error'),chunk
                stamp=time.monotonic_ns();row['chunks'].append(chunk)
                for choice in chunk.get('choices',[]):
                    tokens=choice.get('token_ids') or []
                    row['token_ids'].extend(tokens);row['arrival_ns'].extend([stamp]*len(tokens))
        row['finished_ns']=time.monotonic_ns()
        row['usage']=next(chunk['usage'] for chunk in reversed(row['chunks']) if chunk.get('usage'))
        row['text']=''.join(choice.get('text','') for chunk in row['chunks'] for choice in chunk.get('choices',[]))
        assert row['HTTP_status']==200 and row['token_ids']==[expected]
        assert row['text'].strip()==answers[case['case_id']]
        assert row['usage']['completion_tokens']==1 and row['usage']['prompt_tokens']==len(ids)
        row['TTFT_ms']=(row['arrival_ns'][0]-row['sent_ns'])/1e6
        row['HTTP_finish_ms']=(row['finished_ns']-row['sent_ns'])/1e6
        row['complete_answer_pass']=True
        (out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')

    try:
        for case in cases:
            for mode in ['pd-skip','p-complete']:one(case,mode,'functional' if args.functional_only else 'warmup')
        if not args.functional_only:
            for case in cases:
                for mode in report['formal_order']:one(case,mode,'formal')
            summaries=[]
            for case in cases:
                item={'case_id':case['case_id'],'modes':{},'local_over_PD_skip':{}}
                for mode in ['pd-skip','p-complete']:
                    rows=[row for row in report['rows'] if row['phase']=='formal' and row['case_id']==case['case_id'] and row['mode']==mode]
                    assert len(rows)==6
                    item['modes'][mode]={metric:{'median':statistics.median(row[metric] for row in rows),'samples':[row[metric] for row in rows]} for metric in ['TTFT_ms','HTTP_finish_ms']}
                for metric in ['TTFT_ms','HTTP_finish_ms']:
                    item['local_over_PD_skip'][metric]=item['modes']['p-complete'][metric]['median']/item['modes']['pd-skip'][metric]['median']
                summaries.append(item)
            report['summary']={'cases':summaries,'equal_case_geomean':{metric:math.exp(statistics.mean(math.log(item['local_over_PD_skip'][metric]) for item in summaries)) for metric in ['TTFT_ms','HTTP_finish_ms']},'TPOT':None,'formal_requests':24,'warmup_requests':4}
        else:report['summary']={'functional_requests':4,'timing_accepted':False}
        report['completed']=True
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    finally:marker.unlink(missing_ok=True)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
    assert report['completed'],report.get('failure')


if __name__=='__main__':main()
