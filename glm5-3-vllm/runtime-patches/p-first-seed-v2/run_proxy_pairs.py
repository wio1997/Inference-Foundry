#!/usr/bin/env python3
"""Real external stream arrival comparisons on the same experimental frontend."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--order',default='control,seed,publish-seed');p.add_argument('--max-tokens',type=int,choices=(2,64),default=64);a=p.parse_args();assert a.execute
    base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32');assert a.output.is_relative_to(base);a.output.mkdir(exist_ok=False);cases=json.loads((base/'performance-cases-v1/cases.json').read_text())['cases'];order=a.order.split(',');assert all(m in ('control','seed','publish-seed') for m in order)
    report={'completed':False,'rows':[],'scope':'Real wire TTFT/last across stock-style native control, seed, proxy-published confirmed P first. Same CPU proxy/backend instances, bounded C1 fixed inputs.'}
    try:
        for case in cases:
            reference_text=None
            for mode in order:
                rid='proxy-pair-'+uuid.uuid4().hex;body={'model':'glm5','prompt':case['prompt_token_ids'],'max_tokens':a.max_tokens,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'cache_salt':rid,'stream':True,'stream_options':{'include_usage':True}}
                q=urllib.request.Request('http://172.16.12.32:29881/v1/completions',json.dumps(body).encode(),{'Content-Type':'application/json','X-GLM53-Mode':mode,'X-Request-Id':rid});row={'input_tokens':case['input_tokens'],'case_id':case['case_id'],'mode':mode,'request_id':rid,'request':body,'sent_ns':time.monotonic_ns(),'chunks':[],'token_ids':[],'arrival_ns':[]};report['rows'].append(row)
                with urllib.request.urlopen(q,timeout=120) as response:
                    row['HTTP_status']=response.status;row['effective_mode']=response.headers.get('X-GLM53-Effective-Mode');assert row['effective_mode']==mode
                    for line in response:
                        if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
                        chunk=json.loads(line[6:]);assert not chunk.get('error'),chunk;row['chunks'].append(chunk);stamp=time.monotonic_ns()
                        for choice in chunk.get('choices',[]):
                            ids=choice.get('token_ids') or [];row['token_ids'].extend(ids);row['arrival_ns'].extend([stamp]*len(ids))
                row['received_ns']=time.monotonic_ns();row['text']=''.join(c.get('text','') for ch in row['chunks'] for c in ch.get('choices',[]));usage=next(ch['usage'] for ch in reversed(row['chunks']) if ch.get('usage'));row['usage']=usage
                assert row['token_ids']==case['expected_token_ids'][:a.max_tokens] and usage['completion_tokens']==a.max_tokens and len(row['arrival_ns'])==a.max_tokens
                if reference_text is None:reference_text=row['text']
                assert row['text']==reference_text,'Text differs from same-instance native control'
                row['text_matches_control']=True
                row['TTFT_ms']=(row['arrival_ns'][0]-row['sent_ns'])/1e6;row['last_token_ms']=(row['arrival_ns'][-1]-row['sent_ns'])/1e6;row['TPOT_ms']=(row['last_token_ms']-row['TTFT_ms'])/(a.max_tokens-1);row['exact_pass']=True
                (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        report['completed']=True
    except Exception as ex:report['failure']=type(ex).__name__+': '+str(ex)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['completed'],report.get('failure')


if __name__=='__main__':main()
