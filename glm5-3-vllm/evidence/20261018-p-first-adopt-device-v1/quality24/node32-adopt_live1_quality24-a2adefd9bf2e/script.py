#!/usr/bin/env python3
"""Cold native/adopt pairs on the same P/D instances; explicit execution gate."""
import argparse
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-tokens',type=int,choices=(1,2,64),default=2);p.add_argument('--order',default='control,adopt');p.add_argument('--cases-file',type=Path)
    a=p.parse_args()
    if not a.execute:p.error('pass --execute to run this batch')
    order=a.order.split(',');assert order[0]=='control' and all(m in ('control','adopt') for m in order)
    assert str(a.output).startswith('/data/wio/glm5-3-vllm-execution/20261018-state-tail/') and '..' not in a.output.parts
    assert not a.output.exists();a.output.mkdir(parents=True)
    report={'completed':False,'rows':[],'max_tokens':a.max_tokens,'same_instances':True,'scope':'Finite cold C1 control/adopt exact-token comparison. Diagnostic timing; no production acceptance.'}
    def post(node,body,rid):
        req=urllib.request.Request('http://172.16.12.%s:%s/v1/completions'%(node,29481 if node==32 else 29482),json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
        return urllib.request.urlopen(req,timeout=90)
    cases = json.loads(a.cases_file.read_text())['cases'] if a.cases_file else CASES
    try:
        for case in cases:
            ids=case['prompt_token_ids'];assert len(ids) in (256,257);assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==case['prompt_sha256']
            reference=None
            for mode in order:
                rid='adopt-pair-'+uuid.uuid4().hex
                body={'model':'glm5','prompt':ids,'max_tokens':a.max_tokens,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'cache_salt':rid,'stream':True,'stream_options':{'include_usage':True}}
                producer={k:v for k,v in body.items() if k!='stream_options'}
                producer.update(max_tokens=1,stream=False,kv_transfer_params={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None})
                row={'input_tokens':len(ids),'prompt_sha256':case['prompt_sha256'],'mode':mode,'header_request_id':rid,'P_request':producer,'P_send_ns':time.monotonic_ns()};report['rows'].append(row)
                with post(32,producer,rid) as response:row['P_response']=json.load(response)
                row['P_received_ns']=time.monotonic_ns();params=row['P_response']['kv_transfer_params']
                assert params['remote_num_prompt_tokens']==len(ids) and row['P_response']['choices'][0]['token_ids']==[params['last_token_id']]
                body['kv_transfer_params']={**params,'_glm53_adopt_first':mode=='adopt'}
                row.update(D_request=body,D_send_ns=time.monotonic_ns(),D_chunks=[],token_ids=[],token_arrival_ns=[])
                usage=None
                with post(33,body,rid) as response:
                    assert response.status==200
                    for line in response:
                        if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
                        chunk=json.loads(line[6:]);assert not chunk.get('error'),chunk
                        stamp=time.monotonic_ns();row['D_chunks'].append(chunk)
                        if chunk.get('usage'):usage=chunk['usage']
                        for choice in chunk.get('choices',[]):
                            tokens=choice.get('token_ids') or [];row['token_ids'].extend(tokens);row['token_arrival_ns'].extend([stamp]*len(tokens))
                row['D_received_ns']=time.monotonic_ns();row['D_usage']=usage
                row['text']=''.join(c.get('text','') for chunk in row['D_chunks'] for c in chunk.get('choices',[]))
                if 'expected' in case:
                    match=re.search(r'[-+]?\d+(?:\.\d+)?',row['text']);row['expected']=case['expected'];row['numeric_pass']=bool(match and Decimal(match.group())==case['expected'])
                assert len(row['token_ids'])==a.max_tokens and usage['completion_tokens']==a.max_tokens
                assert row['token_ids'][0]==params['last_token_id']
                if reference is None:reference=row['token_ids']
                assert row['token_ids']==reference,'Exact greedy token sequence differs from same-instance native control'
                row['functional_pass']=True;row['matches_native_control']=True
                row['TTFT_ms']=(row['token_arrival_ns'][0]-row['P_send_ns'])/1e6
                row['last_token_ms']=(row['token_arrival_ns'][-1]-row['P_send_ns'])/1e6
                (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        report['completed']=True
    except Exception as exc:report['failure']=type(exc).__name__+': '+str(exc)
    (a.output/'pairs.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report,indent=2)+'\nGLM53_RESULT_JSON_END')
    return 0 if report['completed'] else 2


CASES = [{'input_tokens': 256, 'prompt_sha256': 'e55a62b8e0046a49b7750c8bcc1925aa5e6cd2effcc61c949561dfaff0c7593b', 'prompt_token_ids': [8855, 21465, 19, 64, 102573, 65, 100899, 6066, 19, 66, 118285, 127020, 7628, 102807, 22, 100441, 16, 98503, 6066, 18699, 12, 116265, 13, 4457, 1493, 10514, 8387, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 198, 14572, 25, 3555, 374, 220, 17, 488, 220, 18, 5267, 16127, 25, 220]}, {'input_tokens': 257, 'prompt_sha256': '01e6c332da1bcc7fec4e7f2e3da0131baada26bff210c3904dc848cd8ad4da83', 'prompt_token_ids': [8855, 21465, 19, 64, 102573, 65, 100899, 6066, 19, 66, 118285, 127020, 7628, 102807, 22, 100441, 16, 98503, 6066, 18699, 12, 116265, 13, 4457, 1493, 10514, 8387, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 1779, 624, 785, 8387, 21081, 279, 1973, 315, 279, 17201, 13057, 13, 1096, 374, 264, 71794, 4938, 8316, 198, 14572, 25, 3555, 374, 220, 17, 488, 220, 18, 5267, 16127, 25, 220, 220]}]

if __name__=='__main__':raise SystemExit(main())
