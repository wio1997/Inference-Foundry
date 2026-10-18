#!/usr/bin/env python3
"""Finite synchronized C4 control/seed correctness gate on frozen fixtures."""
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
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--max-tokens',type=int,choices=(2,64),default=2);p.add_argument('--order',default='control,seed,seed,control');p.add_argument('--diagnostic',action='store_true');p.add_argument('--logprobs',type=int,choices=(0,5),default=0);a=p.parse_args();assert a.execute
    order=a.order.split(',');assert all(m in ('control','seed') for m in order)
    assert a.output.is_relative_to('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32')
    a.output.mkdir(exist_ok=False);cases=json.loads(a.cases.read_text())['cases'];assert len(cases)==2
    report={'completed':False,'max_tokens':a.max_tokens,'concurrency':4,'bursts':[],'scope':'Bounded C4 diagnosis with exact C1 differences retained when diagnostic flag set; no sustainable capacity or full-quality acceptance.'}
    def post(n,body,rid):
        req=urllib.request.Request('http://172.16.12.%d:%d/v1/completions'%(n,29481 if n==32 else 29482),json.dumps(body).encode(),{'Content-Type':'application/json','X-Request-Id':rid})
        return urllib.request.urlopen(req,timeout=90)
    def one(index,mode,barrier,bucket):
        case=cases[index%2];rid='seed-c4-'+uuid.uuid4().hex;ids=case['prompt_token_ids'];assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==case['prompt_sha256'];barrier.wait()
        body={'model':'glm5','prompt':ids,'max_tokens':a.max_tokens,'min_tokens':1,'ignore_eos':True,'temperature':0,'top_p':1,'seed':1024,'return_token_ids':True,'cache_salt':rid,'stream':True,'stream_options':{'include_usage':True}}
        if a.logprobs:body['logprobs']=a.logprobs
        producer={k:v for k,v in body.items() if k!='stream_options'};producer.update(max_tokens=1,stream=False,kv_transfer_params={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None,'_glm53_export_p_draft':mode=='seed'})
        row={'mode':mode,'header_request_id':rid,'prompt_sha256':case['prompt_sha256'],'input_tokens':len(ids),'case_id':case['case_id'],'expected':case['expected'],'P_send_ns':time.monotonic_ns(),'token_ids':[],'token_arrival_ns':[],'D_chunks':[]}
        row['P_request']=producer
        bucket.append(row)
        with post(32,producer,rid) as response:row['P_response']=json.load(response)
        row['P_received_ns']=time.monotonic_ns()
        params=row['P_response']['kv_transfer_params'];packet=params.get('p_first_draft');valid=bool(packet and packet['request_id']==params['remote_request_id'] and packet['rank_agreement']==16 and packet['num_prompt_tokens']==len(ids) and packet['first_token_id']==params['last_token_id'])
        body['kv_transfer_params']={**params,'_glm53_adopt_first':mode=='seed' and valid,'_glm53_use_p_draft':mode=='seed' and valid};row['D_request']=body;row['D_send_ns']=time.monotonic_ns()
        with post(33,body,rid) as response:
            for line in response:
                if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
                chunk=json.loads(line[6:]);assert not chunk.get('error'),chunk;row['D_chunks'].append(chunk)
                for choice in chunk.get('choices',[]):
                    ids_out=choice.get('token_ids') or [];row['token_ids'].extend(ids_out);row['token_arrival_ns'].extend([time.monotonic_ns()]*len(ids_out))
        usage=next(c['usage'] for c in reversed(row['D_chunks']) if c.get('usage'));assert usage['completion_tokens']==a.max_tokens
        row['D_done_ns']=time.monotonic_ns();row['seed_packet_valid']=valid if mode=='seed' else None
        assert mode!='seed' or valid,'Missing export; native fallback consumed request'
        row['matches_frozen_C1_ids']=row['token_ids']==case['expected_token_ids'][:a.max_tokens];assert len(row['token_ids'])==a.max_tokens
        row['first_C1_difference']=next((i for i,(x,y) in enumerate(zip(row['token_ids'],case['expected_token_ids'])) if x!=y),None)
        if not a.diagnostic:assert row['matches_frozen_C1_ids']
        else:
            import re
            text=''.join(c.get('text','') for ch in row['D_chunks'] for c in ch.get('choices',[]));match=re.search(r'[-+]?\d+(?:\.\d+)?',text);row['numeric_answer_correct']=bool(match and float(match.group())==case['expected']);row['text']=text
        row['pass']=True;return row
    try:
        for mode in order:
            barrier=threading.Barrier(4);burst={'mode':mode,'rows':[]};report['bursts'].append(burst)
            with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda i:one(i,mode,barrier,burst['rows']),range(4)))
            (a.output/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        report['completed']=True
    except Exception as ex:report['failure']=type(ex).__name__+': '+str(ex)
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END');assert report['completed'],report.get('failure')


if __name__=='__main__':main()
