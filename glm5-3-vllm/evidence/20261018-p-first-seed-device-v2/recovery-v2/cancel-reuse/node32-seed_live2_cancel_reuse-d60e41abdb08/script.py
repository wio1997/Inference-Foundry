#!/usr/bin/env python3
"""Bounded native/adopt cancellation followed immediately by slot reuse."""
import json
from pathlib import Path
import re
import time
import urllib.request
import uuid


def main():
    base=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live1-20261018/node32')
    quality=json.loads((base/'recovery-v2/quality24/pairs.json').read_text());assert quality['completed']
    cases=json.loads((Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node32/adopt-cases-v1/cases.json')).read_text())['cases'][:2]
    by_sha={r['prompt_sha256']:r['token_ids'][:2] for r in quality['rows'] if r['mode']=='control'}
    reference={c['input_tokens']:by_sha[c['prompt_sha256']] for c in cases}
    out=base/'recovery-v2/client-seed-cancel-v1';out.mkdir(exist_ok=False)
    artifacts=Path(__file__).resolve().parent/'artifacts';artifacts.mkdir(exist_ok=False)
    report={'completed':False,'rows':[],'scope':'Eight bounded cancellation attempts with immediate seeded two-token reuse; actual abort metrics separate from client close.'}
    def request(node,body=None,path='/v1/completions',rid=None):
        req=urllib.request.Request('http://172.16.12.%d:%d%s'%(node,29481 if node==32 else 29482,path),None if body is None else json.dumps(body).encode(),{'Content-Type':'application/json',**({'X-Request-Id':rid} if rid else {})})
        return urllib.request.urlopen(req,timeout=90 if body else 5)
    def metrics(label):
        with request(33,path='/metrics') as r:raw=r.read()
        (artifacts/('D-metrics-'+label+'.txt')).write_bytes(raw)
        values={}
        for line in raw.decode().splitlines():
            if line.startswith('#') or not line.strip():continue
            key,value=line.rsplit(' ',1)
            if 'request_success_total' in key and 'abort' in key:values['abort']=values.get('abort',0)+float(value)
            for kind in ('num_requests_running','num_requests_waiting','request_prefill_time_seconds_sum','request_decode_time_seconds_sum'):
                if (':'+kind+'{') in key or key.endswith(':'+kind):values[kind]=values.get(kind,0)+float(value)
        return values
    def prepare(ids,mode,maximum):
        rid='seed-cancel-'+uuid.uuid4().hex
        body={'model':'glm5','prompt':ids,'max_tokens':maximum,'min_tokens':1,'ignore_eos':True,'temperature':0,'seed':1024,'return_token_ids':True,'stream':True,'stream_options':{'include_usage':True},'cache_salt':rid}
        p={k:v for k,v in body.items() if k!='stream_options'};p.update(max_tokens=1,stream=False,kv_transfer_params={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None,'_glm53_export_p_draft':mode=='seed'})
        with request(32,p,rid=rid) as r:producer=json.load(r)
        body['kv_transfer_params']={**producer['kv_transfer_params'],'_glm53_adopt_first':mode=='seed','_glm53_use_p_draft':mode=='seed'}
        return rid,body,producer
    try:
        report['before']=metrics('before')
        for case in cases:
            for mode in ('control','seed'):
                for point in ('headers','first_tokens'):
                    rid,body,producer=prepare(case['prompt_token_ids'],mode,64)
                    row={'input_tokens':case['input_tokens'],'mode':mode,'close_point':point,'request_id':rid,'P_response':producer,'D_request':body,'received_ids':[]};report['rows'].append(row)
                    with request(33,body,rid=rid) as r:
                        if point=='first_tokens':
                            for line in r:
                                if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
                                chunk=json.loads(line[6:]);assert not chunk.get('error'),chunk
                                for choice in chunk.get('choices',[]):row['received_ids'].extend(choice.get('token_ids') or [])
                                if row['received_ids']:break
                    row['client_closed_ns']=time.monotonic_ns()
                    reuse_id,reuse_body,reuse_P=prepare(case['prompt_token_ids'],'seed',2)
                    row.update(reuse_id=reuse_id,reuse_ids=[],reuse_request=reuse_body)
                    usage=None
                    with request(33,reuse_body,rid=reuse_id) as r:
                        for line in r:
                            if not line.startswith(b'data: ') or line.startswith(b'data: [DONE]'):continue
                            chunk=json.loads(line[6:]);assert not chunk.get('error'),chunk
                            if chunk.get('usage'):usage=chunk['usage']
                            for choice in chunk.get('choices',[]):row['reuse_ids'].extend(choice.get('token_ids') or [])
                    assert row['reuse_ids']==reference[case['input_tokens']] and usage['completion_tokens']==2
                    row['reuse_pass']=True;(out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
        deadline=time.monotonic()+30
        while True:
            after=metrics('after')
            if after.get('num_requests_running',0)==after.get('num_requests_waiting',0)==0:break
            assert time.monotonic()<deadline,'D did not return idle';time.sleep(.5)
        report['after']=after;report['abort_metric_delta']=after.get('abort',0)-report['before'].get('abort',0);report['completed']=True
    except Exception as ex:report['failure']=type(ex).__name__+': '+str(ex)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report,indent=2)+'\nGLM53_RESULT_JSON_END')
    return 0 if report['completed'] else 2


if __name__=='__main__':raise SystemExit(main())
