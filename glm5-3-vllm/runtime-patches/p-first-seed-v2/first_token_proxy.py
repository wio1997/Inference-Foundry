#!/usr/bin/env python3
"""Narrow experimental frontend: publish confirmed P y1, verify proposals at D.

The stock official proxy remains on29880. This research-only29881 supports only
frozen greedy streaming token-ID inputs. No engine timestamp or phase rewrite.
"""
import json
import time
import uuid
import httpx
from fastapi import FastAPI,HTTPException,Request
from fastapi.responses import StreamingResponse
import uvicorn

app=FastAPI();client=httpx.AsyncClient(timeout=120,limits=httpx.Limits(max_connections=32,max_keepalive_connections=16))


def record(**row):
    row['monotonic_ns']=time.monotonic_ns()
    print(json.dumps(row,sort_keys=True),flush=True)


@app.on_event('shutdown')
async def close_client():await client.aclose()


@app.get('/healthcheck')
async def health():return {'status':'ok','scope':'CPU experiment only','official_native_port':29880,'P':'172.16.12.32:29481','D':'172.16.12.33:29482'}


@app.post('/v1/completions')
async def complete(request:Request):
    body=await request.json();mode=request.headers.get('X-GLM53-Mode','control');rid=request.headers.get('X-Request-Id') or ('first-proxy-'+uuid.uuid4().hex)
    if not isinstance(body,dict):raise HTTPException(400,'Expected the explicit research request object')
    prompt=body.get('prompt')
    fields={'model','prompt','stream','stream_options','return_token_ids','temperature','top_p','seed','cache_salt','min_tokens','max_tokens','n','ignore_eos'}
    allowed=(isinstance(body,dict) and not (set(body)-fields)
      and mode in ('control','seed','publish-seed') and body.get('model')=='glm5' and body.get('stream') is True
      and body.get('return_token_ids') is True and isinstance(prompt,list) and len(prompt) in (256,257)
      and all(type(x) is int and 0<=x<154880 for x in prompt) and body.get('temperature')==0
      and body.get('top_p')==1 and type(body.get('seed')) is int
      and isinstance(body.get('cache_salt'),str) and bool(body['cache_salt'])
      and body.get('stream_options')=={'include_usage':True} and body.get('ignore_eos') is True
      and type(body.get('min_tokens')) is int and body['min_tokens']==1
      and type(body.get('max_tokens')) is int and body['max_tokens'] in (2,64)
      and type(body.get('n',1)) is int and body.get('n',1)==1
      and not body.get('echo') and not body.get('stop') and not body.get('logprobs') and not body.get('kv_transfer_params'))
    if not allowed:raise HTTPException(400,'Outside the explicit research contract; stock official service is on29880')
    producer={k:v for k,v in body.items() if k!='stream_options'};producer.update(max_tokens=1,stream=False,kv_transfer_params={'do_remote_decode':True,'do_remote_prefill':False,'remote_engine_id':None,'remote_block_ids':None,'remote_host':None,'remote_port':None,'_glm53_export_p_draft':mode!='control'})
    response=await client.post('http://172.16.12.32:29481/v1/completions',json=producer,headers={'X-Request-Id':rid});response.raise_for_status();p=response.json();params=p['kv_transfer_params'];first=params['last_token_id'];assert p['choices'][0]['token_ids']==[first] and params['remote_num_prompt_tokens']==len(prompt)
    packet=params.get('p_first_draft');valid=bool(isinstance(packet,dict) and packet.get('request_id')==params['remote_request_id'] and packet.get('rank_agreement')==16 and packet.get('num_prompt_tokens')==len(prompt) and packet.get('first_token_id')==first and len(packet.get('token_ids',[]))==1)
    effective=mode if mode=='control' or valid else 'control-export-fallback'
    first_text=p['choices'][0].get('text','');publish=effective=='publish-seed' and bool(first_text) and first_text.isascii()
    if effective=='publish-seed' and not publish:effective='seed-text-fallback'
    decoder={**body,'kv_transfer_params':{**params,'_glm53_adopt_first':mode!='control' and valid,'_glm53_use_p_draft':mode!='control' and valid}}
    record(kind='P_confirmed',request_id=rid,requested_mode=mode,effective_mode=effective,first_token_id=first,proposal_packet_valid=valid if mode!='control' else None,prompt_tokens=len(prompt))
    async def generate():
        dropped=False;wire_count=0
        def sse(value):return ('data: '+json.dumps(value,separators=(',',':'))+'\n\n').encode()
        try:
            if publish:
                record(kind='publish_confirmed_P_first',request_id=rid,token_id=first)
                wire_count+=1
                yield sse({'id':p['id'],'object':'text_completion','created':p['created'],'model':p['model'],'choices':[{'index':0,'text':first_text,'token_ids':[first],'finish_reason':None}],'usage':None})
            async with client.stream('POST','http://172.16.12.33:29482/v1/completions',json=decoder,headers={'X-Request-Id':rid}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith('data: '):continue
                    if line=='data: [DONE]':break
                    chunk=json.loads(line[6:]);assert not chunk.get('error'),chunk
                    for choice in chunk.get('choices',[]):
                        ids=choice.get('token_ids') or []
                        if publish and not dropped:
                            if ids:
                                assert ids[0]==first and choice.get('text','').startswith(first_text),'First D token/text differs from confirmed P prefix'
                                choice['token_ids']=ids[1:];choice['text']=choice['text'][len(first_text):];dropped=True
                                record(kind='strip_D_duplicate_confirmed_first',request_id=rid,first_D_chunk_ids=ids)
                            else:assert not choice.get('text'),'Unexpected buffered text before first D token IDs'
                        wire_count+=len(choice.get('token_ids') or [])
                    yield sse(chunk)
            assert wire_count==body['max_tokens'] and (not publish or dropped)
            record(kind='finished',request_id=rid,wire_count=wire_count)
            yield b'data: [DONE]\n\n'
        except Exception as exc:
            record(kind='stream_error',request_id=rid,error=type(exc).__name__+': '+str(exc))
            yield sse({'error':{'message':str(exc),'type':type(exc).__name__}})
    return StreamingResponse(generate(),media_type='text/event-stream',headers={'X-GLM53-Effective-Mode':effective})


if __name__=='__main__':uvicorn.run(app,host='172.16.12.32',port=29881,log_level='warning')
